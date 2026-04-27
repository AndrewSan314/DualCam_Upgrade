from __future__ import annotations

import importlib.util
import time
from functools import lru_cache
from typing import Any

import numpy as np

from pose_pipeline.config import SKELETON_EDGES
from pose_pipeline.pipelines.judgement_alignment import align_candidates_to_base
from pose_pipeline.pipelines.judgement_optimizer import optimize_temporal_multiview_pose
from pose_pipeline.pipelines.judgement_validation import validate_or_fallback_sequence
from pose_pipeline.pipelines.judgement_weights import build_view_weights, selected_source
from pose_pipeline.naming import make_dual_output_paths, make_unified_output_path
from pose_pipeline.schema import pose_data_view_to_standard_schema, save_pose_pkl
from pose_pipeline.state import PipelineState
from pose_pipeline.vendor_paths import POSE_JUDGEMENT_MAIN


@lru_cache(maxsize=1)
def _load_original_main():
    spec = importlib.util.spec_from_file_location(
        "vendor_pose_judgement_optimization_main", POSE_JUDGEMENT_MAIN
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load original pose judgement file: {POSE_JUDGEMENT_MAIN}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_J(state: PipelineState, config: dict[str, Any]) -> PipelineState:
    if state.pose_data is None:
        raise ValueError("PipelineState.pose_data is required for J")
    if state.output_dir is None:
        raise ValueError("PipelineState.output_dir is required for J")
    if state.mode not in {"dual", "unified"}:
        raise ValueError(f"Unsupported state mode for J: {state.mode}")

    new_history = state.history + ["J"]
    output_mode = _output_mode_for_judgement(state, config)
    config = dict(config)
    config["__judgement_output_mode"] = output_mode
    state.pose_data = run_pose_judgement(state.pose_data, config)
    if output_mode == "dual":
        left_path, right_path = make_dual_output_paths(state.output_dir, new_history)
        save_pose_pkl(
            pose_data_view_to_standard_schema(
                state.pose_data,
                "left",
                new_history,
                created_by="pose_pipeline.pipelines.judgement.run_J",
            ),
            left_path,
        )
        save_pose_pkl(
            pose_data_view_to_standard_schema(
                state.pose_data,
                "right",
                new_history,
                created_by="pose_pipeline.pipelines.judgement.run_J",
            ),
            right_path,
        )
        state.mode = "dual"
        state.left_pkl = left_path
        state.right_pkl = right_path
        state.latest_left_pkl = left_path
        state.latest_right_pkl = right_path
        state.artifacts[f"{''.join(new_history)}_left"] = left_path
        state.artifacts[f"{''.join(new_history)}_right"] = right_path
    else:
        output_path = make_unified_output_path(state.output_dir, new_history)
        save_pose_pkl(
            pose_data_view_to_standard_schema(
                state.pose_data,
                "unified",
                new_history,
                created_by="pose_pipeline.pipelines.judgement.run_J",
            ),
            output_path,
        )
        state.mode = "unified"
        state.unified_pkl = output_path
        state.artifacts["".join(new_history)] = output_path

    state.history = new_history
    return state


def _output_mode_for_judgement(state: PipelineState, config: dict[str, Any]) -> str:
    if state.mode != "dual":
        return "unified"
    requested = str(config.get("judgement_output_mode_when_dual", "auto")).lower()
    if requested in {"dual", "unified"}:
        return requested
    if requested != "auto":
        raise ValueError(f"Unsupported judgement_output_mode_when_dual: {requested}")
    remaining = str(config.get("__remaining_sequence", ""))
    return "dual" if "R" in remaining else "unified"


def run_pose_judgement(pose_data: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    original = _load_original_main()
    left = pose_data["left"]["poses_3d"]
    right = pose_data["right"]["poses_3d"]
    joint_names = pose_data["joint_names"]
    total_frame_count = min(left.shape[0], right.shape[0])

    fused_prior = _initial_fused_pose(pose_data, left, right, total_frame_count)
    fused = fused_prior.copy()
    judgement_mode = str(config.get("judgement_mode", "safe_fusion"))
    output_mode = str(config.get("__judgement_output_mode", "unified"))
    pose_update_mode = str(
        config.get("judgement_pose_update")
        or ("full" if judgement_mode == "full_legacy" else "off")
    )
    blend_alpha = float(config.get("judgement_blend_alpha", 0.2))
    max_joint_shift_m = float(config.get("judgement_max_joint_shift_m", 0.15))
    left_candidate = np.asarray(left[:total_frame_count], dtype=np.float32).copy()
    right_candidate = np.asarray(right[:total_frame_count], dtype=np.float32).copy()
    source_view = []
    errors = []
    pose_update_stats = []
    z_buffer_frames = 0
    max_frames = config.get("max_judgement_frames")
    frame_count = total_frame_count
    if max_frames is not None:
        frame_count = min(frame_count, int(max_frames))
    log_interval = int(config.get("judgement_log_interval") or 10)
    progress_log = []
    start = time.perf_counter()

    print(
        "    J: bat dau Pose Judgement "
        f"mode={judgement_mode}, {frame_count}/{total_frame_count} frame, "
        f"log moi {log_interval} frame).",
        flush=True,
    )

    for frame_idx in range(frame_count):
        frame_start = time.perf_counter()
        cam1 = _frame_to_joint_dict(left[frame_idx], joint_names)
        cam2 = _frame_to_joint_dict(right[frame_idx], joint_names)
        verts_by_cam = _verts_by_cam_for_frame(pose_data, frame_idx)
        if verts_by_cam is not None:
            z_buffer_frames += 1
        try:
            result = original.run_phase3_pipeline(
                {"camera1": cam1, "camera2": cam2},
                verts_by_cam=verts_by_cam,
                occlusion_grid=int(config.get("occlusion_grid", 160)),
                occlusion_tau=float(config.get("occlusion_tau", 0.02)),
                regularization=bool(config.get("judgement_regularization", True)),
                regularization_lambda=float(config.get("judgement_regularization_lambda", 10.0)),
                max_joint_move_m=float(
                    config.get("judgement_vendor_max_joint_move_m", 0.10)
                ),
                reject_excessive_displacement=bool(
                    config.get("judgement_vendor_reject_excessive_displacement", True)
                ),
                soft_tail_temperature=float(
                    config.get("soft_tail_temperature", original.SOFT_TAIL_TEMPERATURE)
                ),
                soft_tail_weight=float(config.get("soft_tail_weight", original.SOFT_TAIL_WEIGHT)),
            )
            opt1 = result["optimized"]["camera1"]
            opt2 = result["optimized"]["camera2"]
            left_candidate[frame_idx] = _joint_dict_to_array(opt1, joint_names)
            right_candidate[frame_idx] = _joint_dict_to_array(opt2, joint_names)
            legacy_candidate = _joint_dicts_to_average_array(opt1, opt2, joint_names)
            if judgement_mode == "full_legacy":
                fused[frame_idx], update_stat = _apply_pose_update(
                    fused_prior[frame_idx],
                    legacy_candidate,
                    mode=pose_update_mode,
                    blend_alpha=blend_alpha,
                    max_joint_shift_m=max_joint_shift_m,
                )
            else:
                update_stat = _candidate_shift_stat(fused_prior[frame_idx], legacy_candidate)
            pose_update_stats.append(update_stat)
            source_view.append(
                {
                    "frame": frame_idx,
                    "M": result["M"],
                    "K1": result["K1"],
                    "K2": result["K2"],
                    "A": result["A"],
                    "A_new": result.get("A_new"),
                    "observed_anchors": result.get("observed_anchors", result["A"]),
                    "imputed_joints": result.get("imputed_joints", []),
                    "F": result["F"],
                    "visibility_mode": "z_buffer_mesh" if verts_by_cam is not None else "all_visible_fallback",
                    "visible_camera1": result.get("visibility", {}).get("camera1"),
                    "visible_camera2": result.get("visibility", {}).get("camera2"),
                    "displacement": result.get("displacement"),
                    "attempted_displacement": result.get("attempted_displacement"),
                    "rejected_by_displacement": result.get("rejected_by_displacement", False),
                    "warnings": result.get("warnings", []),
                }
            )
        except Exception as exc:
            fused[frame_idx] = fused_prior[frame_idx]
            errors.append({"frame": frame_idx, "error": str(exc)})
            source_view.append({"frame": frame_idx, "fallback": "prior_pose", "error": str(exc)})
            print(f"    J: frame {frame_idx + 1} loi, fallback prior pose: {exc}", flush=True)

        if _should_log_progress(frame_idx, frame_count, log_interval):
            elapsed = time.perf_counter() - start
            processed = frame_idx + 1
            seconds_per_frame = elapsed / processed
            eta = seconds_per_frame * (frame_count - processed)
            frame_seconds = time.perf_counter() - frame_start
            progress = {
                "processed_frames": processed,
                "total_frames": frame_count,
                "elapsed_seconds": round(elapsed, 3),
                "seconds_per_frame": round(seconds_per_frame, 3),
                "last_frame_seconds": round(frame_seconds, 3),
                "eta_seconds": round(eta, 3),
                "error_count": len(errors),
            }
            progress_log.append(progress)
            print(
                "    J: "
                f"{processed}/{frame_count} frame, "
                f"{seconds_per_frame:.2f}s/frame, "
                f"ETA {_format_duration(eta)}, "
                f"loi {len(errors)}",
                flush=True,
            )

    elapsed = time.perf_counter() - start
    seconds_per_frame = elapsed / frame_count if frame_count else 0.0
    print(
        "    J: hoan tat "
        f"{frame_count} frame trong {_format_duration(elapsed)} "
        f"({seconds_per_frame:.2f}s/frame, loi {len(errors)}).",
        flush=True,
    )
    if frame_count < total_frame_count:
        left_candidate[frame_count:] = fused_prior[frame_count:]
        right_candidate[frame_count:] = fused_prior[frame_count:]

    skeleton_edges = _skeleton_edges_for_joint_names(joint_names)
    foot_indices = _foot_indices_for_joint_names(joint_names)
    judgement_diagnostics: dict[str, Any] = {
        "mode": judgement_mode,
        "candidate_frames": frame_count,
        "total_frames": total_frame_count,
    }
    if judgement_mode == "metadata_only":
        pose_final = fused_prior
        validation = {"accepted": True, "reasons": [], "mode": "metadata_only"}
        confidence_final = _average_confidence(pose_data, total_frame_count)
        selected = np.full(fused_prior.shape[:2], 2, dtype=np.int8)
    elif judgement_mode == "full_legacy":
        pose_final, validation = validate_or_fallback_sequence(
            fused_prior,
            fused,
            {"mode": "full_legacy"},
            skeleton_edges,
            config,
        )
        confidence_final = _average_confidence(pose_data, total_frame_count)
        selected = np.full(fused_prior.shape[:2], 2, dtype=np.int8)
    else:
        left_aligned, right_aligned, alignment_diag = align_candidates_to_base(
            left_candidate, right_candidate, fused_prior, config
        )
        view_weights, weights_diag = build_view_weights(
            pose_data,
            source_view,
            left_aligned,
            right_aligned,
            fused_prior,
            joint_names,
            config,
        )
        if judgement_mode == "safe_fusion":
            pose_judged = _safe_weighted_fusion(
                fused_prior,
                left_aligned,
                right_aligned,
                view_weights,
                skeleton_edges=skeleton_edges,
                max_joint_shift_m=float(
                    config.get("judgement_safe_max_joint_shift_m", max_joint_shift_m)
                ),
                max_bone_deviation_ratio=float(
                    config.get("judgement_max_bone_deviation_ratio", 0.2)
                ),
            )
            pose_judged, smoothing_diag = _smooth_judgement_pose(
                pose_judged,
                fused_prior,
                skeleton_edges,
                config,
            )
            optimizer_diag = {
                "mode": "safe_weighted_fusion",
                "temporal_smoothing": smoothing_diag,
            }
        elif judgement_mode == "temporal_multiview_optimize":
            pose_judged, optimizer_diag = optimize_temporal_multiview_pose(
                fused_prior,
                left_aligned,
                right_aligned,
                view_weights,
                skeleton_edges,
                foot_indices,
                config,
            )
            pose_judged, smoothing_diag = _smooth_judgement_pose(
                pose_judged,
                fused_prior,
                skeleton_edges,
                config,
            )
            optimizer_diag["temporal_smoothing"] = smoothing_diag
        else:
            raise ValueError(f"Unsupported judgement mode: {judgement_mode}")

        judgement_diagnostics.update(
            {
                "alignment": alignment_diag,
                "view_weights": weights_diag,
                "optimizer": optimizer_diag,
            }
        )
        pose_final, validation = validate_or_fallback_sequence(
            fused_prior,
            pose_judged,
            judgement_diagnostics,
            skeleton_edges,
            config,
        )
        confidence_final = np.clip(
            1.0 - view_weights["base"] + 0.5 * view_weights["base"], 0.0, 1.0
        ).astype(np.float32)
        selected = selected_source(view_weights)

    judgement_metadata = {
        "source_file": str(POSE_JUDGEMENT_MAIN),
        "mode": judgement_mode,
        "output_mode": output_mode,
        "visibility_mode": "z_buffer_mesh" if z_buffer_frames else "all_visible_fallback",
        "z_buffer_frames": z_buffer_frames,
        "processed_frames": frame_count,
        "seconds": round(elapsed, 3),
        "seconds_per_frame": round(seconds_per_frame, 3),
        "progress_log": progress_log,
        "pose_update": {
            "mode": pose_update_mode,
            "blend_alpha": blend_alpha,
            "max_joint_shift_m": max_joint_shift_m,
            "stats": _summarize_pose_updates(pose_update_stats),
            "note": "Only full_legacy writes per-frame averaged candidates directly.",
        },
        "selected_source_counts": _selected_source_counts(selected),
        "validation": validation,
        "diagnostics": judgement_diagnostics,
        "errors": errors[:20],
        "error_count": len(errors),
    }
    if output_mode == "dual":
        if validation.get("accepted"):
            pose_data["left"]["poses_3d"] = left_candidate.astype(np.float32)
            pose_data["right"]["poses_3d"] = right_candidate.astype(np.float32)
        pose_data["left"]["metadata"]["pose_judgement_optimization"] = {
            **judgement_metadata,
            "target": "left",
            "applied_candidates": bool(validation.get("accepted")),
        }
        pose_data["right"]["metadata"]["pose_judgement_optimization"] = {
            **judgement_metadata,
            "target": "right",
            "applied_candidates": bool(validation.get("accepted")),
        }
        pose_data["fused"]["poses_3d"] = None
        pose_data["fused"]["confidence"] = None
        pose_data["fused"]["source_view"] = source_view
        pose_data["fused"]["metadata"]["pose_judgement_optimization"] = judgement_metadata
    else:
        pose_data["fused"]["poses_3d"] = pose_final
        pose_data["fused"]["confidence"] = (
            confidence_final
        )
        pose_data["fused"]["source_view"] = source_view
        pose_data["fused"]["metadata"]["pose_judgement_optimization"] = judgement_metadata
    pose_data["logs"].append(
        f"J: mode={judgement_mode}, called original run_phase3_pipeline for "
        f"{frame_count} frames in {elapsed:.2f}s "
        f"({seconds_per_frame:.2f}s/frame, errors={len(errors)}, "
        f"accepted={validation.get('accepted')}, output={output_mode})"
    )
    return pose_data


def _initial_fused_pose(
    pose_data: dict[str, Any],
    left: np.ndarray,
    right: np.ndarray,
    total_frame_count: int,
) -> np.ndarray:
    existing = pose_data.get("fused", {}).get("poses_3d")
    if existing is not None:
        existing = np.asarray(existing, dtype=float)
        if existing.ndim == 3 and existing.shape[0] >= total_frame_count:
            return existing[:total_frame_count].copy()
    return ((left[:total_frame_count] + right[:total_frame_count]) / 2.0).astype(float)


def _apply_pose_update(
    prior: np.ndarray,
    candidate: np.ndarray,
    *,
    mode: str,
    blend_alpha: float,
    max_joint_shift_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    mode = str(mode or "off").lower()
    prior = np.asarray(prior, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    delta = candidate - prior
    shift = np.linalg.norm(delta, axis=1)
    max_shift = float(np.nanmax(shift)) if shift.size else 0.0
    mean_shift = float(np.nanmean(shift)) if shift.size else 0.0

    if mode == "full":
        return candidate, {"mode": mode, "mean_shift_m": mean_shift, "max_shift_m": max_shift}
    if mode == "blend":
        alpha = float(np.clip(blend_alpha, 0.0, 1.0))
        bounded_delta = _limit_joint_shift(delta, max_joint_shift_m)
        updated = prior + alpha * bounded_delta
        applied_shift = np.linalg.norm(updated - prior, axis=1)
        return updated, {
            "mode": mode,
            "alpha": alpha,
            "mean_shift_m": mean_shift,
            "max_shift_m": max_shift,
            "mean_applied_shift_m": float(np.nanmean(applied_shift)),
            "max_applied_shift_m": float(np.nanmax(applied_shift)),
        }
    return prior, {"mode": "off", "mean_shift_m": mean_shift, "max_shift_m": max_shift}


def _limit_joint_shift(delta: np.ndarray, max_joint_shift_m: float) -> np.ndarray:
    if max_joint_shift_m <= 0:
        return np.zeros_like(delta)
    norms = np.linalg.norm(delta, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_joint_shift_m / np.maximum(norms, 1e-8))
    return delta * scale


def _summarize_pose_updates(stats: list[dict[str, Any]]) -> dict[str, Any]:
    if not stats:
        return {"processed_frames": 0}
    return {
        "processed_frames": len(stats),
        "mode": stats[-1].get("mode", "unknown"),
        "mean_candidate_shift_m": _mean_stat(stats, "mean_shift_m"),
        "max_candidate_shift_m": _max_stat(stats, "max_shift_m"),
        "mean_applied_shift_m": _mean_stat(stats, "mean_applied_shift_m"),
        "max_applied_shift_m": _max_stat(stats, "max_applied_shift_m"),
    }


def _mean_stat(stats: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in stats if key in item]
    if not values:
        return None
    return float(np.mean(values))


def _max_stat(stats: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in stats if key in item]
    if not values:
        return None
    return float(np.max(values))


def _should_log_progress(frame_idx: int, frame_count: int, interval: int) -> bool:
    processed = frame_idx + 1
    if processed in (1, frame_count):
        return True
    return interval > 0 and processed % interval == 0


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def _frame_to_joint_dict(frame: np.ndarray, joint_names: list[str]) -> dict[str, np.ndarray]:
    return {name: np.asarray(frame[idx], dtype=float) for idx, name in enumerate(joint_names)}


def _joint_dicts_to_average_array(
    cam1: dict[str, Any], cam2: dict[str, Any], joint_names: list[str]
) -> np.ndarray:
    return (_joint_dict_to_array(cam1, joint_names) + _joint_dict_to_array(cam2, joint_names)) / 2.0


def _joint_dict_to_array(cam: dict[str, Any], joint_names: list[str]) -> np.ndarray:
    rows = []
    for name in joint_names:
        rows.append(np.asarray(cam[name], dtype=np.float32))
    return np.asarray(rows, dtype=np.float32)


def _candidate_shift_stat(prior: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(candidate, dtype=float) - np.asarray(prior, dtype=float)
    shift = np.linalg.norm(delta, axis=1)
    return {
        "mode": "candidate_only",
        "mean_shift_m": float(np.nanmean(shift)) if shift.size else 0.0,
        "max_shift_m": float(np.nanmax(shift)) if shift.size else 0.0,
    }


def _safe_weighted_fusion(
    base: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    weights: dict[str, np.ndarray],
    *,
    skeleton_edges: list[tuple[int, int]],
    max_joint_shift_m: float,
    max_bone_deviation_ratio: float,
) -> np.ndarray:
    denom = np.maximum(weights["left"] + weights["right"] + weights["base"], 1e-8)
    fused = (
        weights["left"][..., None] * left
        + weights["right"][..., None] * right
        + weights["base"][..., None] * base
    ) / denom[..., None]
    delta = _limit_sequence_shift(fused - base, max_joint_shift_m)
    guarded = _fallback_bad_bones_to_base(
        base + delta,
        base,
        skeleton_edges,
        max_bone_deviation_ratio=max_bone_deviation_ratio,
    )
    return guarded.astype(np.float32)


def _limit_sequence_shift(delta: np.ndarray, max_joint_shift_m: float) -> np.ndarray:
    if max_joint_shift_m <= 0:
        return np.zeros_like(delta)
    norms = np.linalg.norm(delta, axis=2, keepdims=True)
    scale = np.minimum(1.0, max_joint_shift_m / np.maximum(norms, 1e-8))
    return delta * scale


def _fallback_bad_bones_to_base(
    candidate: np.ndarray,
    base: np.ndarray,
    skeleton_edges: list[tuple[int, int]],
    *,
    max_bone_deviation_ratio: float,
) -> np.ndarray:
    if not skeleton_edges:
        return candidate
    guarded = np.asarray(candidate, dtype=np.float32).copy()
    base_arr = np.asarray(base, dtype=np.float32)
    for _ in range(4):
        changed = False
        for a, b in skeleton_edges:
            cand_len = np.linalg.norm(guarded[:, a, :] - guarded[:, b, :], axis=-1)
            base_len = np.linalg.norm(base_arr[:, a, :] - base_arr[:, b, :], axis=-1)
            ratio = np.abs(cand_len - base_len) / np.maximum(base_len, 1e-8)
            bad = ratio > max_bone_deviation_ratio
            if np.any(bad):
                guarded[bad, a, :] = base_arr[bad, a, :]
                guarded[bad, b, :] = base_arr[bad, b, :]
                changed = True
        if not changed:
            break
    return guarded


def _smooth_judgement_pose(
    pose: np.ndarray,
    base: np.ndarray,
    skeleton_edges: list[tuple[int, int]],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    window = int(config.get("judgement_temporal_smoothing_window", 5))
    alpha = float(config.get("judgement_temporal_smoothing_alpha", 0.65))
    if window <= 1 or alpha <= 0.0:
        return np.asarray(pose, dtype=np.float32), {"applied": False}
    if window % 2 == 0:
        window += 1

    pose_arr = np.asarray(pose, dtype=np.float32)
    base_arr = np.asarray(base, dtype=np.float32)
    smoothing_target = str(config.get("judgement_temporal_smoothing_target", "correction"))
    median_window = int(config.get("judgement_temporal_median_window", 3))
    if smoothing_target == "correction":
        correction = pose_arr - base_arr
        if median_window > 1:
            correction = _moving_median_pose(correction, median_window)
        smoothed = base_arr + _moving_average_pose(correction, window)
    elif smoothing_target == "pose":
        source = pose_arr
        if median_window > 1:
            source = _moving_median_pose(source, median_window)
        smoothed = _moving_average_pose(source, window)
    else:
        raise ValueError(f"Unsupported judgement smoothing target: {smoothing_target}")

    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = (1.0 - alpha) * pose_arr + alpha * smoothed
    max_shift = float(config.get("judgement_safe_max_joint_shift_m", 0.12))
    limited = base_arr + _limit_sequence_shift(
        blended - base_arr,
        max_shift,
    )
    guarded = _fallback_bad_bones_to_base(
        limited,
        base,
        skeleton_edges,
        max_bone_deviation_ratio=float(config.get("judgement_max_bone_deviation_ratio", 0.2)),
    )
    repaired, spike_diag = _repair_acceleration_spikes(
        guarded,
        base,
        skeleton_edges,
        config,
    )
    return repaired.astype(np.float32), {
        "applied": True,
        "window": window,
        "alpha": alpha,
        "target": smoothing_target,
        "median_window": median_window if median_window > 1 else None,
        "spike_repair": spike_diag,
    }


def _moving_average_pose(pose: np.ndarray, window: int) -> np.ndarray:
    pad = window // 2
    padded = np.pad(pose, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    out = np.empty_like(pose, dtype=np.float32)
    for frame_idx in range(pose.shape[0]):
        out[frame_idx] = np.tensordot(kernel, padded[frame_idx : frame_idx + window], axes=(0, 0))
    return out


def _moving_median_pose(pose: np.ndarray, window: int) -> np.ndarray:
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(np.asarray(pose, dtype=np.float32), ((pad, pad), (0, 0), (0, 0)), mode="edge")
    out = np.empty_like(pose, dtype=np.float32)
    for frame_idx in range(pose.shape[0]):
        out[frame_idx] = np.median(padded[frame_idx : frame_idx + window], axis=0)
    return out


def _repair_acceleration_spikes(
    pose: np.ndarray,
    base: np.ndarray,
    skeleton_edges: list[tuple[int, int]],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if not bool(config.get("judgement_spike_repair", True)):
        return np.asarray(pose, dtype=np.float32), {"applied": False}

    repaired = np.asarray(pose, dtype=np.float32).copy()
    base_arr = np.asarray(base, dtype=np.float32)
    if repaired.shape[0] < 3:
        return repaired, {"applied": False, "reason": "too_few_frames"}

    ratio = float(config.get("judgement_spike_acceleration_ratio_vs_base", 1.8))
    absolute = float(config.get("judgement_spike_acceleration_m_per_frame2", 0.12))
    alpha = float(np.clip(config.get("judgement_spike_repair_alpha", 0.85), 0.0, 1.0))
    max_shift = float(config.get("judgement_safe_max_joint_shift_m", 0.12))
    max_bone_deviation = float(config.get("judgement_max_bone_deviation_ratio", 0.2))
    passes = max(1, int(config.get("judgement_spike_repair_passes", 2)))

    total_repaired = 0
    worst_before = 0.0
    worst_after = 0.0
    for _ in range(passes):
        pose_acc = np.linalg.norm(repaired[2:] - 2 * repaired[1:-1] + repaired[:-2], axis=-1)
        base_acc = np.linalg.norm(base_arr[2:] - 2 * base_arr[1:-1] + base_arr[:-2], axis=-1)
        threshold = np.maximum(absolute, base_acc * ratio)
        bad = pose_acc > threshold
        if not np.any(bad):
            worst_before = max(worst_before, float(np.max(pose_acc)) if pose_acc.size else 0.0)
            break

        worst_before = max(worst_before, float(np.max(pose_acc)))
        frames, joints = np.where(bad)
        for acc_frame, joint_idx in zip(frames, joints):
            frame_idx = int(acc_frame) + 1
            joint_idx = int(joint_idx)
            interpolated = 0.5 * (repaired[frame_idx - 1, joint_idx] + repaired[frame_idx + 1, joint_idx])
            repaired[frame_idx, joint_idx] = (
                (1.0 - alpha) * repaired[frame_idx, joint_idx] + alpha * interpolated
            )
        repaired = base_arr + _limit_sequence_shift(repaired - base_arr, max_shift)
        repaired = _fallback_bad_bones_to_base(
            repaired,
            base_arr,
            skeleton_edges,
            max_bone_deviation_ratio=max_bone_deviation,
        )
        total_repaired += int(len(frames))

    final_acc = np.linalg.norm(repaired[2:] - 2 * repaired[1:-1] + repaired[:-2], axis=-1)
    worst_after = float(np.max(final_acc)) if final_acc.size else 0.0
    return repaired.astype(np.float32), {
        "applied": True,
        "repaired_points": total_repaired,
        "ratio_vs_base": ratio,
        "absolute_m_per_frame2": absolute,
        "alpha": alpha,
        "passes": passes,
        "max_acceleration_before": worst_before,
        "max_acceleration_after": worst_after,
    }


def _average_confidence(pose_data: dict[str, Any], frame_count: int) -> np.ndarray:
    return (
        pose_data["left"]["confidence"][:frame_count] + pose_data["right"]["confidence"][:frame_count]
    ) / 2.0


def _skeleton_edges_for_joint_names(joint_names: list[str]) -> list[tuple[int, int]]:
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    return [
        (name_to_idx[start], name_to_idx[end])
        for start, end in SKELETON_EDGES
        if start in name_to_idx and end in name_to_idx
    ]


def _foot_indices_for_joint_names(joint_names: list[str]) -> list[int]:
    foot_names = (
        "left_ankle",
        "right_ankle",
        "left_foot",
        "right_foot",
        "left_heel",
        "right_heel",
        "left_big_toe",
        "right_big_toe",
    )
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    return [name_to_idx[name] for name in foot_names if name in name_to_idx]


def _selected_source_counts(selected: np.ndarray) -> dict[str, int]:
    labels = {0: "left", 1: "right", 2: "base"}
    values, counts = np.unique(selected, return_counts=True)
    return {labels.get(int(value), str(int(value))): int(count) for value, count in zip(values, counts)}


def _verts_by_cam_for_frame(pose_data: dict[str, Any], frame_idx: int) -> dict[str, dict[str, Any]] | None:
    left = _frame_mesh_payload(pose_data["left"], frame_idx)
    right = _frame_mesh_payload(pose_data["right"], frame_idx)
    if left is None or right is None:
        return None
    return {"camera1": left, "camera2": right}


def _frame_mesh_payload(side_data: dict[str, Any], frame_idx: int) -> dict[str, Any] | None:
    raw_person = side_data.get("raw_person")
    if not isinstance(raw_person, dict):
        return None

    for key in ("verts_cam", "verts"):
        value = raw_person.get(key)
        if value is None:
            continue
        verts = np.asarray(value, dtype=float)
        if verts.ndim == 3 and frame_idx < verts.shape[0] and verts.shape[2] == 3:
            return {
                "vertices": verts[frame_idx],
                "faces": _mesh_faces(raw_person),
                "camera": _visibility_camera(side_data),
                "image_size": _image_size(side_data),
            }
    return None


def _mesh_faces(raw_person: dict[str, Any]) -> np.ndarray | None:
    faces = raw_person.get("faces")
    if faces is None:
        return None
    arr = np.asarray(faces, dtype=int)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    return None


def _visibility_camera(side_data: dict[str, Any]) -> dict[str, Any] | None:
    intrinsics = side_data.get("camera_intrinsics")
    if not isinstance(intrinsics, dict):
        return None
    if all(key in intrinsics for key in ("fx", "fy", "cx", "cy")):
        return {
            "K": np.asarray(
                [
                    [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
                    [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
        }
    matrix = intrinsics.get("K") or intrinsics.get("intrinsics") or intrinsics.get("camera_matrix")
    if matrix is not None:
        arr = np.asarray(matrix, dtype=float)
        if arr.shape == (3, 3):
            return {"K": arr}
    return None


def _image_size(side_data: dict[str, Any]) -> tuple[int, int] | None:
    video_info = side_data.get("video_info")
    if not isinstance(video_info, dict):
        return None
    width = video_info.get("width")
    height = video_info.get("height")
    if width and height:
        return int(width), int(height)
    return None
