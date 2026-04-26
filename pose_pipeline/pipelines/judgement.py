from __future__ import annotations

import importlib.util
import time
from functools import lru_cache
from typing import Any

import numpy as np

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


def run_pose_judgement(pose_data: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    original = _load_original_main()
    left = pose_data["left"]["poses_3d"]
    right = pose_data["right"]["poses_3d"]
    joint_names = pose_data["joint_names"]
    total_frame_count = min(left.shape[0], right.shape[0])

    fused = ((left[:total_frame_count] + right[:total_frame_count]) / 2.0).astype(float)
    source_view = []
    errors = []
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
        f"({frame_count}/{total_frame_count} frame, "
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
                regularization=bool(config.get("judgement_regularization", False)),
                regularization_lambda=float(config.get("judgement_regularization_lambda", 1.0)),
                soft_tail_temperature=float(
                    config.get("soft_tail_temperature", original.SOFT_TAIL_TEMPERATURE)
                ),
                soft_tail_weight=float(config.get("soft_tail_weight", original.SOFT_TAIL_WEIGHT)),
            )
            opt1 = result["optimized"]["camera1"]
            opt2 = result["optimized"]["camera2"]
            fused[frame_idx] = _joint_dicts_to_average_array(opt1, opt2, joint_names)
            source_view.append(
                {
                    "frame": frame_idx,
                    "M": result["M"],
                    "K1": result["K1"],
                    "K2": result["K2"],
                    "A": result["A"],
                    "F": result["F"],
                    "visibility_mode": "z_buffer_mesh" if verts_by_cam is not None else "all_visible_fallback",
                    "visible_camera1": result.get("visibility", {}).get("camera1"),
                    "visible_camera2": result.get("visibility", {}).get("camera2"),
                }
            )
        except Exception as exc:
            fused[frame_idx] = (left[frame_idx] + right[frame_idx]) / 2.0
            errors.append({"frame": frame_idx, "error": str(exc)})
            source_view.append({"frame": frame_idx, "fallback": "mean", "error": str(exc)})
            print(f"    J: frame {frame_idx + 1} loi, fallback mean: {exc}", flush=True)

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

    pose_data["fused"]["poses_3d"] = fused
    pose_data["fused"]["confidence"] = (
        pose_data["left"]["confidence"][:total_frame_count] + pose_data["right"]["confidence"][:total_frame_count]
    ) / 2.0
    pose_data["fused"]["source_view"] = source_view
    pose_data["fused"]["metadata"]["pose_judgement_optimization"] = {
        "source_file": str(POSE_JUDGEMENT_MAIN),
        "mode": "original_run_phase3_pipeline",
        "visibility_mode": "z_buffer_mesh" if z_buffer_frames else "all_visible_fallback",
        "z_buffer_frames": z_buffer_frames,
        "processed_frames": frame_count,
        "seconds": round(elapsed, 3),
        "seconds_per_frame": round(seconds_per_frame, 3),
        "progress_log": progress_log,
        "errors": errors[:20],
        "error_count": len(errors),
    }
    pose_data["logs"].append(
        "J: called original run_phase3_pipeline for "
        f"{frame_count} frames in {elapsed:.2f}s "
        f"({seconds_per_frame:.2f}s/frame, errors={len(errors)})"
    )
    return pose_data


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
    rows = []
    for name in joint_names:
        p1 = np.asarray(cam1[name], dtype=float)
        p2 = np.asarray(cam2[name], dtype=float)
        rows.append((p1 + p2) / 2.0)
    return np.asarray(rows, dtype=float)


def _verts_by_cam_for_frame(pose_data: dict[str, Any], frame_idx: int) -> dict[str, np.ndarray] | None:
    left = _frame_verts(pose_data["left"], frame_idx)
    right = _frame_verts(pose_data["right"], frame_idx)
    if left is None or right is None:
        return None
    return {"camera1": left, "camera2": right}


def _frame_verts(side_data: dict[str, Any], frame_idx: int) -> np.ndarray | None:
    raw_person = side_data.get("raw_person")
    if not isinstance(raw_person, dict):
        return None

    for key in ("verts_cam", "verts"):
        value = raw_person.get(key)
        if value is None:
            continue
        verts = np.asarray(value, dtype=float)
        if verts.ndim == 3 and frame_idx < verts.shape[0] and verts.shape[2] == 3:
            return verts[frame_idx]
    return None
