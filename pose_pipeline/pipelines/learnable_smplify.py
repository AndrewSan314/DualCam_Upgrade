from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from pose_pipeline.naming import make_dual_output_paths, make_unified_output_path
from pose_pipeline.schema import pose_data_view_to_standard_schema, save_pose_pkl
from pose_pipeline.state import PipelineState
from pose_pipeline.vendor_paths import (
    LEARNABLE_CHECKPOINT,
    LEARNABLE_MODEL_SRC,
    LEARNABLE_ROOT,
    LEARNABLE_SMPL_DIR,
)


def run_L(state: PipelineState, config: dict[str, Any]) -> PipelineState:
    if state.pose_data is None:
        raise ValueError("PipelineState.pose_data is required for L")
    if state.output_dir is None:
        raise ValueError("PipelineState.output_dir is required for L")
    if state.mode not in {"dual", "unified"}:
        raise ValueError(f"Unsupported state mode for L: {state.mode}")

    new_history = state.history + ["L"]
    state.pose_data = run_learnable_smplify(state.pose_data, config)
    if state.mode == "dual" and state.pose_data["fused"].get("poses_3d") is None:
        left_path, right_path = make_dual_output_paths(state.output_dir, new_history)
        save_pose_pkl(
            pose_data_view_to_standard_schema(
                state.pose_data,
                "left",
                new_history,
                created_by="pose_pipeline.pipelines.learnable_smplify.run_L",
            ),
            left_path,
        )
        save_pose_pkl(
            pose_data_view_to_standard_schema(
                state.pose_data,
                "right",
                new_history,
                created_by="pose_pipeline.pipelines.learnable_smplify.run_L",
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
                created_by="pose_pipeline.pipelines.learnable_smplify.run_L",
            ),
            output_path,
        )
        state.mode = "unified"
        state.unified_pkl = output_path
        state.artifacts["".join(new_history)] = output_path

    state.history = new_history
    return state


def run_learnable_smplify(pose_data: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(LEARNABLE_ROOT))
    try:
        from inference import InferenceConfig, LearnableInverseKinematicSolver
        from pkl_io import PoseSequence

        solver = LearnableInverseKinematicSolver(
            InferenceConfig(
                checkpoint=_optional_existing_path(
                    config.get("learnable_checkpoint") or LEARNABLE_CHECKPOINT
                ),
                learnable_smplify_src=_optional_existing_path(
                    config.get("learnable_smplify_src") or LEARNABLE_MODEL_SRC
                ),
                smpl_model_dir=_optional_existing_path(
                    config.get("smpl_model_dir") or LEARNABLE_SMPL_DIR
                ),
                device=str(config.get("device", "auto")),
                fallback_refiner=str(config.get("fallback_refiner", "smooth")),
                smooth_window=int(config.get("smooth_window", 3)),
            )
        )

        side_has_raw_smpl = {
            side: _has_raw_smpl_params(pose_data[side].get("raw_person"))
            for side in ("left", "right")
        }
        fused_has_raw_smpl = _has_raw_smpl_params(
            pose_data.get("fused", {}).get("raw_person")
        )

        if any(side_has_raw_smpl.values()):
            for side in ("left", "right"):
                sequence = _array_to_pose_sequence(
                    pose_data[side]["poses_3d"],
                    pose_data["joint_names"],
                    Path(pose_data[side]["source_path"]),
                    PoseSequence,
                    raw_person=pose_data[side].get("raw_person"),
                    raw_data=pose_data[side].get("raw_data"),
                )
                refined = solver.refine(sequence)
                pose_data[side]["poses_3d"] = _pose_sequence_to_array(refined, pose_data["joint_names"])
                pose_data[side]["metadata"]["learnable_smplify"] = _metadata(
                    solver,
                    side,
                    has_raw_smpl=side_has_raw_smpl[side],
                    refined_metadata=refined.metadata,
                )

            fused_update = _update_fused_after_side_refine(pose_data, config)
            pose_data["fused"]["metadata"]["learnable_smplify"] = _fused_from_sides_metadata(
                solver,
                side_has_raw_smpl=side_has_raw_smpl,
                fused_update=fused_update,
            )
            pose_data["logs"].append(
                "L: ran vendored LearnableInverseKinematicSolver on left/right raw SMPL "
                f"(fused_update={fused_update})"
            )
        elif fused_has_raw_smpl and pose_data["fused"]["poses_3d"] is not None:
            sequence = _array_to_pose_sequence(
                pose_data["fused"]["poses_3d"],
                pose_data["joint_names"],
                Path(pose_data["input_dir"]) / "fused.pkl",
                PoseSequence,
                raw_person=pose_data["fused"].get("raw_person"),
                raw_data=pose_data["fused"].get("raw_data"),
            )
            refined = solver.refine(sequence)
            pose_data["fused"]["poses_3d"] = _pose_sequence_to_array(refined, pose_data["joint_names"])
            pose_data["fused"]["metadata"]["learnable_smplify"] = _metadata(
                solver,
                "fused",
                has_raw_smpl=True,
                refined_metadata=refined.metadata,
            )
            pose_data["logs"].append("L: used vendored LearnableInverseKinematicSolver on fused raw SMPL")
        else:
            pose_data["fused"]["metadata"]["learnable_smplify"] = _metadata(
                solver,
                "skipped",
                has_raw_smpl=False,
                note=(
                    "Learnable-SMPLify requires raw SMPL pose/betas/trans. "
                    "No valid left/right or fused raw_person payload was available."
                ),
                refined_metadata={"refined_by": "none"},
            )
            pose_data["logs"].append("L: skipped Learnable-SMPLify; no valid raw SMPL payload")

        return pose_data
    finally:
        try:
            sys.path.remove(str(LEARNABLE_ROOT))
        except ValueError:
            pass


def _array_to_pose_sequence(
    poses: np.ndarray,
    joint_names: list[str],
    source_path: Path,
    pose_sequence_cls: Any,
    *,
    raw_person: Any,
    raw_data: Any,
):
    frames = {}
    for frame_idx, frame in enumerate(np.asarray(poses, dtype=float)):
        frames[frame_idx] = {
            name: np.asarray(frame[joint_idx], dtype=float)
            for joint_idx, name in enumerate(joint_names)
        }
    return pose_sequence_cls(
        source_path=source_path,
        frames=frames,
        metadata={"representation": "pose_pipeline_array"},
        raw_data=raw_data,
        raw_person=raw_person,
    )


def _pose_sequence_to_array(sequence: Any, joint_names: list[str]) -> np.ndarray:
    rows = []
    for frame_id in sequence.frame_ids:
        frame = sequence.frames[frame_id]
        rows.append([np.asarray(frame[name], dtype=float) for name in joint_names])
    return np.asarray(rows, dtype=float)


def _metadata(
    solver: Any,
    target: str,
    *,
    has_raw_smpl: bool,
    refined_metadata: dict[str, Any] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "source_root": str(LEARNABLE_ROOT),
        "source_class": "inference.LearnableInverseKinematicSolver",
        "target": target,
        "has_raw_smpl_params": bool(has_raw_smpl),
        "model_loaded": bool(solver.is_model_loaded),
        "load_error": solver.load_error,
        "refined_by": (refined_metadata or {}).get("refined_by"),
        "status": _status(solver, refined_metadata),
        "note": note,
    }


def _optional_existing_path(value: Any) -> str | None:
    if value in (None, ""):
        return None
    path = Path(value)
    return str(path) if path.exists() else None


def _update_fused_after_side_refine(
    pose_data: dict[str, Any], config: dict[str, Any]
) -> str:
    fused = pose_data["fused"]
    if fused.get("poses_3d") is None:
        return "not_present_kept_dual"

    update_mode = str(config.get("learnable_fused_update", "auto")).lower()
    has_judgement_pose = "pose_judgement_optimization" in fused.get("metadata", {})
    if update_mode == "off":
        return "preserved_existing_fused_pose"

    left = np.asarray(pose_data["left"]["poses_3d"], dtype=np.float32)
    right = np.asarray(pose_data["right"]["poses_3d"], dtype=np.float32)
    frame_count = min(left.shape[0], right.shape[0])
    existing_fused = np.asarray(fused["poses_3d"][:frame_count], dtype=np.float32)
    if update_mode not in {"auto", "average", "judgement_weights"}:
        raise ValueError(f"Unsupported learnable_fused_update: {update_mode}")

    weights = _judgement_view_weights(fused, frame_count)
    if update_mode in {"auto", "judgement_weights"} and weights is not None:
        denom = np.maximum(weights["left"] + weights["right"] + weights["base"], 1e-8)
        fused["poses_3d"] = (
            weights["left"][..., None] * left[:frame_count]
            + weights["right"][..., None] * right[:frame_count]
            + weights["base"][..., None] * existing_fused
        ) / denom[..., None]
        fused["poses_3d"] = fused["poses_3d"].astype(np.float32)
        update_status = "recomputed_from_left_right_with_judgement_weights"
    elif update_mode == "judgement_weights":
        return "preserved_existing_fused_pose_missing_judgement_weights"
    else:
        fused["poses_3d"] = ((left[:frame_count] + right[:frame_count]) / 2.0).astype(np.float32)
        update_status = (
            "recomputed_from_left_right_average_after_judgement"
            if has_judgement_pose
            else "recomputed_from_left_right"
        )

    left_conf = pose_data["left"].get("confidence")
    right_conf = pose_data["right"].get("confidence")
    if left_conf is not None and right_conf is not None:
        fused["confidence"] = (
            (
                np.asarray(left_conf[:frame_count], dtype=np.float32)
                + np.asarray(right_conf[:frame_count], dtype=np.float32)
            )
            / 2.0
        ).astype(np.float32)

    return update_status


def _judgement_view_weights(
    fused: dict[str, Any],
    frame_count: int,
) -> dict[str, np.ndarray] | None:
    metadata = fused.get("metadata", {}).get("pose_judgement_optimization", {})
    diagnostics = metadata.get("diagnostics", {})
    view_weights = diagnostics.get("view_weights") if isinstance(diagnostics, dict) else None
    if not isinstance(view_weights, dict):
        return None

    pose = np.asarray(fused.get("poses_3d"), dtype=np.float32)
    if pose.ndim != 3 or pose.shape[0] < frame_count:
        return None
    joint_count = pose.shape[1]
    weights = {}
    for key in ("left", "right", "base"):
        value = view_weights.get(key)
        if value is None:
            value = view_weights.get(f"{key}_mean")
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            arr = np.full((frame_count, joint_count), float(arr), dtype=np.float32)
        elif arr.ndim == 1:
            if arr.shape[0] == frame_count:
                arr = np.repeat(arr[:, None], joint_count, axis=1)
            elif arr.shape[0] == joint_count:
                arr = np.repeat(arr[None, :], frame_count, axis=0)
            else:
                return None
        elif arr.ndim == 2:
            if arr.shape[0] < frame_count or arr.shape[1] < joint_count:
                return None
            arr = arr[:frame_count, :joint_count]
        else:
            return None
        weights[key] = np.clip(arr, 0.0, 1.0).astype(np.float32)
    return weights


def _fused_from_sides_metadata(
    solver: Any,
    *,
    side_has_raw_smpl: dict[str, bool],
    fused_update: str,
) -> dict[str, Any]:
    return {
        "source_root": str(LEARNABLE_ROOT),
        "source_class": "inference.LearnableInverseKinematicSolver",
        "target": "left_right",
        "side_has_raw_smpl_params": dict(side_has_raw_smpl),
        "has_raw_smpl_params": any(side_has_raw_smpl.values()),
        "model_loaded": bool(solver.is_model_loaded),
        "load_error": solver.load_error,
        "refined_by": "left_right_raw_smpl",
        "status": fused_update,
        "note": (
            "Learnable-SMPLify ran on left/right raw SMPL payloads. "
            "Fused pose is recomputed only when it is not an existing J result."
        ),
    }


def _has_raw_smpl_params(raw_person: Any) -> bool:
    if not isinstance(raw_person, dict):
        return False
    return ("pose" in raw_person or "poses" in raw_person) and "betas" in raw_person and "trans" in raw_person


def _status(solver: Any, refined_metadata: dict[str, Any] | None) -> str:
    if solver.is_model_loaded and (refined_metadata or {}).get("refined_by") == "learnable_smplify":
        return "learnable_smplify_model"
    if (refined_metadata or {}).get("refined_by") == "temporal_smoothing_fallback":
        return "fallback_temporal_smoothing"
    if (refined_metadata or {}).get("refined_by") == "none":
        return "not_refined"
    return "unknown"
