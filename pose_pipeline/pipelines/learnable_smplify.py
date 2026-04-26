from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from pose_pipeline.vendor_paths import (
    LEARNABLE_CHECKPOINT,
    LEARNABLE_MODEL_SRC,
    LEARNABLE_ROOT,
    LEARNABLE_SMPL_DIR,
)


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

        target = "fused" if pose_data["fused"]["poses_3d"] is not None else "left_right"
        if target == "fused":
            sequence = _array_to_pose_sequence(
                pose_data["fused"]["poses_3d"],
                pose_data["joint_names"],
                Path(pose_data["input_dir"]) / "fused.pkl",
                PoseSequence,
                raw_person=None,
                raw_data=None,
            )
            refined = solver.refine(sequence)
            pose_data["fused"]["poses_3d"] = _pose_sequence_to_array(refined, pose_data["joint_names"])
            pose_data["fused"]["metadata"]["learnable_smplify"] = _metadata(
                solver,
                "fused",
                has_raw_smpl=False,
                note="Fused pose has no raw SMPL pose/betas/trans; model inference cannot run on this target yet.",
                refined_metadata=refined.metadata,
            )
        else:
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
                    has_raw_smpl=_has_raw_smpl_params(pose_data[side].get("raw_person")),
                    refined_metadata=refined.metadata,
                )

        pose_data["logs"].append("L: used vendored LearnableInverseKinematicSolver")
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
