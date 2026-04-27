from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pose_pipeline.config import SKELETON_EDGES
from pose_pipeline.io_utils.pkl_loader import load_wham_pkl


def load_pose_pkl(path: Path) -> dict[str, Any]:
    source = Path(path)
    try:
        with source.open("rb") as handle:
            data = pickle.load(handle)
    except Exception:
        loaded = load_wham_pkl(source)
        standard = wham_loaded_to_standard_schema(loaded, view="unknown", history=[])
        validate_pose_schema(standard)
        return standard
    if isinstance(data, Mapping) and "poses_3d" in data and "source" in data:
        standard = dict(data)
    else:
        loaded = load_wham_pkl(source)
        standard = wham_loaded_to_standard_schema(loaded, view="unknown", history=[])
    validate_pose_schema(standard)
    return standard


def save_pose_pkl(data: dict[str, Any], path: Path) -> None:
    validate_pose_schema(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def validate_pose_schema(data: dict[str, Any]) -> None:
    poses_3d = np.asarray(data.get("poses_3d"))
    if poses_3d.ndim != 3 or poses_3d.shape[-1] != 3:
        raise ValueError(f"poses_3d must have shape [T, J, 3], got {poses_3d.shape}")
    if not np.isfinite(poses_3d).all():
        raise ValueError("poses_3d contains NaN or Inf")
    joint_names = data.get("joint_names")
    if not isinstance(joint_names, list) or len(joint_names) != poses_3d.shape[1]:
        raise ValueError("joint_names must match poses_3d joint dimension")
    confidence = data.get("confidence")
    if confidence is not None and np.asarray(confidence).shape[:2] != poses_3d.shape[:2]:
        raise ValueError("confidence must have shape [T, J]")


def pose_data_view_to_standard_schema(
    pose_data: dict[str, Any],
    view: str,
    history: list[str],
    *,
    created_by: str,
) -> dict[str, Any]:
    if view == "unified":
        view_data = pose_data["fused"]
        source_path = None
    else:
        view_data = pose_data[view]
        source_path = view_data.get("source_path")

    poses_3d = np.asarray(view_data["poses_3d"], dtype=np.float32)
    confidence = view_data.get("confidence")
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float32)

    joint_names = list(pose_data["joint_names"])
    return {
        "poses_3d": poses_3d,
        "poses_2d": _optional_array(view_data.get("poses_2d")),
        "confidence": confidence,
        "smpl_params": _extract_smpl_params(view_data.get("raw_person")),
        "camera": {
            "intrinsics": view_data.get("camera_intrinsics"),
            "video_info": view_data.get("video_info"),
        },
        "joint_names": joint_names,
        "skeleton_edges": _edge_indices(joint_names),
        "source": {
            "view": view,
            "pipeline_history": list(history),
            "input_files": [str(source_path)] if source_path else [],
        },
        "metadata": {
            "fps": _fps(view_data),
            "num_frames": int(poses_3d.shape[0]),
            "coordinate_system": "x-right_y-up_z-depth",
            "created_by": created_by,
            "extra": dict(view_data.get("metadata", {})),
        },
    }


def wham_loaded_to_standard_schema(
    loaded: dict[str, Any], *, view: str, history: list[str]
) -> dict[str, Any]:
    joint_names = list(loaded["joint_names"])
    return {
        "poses_3d": np.asarray(loaded["poses_3d"], dtype=np.float32),
        "poses_2d": _optional_array(loaded.get("poses_2d")),
        "confidence": _optional_array(loaded.get("confidence")),
        "smpl_params": _extract_smpl_params(loaded.get("raw_person")),
        "camera": None,
        "joint_names": joint_names,
        "skeleton_edges": _edge_indices(joint_names),
        "source": {
            "view": view,
            "pipeline_history": list(history),
            "input_files": [str(loaded.get("source_path"))],
        },
        "metadata": {
            "fps": None,
            "num_frames": int(np.asarray(loaded["poses_3d"]).shape[0]),
            "coordinate_system": "x-right_y-up_z-depth",
            "created_by": "pose_pipeline.io_utils.pkl_loader",
            "extra": dict(loaded.get("metadata", {})),
        },
    }


def convert_wham_to_standard_schema(raw_data: dict[str, Any], view: str) -> dict[str, Any]:
    loaded = dict(raw_data)
    return wham_loaded_to_standard_schema(loaded, view=view, history=[])


def convert_standard_to_opencap_input(data: dict[str, Any]) -> dict[str, Any]:
    validate_pose_schema(data)
    return data


def convert_opencap_output_to_standard(data: dict[str, Any]) -> dict[str, Any]:
    validate_pose_schema(data)
    return data


def convert_standard_to_learnable_smplify_input(data: dict[str, Any]) -> dict[str, Any]:
    validate_pose_schema(data)
    return data


def convert_learnable_smplify_output_to_standard(data: dict[str, Any]) -> dict[str, Any]:
    validate_pose_schema(data)
    return data


def _optional_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)


def _extract_smpl_params(raw_person: Any) -> dict[str, Any] | None:
    if not isinstance(raw_person, Mapping):
        return None
    params = {}
    for key in ("pose", "poses", "pose_world", "trans", "trans_world", "betas"):
        if key in raw_person:
            params[key] = raw_person[key]
    return params or None


def _edge_indices(joint_names: list[str]) -> list[tuple[int, int]]:
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    edges = []
    for start, end in SKELETON_EDGES:
        if start in name_to_idx and end in name_to_idx:
            edges.append((name_to_idx[start], name_to_idx[end]))
    return edges


def _fps(view_data: dict[str, Any]) -> float | None:
    video_info = view_data.get("video_info")
    if isinstance(video_info, Mapping) and video_info.get("fps") is not None:
        return float(video_info["fps"])
    return None
