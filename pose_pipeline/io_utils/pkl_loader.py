from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pose_pipeline.config import BODY25_JOINT_NAMES, PROJECT_ROOT, SMPL24_JOINT_NAMES


EXPLICIT_JOINT_KEYS = (
    "joints_3d",
    "keypoints_3d",
    "pred_joints",
    "smpl_joints",
    "joints",
)


def load_wham_pkl(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Missing PKL file: {source}")

    raw = _load_pickle(source)
    person = _select_person(raw)
    frame_ids = _extract_frame_ids(person)
    poses_3d, joint_names, representation = _extract_pose_array(person)

    if frame_ids is None:
        frame_ids = np.arange(poses_3d.shape[0], dtype=int)
    frame_ids = np.asarray(frame_ids).reshape(-1)
    frame_count = min(len(frame_ids), poses_3d.shape[0])

    poses_3d = np.asarray(poses_3d[:frame_count], dtype=np.float32)
    poses_3d, coordinate_transform = canonicalize_pose_coordinates(poses_3d, joint_names)
    confidence = np.ones(poses_3d.shape[:2], dtype=np.float32)
    poses_2d = _extract_2d_keypoints(person, frame_count)
    if poses_2d is not None and poses_2d.shape[-1] >= 3:
        confidence = poses_2d[:, : poses_3d.shape[1], 2]

    return {
        "source_path": str(source.resolve()),
        "poses_3d": poses_3d,
        "poses_2d": poses_2d,
        "confidence": confidence,
        "joint_names": joint_names,
        "frame_ids": frame_ids[:frame_count].astype(int).tolist(),
        "metadata": {
            "representation": representation,
            "frame_count": int(frame_count),
            "joint_count": int(poses_3d.shape[1]),
            "coordinate_transform": coordinate_transform,
        },
        "raw_data": raw,
        "raw_person": person,
        "raw_person_keys": sorted(str(k) for k in person.keys()) if isinstance(person, Mapping) else [],
    }


def _load_pickle(path: Path) -> Any:
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        with path.open("rb") as handle:
            return pickle.load(handle)


def _select_person(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping):
                return item
    if isinstance(payload, Mapping):
        if _looks_like_person(payload):
            return payload
        for key in sorted(payload.keys(), key=lambda value: str(value)):
            item = payload[key]
            if isinstance(item, Mapping):
                return item
    raise ValueError(f"Unsupported PKL structure: {type(payload)!r}")


def _looks_like_person(payload: Mapping[str, Any]) -> bool:
    keys = set(payload.keys())
    return bool(keys & {"verts", "verts_cam", "pose", "poses", *EXPLICIT_JOINT_KEYS})


def _extract_frame_ids(person: Mapping[str, Any]) -> np.ndarray | None:
    for key in ("frame_ids", "frame_id"):
        if key in person:
            ids = np.asarray(person[key]).reshape(-1)
            if ids.size:
                return ids.astype(int)
    return None


def _extract_pose_array(person: Mapping[str, Any]) -> tuple[np.ndarray, list[str], str]:
    for key in EXPLICIT_JOINT_KEYS:
        if key not in person:
            continue
        arr = np.asarray(person[key], dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            return arr, _joint_names_for_count(arr.shape[1]), f"explicit:{key}"

    verts_key = "verts" if "verts" in person else "verts_cam" if "verts_cam" in person else None
    if verts_key:
        verts = np.asarray(person[verts_key], dtype=np.float32)
        if verts.ndim != 3 or verts.shape[-1] != 3:
            raise ValueError(f"Unexpected {verts_key} shape: {verts.shape}")
        regressor = _load_body25_regressor()
        if regressor.shape[1] != verts.shape[1]:
            raise ValueError(
                f"Regressor and verts do not match: reg={regressor.shape}, verts={verts.shape}"
            )
        joints = np.einsum("jv,fvc->fjc", regressor, verts)
        return joints, BODY25_JOINT_NAMES[: joints.shape[1]], f"{verts_key}:J_regressor_body25"

    pose_key = "pose" if "pose" in person else "poses" if "poses" in person else None
    if pose_key:
        poses = np.asarray(person[pose_key], dtype=np.float32)
        if poses.ndim != 2:
            raise ValueError(f"Unexpected {pose_key} shape: {poses.shape}")
        trans = np.asarray(person.get("trans", np.zeros((len(poses), 3))), dtype=np.float32)
        joints = _surrogate_joints_from_smpl_pose(poses, trans)
        return joints, SMPL24_JOINT_NAMES, "smpl_pose_surrogate"

    raise ValueError("PKL does not contain supported 3D joints, verts, or SMPL pose arrays.")


def _load_body25_regressor() -> np.ndarray:
    candidates = [
        PROJECT_ROOT / "assets" / "J_regressor_body25.npy",
        Path(__file__).resolve().parents[3]
        / "learnable-simplify-for-inverse-kinematic-main"
        / "learnable-simplify-for-inverse-kinematic-main"
        / "smpl"
        / "J_regressor_body25.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            reg = np.load(candidate).astype(np.float32, copy=False)
            if reg.ndim != 2:
                raise ValueError(f"Unexpected regressor shape: {reg.shape}")
            return reg
    raise FileNotFoundError(
        "Missing assets/J_regressor_body25.npy. Copy it into pose_pipeline_project/assets/."
    )


def _joint_names_for_count(count: int) -> list[str]:
    if count == len(BODY25_JOINT_NAMES):
        return BODY25_JOINT_NAMES.copy()
    if count == len(SMPL24_JOINT_NAMES):
        return SMPL24_JOINT_NAMES.copy()
    return [f"joint_{idx:02d}" for idx in range(count)]


def _extract_2d_keypoints(person: Mapping[str, Any], frame_count: int) -> np.ndarray | None:
    track = person.get("tracking_results_for_reproj")
    if isinstance(track, Mapping) and "keypoints" in track:
        arr = np.asarray(track["keypoints"], dtype=float)
        if arr.ndim == 3:
            return arr[:frame_count]
    for key in ("keypoints_2d", "joints_2d"):
        if key in person:
            arr = np.asarray(person[key], dtype=float)
            if arr.ndim == 3:
                return arr[:frame_count]
    return None


def canonicalize_pose_coordinates(
    poses_3d: np.ndarray, joint_names: list[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert camera/world pose variants to x-right, y-up, z-depth coordinates."""
    poses = np.asarray(poses_3d, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[-1] != 3 or poses.shape[1] == 0:
        return poses, {"applied": False, "reason": "unsupported_shape"}

    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    vertical_axis, vertical_sign = _infer_vertical_axis(poses, name_to_idx)
    horizontal_axis, horizontal_sign = _infer_horizontal_axis(
        poses, name_to_idx, vertical_axis
    )
    depth_axis = next(axis for axis in range(3) if axis not in {horizontal_axis, vertical_axis})

    canonical = np.empty_like(poses, dtype=np.float32)
    canonical[..., 0] = poses[..., horizontal_axis] * horizontal_sign
    canonical[..., 1] = poses[..., vertical_axis] * vertical_sign
    canonical[..., 2] = poses[..., depth_axis]
    return canonical, {
        "applied": True,
        "source_axes": {
            "x": int(horizontal_axis),
            "y": int(vertical_axis),
            "z": int(depth_axis),
        },
        "signs": {
            "x": float(horizontal_sign),
            "y": float(vertical_sign),
            "z": 1.0,
        },
    }


def _infer_vertical_axis(
    poses: np.ndarray, name_to_idx: dict[str, int]
) -> tuple[int, float]:
    upper = _joint_center(
        poses,
        name_to_idx,
        ("nose", "head", "neck", "left_shoulder", "right_shoulder"),
    )
    lower = _joint_center(
        poses,
        name_to_idx,
        (
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot",
            "right_foot",
            "mid_hip",
            "pelvis",
        ),
    )
    if upper is not None and lower is not None:
        delta = _nanmedian(lower - upper)
        axis = int(np.nanargmax(np.abs(delta)))
        sign = -1.0 if delta[axis] > 0 else 1.0
        return axis, sign

    spans = _nanmedian(np.nanmax(poses, axis=1) - np.nanmin(poses, axis=1))
    axis = int(np.nanargmax(spans))
    return axis, 1.0


def _infer_horizontal_axis(
    poses: np.ndarray, name_to_idx: dict[str, int], vertical_axis: int
) -> tuple[int, float]:
    left = _joint_center(poses, name_to_idx, ("left_shoulder", "left_hip"))
    right = _joint_center(poses, name_to_idx, ("right_shoulder", "right_hip"))
    remaining_axes = [axis for axis in range(3) if axis != vertical_axis]
    if left is not None and right is not None:
        delta = _nanmedian(left - right)
        axis = max(remaining_axes, key=lambda candidate: abs(delta[candidate]))
        sign = 1.0 if delta[axis] >= 0 else -1.0
        return int(axis), sign

    spans = _nanmedian(np.nanmax(poses, axis=1) - np.nanmin(poses, axis=1))
    axis = max(remaining_axes, key=lambda candidate: spans[candidate])
    return int(axis), 1.0


def _joint_center(
    poses: np.ndarray, name_to_idx: dict[str, int], names: tuple[str, ...]
) -> np.ndarray | None:
    indices = [name_to_idx[name] for name in names if name in name_to_idx]
    if not indices:
        return None
    return np.nanmean(poses[:, indices, :], axis=1)


def _nanmedian(values: np.ndarray) -> np.ndarray:
    return np.nanmedian(np.asarray(values, dtype=np.float32), axis=0)


def _surrogate_joints_from_smpl_pose(poses: np.ndarray, trans: np.ndarray) -> np.ndarray:
    offsets = np.asarray(
        [
            [0.000, 0.000, 0.000],
            [0.090, -0.090, 0.000],
            [-0.090, -0.090, 0.000],
            [0.000, 0.120, 0.000],
            [0.000, -0.430, 0.020],
            [0.000, -0.430, 0.020],
            [0.000, 0.120, 0.000],
            [0.000, -0.420, -0.010],
            [0.000, -0.420, -0.010],
            [0.000, 0.120, 0.000],
            [0.000, -0.080, 0.120],
            [0.000, -0.080, 0.120],
            [0.000, 0.130, 0.000],
            [0.070, 0.040, 0.000],
            [-0.070, 0.040, 0.000],
            [0.000, 0.160, 0.020],
            [0.150, 0.020, 0.000],
            [-0.150, 0.020, 0.000],
            [0.280, 0.000, 0.000],
            [-0.280, 0.000, 0.000],
            [0.250, 0.000, 0.000],
            [-0.250, 0.000, 0.000],
            [0.080, 0.000, 0.000],
            [-0.080, 0.000, 0.000],
        ],
        dtype=float,
    )
    if trans.ndim == 1:
        trans = np.repeat(trans[np.newaxis, :], len(poses), axis=0)
    if trans.shape[0] < len(poses):
        pad = np.repeat(trans[-1:], len(poses) - trans.shape[0], axis=0)
        trans = np.vstack([trans, pad])
    return offsets[np.newaxis, :, :] + trans[: len(poses), np.newaxis, :3]
