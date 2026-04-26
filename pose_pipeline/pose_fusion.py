from __future__ import annotations

from typing import Any

import numpy as np


STABLE_ALIGNMENT_JOINTS = (
    "neck",
    "mid_hip",
    "pelvis",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)


def fuse_aligned_poses(
    left: np.ndarray, right: np.ndarray, joint_names: list[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    frame_count = min(left.shape[0], right.shape[0])
    joint_count = min(left.shape[1], right.shape[1], len(joint_names))
    left = np.asarray(left[:frame_count, :joint_count], dtype=np.float32)
    right = np.asarray(right[:frame_count, :joint_count], dtype=np.float32)
    indices = _alignment_indices(joint_names[:joint_count])

    fused = np.empty_like(left, dtype=np.float32)
    rms_before = []
    rms_after = []
    aligned_frames = 0
    fallback_frames = 0

    for frame_idx in range(frame_count):
        aligned, stats = align_pose_frame(right[frame_idx], left[frame_idx], indices)
        fused[frame_idx] = (left[frame_idx] + aligned) / 2.0
        if stats["aligned"]:
            aligned_frames += 1
            rms_before.append(stats["rms_before"])
            rms_after.append(stats["rms_after"])
        else:
            fallback_frames += 1

    return fused, {
        "mode": "right_to_left_rigid_kabsch",
        "reference": "left",
        "alignment_joint_count": len(indices),
        "aligned_frames": aligned_frames,
        "fallback_frames": fallback_frames,
        "mean_rms_before": _mean_or_none(rms_before),
        "mean_rms_after": _mean_or_none(rms_after),
    }


def fuse_aligned_frames(
    left_frame: np.ndarray, right_frame: np.ndarray, joint_names: list[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    joint_count = min(left_frame.shape[0], right_frame.shape[0], len(joint_names))
    left = np.asarray(left_frame[:joint_count], dtype=np.float32)
    right = np.asarray(right_frame[:joint_count], dtype=np.float32)
    aligned, stats = align_pose_frame(right, left, _alignment_indices(joint_names[:joint_count]))
    return ((left + aligned) / 2.0).astype(np.float32), stats


def align_pose_frame(
    moving: np.ndarray, reference: np.ndarray, indices: list[int]
) -> tuple[np.ndarray, dict[str, Any]]:
    moving = np.asarray(moving, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    valid = [
        idx
        for idx in indices
        if idx < len(moving)
        and idx < len(reference)
        and np.isfinite(moving[idx]).all()
        and np.isfinite(reference[idx]).all()
    ]
    if len(valid) < 3:
        return moving, {"aligned": False, "reason": "fewer_than_3_valid_joints"}

    mov = moving[valid]
    ref = reference[valid]
    mov_center = mov.mean(axis=0)
    ref_center = ref.mean(axis=0)
    mov0 = mov - mov_center
    ref0 = ref - ref_center

    try:
        u, _singular_values, vt = np.linalg.svd(mov0.T @ ref0)
    except np.linalg.LinAlgError:
        return moving, {"aligned": False, "reason": "svd_failed"}

    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T

    aligned = (moving - mov_center) @ rotation + ref_center
    rms_before = _rms(mov - ref)
    rms_after = _rms(aligned[valid] - ref)
    return aligned.astype(np.float32), {
        "aligned": True,
        "valid_joint_count": len(valid),
        "rms_before": float(rms_before),
        "rms_after": float(rms_after),
    }


def _alignment_indices(joint_names: list[str]) -> list[int]:
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    indices = [name_to_idx[name] for name in STABLE_ALIGNMENT_JOINTS if name in name_to_idx]
    if len(indices) >= 3:
        return indices
    return list(range(len(joint_names)))


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.asarray(values, dtype=np.float32) ** 2, axis=-1))))


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))
