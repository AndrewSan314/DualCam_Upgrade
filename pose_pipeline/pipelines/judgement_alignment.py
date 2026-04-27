from __future__ import annotations

from typing import Any

import numpy as np


def align_candidates_to_base(
    left_candidate: np.ndarray,
    right_candidate: np.ndarray,
    base_pose: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    method = str(config.get("judgement_coordinate_alignment", "sequence_umeyama"))
    if method == "none":
        return left_candidate, right_candidate, {"method": method}
    if method == "root_scale":
        return (
            sequence_scale_to_base(left_candidate, base_pose),
            sequence_scale_to_base(right_candidate, base_pose),
            {"method": method},
        )
    if method == "sequence_umeyama":
        left, left_diag = align_sequence_umeyama(left_candidate, base_pose)
        right, right_diag = align_sequence_umeyama(right_candidate, base_pose)
        return left, right, {"method": method, "left": left_diag, "right": right_diag}
    raise ValueError(f"Unsupported judgement alignment method: {method}")


def root_center_pose(pose: np.ndarray, root_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(pose, dtype=np.float32)
    root = arr[:, root_idx : root_idx + 1, :]
    return arr - root, root


def sequence_scale_to_base(
    candidate: np.ndarray, base: np.ndarray, root_idx: int = 0
) -> np.ndarray:
    base_centered, base_root = root_center_pose(base, root_idx)
    cand_centered, _ = root_center_pose(candidate, root_idx)
    base_norm = _mean_norm(base_centered)
    cand_norm = _mean_norm(cand_centered)
    if not np.isfinite(cand_norm) or cand_norm < 1e-8:
        return np.asarray(base, dtype=np.float32).copy()
    scale = base_norm / cand_norm
    return (cand_centered * scale + base_root).astype(np.float32)


def align_sequence_umeyama(
    candidate: np.ndarray, base: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    transform, diagnostics = estimate_sequence_umeyama(candidate, base)
    if transform is None:
        if diagnostics.get("fallback") == "root_scale":
            return sequence_scale_to_base(candidate, base), diagnostics
        return np.asarray(base, dtype=np.float32).copy(), diagnostics

    aligned = apply_similarity_transform(candidate, transform)
    return aligned.astype(np.float32), diagnostics


def estimate_sequence_umeyama(
    candidate: np.ndarray, base: np.ndarray
) -> tuple[dict[str, np.ndarray | float] | None, dict[str, Any]]:
    source = np.asarray(candidate, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(base, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[finite]
    target = target[finite]
    if source.shape[0] < 4:
        return None, {"fallback": "root_scale"}

    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean
    variance = np.mean(np.sum(src_centered**2, axis=1))
    if variance < 1e-12:
        return None, {"fallback": "base_zero_variance"}

    covariance = (src_centered.T @ tgt_centered) / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    scale = float(np.sum(singular_values) / variance)
    translation = tgt_mean - scale * (src_mean @ rotation.T)

    return {"scale": scale, "rotation": rotation, "translation": translation}, {
        "scale": scale,
        "rotation": rotation.astype(float).tolist(),
        "translation": translation.astype(float).tolist(),
        "valid_points": int(source.shape[0]),
    }


def apply_similarity_transform(
    points: np.ndarray,
    transform: dict[str, np.ndarray | float],
) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    rotation = np.asarray(transform["rotation"], dtype=np.float64)
    translation = np.asarray(transform["translation"], dtype=np.float64)
    scale = float(transform["scale"])
    return scale * (arr @ rotation.T) + translation


def _mean_norm(values: np.ndarray) -> float:
    norms = np.linalg.norm(values, axis=-1)
    return float(np.nanmean(norms))
