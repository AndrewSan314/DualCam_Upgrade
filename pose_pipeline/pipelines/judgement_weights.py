from __future__ import annotations

from typing import Any

import numpy as np


def build_view_weights(
    pose_data: dict[str, Any],
    source_view: list[dict[str, Any]],
    left_candidate: np.ndarray,
    right_candidate: np.ndarray,
    base_pose: np.ndarray,
    joint_names: list[str],
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    frame_count, joint_count = base_pose.shape[:2]
    left_conf = _confidence(pose_data["left"].get("confidence"), frame_count, joint_count)
    right_conf = _confidence(pose_data["right"].get("confidence"), frame_count, joint_count)
    left_visibility, right_visibility = _visibility_arrays(source_view, frame_count, joint_names)

    w_left, w_right = build_basic_view_weights(
        left_conf, right_conf, left_visibility, right_visibility
    )
    w_left, w_right, conflict_ratio = downweight_disagreement(
        left_candidate,
        right_candidate,
        w_left,
        w_right,
        threshold_m=float(config.get("judgement_camera_disagreement_threshold_m", 0.25)),
    )
    w_left_raw = w_left.copy()
    w_right_raw = w_right.copy()
    w_left, w_right, smoothing_diag = smooth_view_weights_temporally(
        w_left,
        w_right,
        window=int(config.get("judgement_weight_smoothing_window", 5)),
        alpha=float(config.get("judgement_weight_smoothing_alpha", 0.65)),
    )
    w_base = compute_base_prior_weight(
        w_left,
        w_right,
        min_prior=float(config.get("judgement_min_base_prior_weight", 0.2)),
        max_prior=float(config.get("judgement_max_base_prior_weight", 1.0)),
    )

    weights = {
        "left": w_left.astype(np.float32),
        "right": w_right.astype(np.float32),
        "base": w_base.astype(np.float32),
    }
    diagnostics = {
        "left_mean": float(np.mean(w_left)),
        "right_mean": float(np.mean(w_right)),
        "base_mean": float(np.mean(w_base)),
        "conflict_ratio": conflict_ratio,
        "weight_smoothing": smoothing_diag,
        "left_raw_mean": float(np.mean(w_left_raw)),
        "right_raw_mean": float(np.mean(w_right_raw)),
    }
    return weights, diagnostics


def build_basic_view_weights(
    left_conf: np.ndarray,
    right_conf: np.ndarray,
    left_visibility: np.ndarray,
    right_visibility: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    w_left = np.nan_to_num(left_conf * left_visibility, nan=0.0, posinf=0.0, neginf=0.0)
    w_right = np.nan_to_num(right_conf * right_visibility, nan=0.0, posinf=0.0, neginf=0.0)
    denom = np.maximum(w_left + w_right, 1e-8)
    return w_left / denom, w_right / denom


def downweight_disagreement(
    left: np.ndarray,
    right: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
    *,
    threshold_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    dist = np.linalg.norm(np.asarray(left) - np.asarray(right), axis=-1)
    conflict = dist > threshold_m
    left_out = w_left.copy()
    right_out = w_right.copy()
    left_out[conflict] *= 0.3
    right_out[conflict] *= 0.3
    return left_out, right_out, float(np.mean(conflict)) if conflict.size else 0.0


def compute_base_prior_weight(
    w_left: np.ndarray,
    w_right: np.ndarray,
    *,
    min_prior: float,
    max_prior: float,
) -> np.ndarray:
    evidence = np.clip(w_left + w_right, 0.0, 1.0)
    return max_prior - evidence * (max_prior - min_prior)


def smooth_view_weights_temporally(
    w_left: np.ndarray,
    w_right: np.ndarray,
    *,
    window: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if window <= 1 or alpha <= 0:
        return w_left, w_right, {"applied": False}
    if window % 2 == 0:
        window += 1
    alpha = float(np.clip(alpha, 0.0, 1.0))
    left_smooth = _moving_average_2d(w_left, window)
    right_smooth = _moving_average_2d(w_right, window)
    left_out = (1.0 - alpha) * w_left + alpha * left_smooth
    right_out = (1.0 - alpha) * w_right + alpha * right_smooth
    denom = np.maximum(left_out + right_out, 1e-8)
    evidence = np.clip(w_left + w_right, 0.0, 1.0)
    left_out = (left_out / denom) * evidence
    right_out = (right_out / denom) * evidence
    return left_out, right_out, {
        "applied": True,
        "window": window,
        "alpha": alpha,
        "mean_abs_left_delta": float(np.mean(np.abs(left_out - w_left))),
        "mean_abs_right_delta": float(np.mean(np.abs(right_out - w_right))),
    }


def _moving_average_2d(values: np.ndarray, window: int) -> np.ndarray:
    pad = window // 2
    padded = np.pad(np.asarray(values, dtype=np.float32), ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    out = np.empty_like(values, dtype=np.float32)
    for frame_idx in range(values.shape[0]):
        out[frame_idx] = np.tensordot(kernel, padded[frame_idx : frame_idx + window], axes=(0, 0))
    return out


def selected_source(weights: dict[str, np.ndarray]) -> np.ndarray:
    stacked = np.stack([weights["left"], weights["right"], weights["base"]], axis=-1)
    return np.argmax(stacked, axis=-1).astype(np.int8)


def _confidence(value: Any, frame_count: int, joint_count: int) -> np.ndarray:
    if value is None:
        return np.ones((frame_count, joint_count), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    out = np.ones((frame_count, joint_count), dtype=np.float32)
    rows = min(frame_count, arr.shape[0])
    cols = min(joint_count, arr.shape[1])
    out[:rows, :cols] = arr[:rows, :cols]
    return np.clip(out, 0.0, 1.0)


def _visibility_arrays(
    source_view: list[dict[str, Any]],
    frame_count: int,
    joint_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    left = np.ones((frame_count, len(joint_names)), dtype=np.float32)
    right = np.ones_like(left)
    for item in source_view:
        frame_idx = int(item.get("frame", -1))
        if frame_idx < 0 or frame_idx >= frame_count:
            continue
        left[frame_idx] = _visibility_to_row(item.get("visible_camera1"), joint_names)
        right[frame_idx] = _visibility_to_row(item.get("visible_camera2"), joint_names)
    return left, right


def _visibility_to_row(value: Any, joint_names: list[str]) -> np.ndarray:
    if value is None:
        return np.ones(len(joint_names), dtype=np.float32)
    if isinstance(value, dict):
        return np.asarray([float(bool(value.get(name, True))) for name in joint_names], dtype=np.float32)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return np.full(len(joint_names), float(bool(arr)), dtype=np.float32)
    row = np.ones(len(joint_names), dtype=np.float32)
    flat = arr.reshape(-1)
    count = min(len(row), len(flat))
    row[:count] = flat[:count].astype(np.float32)
    return np.clip(row, 0.0, 1.0)
