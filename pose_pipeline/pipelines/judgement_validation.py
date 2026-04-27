from __future__ import annotations

from typing import Any

import numpy as np


def validate_or_fallback_sequence(
    base_pose: np.ndarray,
    pose_judged: np.ndarray,
    diagnostics: dict[str, Any],
    skeleton_edges: list[tuple[int, int]],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    metrics_base = sequence_metrics(base_pose, base_pose, skeleton_edges)
    metrics_judged = sequence_metrics(pose_judged, base_pose, skeleton_edges)
    validation = {
        "accepted": True,
        "reasons": [],
        "base_metrics": metrics_base,
        "judged_metrics": metrics_judged,
    }

    max_bone_dev = metrics_judged["max_bone_deviation_ratio"]
    if max_bone_dev > float(config.get("judgement_max_bone_deviation_ratio", 0.2)):
        validation["accepted"] = False
        validation["reasons"].append(f"max bone deviation too high: {max_bone_dev:.4f}")

    max_velocity = metrics_judged["max_joint_velocity_m_per_frame"]
    if max_velocity > float(config.get("judgement_max_joint_velocity_m_per_frame", 0.35)):
        validation["accepted"] = False
        validation["reasons"].append(f"max joint velocity too high: {max_velocity:.4f}")

    max_acc = metrics_judged["max_joint_acceleration_m_per_frame2"]
    if max_acc > float(config.get("judgement_max_joint_acceleration_m_per_frame2", 0.45)):
        validation["accepted"] = False
        validation["reasons"].append(f"max acceleration too high: {max_acc:.4f}")

    mean_velocity_ratio = _ratio(
        metrics_judged["mean_joint_velocity_m_per_frame"],
        metrics_base["mean_joint_velocity_m_per_frame"],
    )
    max_velocity_ratio = _ratio(
        metrics_judged["max_joint_velocity_m_per_frame"],
        metrics_base["max_joint_velocity_m_per_frame"],
    )
    mean_acc_ratio = _ratio(
        metrics_judged["mean_joint_acceleration_m_per_frame2"],
        metrics_base["mean_joint_acceleration_m_per_frame2"],
    )
    max_acc_ratio = _ratio(
        metrics_judged["max_joint_acceleration_m_per_frame2"],
        metrics_base["max_joint_acceleration_m_per_frame2"],
    )
    validation["smoothness_ratios_vs_base"] = {
        "mean_velocity": mean_velocity_ratio,
        "max_velocity": max_velocity_ratio,
        "mean_acceleration": mean_acc_ratio,
        "max_acceleration": max_acc_ratio,
    }

    if mean_acc_ratio > float(config.get("judgement_max_mean_acceleration_ratio_vs_base", 1.35)):
        validation["accepted"] = False
        validation["reasons"].append(
            f"mean acceleration ratio vs base too high: {mean_acc_ratio:.4f}"
        )

    if max_acc_ratio > float(config.get("judgement_max_acceleration_ratio_vs_base", 2.25)):
        validation["accepted"] = False
        validation["reasons"].append(
            f"max acceleration ratio vs base too high: {max_acc_ratio:.4f}"
        )

    validation["diagnostics"] = diagnostics
    if validation["accepted"]:
        return np.asarray(pose_judged, dtype=np.float32), validation
    return np.asarray(base_pose, dtype=np.float32).copy(), validation


def sequence_metrics(
    pose: np.ndarray, base: np.ndarray, skeleton_edges: list[tuple[int, int]]
) -> dict[str, float]:
    bone_dev = compute_bone_deviation_ratio(pose, base, skeleton_edges)
    velocity = compute_joint_velocity(pose)
    acceleration = compute_joint_acceleration(pose)
    return {
        "mean_bone_deviation_ratio": _safe_stat(bone_dev, np.mean),
        "max_bone_deviation_ratio": _safe_stat(bone_dev, np.max),
        "mean_joint_velocity_m_per_frame": _safe_stat(velocity, np.mean),
        "max_joint_velocity_m_per_frame": _safe_stat(velocity, np.max),
        "mean_joint_acceleration_m_per_frame2": _safe_stat(acceleration, np.mean),
        "max_joint_acceleration_m_per_frame2": _safe_stat(acceleration, np.max),
    }


def compute_bone_deviation_ratio(
    pose: np.ndarray, base: np.ndarray, skeleton_edges: list[tuple[int, int]]
) -> np.ndarray:
    if not skeleton_edges:
        return np.zeros((1,), dtype=np.float32)
    deviations = []
    pose_arr = np.asarray(pose, dtype=np.float32)
    base_arr = np.asarray(base, dtype=np.float32)
    for a, b in skeleton_edges:
        len_pose = np.linalg.norm(pose_arr[:, a, :] - pose_arr[:, b, :], axis=-1)
        len_base = np.linalg.norm(base_arr[:, a, :] - base_arr[:, b, :], axis=-1)
        deviations.append(np.abs(len_pose - len_base) / np.maximum(len_base, 1e-8))
    return np.stack(deviations, axis=-1)


def compute_joint_velocity(pose: np.ndarray) -> np.ndarray:
    arr = np.asarray(pose, dtype=np.float32)
    if arr.shape[0] < 2:
        return np.zeros((1,), dtype=np.float32)
    return np.linalg.norm(arr[1:] - arr[:-1], axis=-1)


def compute_joint_acceleration(pose: np.ndarray) -> np.ndarray:
    arr = np.asarray(pose, dtype=np.float32)
    if arr.shape[0] < 3:
        return np.zeros((1,), dtype=np.float32)
    return np.linalg.norm(arr[2:] - 2 * arr[1:-1] + arr[:-2], axis=-1)


def _safe_stat(values: np.ndarray, fn: Any) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(fn(arr))


def _ratio(value: float, baseline: float) -> float:
    if baseline <= 1e-8:
        return 1.0 if value <= 1e-8 else float("inf")
    return float(value / baseline)
