from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ANGLE_SPECS = {
    "waveform_analysis_left_arm_wrist.png": ("left_shoulder", "left_elbow", ("left_wrist", "left_hand")),
    "waveform_analysis_right_arm_wrist.png": ("right_shoulder", "right_elbow", ("right_wrist", "right_hand")),
    "waveform_analysis_left_thigh_lower_leg.png": ("left_hip", "left_knee", ("left_ankle",)),
    "waveform_analysis_right_thigh_lower_leg.png": ("right_hip", "right_knee", ("right_ankle",)),
}


def draw_waveform_analysis(poses: np.ndarray, joint_names: list[str], output_dir: str | Path) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, spec in ANGLE_SPECS.items():
        values = _angle_series(poses, joint_names, spec)
        path = out / filename
        plt.figure(figsize=(10, 4))
        plt.plot(values)
        plt.xlabel("Frame")
        plt.ylabel("Angle (deg)")
        plt.ylim(0, 180)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(path)
    return paths


def _angle_series(poses: np.ndarray, joint_names: list[str], spec: tuple[str, str, tuple[str, ...]]) -> np.ndarray:
    a_name, center_name, end_candidates = spec
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    end_name = next((name for name in end_candidates if name in name_to_idx), None)
    if a_name not in name_to_idx or center_name not in name_to_idx or end_name is None:
        return np.zeros(poses.shape[0], dtype=float)
    a = poses[:, name_to_idx[a_name]]
    center = poses[:, name_to_idx[center_name]]
    b = poses[:, name_to_idx[end_name]]
    return angle_between(a - center, b - center)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    denom = np.maximum(denom, 1e-8)
    cos_theta = np.sum(v1 * v2, axis=1) / denom
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

