from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


JointMap = Mapping[str, np.ndarray]


@dataclass(frozen=True)
class AnalysisSettings:
    orientation_epsilon: float = 1e-2
    distance_rule: str = "hybrid"
    mad_factor: float = 2.0


@dataclass
class FrameAnalysis:
    frame_id: int | str
    m_conflicts: List[str]
    d_distance_errors: List[str]
    q1: float | None
    q3: float | None
    mean: float | None
    threshold: float
    per_joint_errors: Dict[str, float] = field(default_factory=dict)
    pair_errors: List[float] = field(default_factory=list)
    orientation_flags_a: Dict[str, int] = field(default_factory=dict)
    orientation_flags_b: Dict[str, int] = field(default_factory=dict)

    @property
    def inlier_count(self) -> int:
        return len(
            [
                name
                for name in self.per_joint_errors
                if name not in self.m_conflicts and name not in self.d_distance_errors
            ]
        )


LANDMARK_ALIASES = {
    "right_shoulder": (
        "right_shoulder",
        "r_shoulder",
        "rightshoulder",
        "right_collar",
        "rshoulder",
    ),
    "left_shoulder": (
        "left_shoulder",
        "l_shoulder",
        "leftshoulder",
        "left_collar",
        "lshoulder",
    ),
    "right_hip": ("right_hip", "r_hip", "righthip"),
    "left_hip": ("left_hip", "l_hip", "lefthip"),
}


def analyze_frame(
    frame_id: int | str,
    joints_a: JointMap,
    joints_b: JointMap,
    settings: AnalysisSettings | None = None,
) -> FrameAnalysis:
    settings = settings or AnalysisSettings()
    common_names = sorted(set(joints_a).intersection(joints_b) - {"frame_id"})
    if not common_names:
        raise ValueError(f"Frame {frame_id!r} has no common joints to compare.")

    flags_a = get_orientation_flags(joints_a, settings.orientation_epsilon)
    flags_b = get_orientation_flags(joints_b, settings.orientation_epsilon)

    m_conflicts: List[str] = []
    stable_names: List[str] = []
    for name in common_names:
        fa = flags_a.get(name, 0)
        fb = flags_b.get(name, 0)
        if (fa == 1 and fb == -1) or (fa == -1 and fb == 1):
            m_conflicts.append(name)
        else:
            stable_names.append(name)

    per_joint_errors = _per_joint_geometry_errors(joints_a, joints_b, stable_names)
    threshold = _distance_threshold(list(per_joint_errors.values()), settings)
    d_distance_errors = [
        name
        for name, error in per_joint_errors.items()
        if threshold > 0.0 and error >= threshold
    ]

    inlier_names = [name for name in stable_names if name not in set(d_distance_errors)]
    pair_errors = _pair_distance_errors(joints_a, joints_b, inlier_names)
    q1, q3, mean = _summary(pair_errors)

    return FrameAnalysis(
        frame_id=frame_id,
        m_conflicts=sorted(m_conflicts),
        d_distance_errors=sorted(d_distance_errors),
        q1=q1,
        q3=q3,
        mean=mean,
        threshold=threshold,
        per_joint_errors=per_joint_errors,
        pair_errors=pair_errors,
        orientation_flags_a={name: flags_a.get(name, 0) for name in common_names},
        orientation_flags_b={name: flags_b.get(name, 0) for name in common_names},
    )


def analyze_sequences(
    frames_a: Mapping[int | str, JointMap],
    frames_b: Mapping[int | str, JointMap],
    settings: AnalysisSettings | None = None,
) -> Dict[int | str, FrameAnalysis]:
    settings = settings or AnalysisSettings()
    common_frame_ids = sort_frame_ids(set(frames_a).intersection(frames_b))
    return {
        frame_id: analyze_frame(
            frame_id, frames_a[frame_id], frames_b[frame_id], settings
        )
        for frame_id in common_frame_ids
    }


def get_orientation_flags(joints: JointMap, epsilon: float = 1e-2) -> Dict[str, int]:
    rs = _joint(joints, "right_shoulder")
    ls = _joint(joints, "left_shoulder")
    rh = _joint(joints, "right_hip")
    lh = _joint(joints, "left_hip")

    mid_shoulders = (rs + ls) / 2.0
    mid_hips = (rh + lh) / 2.0
    v_lr = rs - ls
    v_spine = mid_shoulders - mid_hips
    forward_vec = np.cross(v_lr, v_spine)
    norm = np.linalg.norm(forward_vec)
    if norm < 1e-12:
        raise ValueError(
            "Cannot determine body-facing direction: torso plane is degenerate."
        )
    forward_vec = forward_vec / norm
    torso_center = (mid_shoulders + mid_hips) / 2.0

    flags: Dict[str, int] = {}
    for name, pos in joints.items():
        dot_product = float(
            np.dot(np.asarray(pos, dtype=float) - torso_center, forward_vec)
        )
        if abs(dot_product) < epsilon:
            flags[name] = 0
        else:
            flags[name] = 1 if dot_product > 0 else -1
    return flags


def sort_frame_ids(frame_ids: Iterable[int | str]) -> List[int | str]:
    def key(value: int | str) -> tuple[int, int | str]:
        if isinstance(value, (int, np.integer)):
            return (0, int(value))
        text = str(value)
        if text.lstrip("-").isdigit():
            return (0, int(text))
        return (1, text)

    return sorted(frame_ids, key=key)


def _joint(joints: JointMap, canonical: str) -> np.ndarray:
    lowered = {_normalize_name(name): name for name in joints}
    for alias in LANDMARK_ALIASES[canonical]:
        real_name = lowered.get(_normalize_name(alias))
        if real_name is not None:
            return np.asarray(joints[real_name], dtype=float)
    raise KeyError(f"Missing required torso landmark: {canonical}")


def _normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _per_joint_geometry_errors(
    joints_a: JointMap, joints_b: JointMap, names: Sequence[str]
) -> Dict[str, float]:
    if not names:
        return {}
    coords_a = np.asarray([joints_a[name] for name in names], dtype=float)
    coords_b = np.asarray([joints_b[name] for name in names], dtype=float)
    dist_a = _distance_matrix(coords_a)
    dist_b = _distance_matrix(coords_b)
    errors = np.linalg.norm(dist_a - dist_b, axis=1)
    return {name: float(error) for name, error in zip(names, errors)}


def _pair_distance_errors(
    joints_a: JointMap, joints_b: JointMap, names: Sequence[str]
) -> List[float]:
    errors: List[float] = []
    for left_idx, left in enumerate(names):
        for right in names[left_idx + 1 :]:
            da = float(
                np.linalg.norm(np.asarray(joints_a[left]) - np.asarray(joints_a[right]))
            )
            db = float(
                np.linalg.norm(np.asarray(joints_b[left]) - np.asarray(joints_b[right]))
            )
            errors.append(abs(da - db))
    return errors


def _distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _distance_threshold(errors: Sequence[float], settings: AnalysisSettings) -> float:
    if not errors:
        return 0.0
    values = np.asarray(errors, dtype=float)
    if np.allclose(values, 0.0):
        return 0.0

    if settings.distance_rule == "q3":
        return float(np.percentile(values, 75))
    if settings.distance_rule == "mad":
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return median + settings.mad_factor * 1.4826 * mad
    if settings.distance_rule == "hybrid":
        return float((np.mean(values) + np.median(values)) / 2.0)

    raise ValueError(
        "distance_rule must be one of: 'hybrid', 'mad', 'q3'. "
        f"Got {settings.distance_rule!r}."
    )


def _summary(
    values: Sequence[float],
) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    array = np.asarray(values, dtype=float)
    return (
        float(np.percentile(array, 25)),
        float(np.percentile(array, 75)),
        float(np.mean(array)),
    )
