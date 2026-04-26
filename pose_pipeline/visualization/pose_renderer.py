from __future__ import annotations

import numpy as np

from pose_pipeline.config import BODY25_SKELETON_EDGES, SMPL24_SKELETON_EDGES


def render_3d_pose(frame: np.ndarray, joint_names: list[str], size: tuple[int, int]) -> np.ndarray:
    import cv2

    width, height = size
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    points = _project_points(frame, width, height)
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    edges = _skeleton_edges(name_to_idx)

    for a, b in edges:
        if a not in name_to_idx or b not in name_to_idx:
            continue
        pa = tuple(points[name_to_idx[a]].astype(int))
        pb = tuple(points[name_to_idx[b]].astype(int))
        cv2.line(canvas, pa, pb, (35, 91, 180), 2, cv2.LINE_AA)

    for point in points:
        cv2.circle(canvas, tuple(point.astype(int)), 4, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.putText(canvas, "3D pose", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    return canvas


def _skeleton_edges(name_to_idx: dict[str, int]) -> list[tuple[str, str]]:
    if {"mid_hip", "nose"}.issubset(name_to_idx):
        return BODY25_SKELETON_EDGES
    if {"pelvis", "spine1"}.issubset(name_to_idx):
        return SMPL24_SKELETON_EDGES
    return BODY25_SKELETON_EDGES + SMPL24_SKELETON_EDGES


def _project_points(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    pts = np.asarray(frame, dtype=float)
    xy = pts[:, [0, 1]].copy()
    xy[:, 1] *= -1.0
    min_xy = np.nanmin(xy, axis=0)
    max_xy = np.nanmax(xy, axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    normalized = (xy - min_xy) / span
    scale = min(width, height) * 0.76
    offset = np.array([(width - scale) / 2.0, (height - scale) / 2.0])
    return normalized * scale + offset
