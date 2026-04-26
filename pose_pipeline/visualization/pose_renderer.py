from __future__ import annotations

import numpy as np

from typing import Any

from pose_pipeline.config import BODY25_SKELETON_EDGES, SMPL24_SKELETON_EDGES


def render_3d_pose(
    frame: np.ndarray,
    joint_names: list[str],
    size: tuple[int, int],
    view: dict[str, Any] | None = None,
) -> np.ndarray:
    import cv2

    width, height = size
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    points = _project_points(frame, width, height, view)
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


def build_pose_view(
    poses: np.ndarray,
    joint_names: list[str],
    size: tuple[int, int],
    zoom: float = 1.0,
) -> dict[str, Any]:
    width, height = size
    xy = _pose_xy(np.asarray(poses, dtype=float))
    if xy.ndim != 3 or xy.shape[0] == 0:
        return {"scale": 1.0, "center_rel": np.zeros(2), "root_idx": None}

    root_idx = _root_index(joint_names)
    if root_idx is not None and root_idx < xy.shape[1]:
        roots = xy[:, root_idx, :]
    else:
        roots = np.nanmean(xy, axis=1)

    rel = xy - roots[:, np.newaxis, :]
    finite = np.isfinite(rel).all(axis=2)
    if not finite.any():
        return {"scale": 1.0, "center_rel": np.zeros(2), "root_idx": root_idx}

    rel_points = rel[finite]
    min_rel = np.nanpercentile(rel_points, 1, axis=0)
    max_rel = np.nanpercentile(rel_points, 99, axis=0)
    center_rel = (min_rel + max_rel) / 2.0
    span = np.maximum(max_rel - min_rel, 1e-6)
    scale = min(width, height) * 0.76 * max(float(zoom), 0.1) / float(np.max(span))
    return {
        "scale": scale,
        "center_rel": center_rel,
        "root_idx": root_idx,
    }


def _skeleton_edges(name_to_idx: dict[str, int]) -> list[tuple[str, str]]:
    if {"mid_hip", "nose"}.issubset(name_to_idx):
        return BODY25_SKELETON_EDGES
    if {"pelvis", "spine1"}.issubset(name_to_idx):
        return SMPL24_SKELETON_EDGES
    return BODY25_SKELETON_EDGES + SMPL24_SKELETON_EDGES


def _project_points(
    frame: np.ndarray, width: int, height: int, view: dict[str, Any] | None = None
) -> np.ndarray:
    xy = _pose_xy(np.asarray(frame, dtype=float))
    finite = np.isfinite(xy).all(axis=1)
    if not finite.any():
        return np.full((len(xy), 2), [width / 2.0, height / 2.0], dtype=float)

    if view:
        root_idx = view.get("root_idx")
        if root_idx is not None and root_idx < len(xy) and np.isfinite(xy[root_idx]).all():
            center = xy[root_idx] + np.asarray(view["center_rel"], dtype=float)
        else:
            center = np.nanmean(xy[finite], axis=0)
        scale = float(view["scale"])
    else:
        min_xy = np.nanmin(xy[finite], axis=0)
        max_xy = np.nanmax(xy[finite], axis=0)
        center = (min_xy + max_xy) / 2.0
        span = np.maximum(max_xy - min_xy, 1e-6)
        scale = min(width, height) * 0.76 / float(np.max(span))

    canvas_center = np.array([width / 2.0, height / 2.0])
    projected = (xy - center) * scale + canvas_center
    projected[~finite] = canvas_center
    return projected


def _pose_xy(points: np.ndarray) -> np.ndarray:
    xy = points[..., [0, 1]].copy()
    xy[..., 1] *= -1.0
    return xy


def _root_index(joint_names: list[str]) -> int | None:
    for name in ("mid_hip", "pelvis"):
        if name in joint_names:
            return joint_names.index(name)
    return None
