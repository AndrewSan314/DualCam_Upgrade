from __future__ import annotations

import numpy as np

from typing import Any

from pose_pipeline.config import BODY25_SKELETON_EDGES, SMPL24_SKELETON_EDGES


LEFT_LIMB_COLOR = (66, 150, 78)
RIGHT_LIMB_COLOR = (36, 118, 220)
CENTER_LIMB_COLOR = (88, 88, 88)
JOINT_COLOR = (36, 36, 220)
GROUND_GRID_COLOR = (212, 212, 212)
GROUND_AXIS_COLOR = (178, 178, 178)


def render_3d_pose(
    frame: np.ndarray,
    joint_names: list[str],
    size: tuple[int, int],
    view: dict[str, Any] | None = None,
) -> np.ndarray:
    import cv2

    width, height = size
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    projection = _projection_params(frame, width, height, view)
    _draw_ground_plane(cv2, canvas, frame, joint_names, projection, view)
    points = _project_points(frame, width, height, view, projection)
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    edges = _skeleton_edges(name_to_idx)

    for a, b in edges:
        if a not in name_to_idx or b not in name_to_idx:
            continue
        pa = tuple(points[name_to_idx[a]].astype(int))
        pb = tuple(points[name_to_idx[b]].astype(int))
        cv2.line(canvas, pa, pb, _edge_color(a, b), 3, cv2.LINE_AA)

    for name, point in zip(joint_names, points):
        cv2.circle(canvas, tuple(point.astype(int)), 4, _joint_color(name), -1, cv2.LINE_AA)
    cv2.putText(canvas, "3D pose", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    return canvas


def build_pose_view(
    poses: np.ndarray,
    joint_names: list[str],
    size: tuple[int, int],
    zoom: float = 1.0,
    mode: str = "front",
    yaw_deg: float = 45.0,
    pitch_deg: float = 55.0,
    y_up: bool | None = None,
) -> dict[str, Any]:
    width, height = size
    pose_array = np.asarray(poses, dtype=float)
    inferred_y_up = _infer_y_up(pose_array, joint_names) if y_up is None else bool(y_up)
    basis = _view_basis(mode, yaw_deg, pitch_deg, y_up=inferred_y_up)
    xy = _project_pose_to_view(pose_array, basis)
    if xy.ndim != 3 or xy.shape[0] == 0:
        return {
            "scale": 1.0,
            "center_rel": np.zeros(2),
            "root_idx": None,
            "basis": basis,
            "y_up": inferred_y_up,
        }

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
    ground_y = _ground_y(poses=pose_array, joint_names=joint_names, y_up=inferred_y_up)
    ground_extent = _ground_extent(poses=pose_array)
    return {
        "scale": scale,
        "center_rel": center_rel,
        "root_idx": root_idx,
        "basis": basis,
        "y_up": inferred_y_up,
        "ground_y": ground_y,
        "ground_extent": ground_extent,
        "draw_ground": True,
    }


def _skeleton_edges(name_to_idx: dict[str, int]) -> list[tuple[str, str]]:
    if {"mid_hip", "nose"}.issubset(name_to_idx):
        return BODY25_SKELETON_EDGES
    if {"pelvis", "spine1"}.issubset(name_to_idx):
        return SMPL24_SKELETON_EDGES
    return BODY25_SKELETON_EDGES + SMPL24_SKELETON_EDGES


def _project_points(
    frame: np.ndarray,
    width: int,
    height: int,
    view: dict[str, Any] | None = None,
    projection: dict[str, Any] | None = None,
) -> np.ndarray:
    if projection is None:
        projection = _projection_params(frame, width, height, view)
    return _project_world_points(np.asarray(frame, dtype=float), projection, width, height)


def _projection_params(
    frame: np.ndarray, width: int, height: int, view: dict[str, Any] | None = None
) -> dict[str, Any]:
    basis = (
        np.asarray(view.get("basis"), dtype=float)
        if view and "basis" in view
        else _view_basis("front", 0.0, 0.0, y_up=True)
    )
    xy = _project_pose_to_view(np.asarray(frame, dtype=float), basis)
    finite = np.isfinite(xy).all(axis=1)
    if not finite.any():
        return {
            "basis": basis,
            "center": np.zeros(2, dtype=float),
            "scale": 1.0,
            "canvas_center": np.array([width / 2.0, height / 2.0], dtype=float),
        }

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

    return {
        "basis": basis,
        "center": np.asarray(center, dtype=float),
        "scale": float(scale),
        "canvas_center": np.array([width / 2.0, height / 2.0], dtype=float),
    }


def _project_world_points(
    points: np.ndarray,
    projection: dict[str, Any],
    width: int,
    height: int,
) -> np.ndarray:
    xy = _project_pose_to_view(np.asarray(points, dtype=float), projection["basis"])
    finite = np.isfinite(xy).all(axis=1)
    projected = (xy - projection["center"]) * projection["scale"] + projection["canvas_center"]
    canvas_center = np.array([width / 2.0, height / 2.0], dtype=float)
    projected[~finite] = canvas_center
    return projected


def _project_pose_to_view(points: np.ndarray, basis: np.ndarray) -> np.ndarray:
    xy = np.asarray(points, dtype=float) @ basis.T
    xy[..., 1] *= -1.0
    return xy


def _view_basis(
    mode: str,
    yaw_deg: float,
    pitch_deg: float,
    *,
    y_up: bool = True,
) -> np.ndarray:
    """Return orthographic screen axes with fixed Y-up and zero camera roll."""
    up_axis = np.asarray([0.0, 1.0, 0.0] if y_up else [0.0, -1.0, 0.0], dtype=float)
    x_axis = np.asarray([1.0, 0.0, 0.0], dtype=float)
    z_axis = np.asarray([0.0, 0.0, 1.0], dtype=float)

    if mode == "front":
        return np.vstack([x_axis, up_axis])
    if mode == "side":
        return np.vstack([z_axis, up_axis])
    if mode == "top":
        return np.vstack([x_axis, z_axis])
    if mode != "orbit":
        raise ValueError(f"Unsupported render view mode: {mode}")

    yaw = np.radians(float(yaw_deg))
    pitch = np.radians(float(pitch_deg))

    horizontal = np.sin(yaw) * x_axis + np.cos(yaw) * z_axis
    camera_pos = np.cos(pitch) * horizontal + np.sin(pitch) * up_axis
    forward = -camera_pos / np.linalg.norm(camera_pos)
    right = np.cross(forward, up_axis)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return np.vstack([right, up])


def _infer_y_up(poses: np.ndarray, joint_names: list[str]) -> bool:
    root_idx = _root_index(joint_names)
    top_idx = _top_index(joint_names)
    if root_idx is None or top_idx is None:
        return True
    arr = np.asarray(poses, dtype=float)
    if arr.ndim != 3 or max(root_idx, top_idx) >= arr.shape[1]:
        return True
    dy = arr[:, top_idx, 1] - arr[:, root_idx, 1]
    dy = dy[np.isfinite(dy)]
    if dy.size == 0:
        return True
    return bool(float(np.nanmedian(dy)) > 0.0)


def _draw_ground_plane(
    cv2: Any,
    canvas: np.ndarray,
    frame: np.ndarray,
    joint_names: list[str],
    projection: dict[str, Any],
    view: dict[str, Any] | None,
) -> None:
    if view is not None and not bool(view.get("draw_ground", True)):
        return

    frame_arr = np.asarray(frame, dtype=float)
    if frame_arr.ndim != 2 or frame_arr.shape[1] != 3:
        return
    finite = np.isfinite(frame_arr).all(axis=1)
    if not finite.any():
        return

    y_up = bool(view.get("y_up", True)) if view else True
    ground_y = (
        float(view["ground_y"])
        if view and view.get("ground_y") is not None and np.isfinite(view.get("ground_y"))
        else _ground_y(frame_arr[np.newaxis, ...], joint_names, y_up)
    )
    extent = (
        float(view["ground_extent"])
        if view and view.get("ground_extent") is not None and np.isfinite(view.get("ground_extent"))
        else _ground_extent(frame_arr[np.newaxis, ...])
    )
    extent = max(extent, 1e-3)

    center_xz = _frame_root_xz(frame_arr, joint_names, finite)
    if center_xz is None:
        return

    width = canvas.shape[1]
    height = canvas.shape[0]
    x0, z0 = center_xz
    if _floor_projection_is_degenerate(projection):
        _draw_flat_view_ground_grid(
            cv2,
            canvas,
            projection,
            center=np.asarray([x0, ground_y, z0], dtype=float),
            extent=extent,
        )
        return

    xs = np.linspace(x0 - extent, x0 + extent, 9)
    zs = np.linspace(z0 - extent, z0 + extent, 9)
    endpoints = []
    for x in xs:
        endpoints.append(np.asarray([[x, ground_y, z0 - extent], [x, ground_y, z0 + extent]]))
    for z in zs:
        endpoints.append(np.asarray([[x0 - extent, ground_y, z], [x0 + extent, ground_y, z]]))

    for idx, segment in enumerate(endpoints):
        projected = _project_world_points(segment, projection, width, height)
        if np.linalg.norm(projected[0] - projected[1]) < 1.0:
            continue
        color = GROUND_AXIS_COLOR if idx in (4, 13) else GROUND_GRID_COLOR
        cv2.line(
            canvas,
            tuple(projected[0].astype(int)),
            tuple(projected[1].astype(int)),
            color,
            1,
            cv2.LINE_AA,
        )


def _floor_projection_is_degenerate(projection: dict[str, Any]) -> bool:
    basis = np.asarray(projection["basis"], dtype=float)
    x_screen = basis @ np.asarray([1.0, 0.0, 0.0])
    z_screen = basis @ np.asarray([0.0, 0.0, 1.0])
    return min(float(np.linalg.norm(x_screen)), float(np.linalg.norm(z_screen))) < 1e-6


def _draw_flat_view_ground_grid(
    cv2: Any,
    canvas: np.ndarray,
    projection: dict[str, Any],
    *,
    center: np.ndarray,
    extent: float,
) -> None:
    width = canvas.shape[1]
    height = canvas.shape[0]
    tangent = _visible_floor_tangent(projection)
    left_world = center - extent * tangent
    right_world = center + extent * tangent
    line = _project_world_points(np.vstack([left_world, right_world]), projection, width, height)
    left = line[0]
    right = line[1]
    if np.linalg.norm(right - left) < 1.0:
        return

    ground_y = float(np.clip((left[1] + right[1]) / 2.0, 0.0, height - 1.0))
    center_x = float(np.clip((left[0] + right[0]) / 2.0, 0.0, width - 1.0))
    left_x = float(left[0])
    right_x = float(right[0])
    bottom_y = min(height - 1.0, ground_y + max(90.0, height * 0.32))
    if bottom_y - ground_y < 8.0:
        cv2.line(
            canvas,
            (int(left_x), int(ground_y)),
            (int(right_x), int(ground_y)),
            GROUND_AXIS_COLOR,
            2,
            cv2.LINE_AA,
        )
        return

    for idx, alpha in enumerate(np.linspace(0.0, 1.0, 9)):
        start_x = center_x + (left_x - center_x) * (1.0 - alpha)
        end_x = center_x + (right_x - center_x) * (1.0 - alpha)
        bottom_left_x = center_x + (left_x - center_x) * 1.85
        bottom_right_x = center_x + (right_x - center_x) * 1.85
        target_x = bottom_left_x + (bottom_right_x - bottom_left_x) * alpha
        color = GROUND_AXIS_COLOR if idx == 4 else GROUND_GRID_COLOR
        cv2.line(
            canvas,
            (int(start_x), int(ground_y)),
            (int(target_x), int(bottom_y)),
            color,
            1,
            cv2.LINE_AA,
        )

    for idx, frac in enumerate(np.linspace(0.0, 1.0, 7)):
        eased = frac**0.72
        y = ground_y + (bottom_y - ground_y) * eased
        spread = 1.0 + 0.85 * eased
        lx = center_x + (left_x - center_x) * spread
        rx = center_x + (right_x - center_x) * spread
        color = GROUND_AXIS_COLOR if idx == 0 else GROUND_GRID_COLOR
        thickness = 2 if idx == 0 else 1
        cv2.line(
            canvas,
            (int(lx), int(y)),
            (int(rx), int(y)),
            color,
            thickness,
            cv2.LINE_AA,
        )


def _visible_floor_tangent(projection: dict[str, Any]) -> np.ndarray:
    basis = np.asarray(projection["basis"], dtype=float)
    x_axis = np.asarray([1.0, 0.0, 0.0], dtype=float)
    z_axis = np.asarray([0.0, 0.0, 1.0], dtype=float)
    x_len = float(np.linalg.norm(basis @ x_axis))
    z_len = float(np.linalg.norm(basis @ z_axis))
    return x_axis if x_len >= z_len else z_axis


def _ground_y(poses: np.ndarray, joint_names: list[str], y_up: bool) -> float:
    arr = np.asarray(poses, dtype=float)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return 0.0
    indices = _ground_joint_indices(joint_names, arr.shape[1])
    samples = arr[:, indices, 1] if indices else arr[..., 1]
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return 0.0
    percentile = 2 if y_up else 98
    return float(np.nanpercentile(samples, percentile))


def _ground_extent(poses: np.ndarray) -> float:
    arr = np.asarray(poses, dtype=float)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return 1.0
    flat = arr.reshape(-1, 3)
    flat = flat[np.isfinite(flat).all(axis=1)]
    if flat.size == 0:
        return 1.0
    xz = flat[:, [0, 2]]
    low = np.nanpercentile(xz, 5, axis=0)
    high = np.nanpercentile(xz, 95, axis=0)
    vertical = np.nanpercentile(flat[:, 1], 95) - np.nanpercentile(flat[:, 1], 5)
    span = max(float(np.max(high - low)), float(vertical) * 0.65, 1e-3)
    return span * 1.15


def _ground_joint_indices(joint_names: list[str], joint_count: int) -> list[int]:
    ground_names = {
        "left_ankle",
        "right_ankle",
        "left_foot",
        "right_foot",
        "left_heel",
        "right_heel",
        "left_big_toe",
        "right_big_toe",
        "left_small_toe",
        "right_small_toe",
    }
    return [idx for idx, name in enumerate(joint_names[:joint_count]) if name in ground_names]


def _frame_root_xz(
    frame: np.ndarray, joint_names: list[str], finite: np.ndarray
) -> tuple[float, float] | None:
    root_idx = _root_index(joint_names)
    if root_idx is not None and root_idx < len(frame) and finite[root_idx]:
        return float(frame[root_idx, 0]), float(frame[root_idx, 2])
    valid = frame[finite]
    if valid.size == 0:
        return None
    center = np.nanmedian(valid[:, [0, 2]], axis=0)
    return float(center[0]), float(center[1])


def _edge_color(a: str, b: str) -> tuple[int, int, int]:
    sides = {_joint_side(a), _joint_side(b)}
    if sides == {"left"} or sides == {"left", "center"}:
        return LEFT_LIMB_COLOR
    if sides == {"right"} or sides == {"right", "center"}:
        return RIGHT_LIMB_COLOR
    return CENTER_LIMB_COLOR


def _joint_color(name: str) -> tuple[int, int, int]:
    return JOINT_COLOR


def _joint_side(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("left_"):
        return "left"
    if lowered.startswith("right_"):
        return "right"
    return "center"


def _root_index(joint_names: list[str]) -> int | None:
    for name in ("mid_hip", "pelvis"):
        if name in joint_names:
            return joint_names.index(name)
    return None


def _top_index(joint_names: list[str]) -> int | None:
    for name in ("neck", "spine3", "head", "nose"):
        if name in joint_names:
            return joint_names.index(name)
    return None
