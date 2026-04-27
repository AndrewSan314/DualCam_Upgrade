# -*- coding: utf-8 -*-
"""Phase-3 prototype with mesh-based occlusion, RANSAC+Umeyama consensus,
and constrained optimization.

Chế độ hiện tại:
- Chạy độc lập bằng dữ liệu hardcoded trong file.
- Chưa nối vào pipeline/opencap-monocular.

Mục tiêu file này:
- Mô phỏng đầy đủ logic phase 3 theo workflow mới ở mức 1 frame:
  1) Tìm tập mâu thuẫn hướng M.
  2) Ước lượng khuất K1/K2 (mesh-based, nếu có verts).
  3) Tìm tập đồng thuận A bằng RANSAC + Umeyama.
  4) Điền lại điểm khuất bằng phép biến đổi tương ứng.
  5) Tối ưu trên tập F = P \\ A_new với ràng buộc xương cứng.
"""

import argparse
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def configure_stdout_encoding():
    """Best-effort UTF-8 stdout to avoid UnicodeEncodeError on Windows console."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except OSError:
            pass


def _parse_bool(value):
    """Parse argparse booleans while allowing --flag and --flag=true."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


# ------------------------------------------------------------------------------
# Demo fallback data (single-frame) when pkl paths are not provided.
# ------------------------------------------------------------------------------
HARDCODED_DATA = {
    "camera1": {
        "neck": [-0.3228432, -0.3532069, 2.0107348],
        "right_shoulder": [-0.4150482, -0.327897, 1.9074978],
        "right_elbow": [-0.2547746, -0.1923742, 1.6893969],
        "right_hand": [-0.0746337, -0.2165172, 1.4921814],
        "left_shoulder": [-0.2191069, -0.3928319, 2.1184506],
        "left_elbow": [-0.2857845, -0.3392255, 2.3977368],
        "left_hand": [-0.2576031, -0.3526569, 2.6619413],
        "right_hip": [-0.3732703, 0.125791, 2.1726549],
        "right_knee": [-0.3388955, 0.5416505, 2.2547374],
        "right_ankle": [-0.3290971, 0.9297508, 2.4316511],
        "left_hip": [-0.1905955, 0.0709873, 2.2118559],
        "left_knee": [-0.0151706, 0.2265235, 1.8729444],
        "left_ankle": [-0.0946184, 0.5480239, 2.1125834],
    },
    "camera2": {
        "neck": [-0.1128288, -0.2544436, 1.3556519],
        "right_shoulder": [-0.2444652, -0.2447644, 1.3080432],
        "right_elbow": [-0.2931232, -0.091388, 1.0481379],
        "right_hand": [-0.0804789, -0.0911379, 0.8917167],
        "left_shoulder": [0.0395474, -0.2649385, 1.3943661],
        "left_elbow": [0.1189277, -0.2172126, 1.6778377],
        "left_hand": [0.237222, -0.1334405, 1.8995144],
        "right_hip": [-0.1869255, 0.1978373, 1.5920346],
        "right_knee": [-0.1030911, 0.5897393, 1.7876432],
        "right_ankle": [-0.1138326, 0.6218347, 2.2040603],
        "left_hip": [0.0108481, 0.1802105, 1.5820299],
        "left_knee": [-0.14978, 0.5412838, 1.7897142],
        "left_ankle": [-0.079087, 0.9045545, 2.0045514],
    },
}

HEIGHT = 1.72
# RIGID_BONES_RATIO: tỉ lệ chiều dài xương theo chiều cao cơ thể.
# Dùng làm ràng buộc sinh lý để tránh tối ưu ra pose méo.
RIGID_BONES_RATIO = {
    ("left_elbow", "left_shoulder"): 0.186,
    ("left_hand", "left_elbow"): 0.146,
    ("right_elbow", "right_shoulder"): 0.186,
    ("right_hand", "right_elbow"): 0.146,
    ("left_knee", "left_hip"): 0.245,
    ("left_ankle", "left_knee"): 0.246,
    ("right_knee", "right_hip"): 0.245,
    ("right_ankle", "right_knee"): 0.246,
}

# RANSAC cấu hình theo kiểu trong ransac_ume.py
RANSAC_THRESHOLD = 0.05  # m (tương đương 50 mm)
RANSAC_MAX_COMBOS = 500
DIST_CONF_REF = 2.0  # mốc khoảng cách cho confidence theo distance
SOFT_TAIL_TEMPERATURE = 0.05  # m, càng nhỏ càng tập trung vào diff lớn
SOFT_TAIL_WEIGHT = 1.0
DEFAULT_REGULARIZATION_LAMBDA = 10.0
MAX_JOINT_MOVE_M = 0.10


def _as_xyz(point):
    """
    Chuẩn hóa một điểm bất kỳ về vector numpy shape (3,).

    Input:
    - point: iterable có 3 phần tử (x, y, z).

    Output:
    - ndarray shape (3,) dtype float.

    Lỗi:
    - ValueError nếu không đúng 3 chiều.
    """
    arr = np.asarray(point, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"Expected shape (3,), got {arr.shape}")
    return arr


def get_orientation_flag(joints, epsilon=1e-2):
    """
    Phân loại mỗi joint theo hướng trước/sau so với mặt phẳng thân người.

    Ý tưởng:
    - Dùng 4 mốc torso (2 vai, 2 hông) để dựng pháp tuyến "forward".
    - Với mỗi joint, lấy tích vô hướng với vector forward:
      > 0: trước (+1), < 0: sau (-1), gần 0: nằm trên mặt phẳng (0).

    Input:
    - joints: map joint_name -> [x,y,z].
    - epsilon: ngưỡng coi là "trên mặt phẳng".

    Output:
    - dict joint_name -> {-1,0,1}.
    """
    required = ("right_shoulder", "left_shoulder", "right_hip", "left_hip")
    # Nếu thiếu mốc torso thì không thể suy ra hướng ổn định.
    if not all(name in joints for name in required):
        return {name: 0 for name in joints}

    rs, ls = _as_xyz(joints["right_shoulder"]), _as_xyz(joints["left_shoulder"])
    rh, lh = _as_xyz(joints["right_hip"]), _as_xyz(joints["left_hip"])

    mid_shoulders = (rs + ls) / 2.0
    mid_hips = (rh + lh) / 2.0
    v_lr = rs - ls
    v_spine = mid_shoulders - mid_hips

    # Pháp tuyến mặt phẳng torso (xấp xỉ hướng nhìn).
    forward_vec = np.cross(v_lr, v_spine)
    norm = float(np.linalg.norm(forward_vec))
    # Trường hợp suy biến: vai/hông gần thẳng hàng => không xác định được hướng.
    if norm < 1e-8:
        return {name: 0 for name in joints}
    forward_vec /= norm

    torso_center = (mid_shoulders + mid_hips) / 2.0
    flags = {}
    for name, pos in joints.items():
        dot_product = float(np.dot(_as_xyz(pos) - torso_center, forward_vec))
        if abs(dot_product) < epsilon:
            flags[name] = 0
        else:
            flags[name] = 1 if dot_product > 0 else -1
    return flags


def compute_visibility_from_mesh_vertices(
    joints,
    verts,
    *,
    faces=None,
    camera=None,
    image_size=None,
    grid_size=160,
    occlusion_tau=0.02,
):
    """
    Ước lượng visibility bằng camera-space z-buffer.

    Cách làm:
    - Đưa vertices/joints về camera coordinates nếu có extrinsics.
    - Project bằng pinhole camera:
      * nếu có intrinsics K: u = fx*x/z + cx, v = fy*y/z + cy;
      * nếu chưa có K: dùng normalized image plane u=x/z, v=y/z.
    - Raster triangle faces nếu có; nếu không có faces thì vertex-splat fallback.
    - Depth là z theo trục nhìn camera, không dùng abs(z).
    - Joint được coi là khuất nếu depth joint nằm sau z-buffer quá
      occlusion_tau tại pixel/ray tương ứng.

    Input:
    - joints: map joint_name -> [x,y,z].
    - verts: ndarray (N,3) của mesh tại frame đang xét.
    - faces: optional ndarray (F,3) indices. Nếu có thì raster tam giác.
    - camera: optional dict chứa một trong các dạng:
        K/intrinsics/camera_matrix: 3x3 intrinsics;
        R + t hoặc extrinsics/world_to_camera: world->camera transform.
    - image_size: optional (width,height). Nếu có K thì dùng để map pixel.
    - grid_size: độ phân giải lưới depth.
    - occlusion_tau: biên dung sai depth (m).

    Output:
    - dict joint_name -> bool (True = thấy được).
    """
    verts = np.asarray(verts, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"Expected verts shape (N,3), got {verts.shape}")
    if grid_size < 8:
        raise ValueError("grid_size must be at least 8")

    camera = camera or {}
    verts_cam = _world_to_camera_points(verts, camera)
    vert_uv, vert_depth, projection = _project_camera_points(
        verts_cam,
        camera=camera,
        image_size=image_size,
    )
    valid_verts = np.isfinite(vert_uv).all(axis=1) & np.isfinite(vert_depth) & (vert_depth > 1e-6)
    if not np.any(valid_verts):
        # Không có surface hợp lệ theo camera model: đừng tạo mask sai.
        return {name: True for name in joints}

    uv_min, uv_max = _projection_bounds(vert_uv[valid_verts], projection, image_size)
    depth_grid = np.full((grid_size, grid_size), np.inf, dtype=float)

    faces_arr = _valid_faces(faces, len(verts)) if faces is not None else None
    if faces_arr is not None and len(faces_arr):
        _rasterize_faces_to_depth_grid(
            depth_grid,
            vert_uv,
            vert_depth,
            faces_arr,
            valid_verts,
            uv_min,
            uv_max,
        )
    else:
        _splat_vertices_to_depth_grid(
            depth_grid,
            vert_uv[valid_verts],
            vert_depth[valid_verts],
            uv_min,
            uv_max,
        )

    visibility = {}
    for name, pos in joints.items():
        p_cam = _world_to_camera_points(_as_xyz(pos)[None, :], camera)[0]
        uv, depth, _ = _project_camera_points(
            p_cam[None, :],
            camera=camera,
            image_size=image_size,
        )
        uv = uv[0]
        z_joint = float(depth[0])
        if not np.isfinite(uv).all() or not np.isfinite(z_joint) or z_joint <= 1e-6:
            visibility[name] = True
            continue

        xn = (uv[0] - uv_min[0]) / max(float(uv_max[0] - uv_min[0]), 1e-12)
        yn = (uv[1] - uv_min[1]) / max(float(uv_max[1] - uv_min[1]), 1e-12)

        # Joint nằm ngoài bbox mesh => coi như không bị mesh che.
        if xn < 0.0 or xn > 1.0 or yn < 0.0 or yn > 1.0:
            visibility[name] = True
            continue

        xi = int(np.clip(xn * (grid_size - 1), 0, grid_size - 1))
        yi = int(np.clip(yn * (grid_size - 1), 0, grid_size - 1))
        z_surface = depth_grid[yi, xi]
        # Không có bề mặt mesh tại cell này => coi là thấy.
        if not np.isfinite(z_surface):
            visibility[name] = True
            continue

        # Nếu joint nằm sau bề mặt theo ray camera quá ngưỡng -> bị khuất.
        visibility[name] = z_joint <= (z_surface + occlusion_tau)

    return visibility


def _camera_intrinsics(camera):
    for key in ("K", "intrinsics", "camera_matrix"):
        if key in camera and camera[key] is not None:
            k = np.asarray(camera[key], dtype=float)
            if k.shape == (3, 3):
                return k
    return None


def _world_to_camera_points(points, camera):
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points shape (N,3), got {pts.shape}")

    for key in ("world_to_camera", "extrinsics", "T_wc"):
        if key in camera and camera[key] is not None:
            mat = np.asarray(camera[key], dtype=float)
            if mat.shape == (4, 4):
                homog = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
                return (homog @ mat.T)[:, :3]
            if mat.shape == (3, 4):
                homog = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
                return homog @ mat.T

    r = camera.get("R")
    t = camera.get("t", camera.get("T"))
    if r is not None:
        r = np.asarray(r, dtype=float)
        if r.shape != (3, 3):
            raise ValueError(f"Expected camera R shape (3,3), got {r.shape}")
        t = np.zeros(3, dtype=float) if t is None else np.asarray(t, dtype=float).reshape(3)
        return (r @ pts.T).T + t

    return pts.copy()


def _project_camera_points(points_cam, *, camera, image_size=None):
    pts = np.asarray(points_cam, dtype=float)
    z = pts[:, 2].astype(float)
    uv = np.full((len(pts), 2), np.nan, dtype=float)
    valid = np.isfinite(pts).all(axis=1) & (z > 1e-6)
    if not np.any(valid):
        return uv, z, "invalid"

    k = _camera_intrinsics(camera)
    if k is not None:
        x = pts[valid, 0] / z[valid]
        y = pts[valid, 1] / z[valid]
        uv[valid, 0] = k[0, 0] * x + k[0, 2]
        uv[valid, 1] = k[1, 1] * y + k[1, 2]
        return uv, z, "pixel"

    # Normalized pinhole plane. Đây vẫn là camera projection đúng hơn nhiều so
    # với dùng trực tiếp x,y 3D, vì mọi điểm cùng ray có cùng x/z,y/z.
    uv[valid, 0] = pts[valid, 0] / z[valid]
    uv[valid, 1] = pts[valid, 1] / z[valid]
    return uv, z, "normalized"


def _projection_bounds(uv, projection, image_size):
    if projection == "pixel" and image_size is not None:
        width, height = _parse_image_size(image_size)
        if width > 0 and height > 0:
            return np.asarray([0.0, 0.0]), np.asarray([float(width - 1), float(height - 1)])

    uv_min = np.percentile(uv, 1, axis=0)
    uv_max = np.percentile(uv, 99, axis=0)
    span = np.maximum(uv_max - uv_min, 1e-6)
    return uv_min, uv_min + span


def _parse_image_size(image_size):
    if isinstance(image_size, dict):
        width = image_size.get("width", image_size.get("w", 0))
        height = image_size.get("height", image_size.get("h", 0))
        return int(width), int(height)
    values = list(image_size)
    if len(values) != 2:
        return 0, 0
    return int(values[0]), int(values[1])


def _valid_faces(faces, vert_count):
    arr = np.asarray(faces, dtype=int)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected faces shape (F,3), got {arr.shape}")
    valid = np.all((arr >= 0) & (arr < vert_count), axis=1)
    return arr[valid]


def _uv_to_grid(uv, uv_min, uv_max, grid_size):
    denom = np.maximum(uv_max - uv_min, 1e-12)
    norm = (uv - uv_min) / denom
    idx = np.clip((norm * (grid_size - 1)).astype(int), 0, grid_size - 1)
    return idx


def _splat_vertices_to_depth_grid(depth_grid, uv, depth, uv_min, uv_max):
    grid_size = depth_grid.shape[0]
    idx = _uv_to_grid(uv, uv_min, uv_max, grid_size)
    for (xi, yi), zi in zip(idx, depth):
        if zi < depth_grid[yi, xi]:
            depth_grid[yi, xi] = zi


def _rasterize_faces_to_depth_grid(depth_grid, uv, depth, faces, valid_verts, uv_min, uv_max):
    grid_size = depth_grid.shape[0]
    grid_xy = _uv_to_grid(uv, uv_min, uv_max, grid_size).astype(float)
    for face in faces:
        if not np.all(valid_verts[face]):
            continue
        tri = grid_xy[face]
        tri_depth = depth[face]
        x0 = max(int(np.floor(np.min(tri[:, 0]))), 0)
        x1 = min(int(np.ceil(np.max(tri[:, 0]))), grid_size - 1)
        y0 = max(int(np.floor(np.min(tri[:, 1]))), 0)
        y1 = min(int(np.ceil(np.max(tri[:, 1]))), grid_size - 1)
        if x1 < x0 or y1 < y0:
            continue

        denom = _triangle_barycentric_denominator(tri)
        if abs(denom) < 1e-12:
            continue
        for yi in range(y0, y1 + 1):
            for xi in range(x0, x1 + 1):
                bary = _triangle_barycentric(np.asarray([xi + 0.5, yi + 0.5]), tri, denom)
                if np.min(bary) < -1e-6:
                    continue
                z = float(np.dot(bary, tri_depth))
                if z < depth_grid[yi, xi]:
                    depth_grid[yi, xi] = z


def _triangle_barycentric_denominator(tri):
    a, b, c = tri
    return (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])


def _triangle_barycentric(point, tri, denom):
    a, b, c = tri
    w0 = ((b[1] - c[1]) * (point[0] - c[0]) + (c[0] - b[0]) * (point[1] - c[1])) / denom
    w1 = ((c[1] - a[1]) * (point[0] - c[0]) + (a[0] - c[0]) * (point[1] - c[1])) / denom
    return np.asarray([w0, w1, 1.0 - w0 - w1], dtype=float)


def _mesh_visibility_payload(value):
    if isinstance(value, dict):
        vertices = value.get("vertices", value.get("verts"))
        if vertices is None:
            raise ValueError("Mesh payload must contain vertices or verts")
        return {
            "vertices": vertices,
            "faces": value.get("faces"),
            "camera": value.get("camera"),
            "image_size": value.get("image_size"),
        }
    return {"vertices": value, "faces": None, "camera": None, "image_size": None}


def _to_arrays(cam1, cam2, names):
    """
    Chuyển danh sách tên joint thành 2 ma trận điểm tương ứng giữa 2 camera.

    Output:
    - src: (N,3) từ camera1.
    - dst: (N,3) từ camera2.
    """
    src = np.array([_as_xyz(cam1[name]) for name in names], dtype=float)
    dst = np.array([_as_xyz(cam2[name]) for name in names], dtype=float)
    return src, dst


def estimate_umeyama(src, dst):
    """
    Ước lượng phép biến đổi similarity bằng Umeyama:
        dst ~= s * R * src + t

    Input:
    - src, dst: (N,3), N>=3.

    Output:
    - scale s (float), rotation R (3x3), translation t (3,).
    """
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src/dst must have shape (N,3) and match")
    n, m = src.shape
    if n < 3:
        raise ValueError("Need at least 3 points for Umeyama")

    mu_src = src.mean(0)
    mu_dst = dst.mean(0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    sigma_src = np.mean(np.sum(src_c ** 2, axis=1))
    h = (dst_c.T @ src_c) / n
    u, d, vt = np.linalg.svd(h)

    s_mat = np.eye(m)
    # Ép orientation đúng tay phải (det(R) gần +1), tránh lật gương.
    if np.linalg.det(u) * np.linalg.det(vt.T) < 0:
        s_mat[m - 1, m - 1] = -1

    r = u @ s_mat @ vt
    # Nếu var_src quá nhỏ thì scale không ổn định, fallback = 1.
    if sigma_src < 1e-12:
        scale = 1.0
    else:
        scale = float(np.trace(np.diag(d) @ s_mat) / sigma_src)
    t = mu_dst - scale * (r @ mu_src)
    return scale, r, t


def apply_similarity(point, transform):
    """
    Áp dụng transform similarity (s, R, t) cho một điểm 3D.
    """
    scale, r, t = transform
    p = _as_xyz(point)
    return scale * (r @ p) + t


def pair_distance_absolute_confidence(
    names,
    dist_cam1,
    dist_cam2,
    ref_dist=DIST_CONF_REF,
):
    """
    Confidence tuyệt đối theo distance cho từng camera:
      conf = 1 / (1 + (dist / ref_dist)^2)

    Ý nghĩa:
    - dist = 0      -> conf ~ 1.0
    - dist = ref    -> conf = 0.5
    - dist = 2*ref  -> conf = 0.2
    - dist >> ref   -> conf giảm mạnh hơn công thức tuyến tính cũ

    Output:
    - tuple (conf1, conf2), mỗi phần tử là dict joint_name -> confidence [0, 1].
    """
    if not names:
        return {}, {}

    known1 = [float(dist_cam1[n]) for n in names if n in dist_cam1]
    known2 = [float(dist_cam2[n]) for n in names if n in dist_cam2]
    fallback1 = float(np.median(known1)) if known1 else float(ref_dist)
    fallback2 = float(np.median(known2)) if known2 else float(ref_dist)

    conf1 = {}
    conf2 = {}
    for n in names:
        # Chặn min 1cm để tránh chia 0.
        d1 = max(float(dist_cam1.get(n, fallback1)), 0.01)
        d2 = max(float(dist_cam2.get(n, fallback2)), 0.01)

        conf1[n] = float(1.0 / (1.0 + (d1 / ref_dist) ** 2))
        conf2[n] = float(1.0 / (1.0 + (d2 / ref_dist) ** 2))
    return conf1, conf2


def ransac_umeyama(
    cam1,
    cam2,
    names,
    *,
    threshold=RANSAC_THRESHOLD,
    max_combos=RANSAC_MAX_COMBOS,
    rng=None,
):
    """
    RANSAC + Umeyama để tìm tập inlier đồng thuận và transform ổn định.

    Input:
    - cam1/cam2: map joint -> điểm 3D.
    - names: tập điểm ứng viên (thường là L).

    Output:
    - transform tốt nhất (s,R,t) sau khi refine trên inlier.
    - danh sách tên inlier (chính là tập A trong workflow).
    """
    if len(names) < 3:
        return (1.0, np.eye(3), np.zeros(3)), list(names)

    src_all, dst_all = _to_arrays(cam1, cam2, names)
    n = len(names)
    all_idx = list(range(n))
    c3 = n * (n - 1) * (n - 2) // 6
    if rng is None:
        rng = np.random.default_rng(42)

    if c3 <= max_combos:
        triplets = list(combinations(all_idx, 3))
    else:
        triplets = [
            tuple(rng.choice(all_idx, 3, replace=False))
            for _ in range(max_combos)
        ]

    best_inliers = []
    for triplet in triplets:
        tri = list(triplet)
        try:
            transform = estimate_umeyama(src_all[tri], dst_all[tri])
        except np.linalg.LinAlgError:
            continue

        # Chiếu toàn bộ src sang dst rồi tính residual từng điểm.
        pred = np.array(
            [apply_similarity(src_all[i], transform) for i in range(len(names))]
        )
        err = np.linalg.norm(pred - dst_all, axis=1)
        inliers = np.where(err < threshold)[0].tolist()
        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    # Không tìm được model ổn định: fallback identity.
    if not best_inliers:
        best_inliers = all_idx

    # Refine lại transform trên toàn bộ inlier để giảm nhiễu.
    inlier_names = [names[i] for i in best_inliers]
    transform_refined = estimate_umeyama(
        src_all[best_inliers], dst_all[best_inliers]
    )
    return transform_refined, inlier_names


def get_diff_f(
    f_name,
    anchors,
    cam1,
    cam2,
    conf1=None,
    conf2=None,
    vis1=None,
    vis2=None,
    occluded_factor=0.25,
):
    """
    Tính độ lệch (disagreement) của điểm f so với tập các điểm neo (anchors).
    """
    # 1. Lấy tọa độ của f ở 2 camera
    p1_f = _as_xyz(cam1[f_name])
    p2_f = _as_xyz(cam2[f_name])

    sum_wdiff = 0.0
    sum_w = 0.0

    for a in anchors:
        # 2. Chỉ lấy trọng số của ANCHOR (ca, va)
        ca1 = float(conf1.get(a, 1.0))
        ca2 = float(conf2.get(a, 1.0))
        va1 = 1.0 if vis1.get(a, True) else float(occluded_factor)
        va2 = 1.0 if vis2.get(a, True) else float(occluded_factor)

        # 3. Cải tiến trọng số anchor (w_anchor):
        # Dùng trung bình thay vì nhân để tránh trọng số bị tiến về 0 quá nhanh.
        # w_anchor cao khi anchor đó được cả 2 camera nhìn rõ và gần.
        w_anchor = ((ca1 + ca2) / 2.0) * (va1 * va2)

        # 4. Tính khoảng cách hình học
        d1 = np.linalg.norm(p1_f - _as_xyz(cam1[a]))
        d2 = np.linalg.norm(p2_f - _as_xyz(cam2[a]))

        # 5. Cộng dồn sai số có trọng số
        sum_wdiff += w_anchor * abs(d1 - d2)
        sum_w += w_anchor

    # 6. Trả về sai số trung bình (đã loại bỏ triệt tiêu trọng số của f)
    return sum_wdiff / max(sum_w, 1e-12)



def calculate_diff_values(
    cam1,
    cam2,
    f_list,
    anchors,
    conf1=None,
    conf2=None,
    vis1=None,
    vis2=None,
    occluded_factor=0.25,
    f_weights=None,
):
    """
    Tính danh sách diff(f) sau khi áp trọng số của từng joint F.
    """
    if not f_list or not anchors:
        return []
    diffs = [
        get_diff_f(
            f_name,
            anchors,
            cam1,
            cam2,
            conf1=conf1,
            conf2=conf2,
            vis1=vis1,
            vis2=vis2,
            occluded_factor=occluded_factor,
        )
        for f_name in f_list
    ]

    if f_weights is not None:
        return [float(d * f_weights[name]) for d, name in zip(diffs, f_list)]
    return [float(d) for d in diffs]


def soft_tail(values, temperature=SOFT_TAIL_TEMPERATURE):
    """
    Soft alternative to a hard tail statistic such as Q3/max.

    Values with larger diff get larger softmax weights, but the function remains
    smooth for finite temperature.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    logits = (values - float(np.max(values))) / temperature
    weights = np.exp(logits)
    weights /= max(float(np.sum(weights)), 1e-12)
    return float(np.sum(weights * values))


def calculate_stats(
    cam1,
    cam2,
    f_list,
    anchors,
    conf1=None,
    conf2=None,
    vis1=None,
    vis2=None,
    occluded_factor=0.25,
    f_weights=None,
):
    """
    Tính thống kê robust trên tập diff(f):
    - Q1, Q3, Mean, Median.
    """
    processed_diffs = calculate_diff_values(
        cam1,
        cam2,
        f_list,
        anchors,
        conf1=conf1,
        conf2=conf2,
        vis1=vis1,
        vis2=vis2,
        occluded_factor=occluded_factor,
        f_weights=f_weights,
    )

    if not processed_diffs:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(np.percentile(processed_diffs, 25)),
        float(np.percentile(processed_diffs, 75)),
        float(np.mean(processed_diffs)),
        float(np.median(processed_diffs)),
    )


def optimize_f_points(
    data,
    anchors,
    f_list,
    conf1=None,
    conf2=None,
    vis1=None,
    vis2=None,
    occluded_factor=0.25,
    regularization=True,
    regularization_lambda=DEFAULT_REGULARIZATION_LAMBDA,
    soft_tail_temperature=SOFT_TAIL_TEMPERATURE,
    soft_tail_weight=SOFT_TAIL_WEIGHT,
):
    """
    Tối ưu toạ độ các điểm trong F cho cả hai camera bằng SLSQP.

    Hàm mục tiêu:
    - minimize Mean(diff) + soft_tail_weight * SoftTail(diff)
    - nếu regularization=True, cộng thêm weighted proximity:
      lambda * sum_i sum_c conf_i,c * ||x_i,c - x_old_i,c||^2

    Ràng buộc:
    - Chiều dài các xương trong RIGID_BONES_RATIO không lệch quá 5%.

    Output:
    - Pose đã tối ưu cho camera1/camera2.
    - Kết quả solver (hoặc None nếu F rỗng). Nếu solver fail, trả lại pose
      đầu vào và res.success=False để caller fallback an toàn.
    """
    cam1 = {k: _as_xyz(v) for k, v in data["camera1"].items()}
    cam2 = {k: _as_xyz(v) for k, v in data["camera2"].items()}
    regularization_lambda = float(regularization_lambda)
    if regularization_lambda < 0.0:
        raise ValueError("regularization_lambda must be non-negative")
    soft_tail_temperature = float(soft_tail_temperature)
    if soft_tail_temperature <= 0.0:
        raise ValueError("soft_tail_temperature must be positive")
    soft_tail_weight = float(soft_tail_weight)
    if soft_tail_weight < 0.0:
        raise ValueError("soft_tail_weight must be non-negative")

    # --- TÍNH TRỌNG SỐ ƯU TIÊN (PRIORITY) CHO TỪNG KHỚP F ---
    f_weights = {}
    for name in f_list:
        c1 = float(conf1.get(name, 1.0)) if conf1 else 1.0
        c2 = float(conf2.get(name, 1.0)) if conf2 else 1.0
        v1 = 1.0 if (vis1 and vis1.get(name, True)) else float(occluded_factor)
        v2 = 1.0 if (vis2 and vis2.get(name, True)) else float(occluded_factor)
        f_weights[name] = ((c1 + c2) / 2.0) * (v1 * v2)

    num_f = len(f_list)

    def proximity_penalty(x):
        penalty = 0.0
        for i, name in enumerate(f_list):
            c1 = float(conf1.get(name, 1.0)) if conf1 else 1.0
            c2 = float(conf2.get(name, 1.0)) if conf2 else 1.0
            p1_delta = x[i * 3 : i * 3 + 3] - cam1[name]
            p2_delta = x[(num_f + i) * 3 : (num_f + i) * 3 + 3] - cam2[name]
            penalty += c1 * float(np.dot(p1_delta, p1_delta))
            penalty += c2 * float(np.dot(p2_delta, p2_delta))
        return penalty

    # Hàm mục tiêu: giảm mean và soft-tail của diff(f).
    def objective(x):
        p1 = dict(cam1)
        p2 = dict(cam2)
        # Vector x chứa lần lượt điểm F của cam1 rồi cam2.
        for i, name in enumerate(f_list):
            p1[name] = x[i * 3 : i * 3 + 3]
            p2[name] = x[(num_f + i) * 3 : (num_f + i) * 3 + 3]
        diff_values = calculate_diff_values(
            p1,
            p2,
            f_list,
            anchors,
            conf1=conf1,
            conf2=conf2,
            vis1=vis1,
            vis2=vis2,
            occluded_factor=occluded_factor,
            f_weights=f_weights,
        )
        mean = float(np.mean(diff_values)) if diff_values else 0.0
        objective_value = mean + soft_tail_weight * soft_tail(
            diff_values,
            temperature=soft_tail_temperature,
        )
        if regularization:
            objective_value += regularization_lambda * proximity_penalty(x)
        return objective_value

    # Ràng buộc chiều dài xương cứng theo tỷ lệ nhân trắc học.
    constraints = []
    for (child, parent), ratio in RIGID_BONES_RATIO.items():
        if child not in cam1 or parent not in cam1 or child not in cam2 or parent not in cam2:
            continue
        if child not in f_list and parent not in f_list:
            continue
        target = ratio * HEIGHT
        for cam_idx in [0, 1]:
            # Mỗi xương sinh một bất đẳng thức cho mỗi camera.
            def make_constr(c=child, p=parent, t=target, cam=cam_idx):
                def bone_dist(x):
                    pts = [dict(cam1), dict(cam2)][cam]
                    for i, name in enumerate(f_list):
                        start = (cam * len(f_list) + i) * 3
                        pts[name] = x[start : start + 3]
                    dist = float(np.linalg.norm(pts[c] - pts[p]))
                    return 0.05 * t - abs(dist - t)

                return {"type": "ineq", "fun": bone_dist}

            constraints.append(make_constr())

    if not f_list:
        return {"camera1": cam1, "camera2": cam2}, None

    # Khởi tạo x0 từ chính toạ độ hiện tại (warm start).
    x0 = []
    for name in f_list:
        x0.extend(cam1[name])
    for name in f_list:
        x0.extend(cam2[name])

    res = minimize(
        objective,
        np.array(x0, dtype=float),
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 1000},
    )
    if not res.success:
        return {"camera1": cam1, "camera2": cam2}, res

    p1_opt = dict(cam1)
    p2_opt = dict(cam2)
    for i, name in enumerate(f_list):
        p1_opt[name] = res.x[i * 3 : i * 3 + 3]
        p2_opt[name] = res.x[(len(f_list) + i) * 3 : (len(f_list) + i) * 3 + 3]

    return {"camera1": p1_opt, "camera2": p2_opt}, res


def compute_displacement_report(before, after, joint_names):
    """
    Tính độ dịch chuyển per-joint giữa pose trước/sau tối ưu.

    Output:
    - dict camera -> joint -> displacement_m.
    """
    report = {}
    for cam in ("camera1", "camera2"):
        report[cam] = {}
        for name in joint_names:
            if name not in before[cam] or name not in after[cam]:
                continue
            report[cam][name] = float(
                np.linalg.norm(_as_xyz(after[cam][name]) - _as_xyz(before[cam][name]))
            )
    return report


def max_displacement_value(displacement_report):
    values = []
    for per_cam in displacement_report.values():
        values.extend(float(value) for value in per_cam.values())
    return float(max(values)) if values else 0.0


def run_phase3_pipeline(
    data_in,
    verts_by_cam,
    *,
    occlusion_grid,
    occlusion_tau,
    visibility_override=None,
    joint_distance_table=None,
    regularization=True,
    regularization_lambda=DEFAULT_REGULARIZATION_LAMBDA,
    soft_tail_temperature=SOFT_TAIL_TEMPERATURE,
    soft_tail_weight=SOFT_TAIL_WEIGHT,
    max_joint_move_m=MAX_JOINT_MOVE_M,
    reject_excessive_displacement=True,
):
    """
    Chạy toàn bộ workflow phase 3 cho một cặp pose 3D (1 frame):
    - Tạo P, M, K1, K2, L, A, A_new, F
    - Tối ưu trên F
    - Trả về thống kê trước/sau
    """
    cam1 = {k: _as_xyz(v) for k, v in data_in["camera1"].items()}
    cam2 = {k: _as_xyz(v) for k, v in data_in["camera2"].items()}
    warnings = []

    # P: tập joint chung giữa 2 camera.
    names = sorted(set(cam1.keys()) & set(cam2.keys()))
    if not names:
        raise ValueError("No common joints between camera1 and camera2")

    cam1 = {k: cam1[k] for k in names}
    cam2 = {k: cam2[k] for k in names}

    # Bước 1: tìm tập M (mâu thuẫn hướng trước/sau giữa hai camera).
    flags1 = get_orientation_flag(cam1)
    flags2 = get_orientation_flag(cam2)
    m_set = {
        n
        for n in names
        if (flags1.get(n, 0) == 1 and flags2.get(n, 0) == -1)
        or (flags1.get(n, 0) == -1 and flags2.get(n, 0) == 1)
    }

    # Bước 2: xác định K1/K2.
    # Ưu tiên thứ tự:
    # 1) visibility_override (dùng để test nhanh với dữ liệu hardcoded),
    # 2) mesh-based occlusion nếu có verts,
    # 3) mặc định mọi điểm đều visible.
    if visibility_override is not None:
        warnings.append("Using visibility_override; not real mesh/camera occlusion")
        vis1_raw = visibility_override.get("camera1", {})
        vis2_raw = visibility_override.get("camera2", {})
        vis1 = {n: bool(vis1_raw.get(n, True)) for n in names}
        vis2 = {n: bool(vis2_raw.get(n, True)) for n in names}
    elif verts_by_cam is not None:
        mesh1 = _mesh_visibility_payload(verts_by_cam["camera1"])
        mesh2 = _mesh_visibility_payload(verts_by_cam["camera2"])
        vis1 = compute_visibility_from_mesh_vertices(
            cam1,
            mesh1["vertices"],
            faces=mesh1.get("faces"),
            camera=mesh1.get("camera"),
            image_size=mesh1.get("image_size"),
            grid_size=occlusion_grid,
            occlusion_tau=occlusion_tau,
        )
        vis2 = compute_visibility_from_mesh_vertices(
            cam2,
            mesh2["vertices"],
            faces=mesh2.get("faces"),
            camera=mesh2.get("camera"),
            image_size=mesh2.get("image_size"),
            grid_size=occlusion_grid,
            occlusion_tau=occlusion_tau,
        )
        if mesh1.get("camera") is None or mesh2.get("camera") is None:
            warnings.append(
                "Using normalized pinhole visibility fallback; camera intrinsics/extrinsics not provided"
            )
        elif mesh1.get("faces") is None or mesh2.get("faces") is None:
            warnings.append("Using projected vertex z-buffer visibility; mesh faces not provided")
        else:
            warnings.append("Using projected mesh triangle z-buffer visibility")
    else:
        warnings.append("No mesh vertices; all joints assumed visible")
        vis1 = {n: True for n in names}
        vis2 = {n: True for n in names}

    # K1: cam1 thấy nhưng cam2 khuất. K2: cam2 thấy nhưng cam1 khuất.
    k1_set = {n for n in names if vis1.get(n, True) and not vis2.get(n, True)}
    k2_set = {n for n in names if vis2.get(n, True) and not vis1.get(n, True)}
    # Confidence theo từng joint/camera chỉ từ bảng khoảng cách.
    # Nếu không truyền bảng ngoài vào, fallback dùng norm vị trí joint.
    if joint_distance_table is not None:
        dist_cam1 = {
            k: float(v)
            for k, v in joint_distance_table.get("camera1", {}).items()
        }
        dist_cam2 = {
            k: float(v)
            for k, v in joint_distance_table.get("camera2", {}).items()
        }
    else:
        dist_cam1 = {n: float(np.linalg.norm(cam1[n])) for n in names}
        dist_cam2 = {n: float(np.linalg.norm(cam2[n])) for n in names}

    conf_cam1, conf_cam2 = pair_distance_absolute_confidence(
        names,
        dist_cam1,
        dist_cam2,
        ref_dist=DIST_CONF_REF,
    )

    # Bước 3: RANSAC + Umeyama trên L = P \ (M U K1 U K2) để lấy A.
    l_list = [
        n
        for n in names
        if n not in m_set and n not in k1_set and n not in k2_set
    ]
    # t12: phép biến đổi ước lượng từ camera1 -> camera2.
    t12, a_list = ransac_umeyama(cam1, cam2, l_list)

    # Tính transform ngược riêng cho chiều camera2 -> camera1.
    if len(a_list) >= 3:
        src_21 = np.array([cam2[n] for n in a_list], dtype=float)
        dst_21 = np.array([cam1[n] for n in a_list], dtype=float)
        t21 = estimate_umeyama(src_21, dst_21)
    else:
        t21 = (1.0, np.eye(3), np.zeros(3))

    # Bước 4: gán lại tọa độ các điểm bị khuất bằng transform tương ứng.
    cam1_corr = dict(cam1)
    cam2_corr = dict(cam2)
    # Với K1: dùng t12 để điền toạ độ phía camera2.
    for n in k1_set:
        cam2_corr[n] = apply_similarity(cam1[n], t12)
    # Với K2: dùng t21 để điền toạ độ phía camera1.
    for n in k2_set:
        cam1_corr[n] = apply_similarity(cam2[n], t21)

    # Bước 5: chỉ dùng observation thật làm anchor mạnh.
    # K1/K2 là điểm được suy luận bằng transform, nên không được nâng thành
    # anchor high-confidence để kéo các joint khác.
    observed_anchors = sorted(set(a_list))
    imputed_joints = sorted(set(k1_set) | set(k2_set))
    a_new = observed_anchors
    f_list = [n for n in names if n not in set(observed_anchors)]

    # TÍNH TRỌNG SỐ ƯU TIÊN CHO TỪNG KHỚP F ĐỂ THỐNG KÊ
    f_weights = {}
    for name in f_list:
        c1 = float(conf_cam1.get(name, 1.0))
        c2 = float(conf_cam2.get(name, 1.0))
        v1 = 1.0 if vis1.get(name, True) else 0.25 # assume occluded_factor=0.25
        v2 = 1.0 if vis2.get(name, True) else 0.25
        f_weights[name] = ((c1 + c2) / 2.0) * (v1 * v2)

    # Đo chất lượng trước tối ưu để so sánh.
    before_stats = calculate_stats(
        cam1_corr,
        cam2_corr,
        f_list,
        a_new,
        conf1=conf_cam1,
        conf2=conf_cam2,
        vis1=vis1,
        vis2=vis2,
        f_weights=f_weights,
    )
    before_opt_data = {"camera1": cam1_corr, "camera2": cam2_corr}
    optimized_data, res = optimize_f_points(
        before_opt_data,
        observed_anchors,
        f_list,
        conf1=conf_cam1,
        conf2=conf_cam2,
        vis1=vis1,
        vis2=vis2,
        regularization=regularization,
        regularization_lambda=regularization_lambda,
        soft_tail_temperature=soft_tail_temperature,
        soft_tail_weight=soft_tail_weight,
    )
    if res is not None and not res.success:
        warnings.append(f"Optimization failed; using pre-optimization pose: {res.message}")
    attempted_displacement = compute_displacement_report(
        before_opt_data,
        optimized_data,
        f_list,
    )
    max_attempted_move = max_displacement_value(attempted_displacement)
    rejected_by_displacement = bool(
        reject_excessive_displacement
        and max_joint_move_m > 0.0
        and max_attempted_move > float(max_joint_move_m)
    )
    if rejected_by_displacement:
        warnings.append(
            "Reject optimization: excessive joint displacement "
            f"({max_attempted_move:.4f} m > {float(max_joint_move_m):.4f} m)"
        )
        optimized_data = {
            "camera1": dict(before_opt_data["camera1"]),
            "camera2": dict(before_opt_data["camera2"]),
        }
    displacement = compute_displacement_report(
        before_opt_data,
        optimized_data,
        f_list,
    )
    # Đo chất lượng sau tối ưu.
    after_stats = calculate_stats(
        optimized_data["camera1"],
        optimized_data["camera2"],
        f_list,
        a_new,
        conf1=conf_cam1,
        conf2=conf_cam2,
        vis1=vis1,
        vis2=vis2,
        f_weights=f_weights,
    )

    return {
        "M": sorted(m_set),
        "K1": sorted(k1_set),
        "K2": sorted(k2_set),
        "L": l_list,
        "A": a_list,
        "A_new": a_new,
        "observed_anchors": observed_anchors,
        "imputed_joints": imputed_joints,
        "F": f_list,
        "before_stats": before_stats,
        "after_stats": after_stats,
        "joint_confidence": {"camera1": conf_cam1, "camera2": conf_cam2},
        "joint_distances": {"camera1": dist_cam1, "camera2": dist_cam2},
        "visibility": {"camera1": vis1, "camera2": vis2},
        "regularization": {
            "enabled": bool(regularization),
            "lambda": float(regularization_lambda),
        },
        "objective": {
            "tail": "soft_tail",
            "soft_tail_temperature": float(soft_tail_temperature),
            "soft_tail_weight": float(soft_tail_weight),
        },
        "displacement": displacement,
        "attempted_displacement": attempted_displacement,
        "max_joint_move_m": float(max_joint_move_m),
        "rejected_by_displacement": rejected_by_displacement,
        "warnings": warnings,
        "optimized": optimized_data,
        "optimization_result": res,
    }


def main():
    """
    Entry point chạy demo độc lập.
    - Chuẩn hóa output console UTF-8.
    - Chạy pipeline phase3 trên dữ liệu hardcoded.
    - In tập M/K1/K2/A/F và bảng thống kê trước/sau.
    """
    configure_stdout_encoding()
    parser = argparse.ArgumentParser(description="Run the phase-3 pose refinement demo.")
    parser.add_argument(
        "--regularization",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool,
        help="Enable weighted proximity regularization. Accepts true/false.",
    )
    parser.add_argument(
        "--regularization-lambda",
        type=float,
        default=DEFAULT_REGULARIZATION_LAMBDA,
        help="Lambda weight for weighted proximity regularization.",
    )
    parser.add_argument(
        "--max-joint-move-m",
        type=float,
        default=MAX_JOINT_MOVE_M,
        help="Reject optimized result if any optimized joint moves farther than this.",
    )
    parser.add_argument(
        "--soft-tail-temperature",
        type=float,
        default=SOFT_TAIL_TEMPERATURE,
        help="Temperature in meters for the soft-tail objective term.",
    )
    parser.add_argument(
        "--soft-tail-weight",
        type=float,
        default=SOFT_TAIL_WEIGHT,
        help="Weight multiplier for the soft-tail objective term.",
    )
    args = parser.parse_args()

    # Chế độ độc lập: dùng dữ liệu mẫu hardcoded, chưa nối vào repo chính.
    data = HARDCODED_DATA
    verts_by_cam = None
    # Mock visibility để test K1/K2 khi chưa có mesh verts.
    # True = nhìn thấy, False = bị khuất.
    visibility_override = {
        "camera1": {
            "left_hand": True,
            "left_elbow": True,
            "right_ankle": False,
        },
        "camera2": {
            "left_hand": False,
            "left_elbow": False,
            "right_ankle": True,
        },
    }
    # Bảng khoảng cách joint -> camera (m), dùng trực tiếp để tính confidence.
    joint_distance_table = {
        "camera1": {
            "right_hand": 1.4989,
            "right_elbow": 1.7115,
            "left_knee": 1.8865,
            "right_shoulder": 1.9754,
            "neck": 2.0628,
            "left_shoulder": 2.1629,
            "left_ankle": 2.1882,
            "right_hip": 2.2086,
            "left_hip": 2.2195,
            "right_knee": 2.3411,
            "left_elbow": 2.4372,
            "right_ankle": 2.6242,
            "left_hand": 2.6946,
        },
        "camera2": {
            "right_hand": 0.8987,
            "right_elbow": 1.0898,
            "right_shoulder": 1.3541,
            "neck": 1.3868,
            "left_shoulder": 1.4239,
            "left_hip": 1.5954,
            "right_hip": 1.6139,
            "left_elbow": 1.6991,
            "right_knee": 1.8800,
            "left_knee": 1.8822,
            "left_hand": 1.9221,
            "left_ankle": 2.2045,
            "right_ankle": 2.2737,
        },
    }
    occlusion_grid = 160
    occlusion_tau = 0.02
    print("Using hardcoded demo data (mesh occlusion disabled because no verts are provided)")
    print(
        "Apply mock occlusion mask: "
        "K1 candidates=(left_hand,left_elbow), K2 candidate=(right_ankle)"
    )
    print(
        "Regularization: "
        f"{'enabled' if args.regularization else 'disabled'} "
        f"(lambda={args.regularization_lambda:g})"
    )
    print(
        "Objective: mean + soft_tail "
        f"(temperature={args.soft_tail_temperature:g}, "
        f"weight={args.soft_tail_weight:g})"
    )

    # Chạy pipeline phase 3 đúng thứ tự workflow.
    result = run_phase3_pipeline(
        data,
        verts_by_cam,
        occlusion_grid=occlusion_grid,
        occlusion_tau=occlusion_tau,
        visibility_override=visibility_override,
        joint_distance_table=joint_distance_table,
        regularization=args.regularization,
        regularization_lambda=args.regularization_lambda,
        max_joint_move_m=args.max_joint_move_m,
        soft_tail_temperature=args.soft_tail_temperature,
        soft_tail_weight=args.soft_tail_weight,
    )

    print("--- PHASE-3 RESULT ---")
    print(f"M (orientation conflicts): {result['M']}")
    print(f"K1 (cam1 visible, cam2 occluded): {result['K1']}")
    print(f"K2 (cam2 visible, cam1 occluded): {result['K2']}")
    mean_c1 = float(np.mean(list(result["joint_confidence"]["camera1"].values())))
    mean_c2 = float(np.mean(list(result["joint_confidence"]["camera2"].values())))
    print(f"Joint confidence mean: cam1={mean_c1:.3f}, cam2={mean_c2:.3f}")
    print(f"A (RANSAC+Umeyama consensus on L): {result['A']}")
    print(f"A_new = observed anchors only: {result['A_new']}")
    print(f"Imputed joints (not anchors): {result['imputed_joints']}")
    print(f"F = P \\ A_new: {result['F']}\n")
    print(f"Warnings: {result['warnings']}")
    print(f"Rejected by displacement: {result['rejected_by_displacement']}")

    q1_i, q3_i, mean_i, med_i = result["before_stats"]
    q1_f, q3_f, mean_f, med_f = result["after_stats"]

    # Bảng tóm tắt các chỉ số robust trước/sau tối ưu.
    df = pd.DataFrame(
        {
            "Metric": ["Q1 (m)", "Q3 (m)", "Mean (m)", "Median (m)"],
            "Before": [f"{q1_i:.6f}", f"{q3_i:.6f}", f"{mean_i:.6f}", f"{med_i:.6f}"],
            "After": [f"{q1_f:.6f}", f"{q3_f:.6f}", f"{mean_f:.6f}", f"{med_f:.6f}"],
        }
    )
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
