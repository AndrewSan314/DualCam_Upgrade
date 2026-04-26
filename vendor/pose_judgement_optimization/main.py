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
    grid_size=160,
    occlusion_tau=0.02,
):
    """
    Ước lượng visibility của joint bằng depth-grid tạo từ vertices của mesh.

    Cách làm (xấp xỉ nhanh):
    - Raster vertices vào lưới 2D (theo trục x,y của hệ đang dùng).
    - Tại mỗi ô lưới, giữ depth gần nhất (z nhỏ hơn theo |z|).
    - Joint được coi là "thấy được" nếu depth joint không nằm sau bề mặt
      quá ngưỡng occlusion_tau.

    Input:
    - joints: map joint_name -> [x,y,z].
    - verts: ndarray (N,3) của mesh tại frame đang xét.
    - grid_size: độ phân giải lưới depth.
    - occlusion_tau: biên dung sai depth (m).

    Output:
    - dict joint_name -> bool (True = thấy được).
    """
    verts = np.asarray(verts, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"Expected verts shape (N,3), got {verts.shape}")

    xy = verts[:, :2]
    z_abs = np.abs(verts[:, 2])

    # Dùng percentile để giảm ảnh hưởng outlier ở biên mesh.
    x_min, y_min = np.percentile(xy, 1, axis=0)
    x_max, y_max = np.percentile(xy, 99, axis=0)
    if x_max <= x_min:
        x_max = x_min + 1e-6
    if y_max <= y_min:
        y_max = y_min + 1e-6

    # Depth grid khởi tạo vô cùng: chưa có bề mặt nào đi qua.
    depth_grid = np.full((grid_size, grid_size), np.inf, dtype=float)

    x_norm = (xy[:, 0] - x_min) / (x_max - x_min)
    y_norm = (xy[:, 1] - y_min) / (y_max - y_min)
    x_idx = np.clip((x_norm * (grid_size - 1)).astype(int), 0, grid_size - 1)
    y_idx = np.clip((y_norm * (grid_size - 1)).astype(int), 0, grid_size - 1)

    # Z-buffer đơn giản: giữ depth gần nhất ở mỗi cell.
    for xi, yi, zi in zip(x_idx, y_idx, z_abs):
        if zi < depth_grid[yi, xi]:
            depth_grid[yi, xi] = zi

    visibility = {}
    for name, pos in joints.items():
        p = _as_xyz(pos)
        xn = (p[0] - x_min) / (x_max - x_min)
        yn = (p[1] - y_min) / (y_max - y_min)

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

        z_joint = abs(float(p[2]))
        # Nếu joint nằm sau bề mặt quá ngưỡng -> bị khuất.
        visibility[name] = z_joint <= (z_surface + occlusion_tau)

    return visibility


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
    regularization=False,
    regularization_lambda=1.0,
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
    - Kết quả solver (hoặc None nếu F rỗng).
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
        raise RuntimeError(f"Optimization failed: {res.message}")

    p1_opt = dict(cam1)
    p2_opt = dict(cam2)
    for i, name in enumerate(f_list):
        p1_opt[name] = res.x[i * 3 : i * 3 + 3]
        p2_opt[name] = res.x[(len(f_list) + i) * 3 : (len(f_list) + i) * 3 + 3]

    return {"camera1": p1_opt, "camera2": p2_opt}, res


def run_phase3_pipeline(
    data_in,
    verts_by_cam,
    *,
    occlusion_grid,
    occlusion_tau,
    visibility_override=None,
    joint_distance_table=None,
    regularization=False,
    regularization_lambda=1.0,
    soft_tail_temperature=SOFT_TAIL_TEMPERATURE,
    soft_tail_weight=SOFT_TAIL_WEIGHT,
):
    """
    Chạy toàn bộ workflow phase 3 cho một cặp pose 3D (1 frame):
    - Tạo P, M, K1, K2, L, A, A_new, F
    - Tối ưu trên F
    - Trả về thống kê trước/sau
    """
    cam1 = {k: _as_xyz(v) for k, v in data_in["camera1"].items()}
    cam2 = {k: _as_xyz(v) for k, v in data_in["camera2"].items()}

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
        vis1_raw = visibility_override.get("camera1", {})
        vis2_raw = visibility_override.get("camera2", {})
        vis1 = {n: bool(vis1_raw.get(n, True)) for n in names}
        vis2 = {n: bool(vis2_raw.get(n, True)) for n in names}
    elif verts_by_cam is not None:
        vis1 = compute_visibility_from_mesh_vertices(
            cam1,
            verts_by_cam["camera1"],
            grid_size=occlusion_grid,
            occlusion_tau=occlusion_tau,
        )
        vis2 = compute_visibility_from_mesh_vertices(
            cam2,
            verts_by_cam["camera2"],
            grid_size=occlusion_grid,
            occlusion_tau=occlusion_tau,
        )
    else:
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

    # Bước 5: cập nhật A và suy ra F = P \ A.
    a_new = sorted(set(a_list) | set(k1_set) | set(k2_set))
    f_list = [n for n in names if n not in set(a_new)]

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
    optimized_data, res = optimize_f_points(
        {"camera1": cam1_corr, "camera2": cam2_corr},
        a_new,
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
        default=False,
        type=_parse_bool,
        help="Enable weighted proximity regularization. Accepts true/false.",
    )
    parser.add_argument(
        "--regularization-lambda",
        type=float,
        default=1.0,
        help="Lambda weight for weighted proximity regularization.",
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
    print(f"A_new = A U K1 U K2: {result['A_new']}")
    print(f"F = P \\ A_new: {result['F']}\n")

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
