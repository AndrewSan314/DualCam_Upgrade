# Kế hoạch implement fix toàn diện cho Pose Judgement Optimization

## 1. Mục tiêu

Sửa pipeline **Pose Judgement Optimization (J)** để sau khi qua J, pose 3D được cải thiện thật sự, thay vì làm hỏng pose ổn định từ **Pose Refinement Optimization (R)**.

Vấn đề hiện tại:

```python
result = original.run_phase3_pipeline(...)
opt1 = result["optimized"]["camera1"]
opt2 = result["optimized"]["camera2"]
fused[frame_idx] = _joint_dicts_to_average_array(opt1, opt2, joint_names)
```

Tức là J đang:

1. tối ưu từng frame độc lập;
2. lấy `optimized camera1` và `optimized camera2`;
3. trung bình hai pose này;
4. ghi đè vào `fused[frame_idx]`.

Hậu quả:

- pose từ R đang ổn bị overwrite;
- frame trước/sau không được xét đến nên dễ jitter;
- nếu hai camera không cùng hệ tọa độ, average sẽ làm pose méo;
- chân/tay có thể bị kéo dài, bắt chéo hoặc sai giải phẫu.

Mục tiêu sau khi sửa:

```text
J không còn là per-frame overwrite module.
J trở thành temporal multi-view pose fusion/optimization module.
J dùng R pose làm prior.
J dùng left/right camera làm evidence.
J dùng visibility/confidence để chọn nguồn tin tốt.
J tối ưu theo window thời gian.
J giữ bone length, temporal smoothness, floor/contact constraints.
J chỉ chấp nhận output nếu tốt hơn hoặc ít nhất không phá pose từ R.
```

---

## 2. Nguyên tắc thiết kế mới

### 2.1. Không average trực tiếp `opt1` và `opt2`

Không dùng:

```python
candidate = (opt1 + opt2) / 2
```

Lý do:

- `opt1` và `opt2` có thể khác coordinate system;
- hai camera có thể khác translation, scale, rotation;
- average raw XYZ không đảm bảo bone length;
- camera bị occlusion có thể kéo joint sai;
- average không có temporal consistency.

Thay vào đó:

```text
left/right pose chỉ là evidence.
pose cuối phải được sinh bởi optimizer có ràng buộc.
```

### 2.2. R pose là base prior

R đã ổn định hơn, nên J phải tối ưu từ R:

```python
base_pose = pose_after_R  # [T, J, 3]
```

J tìm pose mới:

```text
pose_J = argmin loss(pose)
```

Nhưng pose mới phải:

- gần evidence tốt từ hai camera;
- không lệch bone length;
- không jitter;
- không xuyên sàn;
- không tăng foot sliding;
- không tệ hơn R theo validation score.

### 2.3. J phải tối ưu theo window, không theo từng frame

Không làm:

```python
for frame_idx in range(T):
    optimize_one_frame(frame_idx)
```

Mà làm:

```python
for window in sliding_windows(T, window_size=32, stride=8):
    optimize_window(window)
```

Lý do:

- pose người liên tục theo thời gian;
- cần temporal velocity/acceleration loss;
- tối ưu frame độc lập gây giật;
- window giúp sửa occlusion/hallucination tốt hơn.

### 2.4. Phải thống nhất coordinate system trước khi fusion

Nếu có calibration/extrinsics:

```python
P_world = R_cam_to_world @ P_cam + t_cam_to_world
```

Nếu chưa có extrinsics:

```python
left_candidate = align_sequence_to_base(left_candidate, base_pose)
right_candidate = align_sequence_to_base(right_candidate, base_pose)
```

Quan trọng: align theo cả sequence hoặc window, **không align từng frame riêng lẻ**, vì align từng frame sẽ gây jitter.

---

## 3. Kiến trúc J mới

Đề xuất thêm các file:

```text
pose_pipeline/
└── pipelines/
    ├── judgement.py
    ├── judgement_alignment.py
    ├── judgement_weights.py
    ├── judgement_losses.py
    ├── judgement_optimizer.py
    └── judgement_validation.py
```

| File | Nhiệm vụ |
|---|---|
| `judgement.py` | wrapper chính |
| `judgement_alignment.py` | đưa left/right candidate về hệ tọa độ của R |
| `judgement_weights.py` | tính visibility/confidence/view weight |
| `judgement_losses.py` | định nghĩa các loss |
| `judgement_optimizer.py` | temporal multi-view optimizer |
| `judgement_validation.py` | validate output và fallback |

---

## 4. Interface mới cho J

### Input

```python
JInput = {
    "base_pose": np.ndarray,        # [T, J, 3], pose từ R
    "left_pose": np.ndarray,        # [T, J, 3]
    "right_pose": np.ndarray,       # [T, J, 3]
    "left_confidence": np.ndarray,  # [T, J] hoặc None
    "right_confidence": np.ndarray, # [T, J] hoặc None
    "calib_left": dict | None,
    "calib_right": dict | None,
    "joint_names": list[str],
    "skeleton_edges": list[tuple[int, int]],
}
```

### Output

```python
JOutput = {
    "pose_judged": np.ndarray,      # [T, J, 3]
    "confidence_final": np.ndarray, # [T, J]
    "selected_source": np.ndarray,  # [T, J], left/right/base
    "metadata": {
        "loss_before": dict,
        "loss_after": dict,
        "validation": dict,
        "view_weights": dict,
        "rejected_windows": list,
    }
}
```

---

## 5. Logic mới trong `judgement.py`

Thay logic per-frame overwrite bằng flow sau:

```python
def run_pose_judgement(pose_data: dict, config: dict) -> dict:
    base_pose = get_current_fused_or_refined_pose(pose_data)

    raw_result = run_original_judgement_for_metadata(
        pose_data=pose_data,
        config=config,
    )

    left_candidate, right_candidate = extract_judgement_candidates(
        raw_result=raw_result,
        joint_names=config["joint_names"],
        num_frames=base_pose.shape[0],
    )

    left_candidate, right_candidate = align_candidates_to_base(
        left_candidate=left_candidate,
        right_candidate=right_candidate,
        base_pose=base_pose,
        config=config,
    )

    view_weights = build_view_weights(
        raw_result=raw_result,
        pose_data=pose_data,
        left_candidate=left_candidate,
        right_candidate=right_candidate,
        base_pose=base_pose,
        config=config,
    )

    pose_judged, diagnostics = optimize_temporal_multiview_pose(
        base_pose=base_pose,
        left_candidate=left_candidate,
        right_candidate=right_candidate,
        view_weights=view_weights,
        skeleton_edges=config["skeleton_edges"],
        config=config,
    )

    pose_final, validation = validate_or_fallback_sequence(
        base_pose=base_pose,
        pose_judged=pose_judged,
        diagnostics=diagnostics,
        skeleton_edges=config["skeleton_edges"],
        config=config,
    )

    pose_data["fused"]["poses_3d"] = pose_final
    pose_data["fused"]["metadata"]["judgement"] = {
        "raw_result": raw_result,
        "diagnostics": diagnostics,
        "validation": validation,
        "mode": "temporal_multiview_optimize",
    }

    return pose_data
```

---

## 6. Alignment module

File: `pose_pipeline/pipelines/judgement_alignment.py`

### Root-center + scale align

```python
def root_center_pose(pose: np.ndarray, root_idx: int = 0):
    root = pose[:, root_idx:root_idx + 1, :]
    return pose - root, root
```

```python
def sequence_scale_to_base(candidate: np.ndarray, base: np.ndarray, root_idx: int = 0):
    base_centered, base_root = root_center_pose(base, root_idx)
    cand_centered, _ = root_center_pose(candidate, root_idx)

    base_norm = np.sqrt((base_centered ** 2).sum(axis=-1)).mean()
    cand_norm = np.sqrt((cand_centered ** 2).sum(axis=-1)).mean()

    scale = base_norm / (cand_norm + 1e-8)
    return cand_centered * scale + base_root
```

### Optional: Umeyama alignment theo sequence

```python
def align_sequence_umeyama(candidate: np.ndarray, base: np.ndarray) -> np.ndarray:
    source = candidate.reshape(-1, 3)
    target = base.reshape(-1, 3)

    R, s, t = estimate_similarity_transform(source, target)

    aligned = s * (candidate @ R.T) + t
    return aligned
```

### API chính

```python
def align_candidates_to_base(left_candidate, right_candidate, base_pose, config):
    method = config["judgement"].get("coordinate_alignment", "root_scale")

    if method == "root_scale":
        left = sequence_scale_to_base(left_candidate, base_pose)
        right = sequence_scale_to_base(right_candidate, base_pose)
    elif method == "sequence_umeyama":
        left = align_sequence_umeyama(left_candidate, base_pose)
        right = align_sequence_umeyama(right_candidate, base_pose)
    else:
        left = left_candidate
        right = right_candidate

    return left, right
```

---

## 7. View weights module

File: `pose_pipeline/pipelines/judgement_weights.py`

### Mục tiêu

Tạo weight theo từng frame/joint/view:

```python
w_left[t, j]
w_right[t, j]
w_base[t, j]
```

### Weight cơ bản

```python
def build_basic_view_weights(left_conf, right_conf, left_visibility, right_visibility):
    if left_conf is None:
        left_conf = np.ones_like(left_visibility)
    if right_conf is None:
        right_conf = np.ones_like(right_visibility)
    if left_visibility is None:
        left_visibility = np.ones_like(left_conf)
    if right_visibility is None:
        right_visibility = np.ones_like(right_conf)

    w_left = left_conf * left_visibility
    w_right = right_conf * right_visibility

    denom = w_left + w_right + 1e-8
    return w_left / denom, w_right / denom
```

### Giảm weight nếu hai camera mâu thuẫn

```python
def downweight_disagreement(left, right, w_left, w_right, threshold_m=0.25):
    dist = np.linalg.norm(left - right, axis=-1)
    conflict = dist > threshold_m

    w_left = w_left.copy()
    w_right = w_right.copy()

    w_left[conflict] *= 0.3
    w_right[conflict] *= 0.3

    return w_left, w_right
```

### Base prior weight

```python
def compute_base_prior_weight(w_left, w_right, min_prior=0.2, max_prior=1.0):
    evidence = np.clip(w_left + w_right, 0.0, 1.0)
    w_base = max_prior - evidence * (max_prior - min_prior)
    return w_base
```

---

## 8. Loss function cho J optimizer

File: `pose_pipeline/pipelines/judgement_losses.py`

Loss tổng:

```text
L = L_data
  + λ_prior * L_R_prior
  + λ_bone * L_bone
  + λ_temp * L_temporal
  + λ_acc * L_acceleration
  + λ_floor * L_floor
  + λ_contact * L_foot_contact
  + λ_sym * L_symmetry
```

### Data loss

```python
def data_loss(X, left, right, w_left, w_right):
    return (
        w_left[..., None] * (X - left).pow(2)
        + w_right[..., None] * (X - right).pow(2)
    ).mean()
```

### Prior từ R

```python
def base_prior_loss(X, base, w_base):
    return (w_base[..., None] * (X - base).pow(2)).mean()
```

### Bone length loss

```python
def compute_reference_bone_lengths(base, skeleton_edges):
    lengths = []
    for a, b in skeleton_edges:
        bone = base[:, a, :] - base[:, b, :]
        length = torch.linalg.norm(bone, dim=-1)
        lengths.append(torch.median(length))
    return torch.stack(lengths)
```

```python
def bone_length_loss(X, skeleton_edges, ref_lengths):
    losses = []
    for idx, (a, b) in enumerate(skeleton_edges):
        bone = X[:, a, :] - X[:, b, :]
        length = torch.linalg.norm(bone, dim=-1)
        losses.append((length - ref_lengths[idx]).pow(2).mean())
    return torch.stack(losses).mean()
```

### Temporal velocity loss

```python
def temporal_velocity_loss(X):
    if X.shape[0] < 2:
        return X.new_tensor(0.0)
    velocity = X[1:] - X[:-1]
    return velocity.pow(2).mean()
```

### Acceleration loss

```python
def temporal_acceleration_loss(X):
    if X.shape[0] < 3:
        return X.new_tensor(0.0)
    acceleration = X[2:] - 2 * X[1:-1] + X[:-2]
    return acceleration.pow(2).mean()
```

### Floor penetration loss

```python
def floor_penetration_loss(X, foot_indices, floor_axis=2, floor_value=0.0):
    feet = X[:, foot_indices, floor_axis]
    penetration = torch.clamp(floor_value - feet, min=0.0)
    return penetration.pow(2).mean()
```

### Foot contact loss

```python
def foot_contact_loss(X, foot_indices):
    if X.shape[0] < 2:
        return X.new_tensor(0.0)

    feet = X[:, foot_indices, :]
    velocity = torch.linalg.norm(feet[1:] - feet[:-1], dim=-1)

    loss = velocity.pow(2).mean()
    return loss
```

---

## 9. Temporal optimizer

File: `pose_pipeline/pipelines/judgement_optimizer.py`

### Optimizer chính

```python
def optimize_temporal_multiview_pose(
    base_pose,
    left_candidate,
    right_candidate,
    view_weights,
    skeleton_edges,
    config,
):
    import torch

    cfg = config["judgement"]
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    base = torch.tensor(base_pose, dtype=torch.float32, device=device)
    left = torch.tensor(left_candidate, dtype=torch.float32, device=device)
    right = torch.tensor(right_candidate, dtype=torch.float32, device=device)

    w_left = torch.tensor(view_weights["left"], dtype=torch.float32, device=device)
    w_right = torch.tensor(view_weights["right"], dtype=torch.float32, device=device)
    w_base = torch.tensor(view_weights["base"], dtype=torch.float32, device=device)

    T = base.shape[0]
    window_size = cfg.get("window_size", 32)
    stride = cfg.get("stride", 8)

    output_sum = torch.zeros_like(base)
    output_weight = torch.zeros((T, 1, 1), dtype=torch.float32, device=device)

    diagnostics = {"windows": []}

    for start in range(0, T, stride):
        end = min(start + window_size, T)

        if end - start < 3:
            continue

        X_window, window_diag = optimize_one_window(
            base=base[start:end],
            left=left[start:end],
            right=right[start:end],
            w_left=w_left[start:end],
            w_right=w_right[start:end],
            w_base=w_base[start:end],
            skeleton_edges=skeleton_edges,
            config=config,
        )

        weight = make_overlap_weight(end - start, device=device)

        output_sum[start:end] += X_window * weight
        output_weight[start:end] += weight

        diagnostics["windows"].append({
            "start": start,
            "end": end,
            **window_diag,
        })

        if end == T:
            break

    pose_out = output_sum / torch.clamp(output_weight, min=1e-8)
    return pose_out.detach().cpu().numpy(), diagnostics
```

### Optimize one window

```python
def optimize_one_window(base, left, right, w_left, w_right, w_base, skeleton_edges, config):
    import torch

    cfg = config["judgement"]

    X = torch.nn.Parameter(base.clone())

    ref_bone_lengths = compute_reference_bone_lengths(base, skeleton_edges)

    optimizer = torch.optim.Adam([X], lr=cfg.get("lr", 0.03))

    losses_history = []

    for iteration in range(cfg.get("iters", 80)):
        optimizer.zero_grad()

        l_data = data_loss(X, left, right, w_left, w_right)
        l_prior = base_prior_loss(X, base, w_base)
        l_bone = bone_length_loss(X, skeleton_edges, ref_bone_lengths)
        l_temp = temporal_velocity_loss(X)
        l_acc = temporal_acceleration_loss(X)

        l_floor = floor_penetration_loss(
            X,
            foot_indices=cfg["foot_indices"],
            floor_axis=cfg.get("floor_axis", 2),
            floor_value=cfg.get("floor_value", 0.0),
        )

        l_contact = foot_contact_loss(X, foot_indices=cfg["foot_indices"])

        loss = (
            cfg.get("lambda_data", 1.0) * l_data
            + cfg.get("lambda_prior", 0.4) * l_prior
            + cfg.get("lambda_bone", 8.0) * l_bone
            + cfg.get("lambda_temp", 0.3) * l_temp
            + cfg.get("lambda_acc", 1.5) * l_acc
            + cfg.get("lambda_floor", 3.0) * l_floor
            + cfg.get("lambda_contact", 2.0) * l_contact
        )

        loss.backward()
        optimizer.step()

        losses_history.append(float(loss.detach().cpu()))

    return X.detach(), {
        "loss_start": losses_history[0],
        "loss_end": losses_history[-1],
    }
```

---

## 10. Validation và fallback

File: `pose_pipeline/pipelines/judgement_validation.py`

### Bone deviation

```python
def compute_bone_deviation_ratio(pose, base, skeleton_edges):
    deviations = []

    for a, b in skeleton_edges:
        len_pose = np.linalg.norm(pose[:, a, :] - pose[:, b, :], axis=-1)
        len_base = np.linalg.norm(base[:, a, :] - base[:, b, :], axis=-1)

        ratio = np.abs(len_pose - len_base) / (len_base + 1e-8)
        deviations.append(ratio)

    return np.stack(deviations, axis=-1)
```

### Velocity/acceleration

```python
def compute_joint_velocity(pose):
    return np.linalg.norm(pose[1:] - pose[:-1], axis=-1)
```

```python
def compute_joint_acceleration(pose):
    return np.linalg.norm(pose[2:] - 2 * pose[1:-1] + pose[:-2], axis=-1)
```

### Validate sequence

```python
def validate_or_fallback_sequence(base_pose, pose_judged, diagnostics, skeleton_edges, config):
    cfg = config["judgement"]

    validation = {
        "accepted": True,
        "reasons": [],
    }

    bone_dev = compute_bone_deviation_ratio(pose_judged, base_pose, skeleton_edges)
    max_bone_dev = float(np.max(bone_dev))

    if max_bone_dev > cfg.get("max_bone_deviation_ratio", 0.2):
        validation["accepted"] = False
        validation["reasons"].append(f"max bone deviation too high: {max_bone_dev:.4f}")

    velocity = compute_joint_velocity(pose_judged)
    max_velocity = float(np.max(velocity))

    if max_velocity > cfg.get("max_joint_velocity_m_per_frame", 0.35):
        validation["accepted"] = False
        validation["reasons"].append(f"max joint velocity too high: {max_velocity:.4f}")

    acceleration = compute_joint_acceleration(pose_judged)
    max_acc = float(np.max(acceleration))

    if max_acc > cfg.get("max_joint_acceleration_m_per_frame2", 0.45):
        validation["accepted"] = False
        validation["reasons"].append(f"max acceleration too high: {max_acc:.4f}")

    if validation["accepted"]:
        return pose_judged, validation

    return base_pose, validation
```

---

## 11. Config đề xuất

```yaml
judgement:
  mode: temporal_multiview_optimize

  coordinate_alignment: sequence_umeyama
  use_camera_extrinsics_if_available: true

  window_size: 32
  stride: 8
  iters: 80
  lr: 0.03

  camera_disagreement_threshold_m: 0.25
  min_base_prior_weight: 0.2
  max_base_prior_weight: 1.0

  lambda_data: 1.0
  lambda_prior: 0.4
  lambda_bone: 8.0
  lambda_temp: 0.3
  lambda_acc: 1.5
  lambda_floor: 3.0
  lambda_contact: 2.0

  floor_axis: 2
  floor_value: 0.0

  max_bone_deviation_ratio: 0.20
  max_joint_velocity_m_per_frame: 0.35
  max_joint_acceleration_m_per_frame2: 0.45

  fallback_mode: base

  foot_indices:
    - 7
    - 8
    - 10
    - 11
```

Lưu ý: `foot_indices` cần sửa theo joint mapping thực tế của WHAM/SMPL.

---

## 12. CLI đề xuất

```python
parser.add_argument(
    "--judgement-mode",
    choices=[
        "metadata_only",
        "safe_fusion",
        "temporal_multiview_optimize",
        "full_legacy",
    ],
    default="temporal_multiview_optimize",
)

parser.add_argument("--judgement-window-size", type=int, default=32)
parser.add_argument("--judgement-stride", type=int, default=8)
parser.add_argument("--judgement-iters", type=int, default=80)
parser.add_argument("--judgement-lambda-bone", type=float, default=8.0)
parser.add_argument("--judgement-lambda-acc", type=float, default=1.5)
```

Ý nghĩa mode:

| Mode | Ý nghĩa |
|---|---|
| `metadata_only` | J chỉ lấy metadata, không sửa pose |
| `safe_fusion` | align + confidence fusion + validation |
| `temporal_multiview_optimize` | optimizer theo window, dùng constraints |
| `full_legacy` | mode cũ, chỉ để debug |

---

## 13. Lộ trình implement

### Giai đoạn 1: Chặn legacy overwrite

- [ ] Tìm đoạn `fused[frame_idx] = _joint_dicts_to_average_array(...)`.
- [ ] Bỏ khỏi default path.
- [ ] Thêm `judgement_mode`.
- [ ] Giữ mode `full_legacy` để debug.
- [ ] Mode mặc định không dùng overwrite cũ.

Deliverable:

```text
RJL không còn bị J phá pose từ R.
```

### Giai đoạn 2: Extract candidate theo sequence

- [ ] Viết `_joint_dict_to_array()`.
- [ ] Extract `left_candidate`.
- [ ] Extract `right_candidate`.
- [ ] Đảm bảo shape `[T, J, 3]`.
- [ ] Lưu debug `.npy`.

Deliverable:

```text
Có candidate từ J để inspect trước khi optimize.
```

### Giai đoạn 3: Coordinate alignment

- [ ] Implement root/scale alignment.
- [ ] Implement sequence Umeyama alignment nếu cần.
- [ ] Render debug base/left/right trước và sau align.
- [ ] Kiểm tra range XYZ, bone length.

Deliverable:

```text
Left/right candidate cùng hệ tọa độ với pose R.
```

### Giai đoạn 4: View weights

- [ ] Extract confidence/visibility.
- [ ] Build `w_left`, `w_right`, `w_base`.
- [ ] Downweight khi left/right disagreement quá lớn.
- [ ] Tăng base prior khi evidence yếu.

Deliverable:

```text
J biết camera nào đáng tin theo từng frame/joint.
```

### Giai đoạn 5: Safe fusion baseline

- [ ] Implement weighted fusion.
- [ ] Thêm bone validation.
- [ ] Thêm velocity/acceleration validation.
- [ ] Fallback về R nếu fusion hỏng.

Deliverable:

```text
Có bản fusion an toàn, chưa cần optimizer sâu.
```

### Giai đoạn 6: Temporal optimizer

- [ ] Implement loss functions.
- [ ] Implement optimize one window.
- [ ] Implement sliding window.
- [ ] Implement overlap weighting.
- [ ] Render output video.

Deliverable:

```text
J temporal optimize chạy end-to-end và pose mượt hơn.
```

### Giai đoạn 7: Diagnostics

- [ ] Log loss trước/sau.
- [ ] Log bone deviation.
- [ ] Log acceleration spike.
- [ ] Log rejected windows.
- [ ] Xuất `outputs/logs/judgement_diagnostics.json`.

Deliverable:

```text
Có số liệu chứng minh J cải thiện hoặc không cải thiện.
```

### Giai đoạn 8: Partial fallback

- [ ] Detect frame/joint hỏng.
- [ ] Chỉ fallback phần hỏng về base pose.
- [ ] Smooth transition giữa accepted/fallback frames.

Deliverable:

```text
J không bị reject toàn sequence chỉ vì vài frame lỗi.
```

### Giai đoạn 9: Tích hợp Learnable SMPLify

- [ ] Dùng J để tạo view weights/confidence map.
- [ ] Truyền weights sang Learnable SMPLify.
- [ ] Tối ưu SMPL parameters thay vì raw XYZ nếu API hỗ trợ.
- [ ] So sánh `R -> J temporal`, `R -> J temporal -> L`, `R -> J evidence -> L weighted`.

Deliverable:

```text
Pipeline cuối hợp giải phẫu hơn và ổn định hơn.
```

---

## 14. Test cases

Chạy các mode:

```bash
python main.py --sequence RJL --judgement-mode metadata_only
python main.py --sequence RJL --judgement-mode safe_fusion
python main.py --sequence RJL --judgement-mode temporal_multiview_optimize
python main.py --sequence RJL --judgement-mode full_legacy
```

So sánh:

| Output | Kỳ vọng |
|---|---|
| `metadata_only` | baseline an toàn |
| `safe_fusion` | không hỏng hơn baseline |
| `temporal_multiview_optimize` | mượt hơn, ít jitter hơn |
| `full_legacy` | chỉ để chứng minh lỗi cũ |

Metrics cần tính:

- mean/max bone deviation;
- mean/max joint velocity;
- mean/max acceleration;
- floor penetration;
- foot sliding;
- camera disagreement;
- benchmark MPJPE nếu có ground truth.

---

## 15. Tiêu chí nghiệm thu

J được xem là fix thành công khi:

- [ ] Không còn average trực tiếp `opt1/opt2` để ghi pose final.
- [ ] Candidate left/right được align về hệ tọa độ của R.
- [ ] Có view weights theo frame/joint.
- [ ] Có temporal optimizer theo window.
- [ ] Có bone length constraints.
- [ ] Có velocity/acceleration constraints.
- [ ] Có floor/contact constraints.
- [ ] Có validation/fallback.
- [ ] Có diagnostics chứng minh chất lượng.
- [ ] Video output không còn chân/tay kéo dài bất thường.
- [ ] Pose sau J ít jitter hơn pose trước J.
- [ ] J cải thiện ít nhất một metric: temporal smoothness, view consistency, foot sliding, occlusion robustness hoặc benchmark MPJPE.

---

## 16. Kết luận

Cách fix toàn diện cho J không phải là `blend`.

`blend` chỉ là guardrail tạm thời.

Fix đúng là biến J thành:

```text
temporal multi-view pose fusion optimizer
```

Trong đó:

```text
R cung cấp base pose ổn định.
J cung cấp view scoring, visibility, confidence.
Optimizer dùng left/right evidence, R prior, bone constraints, temporal constraints.
Validation đảm bảo J không làm hỏng skeleton.
Learnable SMPLify là hướng nâng cấp để tối ưu trong không gian SMPL thay vì raw joint XYZ.
```

Mục tiêu cuối cùng:

```text
Sau J, pose phải:
- ít jitter hơn;
- ít hallucination do occlusion hơn;
- chọn đúng camera hơn;
- giữ bone length ổn định;
- không tạo chân/tay kéo dài;
- không phá pose tốt từ R;
- có thể chứng minh cải thiện bằng metrics và video comparison.
```
