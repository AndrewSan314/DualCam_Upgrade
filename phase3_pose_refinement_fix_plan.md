# Plan sửa thuật toán Phase-3 Pose Refinement

## 1. Mục tiêu

Mục tiêu của plan này là biến prototype hiện tại thành một thuật toán sửa pose có cơ sở hình học đúng hơn, tránh tình trạng metric giảm nhưng pose bị kéo sai. Trọng tâm là sửa các lỗi lớn trong các phần:

- occlusion/visibility;
- chuyển đổi giữa các hệ camera;
- cách chọn anchor/inlier;
- cách điền điểm bị khuất;
- objective tối ưu;
- ràng buộc nhân trắc học;
- validation để phát hiện pose bị kéo quá xa.

Plan này giả định file hiện tại là prototype single-frame, hardcoded data, chưa nối vào pipeline chính. Vì vậy, ưu tiên là tách lại interface, thêm calibration/camera model, đổi objective, rồi mới tối ưu thêm performance.

---

## 2. Chẩn đoán lỗi hiện tại

### 2.1. Sai mô hình occlusion

Hiện tại `compute_visibility_from_mesh_vertices()` raster mesh bằng tọa độ 3D `(x, y)` trực tiếp và dùng `abs(z)` làm depth. Cách này chỉ là heuristic, không phải occlusion theo camera thật.

Vấn đề:

- Không dùng camera intrinsics/extrinsics.
- Không project vertex/joint vào pixel image plane.
- Không raster triangle faces, chỉ raster vertices.
- Dùng `abs(z)` thay vì depth theo trục nhìn camera.
- Joint nằm ngoài bbox mesh bị mặc định visible, có thể sai nếu bbox/camera frame chưa chuẩn.

Hậu quả: `K1/K2` có thể sai, kéo theo việc fill joint sai và anchor sai.

### 2.2. Transform giữa camera bị fit từ skeleton nhiễu

Hiện tại `ransac_umeyama()` ước lượng similarity transform từ camera1 sang camera2 trên chính các joint trong frame. Cách này nguy hiểm nếu skeleton đã nhiễu hoặc có occlusion.

Vấn đề:

- Camera transform nên đến từ calibration/extrinsics, không fit mỗi frame từ pose lỗi.
- Umeyama similarity có scale tự do, có thể che giấu lỗi scale/depth.
- RANSAC chọn theo số lượng inlier, chưa tie-break bằng residual tổng/median.
- Không dùng confidence/visibility trong scoring.
- Không có min-inlier-ratio hoặc sanity check transform.

Hậu quả: transform có thể hợp thức hóa pose sai, sau đó dùng để fill occluded joints và làm anchor.

### 2.3. Imputed joints bị đưa vào anchor như điểm tin cậy

Hiện tại sau khi fill `K1/K2`, code gộp chúng vào `A_new = A ∪ K1 ∪ K2`, rồi dùng `A_new` làm anchor cho bước optimize.

Vấn đề:

- Joint được suy luận bằng transform không phải observation thật.
- Nếu transform sai, lỗi bị lan truyền vào toàn bộ F.
- Không có trạng thái `observed`, `imputed`, `low_confidence`, `rejected`.

Hậu quả: anchor có thể chứa điểm giả nhưng lại được xem là đáng tin.

### 2.4. Objective tối ưu không đảm bảo pose đúng

Hiện tại `get_diff_f()` đo `abs(distance_cam1(f, anchor) - distance_cam2(f, anchor))`. Sau đó `optimize_f_points()` tối ưu tọa độ F của cả hai camera để giảm mean/soft-tail của diff.

Vấn đề:

- Objective chỉ làm khoảng cách joint-anchor giống nhau giữa hai camera.
- Không dùng reprojection error về 2D keypoints.
- Không dùng camera calibration.
- Không giữ một skeleton world-space duy nhất.
- Tối ưu đồng thời tọa độ F ở cả hai camera tạo quá nhiều bậc tự do.
- Regularization mặc định tắt, nên solver có thể kéo joint đi rất xa.

Hậu quả: số liệu `before/after` đẹp hơn nhưng pose có thể sai vật lý.

### 2.5. Bone constraint quá yếu và hardcoded

Hiện tại `HEIGHT = 1.72`, bone length target = ratio * height, tolerance 5%.

Vấn đề:

- Chiều cao subject hardcoded.
- Chỉ có vài xương, thiếu torso/head/foot constraints.
- Không có joint angle limit.
- Không có temporal smoothness.
- Nếu anchor sai, bone constraint sẽ kéo F theo anchor sai.

Hậu quả: giảm biến dạng thô nhưng không đảm bảo pose đúng.

### 2.6. Confidence đang bị dùng như giảm trách nhiệm sửa lỗi

Low-confidence hoặc occluded joint bị giảm trọng số diff. Điều này có thể làm metric đẹp hơn nhưng không thật sự sửa joint xấu.

Vấn đề:

- Confidence thấp nên giảm trust vào observation, không nên làm lỗi biến mất khỏi báo cáo.
- Cần tách `observation_weight`, `anchor_weight`, `reporting_weight`, `prior_weight`.

Hậu quả: lỗi lớn ở joint kém tin cậy có thể bị che trong metric.

---

## 3. Nguyên tắc thiết kế mới

### 3.1. Một skeleton world-space duy nhất

Không tối ưu hai skeleton camera-space độc lập. Nên có biến chính:

```text
X_world[joint] ∈ R3
```

Sau đó project `X_world` vào từng camera để so với observation.

### 3.2. Calibration là nguồn transform chính

Phải có camera model:

```python
@dataclass
class CameraModel:
    K: np.ndarray      # 3x3 intrinsics
    R: np.ndarray      # 3x3 world -> camera
    t: np.ndarray      # 3 world -> camera
    dist: Optional[np.ndarray] = None
```

Các hàm cần có:

```python
world_to_camera(X_world, cam) -> X_cam
camera_to_world(X_cam, cam) -> X_world
project_world_to_pixel(X_world, cam) -> uv, depth
```

Nếu chưa có calibration, pipeline phải chạy ở chế độ `prototype_heuristic=True` và in warning rõ ràng, không được coi kết quả là production-grade.

### 3.3. Observation phải có trạng thái

Mỗi joint/camera cần metadata:

```python
@dataclass
class JointObservation:
    joint: str
    xyz_camera: Optional[np.ndarray]
    uv: Optional[np.ndarray]
    confidence: float
    visible: bool
    source: Literal["observed", "imputed", "missing", "rejected"]
```

Anchor chỉ được chọn từ observation thật, visible, confidence cao, và residual thấp.

### 3.4. Imputation không được trở thành anchor mạnh

Điểm fill từ transform/fusion phải có `source="imputed"` và weight thấp. Không được đưa thẳng vào anchor set như observation thật.

### 3.5. Objective phải đo đúng thứ cần đúng

Loss chính nên là:

```text
L = L_reprojection
  + λ_3d * L_3d_observation
  + λ_bone * L_bone
  + λ_angle * L_joint_angle
  + λ_temporal * L_temporal
  + λ_prior * L_proximity_prior
```

Trong đó:

- `L_reprojection`: so projected world joints với 2D detections.
- `L_3d_observation`: nếu có 3D camera-space observation đáng tin, transform về world rồi so.
- `L_bone`: giữ chiều dài xương hợp lý.
- `L_joint_angle`: tránh gập khớp phi sinh lý.
- `L_temporal`: nếu có video, tránh jitter frame-to-frame.
- `L_proximity_prior`: giữ gần estimate ban đầu, luôn bật với weight hợp lý.

---

## 4. Target pipeline mới

### 4.1. Input mới

Thay vì chỉ nhận `data_in` và `verts_by_cam`, pipeline nên nhận:

```python
def run_phase3_pipeline_v2(
    observations_by_cam: dict[str, dict[str, JointObservation]],
    cameras: dict[str, CameraModel],
    mesh_by_frame: Optional[MeshData],
    subject_profile: SubjectProfile,
    previous_pose_world: Optional[dict[str, np.ndarray]] = None,
    config: Phase3Config = Phase3Config(),
) -> Phase3Result:
    ...
```

### 4.2. Output mới

Output phải có cả pose và diagnostics:

```python
@dataclass
class Phase3Result:
    pose_world: dict[str, np.ndarray]
    observations_used: dict
    visibility: dict
    anchors: list[str]
    imputed: list[str]
    rejected: list[str]
    losses_before: dict[str, float]
    losses_after: dict[str, float]
    displacement_report: dict[str, float]
    warnings: list[str]
```

### 4.3. Luồng xử lý mới

```text
1. Validate input + camera calibration.
2. Chuẩn hóa observation về world coordinate.
3. Tính visibility bằng projection + z-buffer đúng camera.
4. Loại observation không visible hoặc confidence thấp khỏi anchor candidate.
5. Fuse multi-view joint observations thành X0_world.
6. Chọn anchors bằng robust residual trong world/reprojection space.
7. Impute missing/occluded joints bằng kinematic prior hoặc mirrored/body prior, weight thấp.
8. Optimize một skeleton world-space duy nhất.
9. Validate displacement, bone length, reprojection, temporal jump.
10. Xuất result + diagnostics.
```

---

## 5. Plan sửa theo phase

## Phase 0 - Thêm guardrail ngay để tránh kết quả sai nặng

Mục tiêu: chưa cần rewrite toàn bộ, nhưng ngăn solver tạo pose quá sai.

### Việc cần làm

1. Bật regularization mặc định:

```python
regularization=True
regularization_lambda >= 10.0  # tune bằng validation set
```

2. Thêm max displacement constraint:

```python
MAX_JOINT_MOVE_M = 0.10  # 10 cm default
```

Nếu joint dịch quá 10 cm so với input, reject update hoặc clamp.

3. Không đưa `K1/K2` vào anchor mạnh:

Thay:

```python
a_new = sorted(set(a_list) | set(k1_set) | set(k2_set))
```

bằng:

```python
observed_anchors = sorted(set(a_list))
imputed_joints = sorted(set(k1_set) | set(k2_set))
f_list = [n for n in names if n not in set(observed_anchors)]
```

4. Report thêm displacement:

```python
displacement = norm(x_after[joint] - x_before[joint])
```

5. Nếu `verts_by_cam is None`, không gọi đây là mesh occlusion. Chỉ ghi rõ là `mock_visibility`.

### Acceptance criteria

- Không joint nào bị dịch > 10 cm nếu không có lý do rõ ràng.
- Report có cảnh báo nếu dùng mock visibility.
- `K1/K2` không còn tự động trở thành anchor high-confidence.

---

## Phase 1 - Tách data model và diagnostics

Mục tiêu: chuẩn hóa dữ liệu để tránh nhầm camera-space/world-space.

### Việc cần làm

1. Tạo file mới hoặc section mới:

```text
phase3_types.py
```

2. Thêm dataclass:

```python
@dataclass
class Phase3Config:
    ransac_threshold_m: float = 0.05
    min_anchor_confidence: float = 0.5
    max_joint_move_m: float = 0.10
    regularization_lambda: float = 10.0
    occluded_observation_weight: float = 0.05
    imputed_anchor_weight: float = 0.0
    enable_world_optimization: bool = True
```

3. Thêm `JointState`:

```python
@dataclass
class JointState:
    name: str
    position: np.ndarray
    confidence: float
    visible: bool
    source: str  # observed/imputed/rejected/missing
```

4. Thêm diagnostic helpers:

```python
compute_joint_displacements(before, after)
compute_bone_errors(pose, subject_profile)
compute_reprojection_errors(pose_world, observations_2d, cameras)
```

### Acceptance criteria

- Output không chỉ có stats Q1/Q3/Mean/Median mà có per-joint diagnostics.
- Có phân biệt rõ `observed` và `imputed`.
- Có warning khi input thiếu calibration.

---

## Phase 2 - Sửa camera geometry

Mục tiêu: thay Umeyama per-frame bằng camera calibration.

### Việc cần làm

1. Implement camera transforms:

```python
def world_to_camera(Xw, cam):
    return cam.R @ Xw + cam.t


def camera_to_world(Xc, cam):
    return cam.R.T @ (Xc - cam.t)


def project_camera_to_pixel(Xc, cam):
    x = Xc[0] / Xc[2]
    y = Xc[1] / Xc[2]
    uv_h = cam.K @ np.array([x, y, 1.0])
    return uv_h[:2], Xc[2]
```

2. Nếu có observation camera-space 3D, transform về world:

```python
Xw_obs = camera_to_world(Xc_obs, camera)
```

3. Chỉ dùng `ransac_umeyama()` như fallback debug khi không có calibration:

```python
if cameras is None:
    warnings.append("No calibration provided; using heuristic Umeyama fallback")
```

4. Thêm sanity check cho fallback Umeyama:

- scale phải gần 1 nếu camera-space metric cùng đơn vị;
- determinant rotation gần +1;
- median residual < threshold;
- min inlier ratio >= 0.5 hoặc >= 6 joints.

### Acceptance criteria

- Production path không fit transform camera mỗi frame từ skeleton.
- Fallback Umeyama có warning và không dùng để đánh giá chất lượng chính.
- Có unit test round-trip `world -> camera -> world`.

---

## Phase 3 - Sửa occlusion đúng camera

Mục tiêu: visibility phải dựa trên projection và depth đúng camera.

### Việc cần làm

1. Thay `compute_visibility_from_mesh_vertices()` bằng:

```python
def compute_visibility_from_mesh(
    joints_world,
    mesh_world,
    camera,
    image_size,
    occlusion_tau=0.02,
):
    depth_buffer = rasterize_mesh_faces_to_depth(mesh_world, camera, image_size)
    visibility = {}
    for joint, Xw in joints_world.items():
        uv, z_joint = project_world_to_pixel(Xw, camera)
        if uv outside image:
            visibility[joint] = False
            continue
        z_surface = depth_buffer[round(v), round(u)]
        visibility[joint] = z_joint <= z_surface + occlusion_tau
    return visibility
```

2. Nếu chưa có faces, vẫn có thể làm vertex splatting nhưng phải project vertex vào pixel trước.

3. Không dùng `abs(z)`; depth phải là `Z_camera`.

4. Joint ngoài image không nên mặc định visible. Nên là `visible=False` hoặc `out_of_frame=True`.

5. Thêm mode debug xuất heatmap depth/visibility.

### Acceptance criteria

- Visibility test dùng camera intrinsics/extrinsics.
- Joint depth so với mesh depth cùng camera frame.
- Có test synthetic: mesh plane chắn trước joint -> invisible; joint trước plane -> visible.

---

## Phase 4 - Sửa anchor selection

Mục tiêu: anchor chỉ là các joint quan sát thật, ổn định, residual thấp.

### Việc cần làm

1. Tạo anchor candidate:

```python
candidate = observed and visible and confidence >= min_anchor_confidence
```

2. Tính residual theo world-space agreement:

```python
residual_j = robust_spread([Xw_obs_cam1[j], Xw_obs_cam2[j], ...])
```

hoặc theo reprojection:

```python
reproj_error_j_cam = ||project(Xw_fused[j], cam) - uv_obs[j, cam]||
```

3. Chọn anchor nếu:

```text
residual_j < threshold
observed_count >= 2 hoặc confidence rất cao từ 1 camera
not imputed
not rejected
```

4. RANSAC nếu vẫn cần thì chạy trên world observations, scoring có weighted residual:

```python
score = sum(weight_j * robust_loss(residual_j))
```

5. Tie-break RANSAC bằng median residual, không chỉ bằng count.

### Acceptance criteria

- `imputed` không bao giờ thành anchor mặc định.
- Anchor list có confidence và residual report.
- Nếu anchor ít hơn min count, pipeline fallback sang prior/regularized mode và warning.

---

## Phase 5 - Sửa imputation

Mục tiêu: điền điểm khuất bằng estimate hợp lý nhưng không biến nó thành ground truth.

### Việc cần làm

1. Với joint chỉ thấy ở một camera:

- transform observation về world;
- nếu confidence đủ, dùng nó làm initial estimate;
- vẫn giữ source là `observed_single_view`, không phải anchor mạnh.

2. Với joint missing/occluded cả hai camera:

Ưu tiên impute theo thứ tự:

1. previous frame pose nếu có;
2. kinematic chain từ parent/child visible;
3. symmetry prior từ bên đối diện;
4. subject neutral pose prior.

3. Gắn weight:

```python
source_weight = {
    "observed_multi_view": 1.0,
    "observed_single_view": 0.5,
    "imputed_kinematic": 0.2,
    "imputed_symmetry": 0.1,
    "missing": 0.0,
}
```

### Acceptance criteria

- Mọi imputed joint có cờ source rõ ràng.
- Loss/report không trộn imputed với observed.
- Không dùng imputed joint để kéo observed joint khác, trừ khi weight rất thấp.

---

## Phase 6 - Rewrite objective tối ưu

Mục tiêu: tối ưu một skeleton world-space duy nhất.

### Objective đề xuất

```text
L_total =
    λ_reproj * Σ_cam,j w_cam,j * ρ(||π_cam(X_j) - uv_cam,j||)
  + λ_3d     * Σ_cam,j w_cam,j * ρ(||T_cam_to_world(xyz_cam,j) - X_j||)
  + λ_bone   * Σ_bone ρ(||X_child - X_parent|| - target_bone_len)
  + λ_prior  * Σ_j prior_w_j * ρ(||X_j - X0_j||)
  + λ_temp   * Σ_j ρ(||X_j - X_prev_j||)
```

Dùng Huber/Cauchy loss thay vì L2 thuần:

```python
def huber(r, delta):
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r*r, delta * (a - 0.5 * delta))
```

### Việc cần làm

1. Tạo `optimize_world_pose()` mới, không sửa `cam1/cam2` độc lập.

2. Biến tối ưu:

```python
x = concat([X_world[j] for j in optimizable_joints])
```

3. Anchor observed high-confidence có thể fixed hoặc regularized mạnh.

4. Missing/imputed joints được optimize nhưng bị ràng buộc bằng bone/prior/temporal.

5. Luôn bật proximity prior. Không có mode default hoàn toàn tự do.

6. Thêm hard safety constraint:

```text
||X_j - X0_j|| <= max_joint_move_m
```

hoặc reject result nếu vi phạm.

### Acceptance criteria

- Không còn optimize hai pose camera-space độc lập.
- Loss giảm phải đi kèm reprojection/3D/bone diagnostics tốt hơn.
- Nếu metric giảm nhưng displacement tăng bất thường, result bị reject/warn.

---

## Phase 7 - Bone, angle, subject profile

Mục tiêu: ràng buộc nhân trắc học đúng hơn và tùy subject.

### Việc cần làm

1. Tạo `SubjectProfile`:

```python
@dataclass
class SubjectProfile:
    height_m: Optional[float]
    bone_lengths_m: dict[tuple[str, str], float]
    bone_tolerance_ratio: float = 0.05
```

2. Nếu có calibration/sequence tốt, estimate bone length từ median nhiều frame thay vì hardcoded height.

3. Thêm bone length cho torso:

- shoulder width;
- hip width;
- neck-shoulder;
- hip-shoulder/spine;
- optional foot/head nếu model có.

4. Thêm joint angle constraints cho elbow/knee:

```text
0° <= elbow flexion <= 170°
0° <= knee flexion <= 170°
```

Tùy hệ tọa độ và model, angle constraint nên là soft loss trước, chưa nên hard constraint ngay.

### Acceptance criteria

- Không dùng `HEIGHT = 1.72` hardcoded trong production path.
- Bone error report theo từng bone.
- Subject-specific bone length được cache/ước lượng ổn định.

---

## Phase 8 - Metrics và validation

Mục tiêu: không còn đánh giá bằng `diff` nội bộ dễ đánh lừa.

### Metrics bắt buộc

1. Reprojection error per camera/joint, đơn vị pixel.
2. 3D observation residual per camera/joint, đơn vị meter.
3. Bone length error per bone, đơn vị meter và percent.
4. Joint displacement before/after, đơn vị meter.
5. Temporal velocity/acceleration nếu có video.
6. Visibility confusion nếu có ground truth/synthetic occlusion.

### Reject rules đề xuất

Reject hoặc warning nếu:

```text
max_joint_displacement > 0.10 m
median_reprojection_error_after > before + 5 px
bone_error_after > 8% target length
number_of_anchors < 5
optimization_success == False
```

### Test cases cần có

#### Unit tests

- `camera_to_world(world_to_camera(X)) ≈ X`
- projection known point đúng pixel expected;
- occlusion synthetic plane;
- Huber loss finite gradient;
- bone length error đúng sign/value.

#### Regression tests

- Hardcoded current data: optimizer không được dịch joint > 10 cm khi regularization bật.
- Case one occluded hand: hand được impute nhưng không thành anchor.
- Case noisy knee: residual lớn thì reject hoặc weight thấp.

#### Integration tests

- Multi-frame short sequence: jitter after <= jitter before.
- Reprojection after <= before trên visible joints.
- Bone length median error giảm hoặc không tăng.

---

## 6. Thứ tự triển khai khuyến nghị

### Sprint 1 - Safety patch trong file hiện tại

- Bật regularization mặc định.
- Thêm max displacement report/reject.
- Không đưa K1/K2 vào `A_new` như anchor mạnh.
- Tách observed/imputed status.
- Thêm diagnostics per-joint.

Kết quả: prototype bớt nguy hiểm, vẫn chưa phải thuật toán đúng hoàn toàn.

### Sprint 2 - Camera model + world coordinate

- Thêm `CameraModel`.
- Implement transform/projection.
- Đổi data flow sang world-space.
- Umeyama chỉ còn fallback/debug.

Kết quả: nền hình học đúng hơn.

### Sprint 3 - Visibility đúng camera

- Project mesh vào từng camera.
- Tạo depth buffer theo pixel.
- Tính visibility bằng depth đúng.

Kết quả: `K1/K2` đáng tin hơn.

### Sprint 4 - Rewrite optimization

- Tạo `optimize_world_pose()`.
- Loss gồm reprojection, 3D observation, bone, prior, temporal.
- Dùng robust loss và reject rules.

Kết quả: metric giảm sẽ tương quan tốt hơn với pose đúng.

### Sprint 5 - Validation/benchmark

- Thêm synthetic tests.
- Thêm regression tests từ sample hiện tại.
- Chạy trên vài clip thật, log before/after.

Kết quả: biết thuật toán tốt lên ở đâu, xấu đi ở đâu.

---

## 7. Patch tối thiểu nên làm ngay trong `main.py`

Nếu chưa kịp rewrite lớn, làm ngay các thay đổi sau:

### 7.1. Đổi default regularization

```python
regularization=True
regularization_lambda=10.0
```

### 7.2. Không gộp K1/K2 vào anchor mạnh

Thay logic:

```python
a_new = sorted(set(a_list) | set(k1_set) | set(k2_set))
f_list = [n for n in names if n not in set(a_new)]
```

bằng:

```python
observed_anchors = sorted(set(a_list))
imputed_joints = sorted(set(k1_set) | set(k2_set))
f_list = [n for n in names if n not in set(observed_anchors)]
```

Sau đó truyền `observed_anchors` vào `calculate_stats()` và `optimize_f_points()`.

### 7.3. Thêm displacement guard

Sau optimize:

```python
def max_displacement(before, after, f_list):
    values = {}
    for cam in ["camera1", "camera2"]:
        values[cam] = {
            j: float(np.linalg.norm(after[cam][j] - before[cam][j]))
            for j in f_list
        }
    return values
```

Reject nếu joint dịch quá ngưỡng:

```python
if max_move > config.max_joint_move_m:
    warnings.append("Reject optimization: excessive joint displacement")
    optimized_data = before_data
```

### 7.4. Thêm warning cho mock occlusion

```python
if visibility_override is not None:
    warnings.append("Using visibility_override; not real mesh/camera occlusion")
elif verts_by_cam is None:
    warnings.append("No mesh vertices; all joints assumed visible")
```

### 7.5. Tách metric reporting khỏi optimization metric

Không dùng `weighted diff` làm bằng chứng pose tốt. Report thêm:

- displacement per joint;
- bone length error before/after;
- inlier residual từ RANSAC;
- number of anchors.

---

## 8. Quyết định nên bỏ hoặc giới hạn

### Nên bỏ khỏi production path

- Occlusion bằng raw `(x, y, abs(z))`.
- Fit camera transform per-frame bằng Umeyama từ skeleton lỗi.
- Tối ưu hai camera-space skeleton cùng lúc.
- Dùng `diff distance-to-anchor` làm metric chính.
- Gộp imputed joints vào anchors.

### Có thể giữ làm debug/fallback

- `estimate_umeyama()` giữ lại để kiểm tra consistency hoặc fallback khi thiếu calibration.
- `ransac_umeyama()` giữ lại nhưng cần weighted scoring và warning.
- `soft_tail()` có thể giữ như robust statistic phụ, không phải objective chính.

---

## 9. Definition of Done

Một bản fix được xem là đạt nếu thỏa các điều kiện:

1. Có camera calibration path rõ ràng.
2. Pose cuối là một skeleton world-space duy nhất.
3. Visibility dùng projection/depth đúng camera.
4. Imputed joints không thành anchor mạnh.
5. Objective có reprojection hoặc world-space residual thật.
6. Regularization/prior luôn bật.
7. Có displacement guard.
8. Có per-joint/per-bone diagnostics.
9. Có unit/regression tests.
10. Trên sample hiện tại, không còn hiện tượng metric giảm bằng cách kéo joint hàng chục cm.

---

## 10. Pseudocode pipeline cuối cùng

```python
def run_phase3_pipeline_v2(obs_by_cam, cameras, mesh, subject, prev_pose=None, config=None):
    config = config or Phase3Config()
    warnings = []

    validate_observations(obs_by_cam)
    validate_cameras(cameras)

    # 1. Convert observations to world coordinates.
    obs_world = build_world_observations(obs_by_cam, cameras)

    # 2. Compute visibility using real projection/depth if mesh is available.
    visibility = compute_multicam_visibility(obs_world, mesh, cameras, config)

    # 3. Fuse initial world pose.
    X0, joint_sources = fuse_observations(obs_world, visibility, config)

    # 4. Select anchors from observed high-confidence multi-view joints only.
    anchors, rejected = select_anchors(X0, obs_world, visibility, config)

    # 5. Impute missing/occluded joints with low source weight.
    X_init = impute_missing_joints(X0, anchors, subject, prev_pose, config)

    # 6. Optimize one world-space skeleton.
    X_opt, opt_info = optimize_world_pose(
        X_init,
        obs_by_cam,
        cameras,
        subject,
        anchors,
        joint_sources,
        prev_pose,
        config,
    )

    # 7. Validate and possibly reject.
    diagnostics = compute_diagnostics(X_init, X_opt, obs_by_cam, cameras, subject)
    if should_reject_result(diagnostics, config):
        warnings.append("Optimization rejected by safety checks")
        X_opt = X_init

    return Phase3Result(
        pose_world=X_opt,
        observations_used=obs_by_cam,
        visibility=visibility,
        anchors=anchors,
        imputed=[j for j, s in joint_sources.items() if "imputed" in s],
        rejected=rejected,
        losses_before=diagnostics["before"],
        losses_after=diagnostics["after"],
        displacement_report=diagnostics["displacement"],
        warnings=warnings,
    )
```

---

## 11. Ghi chú triển khai

- Nếu chưa có 2D keypoints, vẫn nên chuyển 3D camera-space observation về world rồi tối ưu một world skeleton. Tuy nhiên, thiếu reprojection error thì kết quả vẫn yếu hơn.
- Nếu chưa có calibration, không nên gọi thuật toán là multi-view correction production. Chỉ nên chạy heuristic debug.
- Nếu chỉ có một frame, temporal loss không dùng được; khi có sequence, temporal prior nên bật vì nó giúp chống pose jump rất mạnh.
- Nếu mesh không có faces, vertex-based visibility chỉ là fallback tạm. Cần ghi warning trong result.
- Mọi metric phải tách theo source: observed vs imputed. Không trộn chung để tránh che lỗi.
