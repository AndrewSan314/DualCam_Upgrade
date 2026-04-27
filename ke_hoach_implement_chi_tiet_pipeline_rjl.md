# Kế hoạch implement chi tiết chương trình Pose 3D với pipeline R/J/L linh hoạt

## 1. Mục tiêu tổng thể

Xây dựng chương trình console cho bài toán nhận diện và xử lý pose 3D từ hai monocular camera.

Input mặc định:

```text
cam_left.mp4
cam_right.mp4
left.pkl
right.pkl
```

Output cuối:

```text
cam_left_right_3D_poses.mp4
waveform_analysis_left_arm_wrist.png
waveform_analysis_right_arm_wrist.png
waveform_analysis_left_thigh_lower_leg.png
waveform_analysis_right_thigh_lower_leg.png
run_log.txt
benchmark_result.json nếu có benchmark
```

Chương trình phải hỗ trợ nhiều thứ tự pipeline:

```text
RJL
LJR
LJRL
LRJ
LRJL
RLJ
RLJL
JLR
JLRL
JRL
```

Trong đó:

| Ký hiệu | Pipeline |
|---|---|
| R | Pose Refinement Optimization |
| J | Pose Judgement Optimization |
| L | Learnable SMPLify |

Các pipeline có thể được chạy theo nhiều thứ tự, nên không được hard-code riêng từng sequence. Cần implement theo kiến trúc **state-based pipeline executor**.

---

## 2. Vấn đề cần xử lý

### 2.1. Không thể giả định input/output cố định

Ví dụ:

### Sequence `RJL`

```text
left.pkl + right.pkl
=> R
=> unify_R.pkl
=> J
=> unify_RJ.pkl
=> L
=> unify_RJL.pkl
```

Ở đây `R` nhận hai file left/right rồi tạo unified pose.

### Sequence `LJR`

```text
left.pkl + right.pkl
=> L
=> left_L.pkl + right_L.pkl
=> J
=> left_LJ.pkl + right_LJ.pkl hoặc unify_LJ.pkl
=> R
=> unify_LJR.pkl
```

Ở đây `L` chạy riêng cho từng camera. Sau đó `J` có thể vẫn giữ dual output hoặc tạo unified output. Cuối cùng `R` mới tạo unified pose.

### Kết luận

Không thể viết code kiểu:

```python
if sequence == "RJL":
    ...
elif sequence == "LJR":
    ...
elif sequence == "LJRL":
    ...
```

Vì sẽ rất khó bảo trì và dễ sai.

Cần viết theo kiểu:

```text
state hiện tại là dual hay unified?
pipeline tiếp theo là R/J/L?
pipeline đó biết xử lý state hiện tại như thế nào?
sau pipeline, state được cập nhật ra sao?
```

---

## 3. Kiến trúc tổng thể

Đề xuất cấu trúc thư mục:

```text
project_root/
│
├── main.py
├── config.py
├── requirements.txt
├── README.md
│
├── pose_pipeline/
│   ├── __init__.py
│   ├── state.py
│   ├── executor.py
│   ├── naming.py
│   ├── validation.py
│   ├── schema.py
│   │
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── refinement.py
│   │   ├── judgement.py
│   │   ├── judgement_alignment.py
│   │   ├── judgement_weights.py
│   │   ├── judgement_losses.py
│   │   ├── judgement_optimizer.py
│   │   ├── judgement_validation.py
│   │   └── learnable_smplify.py
│   │
│   ├── io/
│   │   ├── __init__.py
│   │   ├── pkl_io.py
│   │   ├── video_io.py
│   │   └── calibration_io.py
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── video_composer.py
│   │   ├── pose_renderer.py
│   │   └── waveform.py
│   │
│   └── benchmark/
│       ├── __init__.py
│       └── evaluator.py
│
├── opencap_monocular/
│   ├── optimization.py
│   ├── optimization_formulation.py
│   └── ...
│
├── pose_judgement_optimization/
│   ├── main.py
│   └── ...
│
├── learnable_smplify/
│   └── ...
│
└── outputs/
    ├── intermediate/
    ├── videos/
    ├── figures/
    └── logs/
```

---

## 4. PipelineState

File:

```text
pose_pipeline/state.py
```

### 4.1. Mục tiêu

`PipelineState` lưu trạng thái hiện tại của dữ liệu trong quá trình chạy pipeline.

State có hai mode chính:

```text
dual:
    đang có hai pose riêng cho left/right camera

unified:
    đang có một pose đã hợp nhất
```

### 4.2. Dataclass đề xuất

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any


@dataclass
class PipelineState:
    mode: str  # "dual" hoặc "unified"

    left_pkl: Optional[Path] = None
    right_pkl: Optional[Path] = None
    unified_pkl: Optional[Path] = None

    left_video: Optional[Path] = None
    right_video: Optional[Path] = None

    calib_left: Optional[Path] = None
    calib_right: Optional[Path] = None

    benchmark_path: Optional[Path] = None
    output_dir: Optional[Path] = None

    history: list[str] = field(default_factory=list)

    # Lưu source mới nhất cho left/right để J có thể dùng làm evidence
    latest_left_pkl: Optional[Path] = None
    latest_right_pkl: Optional[Path] = None

    # Lưu path các output trung gian
    artifacts: dict[str, Path] = field(default_factory=dict)

    # Metadata cho từng pipeline
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 4.3. State ban đầu

Sau khi đọc input folder:

```python
state = PipelineState(
    mode="dual",
    left_pkl=input_dir / "left.pkl",
    right_pkl=input_dir / "right.pkl",
    latest_left_pkl=input_dir / "left.pkl",
    latest_right_pkl=input_dir / "right.pkl",
    left_video=input_dir / "cam_left.mp4",
    right_video=input_dir / "cam_right.mp4",
    calib_left=find_optional_calib(input_dir, "left"),
    calib_right=find_optional_calib(input_dir, "right"),
    benchmark_path=benchmark_path,
    output_dir=output_dir,
)
```

---

## 5. Schema chuẩn cho pkl nội bộ

File:

```text
pose_pipeline/schema.py
```

### 5.1. Vì sao cần schema chuẩn?

R, J, L có thể dùng format khác nhau. Nếu mỗi pipeline trả format riêng, pipeline sau sẽ rất khó dùng.

Do đó sau mỗi pipeline, output nên được convert về schema chung.

### 5.2. Schema đề xuất

Mỗi file `.pkl` trung gian nên có dạng:

```python
{
    "poses_3d": np.ndarray,          # [T, J, 3]
    "poses_2d": np.ndarray | None,   # [T, J, 2] nếu có
    "confidence": np.ndarray | None, # [T, J] nếu có
    "smpl_params": dict | None,
    "camera": dict | None,
    "joint_names": list[str],
    "skeleton_edges": list[tuple[int, int]],

    "source": {
        "view": "left" | "right" | "unified",
        "pipeline_history": list[str],
        "input_files": list[str],
    },

    "metadata": {
        "fps": float | None,
        "num_frames": int,
        "coordinate_system": str,
        "created_by": str,
        "extra": dict,
    }
}
```

### 5.3. Các hàm cần có

```python
def load_pose_pkl(path: Path) -> dict:
    ...

def save_pose_pkl(data: dict, path: Path) -> None:
    ...

def validate_pose_schema(data: dict) -> None:
    ...

def convert_wham_to_standard_schema(raw_data: dict, view: str) -> dict:
    ...

def convert_standard_to_opencap_input(data: dict) -> dict:
    ...

def convert_opencap_output_to_standard(data: dict) -> dict:
    ...

def convert_standard_to_learnable_smplify_input(data: dict) -> dict:
    ...

def convert_learnable_smplify_output_to_standard(data: dict) -> dict:
    ...
```

---

## 6. Naming system cho output trung gian

File:

```text
pose_pipeline/naming.py
```

### 6.1. Mục tiêu

Không hard-code kiểu:

```text
left_3.pkl
right_3.pkl
left_3_2.pkl
```

Thay vào đó, dùng history pipeline:

```text
left_L.pkl
right_L.pkl
left_LJ.pkl
right_LJ.pkl
unify_R.pkl
unify_RJ.pkl
unify_RJL.pkl
```

### 6.2. Hàm đặt tên

```python
def make_dual_output_paths(output_dir: Path, history: list[str]) -> tuple[Path, Path]:
    suffix = "".join(history)
    intermediate_dir = output_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    left = intermediate_dir / f"left_{suffix}.pkl"
    right = intermediate_dir / f"right_{suffix}.pkl"

    return left, right


def make_unified_output_path(output_dir: Path, history: list[str]) -> Path:
    suffix = "".join(history)
    intermediate_dir = output_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    return intermediate_dir / f"unify_{suffix}.pkl"
```

### 6.3. Ví dụ tên file

| Sequence | Sau bước | Output |
|---|---|---|
| RJL | R | `unify_R.pkl` |
| RJL | J | `unify_RJ.pkl` |
| RJL | L | `unify_RJL.pkl` |
| LJR | L | `left_L.pkl`, `right_L.pkl` |
| LJR | J | `left_LJ.pkl`, `right_LJ.pkl` hoặc `unify_LJ.pkl` |
| LJR | R | `unify_LJR.pkl` |

---

## 7. Pipeline executor

File:

```text
pose_pipeline/executor.py
```

### 7.1. Pipeline map

```python
from pose_pipeline.pipelines.refinement import run_R
from pose_pipeline.pipelines.judgement import run_J
from pose_pipeline.pipelines.learnable_smplify import run_L

PIPELINE_FUNCS = {
    "R": run_R,
    "J": run_J,
    "L": run_L,
}
```

### 7.2. Sequence executor

```python
def run_pipeline_sequence(sequence: str, state: PipelineState, config: dict) -> PipelineState:
    for step in sequence:
        if step not in PIPELINE_FUNCS:
            raise ValueError(f"Unknown pipeline step: {step}")

        print(f"[Pipeline] Running step {step}")
        print(f"[Pipeline] Current mode before {step}: {state.mode}")

        fn = PIPELINE_FUNCS[step]
        state = fn(state, config)

        validate_state(state)

        print(f"[Pipeline] Finished step {step}")
        print(f"[Pipeline] Current mode after {step}: {state.mode}")
        print_state_files(state)

    return state
```

### 7.3. Chạy tất cả sequence nếu người dùng nhấn Enter

```python
SUPPORTED_SEQUENCES = [
    "RJL",
    "LJR",
    "LJRL",
    "LRJ",
    "LRJL",
    "RLJ",
    "RLJL",
    "JLR",
    "JLRL",
    "JRL",
]


def resolve_sequences(user_input: str) -> list[str]:
    if user_input.strip() == "":
        return SUPPORTED_SEQUENCES

    seq = user_input.strip().upper()
    if seq not in SUPPORTED_SEQUENCES:
        raise ValueError(f"Unsupported sequence: {seq}")

    return [seq]
```

---

## 8. State validation

File:

```text
pose_pipeline/validation.py
```

### 8.1. Validate sau mỗi pipeline

```python
def validate_state(state: PipelineState) -> None:
    if state.mode == "dual":
        if state.left_pkl is None or state.right_pkl is None:
            raise ValueError("Dual state requires left_pkl and right_pkl")

        if not state.left_pkl.exists():
            raise FileNotFoundError(state.left_pkl)

        if not state.right_pkl.exists():
            raise FileNotFoundError(state.right_pkl)

    elif state.mode == "unified":
        if state.unified_pkl is None:
            raise ValueError("Unified state requires unified_pkl")

        if not state.unified_pkl.exists():
            raise FileNotFoundError(state.unified_pkl)

    else:
        raise ValueError(f"Unknown state mode: {state.mode}")
```

### 8.2. Validate pose content

```python
def validate_pose_content(pose_data: dict) -> None:
    poses_3d = pose_data.get("poses_3d")

    if poses_3d is None:
        raise ValueError("pose_data does not contain poses_3d")

    if poses_3d.ndim != 3:
        raise ValueError(f"poses_3d must have shape [T, J, 3], got {poses_3d.shape}")

    if poses_3d.shape[-1] != 3:
        raise ValueError(f"last dimension must be 3, got {poses_3d.shape[-1]}")

    if not np.isfinite(poses_3d).all():
        raise ValueError("poses_3d contains NaN or Inf")
```

---

## 9. Implement pipeline R: Pose Refinement Optimization

File:

```text
pose_pipeline/pipelines/refinement.py
```

R được cài đặt sẵn trong:

```text
opencap_monocular/
```

Có thể tham khảo:

```text
opencap_monocular/optimization.py
opencap_monocular/optimization_formulation.py
```

### 9.1. Vai trò của R

R có nhiệm vụ:

- refine pose;
- giảm foot sliding;
- giảm floor penetration;
- tạo pose unified nếu input là dual;
- cải thiện động học/cấu trúc cơ thể.

### 9.2. R phải hỗ trợ hai input mode

| Input mode | Output mode | Ý nghĩa |
|---|---|---|
| dual | unified | refine + hợp nhất left/right |
| unified | unified | refine tiếp pose unified |

### 9.3. Wrapper chính

```python
def run_R(state: PipelineState, config: dict) -> PipelineState:
    if state.mode == "dual":
        return run_R_from_dual(state, config)

    if state.mode == "unified":
        return run_R_from_unified(state, config)

    raise ValueError(f"Unsupported state mode for R: {state.mode}")
```

### 9.4. R từ dual

```python
def run_R_from_dual(state: PipelineState, config: dict) -> PipelineState:
    new_history = state.history + ["R"]
    output_path = make_unified_output_path(state.output_dir, new_history)

    left_data = load_pose_pkl(state.left_pkl)
    right_data = load_pose_pkl(state.right_pkl)

    opencap_input = prepare_opencap_dual_input(
        left_data=left_data,
        right_data=right_data,
        calib_left=state.calib_left,
        calib_right=state.calib_right,
        config=config,
    )

    opencap_output = call_opencap_optimization(
        opencap_input=opencap_input,
        config=config,
    )

    unified_data = convert_opencap_output_to_standard(
        opencap_output,
        view="unified",
        history=new_history,
    )

    save_pose_pkl(unified_data, output_path)

    state.mode = "unified"
    state.unified_pkl = output_path
    state.history = new_history
    state.artifacts["R_unified"] = output_path

    return state
```

### 9.5. R từ unified

```python
def run_R_from_unified(state: PipelineState, config: dict) -> PipelineState:
    new_history = state.history + ["R"]
    output_path = make_unified_output_path(state.output_dir, new_history)

    unified_data = load_pose_pkl(state.unified_pkl)

    opencap_input = prepare_opencap_unified_input(
        unified_data=unified_data,
        config=config,
    )

    opencap_output = call_opencap_optimization(
        opencap_input=opencap_input,
        config=config,
    )

    refined_data = convert_opencap_output_to_standard(
        opencap_output,
        view="unified",
        history=new_history,
    )

    save_pose_pkl(refined_data, output_path)

    state.mode = "unified"
    state.unified_pkl = output_path
    state.history = new_history
    state.artifacts["R_unified"] = output_path

    return state
```

### 9.6. Việc cần kiểm tra khi tích hợp R

- [ ] `optimization.py` nhận input format gì?
- [ ] Có cần SMPL parameters không?
- [ ] Có cần ground plane không?
- [ ] Có cần camera intrinsics/extrinsics không?
- [ ] Output là joint coordinates hay SMPL params?
- [ ] Coordinate system output là gì?
- [ ] Có cần convert từ WHAM sang OpenCap format không?
- [ ] Có cần batch theo frame không?

---

## 10. Implement pipeline L: Learnable SMPLify

File:

```text
pose_pipeline/pipelines/learnable_smplify.py
```

### 10.1. Vai trò của L

L có nhiệm vụ refine pose bằng Learnable SMPLify.

Mục tiêu:

- làm pose hợp giải phẫu hơn;
- tối ưu nhanh hơn các phương pháp fitting truyền thống;
- refine SMPL/joint pose;
- có thể chạy riêng cho left/right hoặc chạy trên unified pose.

### 10.2. L phải hỗ trợ hai input mode

| Input mode | Output mode | Ý nghĩa |
|---|---|---|
| dual | dual | chạy L riêng cho left/right |
| unified | unified | chạy L trên pose unified |

### 10.3. Wrapper chính

```python
def run_L(state: PipelineState, config: dict) -> PipelineState:
    if state.mode == "dual":
        return run_L_from_dual(state, config)

    if state.mode == "unified":
        return run_L_from_unified(state, config)

    raise ValueError(f"Unsupported state mode for L: {state.mode}")
```

### 10.4. L từ dual

```python
def run_L_from_dual(state: PipelineState, config: dict) -> PipelineState:
    new_history = state.history + ["L"]
    left_out, right_out = make_dual_output_paths(state.output_dir, new_history)

    left_data = load_pose_pkl(state.left_pkl)
    right_data = load_pose_pkl(state.right_pkl)

    left_input = prepare_learnable_smplify_input(left_data, config)
    right_input = prepare_learnable_smplify_input(right_data, config)

    left_result = call_learnable_smplify(left_input, config)
    right_result = call_learnable_smplify(right_input, config)

    left_standard = convert_learnable_smplify_output_to_standard(
        left_result,
        view="left",
        history=new_history,
    )

    right_standard = convert_learnable_smplify_output_to_standard(
        right_result,
        view="right",
        history=new_history,
    )

    save_pose_pkl(left_standard, left_out)
    save_pose_pkl(right_standard, right_out)

    state.mode = "dual"
    state.left_pkl = left_out
    state.right_pkl = right_out
    state.latest_left_pkl = left_out
    state.latest_right_pkl = right_out
    state.history = new_history
    state.artifacts["L_left"] = left_out
    state.artifacts["L_right"] = right_out

    return state
```

### 10.5. L từ unified

```python
def run_L_from_unified(state: PipelineState, config: dict) -> PipelineState:
    new_history = state.history + ["L"]
    output_path = make_unified_output_path(state.output_dir, new_history)

    unified_data = load_pose_pkl(state.unified_pkl)

    l_input = prepare_learnable_smplify_input(unified_data, config)
    l_result = call_learnable_smplify(l_input, config)

    unified_standard = convert_learnable_smplify_output_to_standard(
        l_result,
        view="unified",
        history=new_history,
    )

    save_pose_pkl(unified_standard, output_path)

    state.mode = "unified"
    state.unified_pkl = output_path
    state.history = new_history
    state.artifacts["L_unified"] = output_path

    return state
```

### 10.6. Việc cần kiểm tra khi tích hợp L

- [ ] Learnable SMPLify nhận 2D keypoints, 3D joints hay SMPL params?
- [ ] Có cần SMPL model files không?
- [ ] Có cần shape beta không?
- [ ] Có cần camera params không?
- [ ] Output là SMPL params hay joints?
- [ ] Có API callable không hay chỉ CLI?
- [ ] Có hỗ trợ batch/sequence không?
- [ ] Có cần GPU/CUDA không?

---

## 11. Implement pipeline J: Pose Judgement Optimization

File chính:

```text
pose_pipeline/pipelines/judgement.py
```

Các file phụ:

```text
pose_pipeline/pipelines/judgement_alignment.py
pose_pipeline/pipelines/judgement_weights.py
pose_pipeline/pipelines/judgement_losses.py
pose_pipeline/pipelines/judgement_optimizer.py
pose_pipeline/pipelines/judgement_validation.py
```

### 11.1. Vai trò của J

J không nên chỉ average hai pose.

J mới cần:

- phân tích left/right view;
- lấy visibility/confidence/score;
- xác định joint nào nên tin camera nào hơn;
- align candidate về cùng hệ tọa độ;
- dùng R pose hoặc current pose làm prior;
- tối ưu theo temporal window;
- giữ ràng buộc bone length;
- giảm jitter;
- tránh foot sliding/floor penetration;
- validate output trước khi ghi ra file.

### 11.2. J phải hỗ trợ input mode

| Input mode | Output mode đề xuất | Ý nghĩa |
|---|---|---|
| dual | dual | judgement refine riêng left/right, giữ hai nhánh |
| dual | unified | judgement fusion thành pose unified |
| unified | unified | refine unified pose bằng evidence left/right nếu có |

Config nên cho phép chọn:

```yaml
judgement:
  output_mode_when_dual: dual hoặc unified
```

Khuyến nghị:

```text
Nếu J đứng trước R:
    có thể giữ dual để R unify sau.

Nếu J đứng sau R:
    J nên unified -> unified.

Nếu J là module temporal multi-view mạnh:
    J có thể dual -> unified.
```

### 11.3. Wrapper chính

```python
def run_J(state: PipelineState, config: dict) -> PipelineState:
    if state.mode == "dual":
        return run_J_from_dual(state, config)

    if state.mode == "unified":
        return run_J_from_unified(state, config)

    raise ValueError(f"Unsupported state mode for J: {state.mode}")
```

---

## 12. J từ dual

### 12.1. Trường hợp dual -> dual

Dùng khi sequence như `LJR`, muốn sau J vẫn còn left/right để R xử lý.

```python
def run_J_from_dual(state: PipelineState, config: dict) -> PipelineState:
    output_mode = config["judgement"].get("output_mode_when_dual", "dual")

    if output_mode == "dual":
        return run_J_dual_to_dual(state, config)

    if output_mode == "unified":
        return run_J_dual_to_unified(state, config)

    raise ValueError(f"Unsupported J output mode: {output_mode}")
```

```python
def run_J_dual_to_dual(state: PipelineState, config: dict) -> PipelineState:
    new_history = state.history + ["J"]
    left_out, right_out = make_dual_output_paths(state.output_dir, new_history)

    left_data = load_pose_pkl(state.left_pkl)
    right_data = load_pose_pkl(state.right_pkl)

    raw_result = call_original_judgement_for_metadata(
        left_data=left_data,
        right_data=right_data,
        config=config,
    )

    left_candidate, right_candidate = extract_candidates_from_raw_judgement(
        raw_result=raw_result,
        left_data=left_data,
        right_data=right_data,
        config=config,
    )

    left_refined, right_refined, diagnostics = optimize_or_safe_refine_dual(
        left_base=left_data,
        right_base=right_data,
        left_candidate=left_candidate,
        right_candidate=right_candidate,
        raw_result=raw_result,
        config=config,
    )

    save_pose_pkl(left_refined, left_out)
    save_pose_pkl(right_refined, right_out)

    state.mode = "dual"
    state.left_pkl = left_out
    state.right_pkl = right_out
    state.latest_left_pkl = left_out
    state.latest_right_pkl = right_out
    state.history = new_history
    state.metadata["J"] = diagnostics
    state.artifacts["J_left"] = left_out
    state.artifacts["J_right"] = right_out

    return state
```

### 12.2. Trường hợp dual -> unified

Dùng khi J mới đủ mạnh để fusion trực tiếp.

```python
def run_J_dual_to_unified(state: PipelineState, config: dict) -> PipelineState:
    new_history = state.history + ["J"]
    output_path = make_unified_output_path(state.output_dir, new_history)

    left_data = load_pose_pkl(state.left_pkl)
    right_data = load_pose_pkl(state.right_pkl)

    raw_result = call_original_judgement_for_metadata(
        left_data=left_data,
        right_data=right_data,
        config=config,
    )

    left_candidate, right_candidate = extract_candidates_from_raw_judgement(
        raw_result=raw_result,
        left_data=left_data,
        right_data=right_data,
        config=config,
    )

    base_pose = build_initial_base_pose_for_dual_judgement(
        left_data=left_data,
        right_data=right_data,
        config=config,
    )

    left_candidate, right_candidate = align_candidates_to_base(
        left_candidate=left_candidate,
        right_candidate=right_candidate,
        base_pose=base_pose,
        config=config,
    )

    view_weights = build_view_weights(
        raw_result=raw_result,
        left_data=left_data,
        right_data=right_data,
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
        skeleton_edges=left_data["skeleton_edges"],
        config=config,
    )

    pose_final, validation = validate_or_fallback_sequence(
        base_pose=base_pose,
        pose_judged=pose_judged,
        diagnostics=diagnostics,
        skeleton_edges=left_data["skeleton_edges"],
        config=config,
    )

    unified_data = build_unified_pose_data(
        pose_3d=pose_final,
        left_data=left_data,
        right_data=right_data,
        history=new_history,
        metadata={
            "raw_judgement": raw_result,
            "diagnostics": diagnostics,
            "validation": validation,
        },
    )

    save_pose_pkl(unified_data, output_path)

    state.mode = "unified"
    state.unified_pkl = output_path
    state.history = new_history
    state.metadata["J"] = diagnostics
    state.artifacts["J_unified"] = output_path

    return state
```

---

## 13. J từ unified

Dùng cho sequence như:

```text
RJL
RLJ
JRL nếu J hoặc R đã unify trước đó
```

### 13.1. Input cần có

Khi state đang unified, J vẫn nên dùng left/right evidence nếu có:

```python
left_ref = state.latest_left_pkl
right_ref = state.latest_right_pkl
base = state.unified_pkl
```

Nếu không còn left/right evidence, J chỉ có thể làm temporal/kinematic refine trên unified pose.

### 13.2. Implementation

```python
def run_J_from_unified(state: PipelineState, config: dict) -> PipelineState:
    new_history = state.history + ["J"]
    output_path = make_unified_output_path(state.output_dir, new_history)

    base_data = load_pose_pkl(state.unified_pkl)

    left_data = load_pose_pkl(state.latest_left_pkl) if state.latest_left_pkl else None
    right_data = load_pose_pkl(state.latest_right_pkl) if state.latest_right_pkl else None

    if left_data is not None and right_data is not None:
        raw_result = call_original_judgement_for_metadata(
            left_data=left_data,
            right_data=right_data,
            config=config,
        )

        left_candidate, right_candidate = extract_candidates_from_raw_judgement(
            raw_result=raw_result,
            left_data=left_data,
            right_data=right_data,
            config=config,
        )

        left_candidate, right_candidate = align_candidates_to_base(
            left_candidate=left_candidate,
            right_candidate=right_candidate,
            base_pose=base_data["poses_3d"],
            config=config,
        )

        view_weights = build_view_weights(
            raw_result=raw_result,
            left_data=left_data,
            right_data=right_data,
            left_candidate=left_candidate,
            right_candidate=right_candidate,
            base_pose=base_data["poses_3d"],
            config=config,
        )

        pose_judged, diagnostics = optimize_temporal_multiview_pose(
            base_pose=base_data["poses_3d"],
            left_candidate=left_candidate,
            right_candidate=right_candidate,
            view_weights=view_weights,
            skeleton_edges=base_data["skeleton_edges"],
            config=config,
        )

    else:
        pose_judged, diagnostics = optimize_temporal_single_pose(
            base_pose=base_data["poses_3d"],
            skeleton_edges=base_data["skeleton_edges"],
            config=config,
        )

    pose_final, validation = validate_or_fallback_sequence(
        base_pose=base_data["poses_3d"],
        pose_judged=pose_judged,
        diagnostics=diagnostics,
        skeleton_edges=base_data["skeleton_edges"],
        config=config,
    )

    base_data["poses_3d"] = pose_final
    base_data["source"]["pipeline_history"] = new_history
    base_data["metadata"]["created_by"] = "J"
    base_data["metadata"]["judgement"] = {
        "diagnostics": diagnostics,
        "validation": validation,
    }

    save_pose_pkl(base_data, output_path)

    state.mode = "unified"
    state.unified_pkl = output_path
    state.history = new_history
    state.metadata["J"] = diagnostics
    state.artifacts["J_unified"] = output_path

    return state
```

---

## 14. J optimizer chi tiết

### 14.1. Không dùng legacy overwrite

Đoạn này chỉ được giữ trong mode debug:

```python
opt1 = result["optimized"]["camera1"]
opt2 = result["optimized"]["camera2"]
fused[frame_idx] = _joint_dicts_to_average_array(opt1, opt2, joint_names)
```

Config:

```yaml
judgement:
  mode: temporal_multiview_optimize
  allow_legacy_full_overwrite: false
```

Nếu dùng legacy:

```text
In warning:
"full_legacy mode may destroy refined pose and should not be used for final output"
```

### 14.2. Loss tổng

```text
L = λ_data * L_data
  + λ_prior * L_prior
  + λ_bone * L_bone
  + λ_temp * L_temporal
  + λ_acc * L_acceleration
  + λ_floor * L_floor
  + λ_contact * L_contact
  + λ_sym * L_symmetry
```

### 14.3. Config ban đầu

```yaml
judgement:
  mode: temporal_multiview_optimize

  output_mode_when_dual: dual
  coordinate_alignment: sequence_umeyama

  window_size: 32
  stride: 8
  iters: 80
  lr: 0.03

  lambda_data: 1.0
  lambda_prior: 0.4
  lambda_bone: 8.0
  lambda_temp: 0.3
  lambda_acc: 1.5
  lambda_floor: 3.0
  lambda_contact: 2.0
  lambda_symmetry: 1.0

  camera_disagreement_threshold_m: 0.25
  min_base_prior_weight: 0.2
  max_base_prior_weight: 1.0

  floor_axis: 2
  floor_value: 0.0

  max_bone_deviation_ratio: 0.20
  max_joint_velocity_m_per_frame: 0.35
  max_joint_acceleration_m_per_frame2: 0.45

  fallback_mode: partial
```

---

## 15. Finalization sau khi chạy sequence

File:

```text
pose_pipeline/executor.py
```

### 15.1. Đảm bảo output cuối là unified

Sau khi chạy xong sequence:

```python
def finalize_state(final_state: PipelineState, config: dict) -> PipelineState:
    if final_state.mode == "unified":
        return final_state

    if final_state.mode == "dual":
        return force_unify(final_state, config)

    raise ValueError(f"Unknown final state mode: {final_state.mode}")
```

### 15.2. Force unify nếu cần

```python
def force_unify(state: PipelineState, config: dict) -> PipelineState:
    new_history = state.history + ["U"]
    output_path = make_unified_output_path(state.output_dir, new_history)

    left_data = load_pose_pkl(state.left_pkl)
    right_data = load_pose_pkl(state.right_pkl)

    unified_data = simple_confidence_fusion(
        left_data=left_data,
        right_data=right_data,
        config=config,
    )

    unified_data["source"]["pipeline_history"] = new_history
    unified_data["metadata"]["created_by"] = "force_unify"

    save_pose_pkl(unified_data, output_path)

    state.mode = "unified"
    state.unified_pkl = output_path
    state.history = new_history

    return state
```

Lý do cần force unify:

```text
Renderer, waveform và benchmark đều nên dùng unified pose.
```

---

## 16. Video composer

File:

```text
pose_pipeline/visualization/video_composer.py
```

### 16.1. Input

```python
create_composite_video(
    left_video_path=state.left_video,
    right_video_path=state.right_video,
    unified_pkl=final_state.unified_pkl,
    output_path=output_dir / "videos" / f"{sequence}_cam_left_right_3D_poses.mp4",
)
```

### 16.2. Layout video

```text
+----------------+----------------+----------------+
| cam_left.mp4   | cam_right.mp4  | 3D pose render |
+----------------+----------------+----------------+
```

### 16.3. Cần xử lý

- [ ] Đồng bộ FPS.
- [ ] Đồng bộ frame count.
- [ ] Dùng `min(num_video_frames, num_pose_frames)`.
- [ ] Render 3D theo fixed camera view.
- [ ] Center skeleton theo pelvis/root.
- [ ] Fixed axis limits để không bị zoom/nhảy.
- [ ] Tô màu left/right limbs để debug.
- [ ] Có thể thêm ground plane.

---

## 17. Waveform analysis

File:

```text
pose_pipeline/visualization/waveform.py
```

### 17.1. Input

```python
draw_waveform_analysis(
    unified_pkl=final_state.unified_pkl,
    output_dir=output_dir / "figures",
)
```

### 17.2. Góc cần tính

Góc tay:

```text
vector 1 = elbow -> shoulder
vector 2 = elbow -> hand
```

Góc chân:

```text
vector 1 = knee -> hip
vector 2 = knee -> ankle
```

### 17.3. Công thức

```python
def angle_between(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))
```

### 17.4. Output

```text
waveform_analysis_left_arm_wrist.png
waveform_analysis_right_arm_wrist.png
waveform_analysis_left_thigh_lower_leg.png
waveform_analysis_right_thigh_lower_leg.png
```

### 17.5. Cần xác nhận joint mapping

Cần xác định chính xác:

```python
JOINTS = {
    "left_shoulder": ...,
    "left_elbow": ...,
    "left_hand": ...,
    "right_shoulder": ...,
    "right_elbow": ...,
    "right_hand": ...,
    "left_hip": ...,
    "left_knee": ...,
    "left_ankle": ...,
    "right_hip": ...,
    "right_knee": ...,
    "right_ankle": ...,
}
```

Nếu mapping sai, waveform sẽ sai dù pose đúng.

---

## 18. Benchmark evaluation

File:

```text
pose_pipeline/benchmark/evaluator.py
```

### 18.1. Nếu người dùng nhập `$`

Bỏ qua benchmark.

### 18.2. Nếu có benchmark

```python
evaluate_benchmark(
    prediction_pkl=final_state.unified_pkl,
    benchmark_path=state.benchmark_path,
    output_path=output_dir / "logs" / f"{sequence}_benchmark_result.json",
)
```

### 18.3. Metrics đề xuất

| Metric | Ý nghĩa |
|---|---|
| MPJPE | Sai số trung bình từng joint |
| PA-MPJPE | Sai số sau Procrustes alignment |
| PCK | Tỷ lệ keypoint đúng |
| acceleration error | Sai số động học |
| foot sliding score | Độ trượt chân |
| floor penetration score | Độ lún sàn |

---

## 19. Main console flow

File:

```text
main.py
```

### 19.1. Console output

```python
def main():
    print("Chương trình có các pipeline sau:")
    print("\t(0) đọc hai file video và hai file pkl tương ứng")
    print("\t(1) Pose Refinement Optimization")
    print("\t(2) Pose Judgement Optimization")
    print("\t(3) Learnable SMPLify")
    print("\t(4) tạo video tổng hợp")
    print("\t(5) Vẽ waveform analysis")

    benchmark_input = input("Bạn muốn áp dụng với bộ dữ liệu benchmark nào? (nhập $ để bỏ qua bước này): ")
    input_dir_input = input("Hãy nhập đường dẫn chứa hai file video và hai file pkl (Enter để chọn thư mục hiện tại): ")

    print_supported_sequences()

    sequence_input = input("Bạn cho biết thứ tự bạn muốn thực hiện là gì? (nhấn Enter để thực hiện tất cả): ")

    ...
```

### 19.2. Main logic

```python
def main():
    start_time = time.time()

    config = load_config()

    benchmark_path = resolve_benchmark_path(benchmark_input)
    input_dir = resolve_input_dir(input_dir_input)

    sequences = resolve_sequences(sequence_input)

    for sequence in sequences:
        state = build_initial_state(
            input_dir=input_dir,
            benchmark_path=benchmark_path,
            output_dir=Path("outputs") / sequence,
        )

        final_state = run_pipeline_sequence(
            sequence=sequence,
            state=state,
            config=config,
        )

        final_state = finalize_state(final_state, config)

        video_path = create_composite_video_for_state(
            state=final_state,
            sequence=sequence,
            config=config,
        )

        waveform_paths = draw_waveforms_for_state(
            state=final_state,
            sequence=sequence,
            config=config,
        )

        if benchmark_path is not None:
            benchmark_result = evaluate_benchmark_for_state(
                state=final_state,
                sequence=sequence,
                config=config,
            )

        write_run_log(
            state=final_state,
            video_path=video_path,
            waveform_paths=waveform_paths,
            elapsed=time.time() - start_time,
        )

    print_summary(...)
```

---

## 20. Logging

File:

```text
pose_pipeline/logging_utils.py
```

### 20.1. Log cần ghi

Mỗi run cần ghi:

```text
sequence
input_dir
benchmark
start_time
end_time
total_elapsed
pipeline_history
state transitions
intermediate files
final unified pkl
video output
waveform outputs
benchmark output nếu có
warnings/errors
```

### 20.2. Ví dụ run log

```text
Sequence: RJL
Input dir: ./data/sample_01
Benchmark: skipped

State transitions:
  Start: dual left.pkl right.pkl
  R: dual -> unified outputs/intermediate/unify_R.pkl
  J: unified -> unified outputs/intermediate/unify_RJ.pkl
  L: unified -> unified outputs/intermediate/unify_RJL.pkl

Final pkl:
  outputs/intermediate/unify_RJL.pkl

Video:
  outputs/videos/RJL_cam_left_right_3D_poses.mp4

Waveforms:
  outputs/figures/waveform_analysis_left_arm_wrist.png
  outputs/figures/waveform_analysis_right_arm_wrist.png
  outputs/figures/waveform_analysis_left_thigh_lower_leg.png
  outputs/figures/waveform_analysis_right_thigh_lower_leg.png

Total elapsed:
  01:23:45
```

---

## 21. Test plan

### 21.1. Unit tests

Test các module nhỏ:

- [ ] `make_unified_output_path`
- [ ] `make_dual_output_paths`
- [ ] `validate_state`
- [ ] `load_pose_pkl`
- [ ] `save_pose_pkl`
- [ ] `angle_between`
- [ ] `bone_length_loss`
- [ ] `align_sequence_to_base`
- [ ] `build_view_weights`

### 21.2. Integration tests theo sequence

Chạy lần lượt:

```bash
python main.py --sequence RJL
python main.py --sequence LJR
python main.py --sequence LJRL
python main.py --sequence LRJ
python main.py --sequence LRJL
python main.py --sequence RLJ
python main.py --sequence RLJL
python main.py --sequence JLR
python main.py --sequence JLRL
python main.py --sequence JRL
```

### 21.3. Kiểm tra state transition

Expected:

| Sequence | Expected final mode |
|---|---|
| RJL | unified |
| LJR | unified |
| LJRL | unified |
| LRJ | unified hoặc force unified |
| LRJL | unified |
| RLJ | unified |
| RLJL | unified |
| JLR | unified |
| JLRL | unified |
| JRL | unified |

### 21.4. Visual tests

Render video cho từng sequence:

```text
RJL_cam_left_right_3D_poses.mp4
LJR_cam_left_right_3D_poses.mp4
...
```

Kiểm tra:

- skeleton không lật ngược;
- không có xương dài bất thường;
- không giật mạnh;
- chân/tay không bị kéo ngang vô lý;
- 3D pose tương ứng với chuyển động trong video;
- waveform có dạng liên tục, không spike bất thường.

### 21.5. Regression tests cho lỗi J cũ

Chạy:

```bash
python main.py --sequence RJL --judgement-mode full_legacy
python main.py --sequence RJL --judgement-mode temporal_multiview_optimize
```

So sánh:

- `full_legacy` có thể còn lỗi chân/tay dài;
- `temporal_multiview_optimize` không được có lỗi đó;
- bone deviation thấp hơn;
- acceleration spike thấp hơn.

---

## 22. Kế hoạch implement theo giai đoạn

## Giai đoạn 1: Skeleton project và console

Mục tiêu:

```text
Chạy được chương trình console và parse sequence.
```

Tasks:

- [ ] Tạo cấu trúc thư mục.
- [ ] Tạo `main.py`.
- [ ] Tạo `config.py`.
- [ ] In menu console đúng yêu cầu.
- [ ] Đọc benchmark input.
- [ ] Đọc input folder.
- [ ] Parse sequence.
- [ ] Nếu Enter thì chạy tất cả sequence.

Deliverable:

```text
python main.py chạy được đến bước in sequence cần chạy.
```

---

## Giai đoạn 2: PipelineState và executor

Mục tiêu:

```text
Có state machine xử lý R/J/L linh hoạt.
```

Tasks:

- [ ] Tạo `PipelineState`.
- [ ] Tạo `run_pipeline_sequence`.
- [ ] Tạo `validate_state`.
- [ ] Tạo naming system.
- [ ] Tạo mock `run_R`, `run_J`, `run_L`.
- [ ] Test các sequence bằng mock file.

Deliverable:

```text
Các sequence RJL, LJR, LJRL... chạy được bằng mock pipeline.
```

---

## Giai đoạn 3: Chuẩn hóa pkl schema

Mục tiêu:

```text
Mọi pipeline đọc/ghi cùng một schema nội bộ.
```

Tasks:

- [ ] Viết `load_pose_pkl`.
- [ ] Viết `save_pose_pkl`.
- [ ] Inspect `left.pkl`, `right.pkl`.
- [ ] Convert WHAM pkl sang schema chuẩn.
- [ ] Validate `poses_3d` shape.
- [ ] Validate joint names.
- [ ] Validate skeleton edges.

Deliverable:

```text
Đọc được left.pkl/right.pkl và tạo left_standard.pkl/right_standard.pkl.
```

---

## Giai đoạn 4: Tích hợp R

Mục tiêu:

```text
Gọi được Pose Refinement Optimization từ opencap_monocular.
```

Tasks:

- [ ] Kiểm tra `opencap_monocular/optimization.py`.
- [ ] Kiểm tra `optimization_formulation.py`.
- [ ] Xác định input/output thật.
- [ ] Viết adapter `prepare_opencap_dual_input`.
- [ ] Viết `call_opencap_optimization`.
- [ ] Convert output về schema chuẩn.
- [ ] Implement `run_R_from_dual`.
- [ ] Implement `run_R_from_unified`.
- [ ] Test sequence `R`.

Deliverable:

```text
left.pkl + right.pkl -> unify_R.pkl.
```

---

## Giai đoạn 5: Tích hợp L

Mục tiêu:

```text
Gọi được Learnable SMPLify ở cả dual và unified mode.
```

Tasks:

- [ ] Kiểm tra API/CLI của Learnable SMPLify.
- [ ] Xác định input/output.
- [ ] Viết adapter input.
- [ ] Viết adapter output.
- [ ] Implement `run_L_from_dual`.
- [ ] Implement `run_L_from_unified`.
- [ ] Test `L`.
- [ ] Test `RL`.

Deliverable:

```text
dual -> dual khi L chạy trước.
unified -> unified khi L chạy sau R.
```

---

## Giai đoạn 6: Fix J mode cơ bản

Mục tiêu:

```text
J không còn phá pose từ R.
```

Tasks:

- [ ] Tách legacy code vào `full_legacy`.
- [ ] Không dùng average `opt1/opt2` làm output mặc định.
- [ ] Implement `metadata_only`.
- [ ] Implement `run_J_from_dual`.
- [ ] Implement `run_J_from_unified`.
- [ ] Lưu raw judgement metadata.
- [ ] Test `RJ`.
- [ ] Test `LJ`.

Deliverable:

```text
RJL không còn bị J overwrite pose tốt từ R.
```

---

## Giai đoạn 7: J alignment và weights

Mục tiêu:

```text
J biết align và đánh trọng số left/right evidence.
```

Tasks:

- [ ] Extract left/right candidate từ raw judgement.
- [ ] Implement root-scale alignment.
- [ ] Implement sequence Umeyama alignment.
- [ ] Build `w_left`, `w_right`, `w_base`.
- [ ] Downweight khi left/right disagreement.
- [ ] Debug heatmap/view weights.
- [ ] Test với `J` trên dual.
- [ ] Test với `J` sau R.

Deliverable:

```text
Candidate J không còn lệch hệ tọa độ nghiêm trọng.
```

---

## Giai đoạn 8: J temporal optimizer

Mục tiêu:

```text
J cải thiện pose thật sự theo chuỗi thời gian.
```

Tasks:

- [ ] Implement data loss.
- [ ] Implement base prior loss.
- [ ] Implement bone length loss.
- [ ] Implement velocity loss.
- [ ] Implement acceleration loss.
- [ ] Implement floor penetration loss.
- [ ] Implement foot contact loss.
- [ ] Implement window optimizer.
- [ ] Implement sliding window + overlap.
- [ ] Test `temporal_multiview_optimize`.

Deliverable:

```text
Pose sau J mượt hơn, không tạo xương dài bất thường.
```

---

## Giai đoạn 9: J validation và fallback

Mục tiêu:

```text
J robust hơn, không phá toàn sequence nếu vài frame lỗi.
```

Tasks:

- [ ] Validate bone deviation.
- [ ] Validate velocity spike.
- [ ] Validate acceleration spike.
- [ ] Validate floor penetration.
- [ ] Implement fallback toàn sequence.
- [ ] Implement partial fallback theo frame/joint.
- [ ] Log rejected windows.
- [ ] Xuất diagnostics JSON.

Deliverable:

```text
J có guardrails rõ ràng.
```

---

## Giai đoạn 10: Video render

Mục tiêu:

```text
Tạo video ghép 2 camera + 3D pose.
```

Tasks:

- [ ] Đọc video left/right bằng OpenCV.
- [ ] Đọc final unified pkl.
- [ ] Render skeleton 3D.
- [ ] Center theo pelvis/root.
- [ ] Fixed axis limits.
- [ ] Tô màu left/right.
- [ ] Ghép 3 panel.
- [ ] Xuất MP4.

Deliverable:

```text
cam_left_right_3D_poses.mp4.
```

---

## Giai đoạn 11: Waveform analysis

Mục tiêu:

```text
Vẽ đủ 4 biểu đồ waveform.
```

Tasks:

- [ ] Xác định joint mapping.
- [ ] Tính góc elbow trái/phải.
- [ ] Tính góc knee trái/phải.
- [ ] Vẽ 4 biểu đồ.
- [ ] Xuất PNG.

Deliverable:

```text
waveform_analysis_left_arm_wrist.png
waveform_analysis_right_arm_wrist.png
waveform_analysis_left_thigh_lower_leg.png
waveform_analysis_right_thigh_lower_leg.png
```

---

## Giai đoạn 12: Benchmark

Mục tiêu:

```text
Nếu có benchmark, so sánh output với ground truth.
```

Tasks:

- [ ] Load benchmark.
- [ ] Convert benchmark về cùng joint format.
- [ ] Align frame count.
- [ ] Tính MPJPE.
- [ ] Tính PA-MPJPE.
- [ ] Tính PCK.
- [ ] Tính acceleration error.
- [ ] Tính foot sliding/floor penetration nếu cần.
- [ ] Xuất JSON.

Deliverable:

```text
benchmark_result.json.
```

---

## Giai đoạn 13: End-to-end testing

Mục tiêu:

```text
Chạy được tất cả sequence.
```

Tasks:

- [ ] Test `RJL`.
- [ ] Test `LJR`.
- [ ] Test `LJRL`.
- [ ] Test `LRJ`.
- [ ] Test `LRJL`.
- [ ] Test `RLJ`.
- [ ] Test `RLJL`.
- [ ] Test `JLR`.
- [ ] Test `JLRL`.
- [ ] Test `JRL`.
- [ ] Ghi log từng sequence.
- [ ] So sánh video và metrics.

Deliverable:

```text
Tất cả sequence chạy được, output có video/waveform/log.
```

---

## 23. Ưu tiên nếu thời gian hạn chế

Nếu không đủ thời gian, ưu tiên:

```text
1. PipelineState + executor.
2. Schema chuẩn cho pkl.
3. R adapter dual -> unified.
4. L adapter dual/unified.
5. J metadata_only + không overwrite R.
6. Video render.
7. Waveform.
8. J safe_fusion.
9. J temporal optimizer.
10. Benchmark.
```

Tức là phải có bản chạy ổn trước, rồi mới nâng cấp J.

---

## 24. Tiêu chí nghiệm thu

Chương trình đạt yêu cầu khi:

- [ ] Chạy được bằng `python main.py`.
- [ ] Đọc được `cam_left.mp4`, `cam_right.mp4`, `left.pkl`, `right.pkl`.
- [ ] Hỗ trợ tất cả sequence R/J/L đã liệt kê.
- [ ] Không hard-code riêng từng sequence.
- [ ] Mỗi pipeline wrapper xử lý theo `state.mode`.
- [ ] Sau mỗi step có intermediate file rõ ràng.
- [ ] Cuối sequence có unified pkl.
- [ ] Xuất được video ghép.
- [ ] Xuất được 4 waveform.
- [ ] Có log tổng thời gian.
- [ ] J không còn average `opt1/opt2` để overwrite pose final mặc định.
- [ ] J mới có alignment, weights, validation.
- [ ] Nếu dùng temporal optimizer, pose sau J ít jitter và ít lỗi xương hơn legacy.
- [ ] Nếu có benchmark, xuất được kết quả so sánh.

---

## 25. Kết luận

Kiến trúc đúng cho bài này là:

```text
state-based pipeline executor
```

Không phải:

```text
hard-code từng sequence
```

Cốt lõi:

```text
1. State biết hiện tại đang dual hay unified.
2. Mỗi pipeline R/J/L biết xử lý input mode hiện tại.
3. Mỗi pipeline ghi output theo schema chuẩn.
4. Executor chỉ đọc sequence từng ký tự và gọi wrapper.
5. Cuối sequence ép về unified để render/waveform/benchmark.
```

Đặc biệt với J:

```text
J không được average opt1/opt2 rồi overwrite pose.
J phải được thiết kế lại thành module judgement/fusion có alignment, confidence weight, temporal optimization và validation.
```

Khi làm đúng, chương trình sẽ xử lý được mọi thứ tự `R/J/L` mà không bị vỡ logic input/output.
