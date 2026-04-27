# DualCam Upgrade

Pipeline xu ly pose 3D tu hai camera. Repo nay da gom cac adapter chinh va cac source can dung trong `vendor/`.

## Cau truc input

Mac dinh chuong trinh doc input o folder ben ngoai repo:

```text
..\AnhVTM-20260426T073026Z-3-001\AnhVTM
```

Folder input can co cac file video/pkl trai-phai ma `pose_pipeline/io_utils/input_loader.py` dang tim. Neu dat input o vi tri khac, truyen bang `--input-dir`:

```powershell
python main.py --no-prompt --input-dir E:\path\to\AnhVTM --sequence RJL
```

Khong nen commit video/pkl input va folder `outputs/` len GitHub vi thuong lon va co the la du lieu rieng tu.

## Cai dat

Nen dung virtual environment rieng:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Neu may co GPU/CUDA, cai `torch` theo dung ban CUDA cua may neu can.

## Lenh chay dang on dinh

Lenh test 100 frame, nhan vat nu, R chay 10 iterations:

```powershell
python main.py --no-prompt --sequence RJL --max-judgement-frames 100 --opencap-max-frames 100 --opencap-sex f --opencap-iterations 10 --output-dir outputs\test_RJL_100_iter10
```

Y nghia:

- `--sequence RJL`: chay Pose Refinement, Pose Judgement, Learnable SMPLify.
- `--opencap-iterations 10`: so iteration cua buoc R/OpenCap.
- `--max-judgement-frames 100`: chi cho J xu ly 100 frame dau.
- `--opencap-max-frames 100`: chi cho R xu ly 100 frame dau.
- `--judgement-pose-update off`: mac dinh hien tai; J van chay va ghi metadata nhung khong ghi de pose 3D bang candidate legacy.
- `--render-stage final`: mac dinh; video render pose final cua sequence.

Mac dinh renderer se tu tao/dung reference iter1 de giu goc nhin on dinh:

```text
outputs\<ten_output>\render_reference_iter1\intermediate\unify_R.pkl
outputs\<ten_output>\render_reference_iter1\reference_meta.json
```

Neu `unify_R.pkl` chua co, chuong trinh tu chay prepass `R` voi `opencap_iterations=1`. Neu da co va `reference_meta.json` khop input/config hien tai thi dung lai. Reference nay chi dung cho render video; cac pkl final nhu `unify_R.pkl`, `unify_RJ.pkl`, `unify_RJL.pkl` cua run chinh khong bi sua.

Tat auto iter1 render view:

```powershell
python main.py --no-prompt --sequence RJL --opencap-max-frames 100 --opencap-iterations 10 --no-default-render-align --output-dir outputs\test_no_default_align
```

Override reference render bang file rieng:

```powershell
python main.py --no-prompt --sequence RJL --opencap-max-frames 100 --opencap-iterations 10 --render-align-reference path\to\unify_R.pkl --output-dir outputs\test_custom_render_ref
```

Neu muon cho J tac dong nhe vao pose, dung mode blend:

```powershell
python main.py --no-prompt --sequence RJL --max-judgement-frames 100 --opencap-max-frames 100 --opencap-iterations 1 --judgement-pose-update blend --judgement-blend-alpha 0.2 --judgement-max-joint-shift-m 0.15 --output-dir outputs\test_j_blend_100
```

Khong khuyen dung `--judgement-pose-update full` cho output hien tai, vi mode nay la cach cu da lam skeleton bi giat/meo.

Renderer 3D hien tai:

- default `--render-view front`;
- left/right limb co mau rieng;
- cac joint la cham do;
- co ground grid; voi `front/side`, ground duoc ve thanh luoi san de de nhin hon thay vi chi la mot duong ngang.

Neu muon xem san 3D ro hon:

```powershell
python main.py --no-prompt --sequence RJL --opencap-max-frames 100 --opencap-iterations 10 --render-view orbit --render-yaw-deg 45 --render-pitch-deg 55 --output-dir outputs\test_orbit_view
```

Output nam trong:

```text
outputs\<ten_output>\videos\
outputs\<ten_output>\figures\
outputs\<ten_output>\logs\run_log.json
outputs\<ten_output>\render_reference_iter1\
```

`--judgement-iters` chi co tac dung ro khi dung `--judgement-mode temporal_multiview_optimize`. Voi mac dinh `safe_fusion`, tham so nay gan nhu khong anh huong thoi gian/chuyen dong.

## Debug tung stage

Neu can kiem tra stage nao lam hong pose:

```powershell
python main.py --no-prompt --sequence R --opencap-max-frames 100 --opencap-iterations 1 --render-stage final --output-dir outputs\debug_R_100
```

```powershell
python main.py --no-prompt --sequence RJ --max-judgement-frames 100 --opencap-max-frames 100 --opencap-iterations 1 --render-stage final --output-dir outputs\debug_RJ_100
```

```powershell
python main.py --no-prompt --sequence RJL --max-judgement-frames 100 --opencap-max-frames 100 --opencap-iterations 1 --render-stage final --output-dir outputs\debug_RJL_100
```

Neu `R` dep nhung `RJ` hong, loi nam o J. Neu `RJ` dep nhung `RJL` hong, loi nam o L.

Khi debug global coordinate/view that, them `--no-default-render-align` de render dung pose thuc, khong apply view cua iter1 reference.

## Ghi chu ve cac module vendor

- `vendor/pose_judgement_optimization/main.py`: entrypoint Pose Judgement goc.
- `vendor/opencap-monocular-main/optimization.py`: entrypoint Pose Refinement/OpenCap.
- `vendor/learnable-simplify-for-inverse-kinematic-main/...`: cac module cho Learnable SMPLify.

`vendor/` co chua mot so model file lon. Repo da push du nhung lan clone dau tien co the mat thoi gian.
