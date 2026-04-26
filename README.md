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

De chay 100 frame dau, pipeline van chay du `R -> J -> L`, nhung video render lay pose sau stage `R` vi stage nay dang cho skeleton on dinh nhat:

```powershell
python main.py --no-prompt --sequence RJL --max-judgement-frames 100 --opencap-max-frames 100 --opencap-iterations 1 --opencap-sex f --opencap-height-m 1.58 --opencap-mass-kg 48 --render-stage R --render-zoom 1.15 --output-dir outputs\restored_axis_like_100
```

Y nghia:

- `--sequence RJL`: chay Pose Refinement, Pose Judgement, Learnable SMPLify.
- `--render-stage R`: video 3D render pose sau stage R, tranh pose sau J/L dang bi giat/meo.
- `--judgement-pose-update off`: mac dinh hien tai; J van chay va ghi metadata nhung khong ghi de pose 3D tu R.
- `--render-zoom 1.15`: phong to skeleton 3D trong panel render.
- `--max-judgement-frames 100`: chi cho J xu ly 100 frame dau.
- `--opencap-max-frames 100`: chi cho R xu ly 100 frame dau.

Neu muon cho J tac dong nhe vao pose, dung mode blend:

```powershell
python main.py --no-prompt --sequence RJL --max-judgement-frames 100 --opencap-max-frames 100 --opencap-iterations 1 --judgement-pose-update blend --judgement-blend-alpha 0.2 --judgement-max-joint-shift-m 0.15 --output-dir outputs\test_j_blend_100
```

Khong khuyen dung `--judgement-pose-update full` cho output hien tai, vi mode nay la cach cu da lam skeleton bi giat/meo.

Output nam trong:

```text
outputs\<ten_output>\videos\
outputs\<ten_output>\figures\
outputs\<ten_output>\logs\run_log.json
```

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

## Ghi chu ve cac module vendor

- `vendor/pose_judgement_optimization/main.py`: entrypoint Pose Judgement goc.
- `vendor/opencap-monocular-main/optimization.py`: entrypoint Pose Refinement/OpenCap.
- `vendor/learnable-simplify-for-inverse-kinematic-main/...`: cac module cho Learnable SMPLify.

`vendor/` co chua mot so model file lon. Repo da push du nhung lan clone dau tien co the mat thoi gian.
