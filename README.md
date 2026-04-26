# Pose Pipeline Project

Folder nay duoc dong goi de push rieng. Cac source goc can dung cho workflow chinh da nam trong `vendor/`:

- `vendor/pose_judgement_optimization/main.py`: copy nguyen file `main.py` goc.
- `vendor/opencap-monocular-main/optimization.py`: entrypoint Pose Refinement Optimization goc ma pipeline R dang goi.
- `vendor/learnable-simplify-for-inverse-kinematic-main/.../{inference.py,pkl_io.py,pose_analysis.py}`: cac module goc ma pipeline L dang goi.

## Cai dat

```powershell
pip install -r requirements.txt
```

## Chay pipeline tich hop

Mac dinh input tro toi `..\AnhVTM-20260426T073026Z-3-001\AnhVTM`.

```powershell
python main.py --no-prompt --sequence RJL
```

De test nhanh khong render video:

```powershell
python main.py --no-prompt --sequence RJL --skip-video --skip-waveform --max-judgement-frames 5
```

Pipeline R mac dinh chay `optimization.run_optimization` cho ca `left.pkl` va `right.pkl`, tao `wham_output.pkl` tam thoi trong `outputs/opencap_refinement/`, roi nap lai file `*_optimized.pkl` vao workflow. Co the giam thoi gian test bang:

```powershell
python main.py --no-prompt --sequence R --skip-video --skip-waveform --opencap-iterations 1 --opencap-max-frames 20
```

## Chay lai entrypoint judgement goc

```powershell
python run_original_pose_judgement.py
```

## Output

```text
outputs/videos/
outputs/figures/
outputs/logs/run_log.json
```

## Luu y

Adapter trong `pose_pipeline/pipelines/` chi noi data vao cac module trong `vendor/`.
`vendor/` da duoc thu gon theo workflow chinh, nen khong con chua demo/test/full entrypoint khong duoc goi.
