# Vendored Original Sources

Thu muc nay chua source goc dang duoc workflow chinh cua `pose_pipeline_project` goi.

- `pose_judgement_optimization/main.py` la ban copy nguyen tu `../main.py`.
- `opencap-monocular-main/optimization.py` va cac package phu thuoc truc tiep (`utils`, `slahmr`, `third_party_modified`, `params`, SMPL neutral model) la source goc dung cho pipeline R.
- `learnable-simplify-for-inverse-kinematic-main/.../inference.py`, `pkl_io.py`, `pose_analysis.py` la cac module goc dung cho pipeline L.

Nhung file demo, test, validation, docker, WHAM training/inference khong di qua workflow chinh da duoc xoa de folder nhe hon.
