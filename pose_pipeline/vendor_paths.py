from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PROJECT_ROOT / "vendor"
POSE_JUDGEMENT_MAIN = VENDOR_ROOT / "pose_judgement_optimization" / "main.py"
OPENCAP_ROOT = VENDOR_ROOT / "opencap-monocular-main"
LEARNABLE_ROOT = (
    VENDOR_ROOT
    / "learnable-simplify-for-inverse-kinematic-main"
    / "learnable-simplify-for-inverse-kinematic-main"
)
LEARNABLE_MODEL_SRC = LEARNABLE_ROOT / "Learnable-SMPLify-main" / "src"
LEARNABLE_CHECKPOINT = LEARNABLE_ROOT / "real_best_ckpt.pth"
LEARNABLE_SMPL_DIR = LEARNABLE_ROOT / "smpl" / "models"
