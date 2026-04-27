from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pose_pipeline.state import PipelineState


def validate_state(state: PipelineState) -> None:
    if state.output_dir is None:
        raise ValueError("PipelineState requires output_dir")
    if state.mode == "dual":
        _require_existing(state.left_pkl, "left_pkl")
        _require_existing(state.right_pkl, "right_pkl")
    elif state.mode == "unified":
        _require_existing(state.unified_pkl, "unified_pkl")
    else:
        raise ValueError(f"Unknown state mode: {state.mode}")


def validate_pose_content(pose_data: dict[str, Any]) -> None:
    poses_3d = np.asarray(pose_data.get("poses_3d"))
    if poses_3d.ndim != 3:
        raise ValueError(f"poses_3d must have shape [T, J, 3], got {poses_3d.shape}")
    if poses_3d.shape[-1] != 3:
        raise ValueError(f"last dimension must be 3, got {poses_3d.shape[-1]}")
    if not np.isfinite(poses_3d).all():
        raise ValueError("poses_3d contains NaN or Inf")


def _require_existing(path: Path | None, field_name: str) -> None:
    if path is None:
        raise ValueError(f"{field_name} is required")
    if not Path(path).exists():
        raise FileNotFoundError(path)
