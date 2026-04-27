from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineState:
    mode: str
    left_pkl: Path | None = None
    right_pkl: Path | None = None
    unified_pkl: Path | None = None
    left_video: Path | None = None
    right_video: Path | None = None
    calib_left: Path | None = None
    calib_right: Path | None = None
    benchmark_path: Path | None = None
    output_dir: Path | None = None
    history: list[str] = field(default_factory=list)
    latest_left_pkl: Path | None = None
    latest_right_pkl: Path | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    pose_data: dict[str, Any] | None = None
    snapshots: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pose_data(
        cls,
        pose_data: dict[str, Any],
        *,
        input_dir: Path,
        output_dir: Path,
        benchmark_path: Path | None = None,
    ) -> "PipelineState":
        left_pkl = input_dir / "left.pkl"
        right_pkl = input_dir / "right.pkl"
        state = cls(
            mode="dual",
            left_pkl=left_pkl,
            right_pkl=right_pkl,
            latest_left_pkl=left_pkl,
            latest_right_pkl=right_pkl,
            left_video=input_dir / "cam_left.mp4",
            right_video=input_dir / "cam_right.mp4",
            calib_left=_optional_calibration_path(input_dir, "left"),
            calib_right=_optional_calibration_path(input_dir, "right"),
            benchmark_path=benchmark_path,
            output_dir=output_dir,
            pose_data=pose_data,
        )
        return state


def _optional_calibration_path(input_dir: Path, side: str) -> Path | None:
    for name in (f"calib_{side}.txt", f"{side}_calib.txt", f"camera_{side}.json"):
        path = input_dir / name
        if path.exists():
            return path
    return None
