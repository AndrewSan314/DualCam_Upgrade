from __future__ import annotations

from pathlib import Path
from typing import Any

from pose_pipeline.io_utils.calibration_loader import load_calibration_if_exists
from pose_pipeline.io_utils.pkl_loader import load_wham_pkl
from pose_pipeline.io_utils.video_loader import read_video_info


REQUIRED_FILES = ("cam_left.mp4", "cam_right.mp4", "left.pkl", "right.pkl")


def load_inputs(input_dir: str | Path) -> dict[str, Any]:
    root = Path(input_dir).expanduser().resolve()
    missing = [name for name in REQUIRED_FILES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Input folder {root} is missing: {', '.join(missing)}")

    left_pose = load_wham_pkl(root / "left.pkl")
    right_pose = load_wham_pkl(root / "right.pkl")

    left = {
        **left_pose,
        "video_path": str((root / "cam_left.mp4").resolve()),
        "video_info": read_video_info(root / "cam_left.mp4"),
        "camera_intrinsics": load_calibration_if_exists(root, "left"),
        "metadata": {**left_pose["metadata"], "side": "left"},
    }
    right = {
        **right_pose,
        "video_path": str((root / "cam_right.mp4").resolve()),
        "video_info": read_video_info(root / "cam_right.mp4"),
        "camera_intrinsics": load_calibration_if_exists(root, "right"),
        "metadata": {**right_pose["metadata"], "side": "right"},
    }

    frame_count = min(left["poses_3d"].shape[0], right["poses_3d"].shape[0])
    joint_count = min(left["poses_3d"].shape[1], right["poses_3d"].shape[1])
    joint_names = left["joint_names"][:joint_count]

    for view in (left, right):
        view["poses_3d"] = view["poses_3d"][:frame_count, :joint_count]
        view["confidence"] = view["confidence"][:frame_count, :joint_count]
        view["joint_names"] = joint_names
        view["frame_ids"] = view["frame_ids"][:frame_count]

    return {
        "input_dir": str(root),
        "joint_names": joint_names,
        "left": left,
        "right": right,
        "fused": {
            "poses_3d": None,
            "confidence": None,
            "source_view": None,
            "metadata": {},
        },
        "metadata": {
            "frame_count": int(frame_count),
            "joint_count": int(joint_count),
        },
        "logs": [],
    }

