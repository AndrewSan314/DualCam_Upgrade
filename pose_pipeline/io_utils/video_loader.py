from __future__ import annotations

from pathlib import Path
from typing import Any


def read_video_info(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Missing video file: {source}")
    try:
        import cv2
    except ImportError:
        return {
            "path": str(source.resolve()),
            "fps": None,
            "frame_count": None,
            "width": None,
            "height": None,
            "warning": "opencv-python is not installed; video stats unavailable",
        }
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {source}")
    info = {
        "path": str(source.resolve()),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    return info
