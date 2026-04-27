from __future__ import annotations

from pathlib import Path

import numpy as np

from pose_pipeline.visualization.pose_renderer import build_pose_view, render_3d_pose


def compose_output_video(
    left_video_path: str,
    right_video_path: str,
    pose_3d: np.ndarray,
    joint_names: list[str],
    output_path: str | Path,
    render_zoom: float = 1.0,
    render_view: str = "front",
    render_yaw_deg: float = 45.0,
    render_pitch_deg: float = 55.0,
    render_y_up: bool | None = None,
) -> Path:
    import cv2

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cap_l = cv2.VideoCapture(left_video_path)
    cap_r = cv2.VideoCapture(right_video_path)
    if not cap_l.isOpened() or not cap_r.isOpened():
        cap_l.release()
        cap_r.release()
        raise RuntimeError("Could not open input videos")

    fps = cap_l.get(cv2.CAP_PROP_FPS) or cap_r.get(cv2.CAP_PROP_FPS) or 30.0
    height = int(min(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT), cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720)
    panel_width = int(height * 9 / 16)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (panel_width * 3, height),
    )
    frame_count = min(
        int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT) or len(pose_3d)),
        int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT) or len(pose_3d)),
        len(pose_3d),
    )
    pose_view = build_pose_view(
        pose_3d[:frame_count],
        joint_names,
        (panel_width, height),
        zoom=render_zoom,
        mode=render_view,
        yaw_deg=render_yaw_deg,
        pitch_deg=render_pitch_deg,
        y_up=render_y_up,
    )
    for idx in range(frame_count):
        ok_l, frame_l = cap_l.read()
        ok_r, frame_r = cap_r.read()
        if not ok_l or not ok_r:
            break
        panels = [
            cv2.resize(frame_l, (panel_width, height)),
            cv2.resize(frame_r, (panel_width, height)),
            render_3d_pose(pose_3d[idx], joint_names, (panel_width, height), pose_view),
        ]
        writer.write(np.concatenate(panels, axis=1))

    writer.release()
    cap_l.release()
    cap_r.release()
    return output
