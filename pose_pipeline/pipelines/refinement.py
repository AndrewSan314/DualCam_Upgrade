from __future__ import annotations

import copy
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
from scipy.spatial.transform import Rotation

from pose_pipeline.io_utils.pkl_loader import load_wham_pkl
from pose_pipeline.naming import make_unified_output_path
from pose_pipeline.schema import pose_data_view_to_standard_schema, save_pose_pkl
from pose_pipeline.state import PipelineState
from pose_pipeline.vendor_paths import OPENCAP_ROOT


def run_R(state: PipelineState, config: dict[str, Any]) -> PipelineState:
    if state.pose_data is None:
        raise ValueError("PipelineState.pose_data is required for R")
    if state.output_dir is None:
        raise ValueError("PipelineState.output_dir is required for R")
    if state.mode not in {"dual", "unified"}:
        raise ValueError(f"Unsupported state mode for R: {state.mode}")

    new_history = state.history + ["R"]
    if state.mode == "unified":
        _mirror_unified_pose_to_dual_inputs(state.pose_data)

    state.pose_data = run_pose_refinement(state.pose_data, config)
    output_path = make_unified_output_path(state.output_dir, new_history)
    save_pose_pkl(
        pose_data_view_to_standard_schema(
            state.pose_data,
            "unified",
            new_history,
            created_by="pose_pipeline.pipelines.refinement.run_R",
        ),
        output_path,
    )

    state.mode = "unified"
    state.unified_pkl = output_path
    state.history = new_history
    state.artifacts["".join(new_history)] = output_path
    return state


def run_pose_refinement(pose_data: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(config["output_dir"]).resolve() / "opencap_refinement"
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for side in ("left", "right"):
        print(f"    R: chuan bi du lieu OpenCap cho camera {side}...", flush=True)
        summaries[side] = _run_single_camera_refinement(side, pose_data, config, output_root)

    pose_data["fused"]["poses_3d"] = (pose_data["left"]["poses_3d"] + pose_data["right"]["poses_3d"]) / 2.0
    pose_data["fused"]["confidence"] = (
        pose_data["left"]["confidence"] + pose_data["right"]["confidence"]
    ) / 2.0
    pose_data["fused"]["metadata"]["pose_refinement_optimization"] = {
        "source_root": str(OPENCAP_ROOT),
        "source_function": "optimization.run_optimization",
        "camera_results": summaries,
    }
    pose_data["logs"].append("R: ran opencap-monocular-main optimization.run_optimization for left/right")
    return pose_data


def _mirror_unified_pose_to_dual_inputs(pose_data: dict[str, Any]) -> None:
    fused = pose_data.get("fused", {})
    poses_3d = fused.get("poses_3d")
    if poses_3d is None:
        return
    confidence = fused.get("confidence")
    for side in ("left", "right"):
        pose_data[side]["poses_3d"] = np.asarray(poses_3d, dtype=np.float32).copy()
        if confidence is not None:
            pose_data[side]["confidence"] = np.asarray(confidence, dtype=np.float32).copy()


def _run_single_camera_refinement(
    side: str,
    pose_data: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    side_data = pose_data[side]
    work_dir = output_root / side
    work_dir.mkdir(parents=True, exist_ok=True)

    input_pkl = work_dir / "wham_output.pkl"
    payload_summary = _write_opencap_wham_payload(side, side_data, config, input_pkl)
    source_dir = Path(side_data["source_path"]).resolve().parent
    intrinsics_path = Path(side_data.get("calib_path") or source_dir / f"calib_{side}.txt").resolve()
    video_path = Path(side_data["video_path"]).resolve()

    try:
        with _opencap_import_context():
            from optimization import run_optimization

            output_paths = run_optimization(
                data_dir=str(work_dir),
                trial_name=f"{side}_opencap_refined",
                height_m=float(config.get("opencap_height_m", 1.7)),
                mass_kg=float(config.get("opencap_mass_kg", 70.0)),
                sex=str(config.get("opencap_sex", "m")),
                intrinsics_pth=str(intrinsics_path),
                run_opensim_original_wham=False,
                run_opensim_opt2=False,
                use_gpu=_use_gpu(config),
                static_cam=True,
                optimize_camera=True,
                n_iter_opt2=int(config.get("opencap_iterations", 75)),
                plotting=False,
                save_video_debug=False,
                output_path=str(work_dir),
                video_path=str(video_path),
                activity=config.get("opencap_activity") or "other",
                save_smpl_for_viz=True,
                rotation=0,
                create_contact_visualizations=False,
            )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        _record_refinement_metadata(side_data, "failed", message, payload_summary, None)
        print(f"    R: OpenCap camera {side} loi, giu nguyen pose goc: {message}", flush=True)
        return {
            "status": "failed",
            "error": message,
            "prepared_input": str(input_pkl),
            "payload": payload_summary,
        }

    optimized_pkl = output_paths.get("optimized_pkl") if output_paths else None
    if optimized_pkl and Path(optimized_pkl).exists():
        refined = load_wham_pkl(optimized_pkl)
        _replace_side_pose(side_data, refined)
        _record_refinement_metadata(side_data, "opencap_optimized", None, payload_summary, output_paths)
        print(f"    R: OpenCap camera {side} da xuat optimized pkl.", flush=True)
        return {
            "status": "opencap_optimized",
            "optimized_pkl": optimized_pkl,
            "output_paths": output_paths,
            "payload": payload_summary,
        }

    error = "optimization.run_optimization returned without optimized_pkl"
    _record_refinement_metadata(side_data, "no_optimized_pkl", error, payload_summary, output_paths)
    print(f"    R: OpenCap camera {side} khong tao optimized pkl, giu nguyen pose.", flush=True)
    return {
        "status": "no_optimized_pkl",
        "error": error,
        "output_paths": output_paths,
        "payload": payload_summary,
    }


def _write_opencap_wham_payload(
    side: str,
    side_data: Mapping[str, Any],
    config: Mapping[str, Any],
    destination: Path,
) -> dict[str, Any]:
    person = _normalized_person_payload(side_data)
    max_frames = config.get("opencap_max_frames")
    if max_frames:
        person = _slice_person(person, int(max_frames))

    person["tracking_results_for_reproj"] = {
        "keypoints": _wholebody_keypoints(side_data, len(person["pose_world"]))
    }
    if max_frames:
        person["tracking_results_for_reproj"]["keypoints"] = person["tracking_results_for_reproj"][
            "keypoints"
        ][: int(max_frames)]

    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({0: person}, destination)
    return {
        "side": side,
        "frames": int(len(person["pose_world"])),
        "keys": sorted(person.keys()),
        "path": str(destination),
        "synthetic_2d_keypoints": side_data.get("poses_2d") is None,
    }


def _normalized_person_payload(side_data: Mapping[str, Any]) -> dict[str, Any]:
    raw_person = side_data.get("raw_person")
    if not isinstance(raw_person, Mapping):
        raise ValueError("Input PKL has no raw WHAM person payload.")

    person = copy.deepcopy(dict(raw_person))
    pose_world = _array(person.get("pose_world", person.get("pose")), "pose_world", ndim=2)
    pose_cam = _array(person.get("pose", pose_world), "pose", ndim=2)
    trans_world = _array(person.get("trans_world", person.get("trans")), "trans_world", ndim=2)
    trans_cam = _array(person.get("trans_cam", person.get("trans", trans_world)), "trans_cam", ndim=2)
    betas = _array(person.get("betas"), "betas", ndim=2)

    loaded_frame_count = int(np.asarray(side_data["poses_3d"]).shape[0])
    frame_count = min(len(pose_world), len(trans_world), len(betas), loaded_frame_count)
    pose_world = pose_world[:frame_count]
    pose_cam = pose_cam[:frame_count]
    trans_world = trans_world[:frame_count]
    trans_cam = trans_cam[:frame_count]
    betas = betas[:frame_count]

    person["pose_world"] = pose_world
    person["pose"] = pose_cam
    person["trans_world"] = trans_world
    person["trans"] = trans_cam
    person["trans_cam"] = trans_cam
    person["betas"] = betas
    person["poses_root_world"] = _rotvec_to_matrix(pose_world[:, :3])
    person["poses_root_cam"] = _rotvec_to_matrix(pose_cam[:, :3])
    person["poses_body"] = _rotvec_to_matrix(pose_world[:, 3:72].reshape(frame_count, 23, 3))
    person["contact"] = _contact_probabilities(side_data, frame_count)

    frame_ids = np.asarray(person.get("frame_ids", person.get("frame_id", np.arange(frame_count))), dtype=int)
    person["frame_ids"] = frame_ids[:frame_count]
    person["frame_id"] = person["frame_ids"]

    for key in ("verts", "joints", "keypoints_3d"):
        if key in person and isinstance(person[key], np.ndarray):
            person[key] = person[key][:frame_count]
    return person


def _array(value: Any, name: str, ndim: int) -> np.ndarray:
    if value is None:
        raise ValueError(f"Missing required WHAM field: {name}")
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != ndim:
        raise ValueError(f"Unexpected {name} shape: {arr.shape}")
    return arr


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    flat = np.asarray(rotvec, dtype=np.float32).reshape(-1, 3)
    mats = Rotation.from_rotvec(flat).as_matrix().astype(np.float32)
    return mats.reshape(*rotvec.shape[:-1], 3, 3)


def _contact_probabilities(side_data: Mapping[str, Any], frame_count: int) -> np.ndarray:
    poses = np.asarray(side_data["poses_3d"][:frame_count], dtype=np.float32)
    if poses.shape[1] <= 24:
        return np.full((frame_count, 4), 0.5, dtype=np.float32)

    foot_indices = [19, 21, 22, 24]
    feet = poses[:, foot_indices, :]
    speed = np.zeros((frame_count, 4), dtype=np.float32)
    if frame_count > 1:
        speed[1:] = np.linalg.norm(np.diff(feet, axis=0), axis=2)
    scale = np.percentile(speed, 90) or 1.0
    contact = 1.0 - np.clip(speed / scale, 0.0, 1.0)
    return contact.astype(np.float32)


def _wholebody_keypoints(side_data: Mapping[str, Any], frame_count: int) -> np.ndarray:
    poses_2d = side_data.get("poses_2d")
    if poses_2d is not None:
        body25 = _ensure_xyc(np.asarray(poses_2d[:frame_count], dtype=np.float32))
    else:
        width, height = _video_size(side_data.get("video_path"))
        body25 = _project_body25_to_image(np.asarray(side_data["poses_3d"][:frame_count]), width, height)

    wholebody = np.zeros((frame_count, 133, 3), dtype=np.float32)
    target_slots = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11, 19, 20, 21, 22, 23, 24]
    for source_idx, target_idx in enumerate(target_slots):
        if target_idx < body25.shape[1]:
            wholebody[:, source_idx, :] = body25[:, target_idx, :]
    return wholebody


def _ensure_xyc(arr: np.ndarray) -> np.ndarray:
    if arr.shape[-1] >= 3:
        return arr[..., :3]
    conf = np.ones((*arr.shape[:2], 1), dtype=np.float32)
    return np.concatenate([arr[..., :2], conf], axis=2)


def _project_body25_to_image(poses_3d: np.ndarray, width: int, height: int) -> np.ndarray:
    xy = np.asarray(poses_3d[..., :2], dtype=np.float32)
    xy_min = np.nanmin(xy.reshape(-1, 2), axis=0)
    xy_max = np.nanmax(xy.reshape(-1, 2), axis=0)
    denom = np.maximum(xy_max - xy_min, 1e-6)
    norm = (xy - xy_min) / denom
    out = np.empty((*xy.shape[:2], 3), dtype=np.float32)
    out[..., 0] = width * (0.1 + 0.8 * norm[..., 0])
    out[..., 1] = height * (0.9 - 0.8 * norm[..., 1])
    out[..., 2] = 1.0
    return out


def _video_size(path: Any) -> tuple[int, int]:
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        cap.release()
        return width, height
    except Exception:
        return 1280, 720


def _slice_person(person: dict[str, Any], frame_count: int) -> dict[str, Any]:
    sliced = {}
    for key, value in person.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (len(person["pose_world"]),):
            sliced[key] = value[:frame_count]
        else:
            sliced[key] = value
    return sliced


def _replace_side_pose(side_data: dict[str, Any], refined: dict[str, Any]) -> None:
    side_data["poses_3d"] = refined["poses_3d"]
    side_data["poses_2d"] = refined.get("poses_2d")
    side_data["confidence"] = refined["confidence"]
    side_data["frame_ids"] = refined["frame_ids"]
    side_data["raw_data"] = refined["raw_data"]
    side_data["raw_person"] = refined["raw_person"]
    side_data["raw_person_keys"] = refined["raw_person_keys"]
    side_data["metadata"].update(refined["metadata"])


def _record_refinement_metadata(
    side_data: dict[str, Any],
    status: str,
    error: str | None,
    payload_summary: dict[str, Any],
    output_paths: dict[str, Any] | None,
) -> None:
    side_data["metadata"]["pose_refinement_optimization"] = {
        "status": status,
        "error": error,
        "source_root": str(OPENCAP_ROOT),
        "source_function": "optimization.run_optimization",
        "prepared_payload": payload_summary,
        "output_paths": output_paths,
    }


def _use_gpu(config: Mapping[str, Any]) -> bool:
    if config.get("device") == "cpu":
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@contextmanager
def _opencap_import_context():
    old_cwd = os.getcwd()
    root = str(OPENCAP_ROOT)
    sys.path.insert(0, root)
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        try:
            sys.path.remove(root)
        except ValueError:
            pass
