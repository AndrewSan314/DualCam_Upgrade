from __future__ import annotations

import copy
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pkl_io import PoseSequence


if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

for _alias, _value in {
    "bool": np.bool_,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "str": str,
    "unicode": str,
}.items():
    if not hasattr(np, _alias):
        setattr(np, _alias, _value)


@dataclass
class InferenceConfig:
    checkpoint: str | None = None
    learnable_smplify_src: str | None = None
    smpl_model_dir: str | None = None
    device: str = "auto"
    fallback_refiner: str = "smooth"
    smooth_window: int = 3


class LearnableInverseKinematicSolver:
    """Adapter for Learnable-SMPLify with a deterministic local fallback."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.net: Any = None
        self.torch: Any = None
        self.load_error: str | None = None
        self.loaded_from: str | None = None
        self._load_model_if_requested()

    @property
    def is_model_loaded(self) -> bool:
        return self.net is not None and self.torch is not None

    def refine(self, sequence: PoseSequence) -> PoseSequence:
        if self.is_model_loaded:
            try:
                refined = self._refine_with_learnable_smplify(sequence)
                refined.metadata["refined_by"] = "learnable_smplify"
                refined.metadata["checkpoint"] = self.config.checkpoint
                return refined
            except Exception as exc:
                self.load_error = (
                    f"Learnable-SMPLify inference failed; fallback used: {exc}"
                )

        if self.config.fallback_refiner == "none":
            frames = copy.deepcopy(sequence.frames)
            method = "none"
        else:
            frames = _smooth_frames(sequence.frames, self.config.smooth_window)
            method = "temporal_smoothing_fallback"

        metadata = dict(sequence.metadata)
        metadata.update(
            {
                "refined_by": method,
                "learnable_smplify_status": self.load_error
                or "checkpoint/source not provided",
            }
        )
        return PoseSequence(
            source_path=sequence.source_path,
            frames=frames,
            metadata=metadata,
            raw_data=sequence.raw_data,
            raw_person=sequence.raw_person,
        )

    def _load_model_if_requested(self) -> None:
        if not self.config.checkpoint or not self.config.learnable_smplify_src:
            self.load_error = "Missing --checkpoint or --learnable-smplify-src."
            return

        checkpoint = Path(self.config.checkpoint)
        src = Path(self.config.learnable_smplify_src)
        if not checkpoint.exists():
            self.load_error = f"Checkpoint not found: {checkpoint}"
            return
        if not src.exists():
            self.load_error = f"Learnable-SMPLify src folder not found: {src}"
            return

        try:
            import torch
            import yaml
            from easydict import EasyDict as edict
        except Exception as exc:
            self.load_error = f"Cannot import Learnable-SMPLify dependencies: {exc}"
            return

        sys.path.insert(0, str(src))
        try:
            from module.net_body25 import NetBody25

            config_path = src / "configs" / "net.yaml"
            if not config_path.exists():
                config_path = src / "config" / "net.yaml"
            with config_path.open("r", encoding="utf-8") as handle:
                config = edict(yaml.safe_load(handle))
            if self.config.smpl_model_dir:
                config.model_params.human_model.smpl_dir = self.config.smpl_model_dir

            device = self._device(torch)
            net = NetBody25(config.model_params)
            net = net.to(device)
            net.eval()
            state = torch.load(str(checkpoint), map_location="cpu")
            state_dict = (
                state["model"]
                if isinstance(state, dict) and "model" in state
                else state
            )
            net.load_state_dict(state_dict)

            if hasattr(net, "human_model"):
                for layer_name in net.human_model.layer.keys():
                    net.human_model.layer[layer_name] = net.human_model.layer[
                        layer_name
                    ].to(device)

            self.torch = torch
            self.net = net
            self.loaded_from = str(src)
            self.load_error = None
        except Exception as exc:
            self.load_error = f"Could not load Learnable-SMPLify model: {exc}"
        finally:
            try:
                sys.path.remove(str(src))
            except ValueError:
                pass

    def _refine_with_learnable_smplify(self, sequence: PoseSequence) -> PoseSequence:
        person = sequence.raw_person
        if not isinstance(person, dict):
            raise ValueError(
                "Raw PKL person data must be a dict for Learnable-SMPLify inference."
            )

        pose_key = "pose" if "pose" in person else "poses"
        if pose_key not in person or "betas" not in person or "trans" not in person:
            raise ValueError("PKL must contain pose/poses, betas and trans arrays.")

        poses = np.asarray(person[pose_key], dtype=np.float32)
        betas = np.asarray(person["betas"], dtype=np.float32)
        trans = np.asarray(person["trans"], dtype=np.float32)
        if poses.ndim != 2 or poses.shape[1] < 72 or len(poses) < 2:
            raise ValueError(
                "Expected poses with shape (frames, >=72) and at least two frames."
            )

        torch = self.torch
        assert torch is not None
        device = self._device(torch)

        if betas.ndim == 1:
            betas = np.repeat(betas[np.newaxis, :], len(poses), axis=0)
        item = {
            "poses": torch.as_tensor(
                poses[np.newaxis, :, :72], dtype=torch.float32, device=device
            ),
            "betas": torch.as_tensor(
                betas[np.newaxis, :, :10], dtype=torch.float32, device=device
            ),
            "trans": torch.as_tensor(
                trans[np.newaxis, :, :3], dtype=torch.float32, device=device
            ),
        }

        frames = sequence.frame_ids
        refined_frames = copy.deepcopy(sequence.frames)
        iter_root = None
        iter_body = None
        with torch.no_grad():
            for frame_idx in range(len(poses) - 1):
                model_input = {
                    "start_pose": item["poses"][:, frame_idx].clone(),
                    "end_pose": item["poses"][:, frame_idx + 1].clone(),
                    "betas": item["betas"][:, frame_idx + 1].clone(),
                    "start_trans": item["trans"][:, frame_idx].clone(),
                    "end_trans": item["trans"][:, frame_idx + 1].clone(),
                }
                if iter_root is not None and iter_body is not None:
                    model_input["start_pose"][:, :3] = iter_root.view(1, -1).clone()
                    model_input["start_pose"][:, 3:66] = iter_body.view(1, -1)[
                        :, :63
                    ].clone()

                _, info = self.net(model_input, is_training=False)
                iter_root = info["pred_root_orient"].detach()
                iter_body = info["pred_body_pose"].detach()
                if "pred_smpl_joints" in info:
                    joints = info["pred_smpl_joints"].detach().cpu().numpy()[0]
                    frame_id = frames[frame_idx + 1]
                    names = list(refined_frames[frame_id])
                    for joint_index, name in enumerate(names[: joints.shape[0]]):
                        refined_frames[frame_id][name] = joints[joint_index, :3].astype(
                            float
                        )

        return PoseSequence(
            source_path=sequence.source_path,
            frames=refined_frames,
            metadata=dict(sequence.metadata),
            raw_data=sequence.raw_data,
            raw_person=sequence.raw_person,
        )

    def _device(self, torch: Any) -> str:
        if self.config.device != "auto":
            return self.config.device
        return "cuda" if torch.cuda.is_available() else "cpu"


def _smooth_frames(
    frames: dict[int | str, dict[str, np.ndarray]], window: int
) -> dict[int | str, dict[str, np.ndarray]]:
    from pose_analysis import sort_frame_ids

    frame_ids = sort_frame_ids(frames)
    if len(frame_ids) < 3 or window <= 1:
        return copy.deepcopy(frames)

    radius = max(1, window // 2)
    result = copy.deepcopy(frames)
    joint_names = sorted(
        set.intersection(*(set(frames[frame_id]) for frame_id in frame_ids))
    )

    for index, frame_id in enumerate(frame_ids):
        left = max(0, index - radius)
        right = min(len(frame_ids), index + radius + 1)
        neighborhood = frame_ids[left:right]
        for joint_name in joint_names:
            coords = np.asarray(
                [frames[neighbor][joint_name] for neighbor in neighborhood],
                dtype=float,
            )
            result[frame_id][joint_name] = np.mean(coords, axis=0)
    return result
