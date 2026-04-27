from __future__ import annotations

from typing import Any

import numpy as np
import torch

from pose_pipeline.pipelines.judgement_losses import (
    base_prior_loss,
    bone_length_loss,
    compute_reference_bone_lengths,
    data_loss,
    floor_penetration_loss,
    foot_contact_loss,
    temporal_acceleration_loss,
    temporal_velocity_loss,
)


def optimize_temporal_multiview_pose(
    base_pose: np.ndarray,
    left_candidate: np.ndarray,
    right_candidate: np.ndarray,
    view_weights: dict[str, np.ndarray],
    skeleton_edges: list[tuple[int, int]],
    foot_indices: list[int],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    device = _device(config)
    base = torch.tensor(base_pose, dtype=torch.float32, device=device)
    left = torch.tensor(left_candidate, dtype=torch.float32, device=device)
    right = torch.tensor(right_candidate, dtype=torch.float32, device=device)
    w_left = torch.tensor(view_weights["left"], dtype=torch.float32, device=device)
    w_right = torch.tensor(view_weights["right"], dtype=torch.float32, device=device)
    w_base = torch.tensor(view_weights["base"], dtype=torch.float32, device=device)

    frame_count = int(base.shape[0])
    window_size = int(config.get("judgement_window_size", 32))
    stride = int(config.get("judgement_stride", 8))
    output_sum = torch.zeros_like(base)
    output_weight = torch.zeros((frame_count, 1, 1), dtype=torch.float32, device=device)
    diagnostics: dict[str, Any] = {"windows": [], "device": str(device)}

    for start in range(0, frame_count, max(stride, 1)):
        end = min(start + window_size, frame_count)
        if end - start < 3:
            output_sum[start:end] += base[start:end]
            output_weight[start:end] += 1.0
            if end == frame_count:
                break
            continue

        window, window_diag = optimize_one_window(
            base=base[start:end],
            left=left[start:end],
            right=right[start:end],
            w_left=w_left[start:end],
            w_right=w_right[start:end],
            w_base=w_base[start:end],
            skeleton_edges=skeleton_edges,
            foot_indices=foot_indices,
            config=config,
        )
        weight = make_overlap_weight(end - start, device=device)
        output_sum[start:end] += window * weight
        output_weight[start:end] += weight
        diagnostics["windows"].append({"start": start, "end": end, **window_diag})
        if end == frame_count:
            break

    pose_out = output_sum / torch.clamp(output_weight, min=1e-8)
    return pose_out.detach().cpu().numpy().astype(np.float32), diagnostics


def optimize_one_window(
    *,
    base,
    left,
    right,
    w_left,
    w_right,
    w_base,
    skeleton_edges,
    foot_indices,
    config,
):
    X = torch.nn.Parameter(base.clone())
    ref_bone_lengths = compute_reference_bone_lengths(base, skeleton_edges)
    optimizer = torch.optim.Adam([X], lr=float(config.get("judgement_lr", 0.03)))
    iterations = int(config.get("judgement_iters", 80))
    history = []

    for _ in range(iterations):
        optimizer.zero_grad()
        l_data = data_loss(X, left, right, w_left, w_right)
        l_prior = base_prior_loss(X, base, w_base)
        l_bone = bone_length_loss(X, skeleton_edges, ref_bone_lengths)
        l_temp = temporal_velocity_loss(X)
        l_acc = temporal_acceleration_loss(X)
        l_floor = floor_penetration_loss(
            X,
            foot_indices,
            floor_axis=int(config.get("judgement_floor_axis", 2)),
            floor_value=float(config.get("judgement_floor_value", 0.0)),
        )
        l_contact = foot_contact_loss(X, foot_indices)
        loss = (
            float(config.get("judgement_lambda_data", 1.0)) * l_data
            + float(config.get("judgement_lambda_prior", 0.4)) * l_prior
            + float(config.get("judgement_lambda_bone", 8.0)) * l_bone
            + float(config.get("judgement_lambda_temp", 0.3)) * l_temp
            + float(config.get("judgement_lambda_acc", 1.5)) * l_acc
            + float(config.get("judgement_lambda_floor", 3.0)) * l_floor
            + float(config.get("judgement_lambda_contact", 2.0)) * l_contact
        )
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))

    return X.detach(), {
        "loss_start": history[0] if history else 0.0,
        "loss_end": history[-1] if history else 0.0,
        "iterations": iterations,
    }


def make_overlap_weight(length: int, *, device) -> torch.Tensor:
    if length <= 2:
        return torch.ones((length, 1, 1), dtype=torch.float32, device=device)
    values = torch.ones(length, dtype=torch.float32, device=device)
    ramp = max(1, min(length // 4, 8))
    fade = torch.linspace(0.25, 1.0, ramp, dtype=torch.float32, device=device)
    values[:ramp] = fade
    values[-ramp:] = torch.flip(fade, dims=(0,))
    return values[:, None, None]


def _device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("judgement_device") or config.get("device") or "auto")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)
