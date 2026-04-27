from __future__ import annotations

import torch


def data_loss(X, left, right, w_left, w_right):
    return (
        w_left[..., None] * (X - left).pow(2)
        + w_right[..., None] * (X - right).pow(2)
    ).mean()


def base_prior_loss(X, base, w_base):
    return (w_base[..., None] * (X - base).pow(2)).mean()


def compute_reference_bone_lengths(base, skeleton_edges):
    if not skeleton_edges:
        return base.new_zeros((0,))
    lengths = []
    for a, b in skeleton_edges:
        bone = base[:, a, :] - base[:, b, :]
        lengths.append(torch.median(torch.linalg.norm(bone, dim=-1)))
    return torch.stack(lengths)


def bone_length_loss(X, skeleton_edges, ref_lengths):
    if not skeleton_edges:
        return X.new_tensor(0.0)
    losses = []
    for idx, (a, b) in enumerate(skeleton_edges):
        bone = X[:, a, :] - X[:, b, :]
        length = torch.linalg.norm(bone, dim=-1)
        losses.append((length - ref_lengths[idx]).pow(2).mean())
    return torch.stack(losses).mean()


def temporal_velocity_loss(X):
    if X.shape[0] < 2:
        return X.new_tensor(0.0)
    return (X[1:] - X[:-1]).pow(2).mean()


def temporal_acceleration_loss(X):
    if X.shape[0] < 3:
        return X.new_tensor(0.0)
    return (X[2:] - 2 * X[1:-1] + X[:-2]).pow(2).mean()


def floor_penetration_loss(X, foot_indices, floor_axis=2, floor_value=0.0):
    if not foot_indices:
        return X.new_tensor(0.0)
    feet = X[:, foot_indices, floor_axis]
    penetration = torch.clamp(float(floor_value) - feet, min=0.0)
    return penetration.pow(2).mean()


def foot_contact_loss(X, foot_indices):
    if X.shape[0] < 2 or not foot_indices:
        return X.new_tensor(0.0)
    feet = X[:, foot_indices, :]
    velocity = torch.linalg.norm(feet[1:] - feet[:-1], dim=-1)
    return velocity.pow(2).mean()
