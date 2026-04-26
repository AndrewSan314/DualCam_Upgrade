from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pose_pipeline.io_utils.pkl_loader import load_wham_pkl


def evaluate_benchmark(pred_pose: np.ndarray, benchmark_path: str | Path, output_path: str | Path) -> dict[str, float]:
    gt = load_wham_pkl(benchmark_path)["poses_3d"]
    frames = min(len(pred_pose), len(gt))
    joints = min(pred_pose.shape[1], gt.shape[1])
    pred = pred_pose[:frames, :joints]
    target = gt[:frames, :joints]
    result = {
        "mpjpe": compute_mpjpe(pred, target),
        "pa_mpjpe": compute_pa_mpjpe(pred, target),
        "pck_50mm": compute_pck(pred, target, threshold=0.05),
        "acceleration_error": compute_acceleration_error(pred, target),
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def compute_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.linalg.norm(pred - gt, axis=-1).mean())


def compute_pck(pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    return float((np.linalg.norm(pred - gt, axis=-1) < threshold).mean())


def compute_acceleration_error(pred: np.ndarray, gt: np.ndarray) -> float:
    if len(pred) < 3:
        return 0.0
    acc_p = pred[2:] - 2 * pred[1:-1] + pred[:-2]
    acc_g = gt[2:] - 2 * gt[1:-1] + gt[:-2]
    return float(np.linalg.norm(acc_p - acc_g, axis=-1).mean())


def compute_pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    values = []
    for pred_frame, gt_frame in zip(pred, gt):
        values.append(compute_mpjpe(_procrustes_align(pred_frame, gt_frame), gt_frame))
    return float(np.mean(values)) if values else 0.0


def _procrustes_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    mu_p = pred.mean(axis=0)
    mu_g = gt.mean(axis=0)
    p = pred - mu_p
    g = gt - mu_g
    norm_p = np.sqrt((p**2).sum() / max(1, len(pred)))
    norm_g = np.sqrt((g**2).sum() / max(1, len(gt)))
    if norm_p > 0:
        p /= norm_p
    if norm_g > 0:
        g /= norm_g
    u, _, vt = np.linalg.svd(g.T @ p)
    r = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    return norm_g * (p @ r.T) + mu_g

