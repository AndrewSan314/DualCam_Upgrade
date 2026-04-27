from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pose_pipeline.pipelines.judgement_alignment import (
    apply_similarity_transform,
    estimate_sequence_umeyama,
)


def align_pose_to_render_reference(
    pose: np.ndarray,
    reference_pose: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    transform, diagnostics = estimate_render_alignment_transform(pose, reference_pose)
    if transform is None:
        return np.asarray(pose, dtype=np.float32).copy(), diagnostics

    aligned = apply_render_alignment_transform(pose, transform)
    return aligned.astype(np.float32), diagnostics


def estimate_render_alignment_transform(
    pose: np.ndarray,
    reference_pose: np.ndarray,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    source = np.asarray(pose, dtype=np.float32)
    reference = np.asarray(reference_pose, dtype=np.float32)
    if source.ndim != 3 or reference.ndim != 3 or source.shape[-1] != 3 or reference.shape[-1] != 3:
        raise ValueError(
            "Render alignment requires pose arrays with shape [frames, joints, 3]"
        )

    frame_count = min(source.shape[0], reference.shape[0])
    joint_count = min(source.shape[1], reference.shape[1])
    diagnostics: dict[str, Any] = {
        "method": "sequence_umeyama",
        "frame_count": int(frame_count),
        "joint_count": int(joint_count),
    }
    if frame_count == 0 or joint_count == 0:
        diagnostics["fallback"] = "empty_overlap"
        return None, diagnostics

    transform, transform_diag = estimate_sequence_umeyama(
        source[:frame_count, :joint_count],
        reference[:frame_count, :joint_count],
    )
    diagnostics.update(transform_diag)
    if transform is None:
        diagnostics.setdefault("fallback", "no_similarity_transform")
        return None, diagnostics

    serializable = _serializable_transform(transform)
    diagnostics.update(serializable)
    return serializable, diagnostics


def apply_render_alignment_transform(
    pose: np.ndarray,
    transform: dict[str, Any],
) -> np.ndarray:
    return apply_similarity_transform(np.asarray(pose, dtype=np.float32), transform).astype(np.float32)


def save_render_alignment_transform(
    transform: dict[str, Any],
    path: str | Path,
    diagnostics: dict[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "pose_pipeline.render_alignment_transform.v1",
        "row_vector_formula": "aligned = scale * (pose @ rotation.T) + translation",
        "transform": _serializable_transform(transform),
        "diagnostics": diagnostics or {},
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def load_render_alignment_transform(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    transform = _extract_transform(payload)
    if transform is None:
        raise ValueError(f"No render alignment transform found in {source}")
    return transform


def default_render_alignment_cache_path(reference_path: str | Path) -> Path:
    path = Path(reference_path)
    return path.with_name(f"{path.stem}_render_transform.json")


def transform_diagnostics(transform: dict[str, Any]) -> dict[str, Any]:
    serializable = _serializable_transform(transform)
    return {
        "method": "fixed_similarity_transform",
        **serializable,
    }


def _serializable_transform(transform: dict[str, Any]) -> dict[str, Any]:
    return {
        "scale": float(transform["scale"]),
        "rotation": np.asarray(transform["rotation"], dtype=float).tolist(),
        "translation": np.asarray(transform["translation"], dtype=float).tolist(),
    }


def _extract_transform(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    candidates = [
        payload.get("transform"),
        payload.get("render_alignment"),
        payload,
    ]
    sequences = payload.get("sequences")
    if isinstance(sequences, list):
        for item in sequences:
            if isinstance(item, dict):
                candidates.append(item.get("render_alignment"))

    for candidate in candidates:
        if _looks_like_transform(candidate):
            return _serializable_transform(candidate)
    return None


def _looks_like_transform(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if not {"scale", "rotation", "translation"}.issubset(value):
        return False
    rotation = np.asarray(value["rotation"], dtype=float)
    translation = np.asarray(value["translation"], dtype=float)
    return rotation.shape == (3, 3) and translation.shape == (3,)
