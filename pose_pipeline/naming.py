from __future__ import annotations

from pathlib import Path


def make_dual_output_paths(output_dir: Path, history: list[str]) -> tuple[Path, Path]:
    suffix = _suffix(history)
    intermediate_dir = output_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    return intermediate_dir / f"left_{suffix}.pkl", intermediate_dir / f"right_{suffix}.pkl"


def make_unified_output_path(output_dir: Path, history: list[str]) -> Path:
    suffix = _suffix(history)
    intermediate_dir = output_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    return intermediate_dir / f"unify_{suffix}.pkl"


def _suffix(history: list[str]) -> str:
    suffix = "".join(history)
    if not suffix:
        raise ValueError("Cannot create an intermediate output name with empty history.")
    return suffix
