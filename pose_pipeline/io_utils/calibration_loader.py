from __future__ import annotations

from pathlib import Path
from typing import Any


def load_calibration_if_exists(input_dir: str | Path, side: str) -> dict[str, Any] | None:
    root = Path(input_dir)
    candidates = [
        root / f"calib_{side}.txt",
        root / f"calib_{side}.json",
        root / f"calib_{side}.yaml",
        root / f"calib_{side}.yml",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix.lower() == ".txt":
            values = [float(part) for part in path.read_text(encoding="utf-8").split()]
            if len(values) >= 4:
                return {
                    "path": str(path.resolve()),
                    "fx": values[0],
                    "fy": values[1],
                    "cx": values[2],
                    "cy": values[3],
                }
        return {"path": str(path.resolve()), "raw": path.read_text(encoding="utf-8")}
    return None

