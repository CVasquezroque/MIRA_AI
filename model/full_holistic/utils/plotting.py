from __future__ import annotations

from pathlib import Path


def ensure_figures_dir(results_dir: Path) -> Path:
    figures = results_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    return figures

