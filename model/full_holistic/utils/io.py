from __future__ import annotations

import shutil
from pathlib import Path

from model.full_holistic.paths import stage_dir as resolve_stage_dir


class DependencyError(RuntimeError):
    """Raised when a stage is missing required upstream artifacts."""


def prepare_results_dir(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)


def prepare_stage_dir(results_dir: Path, stage: str, force: bool = False) -> Path:
    prepare_results_dir(results_dir)
    path = resolve_stage_dir(results_dir, stage)
    if force and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: Path, message: str) -> Path:
    if not path.exists():
        raise DependencyError(message)
    return path


def optional_read_csv(path: Path) -> pd.DataFrame:
    import pandas as pd

    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def optional_read_text(path: Path, missing_message: str = "This analysis was not run.") -> str:
    return path.read_text(encoding="utf-8") if path.exists() else missing_message


def write_note(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")
