from __future__ import annotations

from pathlib import Path


def dump_joblib(obj, path: Path) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: Path):
    import joblib

    return joblib.load(path)

