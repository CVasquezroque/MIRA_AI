from __future__ import annotations

from pathlib import Path


class StageLogger:
    def __init__(self, path: Path, title: str = "Stage Run Log"):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(f"# {title}\n\n", encoding="utf-8")

    def write(self, title: str, body: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"## {title}\n\n{body.strip()}\n\n")

