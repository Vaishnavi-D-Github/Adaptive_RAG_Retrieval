"""Per-user query history that survives process and login sessions."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any


def _safe_user_id(user_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(user_id).strip()) or "anonymous"
    return cleaned[:120]


class FileHistoryStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, user_id: str) -> Path:
        return self.root / f"{_safe_user_id(user_id)}.jsonl"

    def append(self, user_id: str, record: dict[str, Any]) -> None:
        path = self._path(user_id)
        payload = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")

    def list_for_user(self, user_id: str) -> list[dict[str, Any]]:
        path = self._path(user_id)
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        rows.reverse()
        return rows
