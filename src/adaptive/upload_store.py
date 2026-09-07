"""Append-only storage for HR document uploads and extracted chunks."""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _safe_name(name: str) -> str:
    stem = Path(name).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "document"
    return cleaned[:180]


class UploadStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.documents_dir = self.root / "documents"
        self.chunks_dir = self.root / "chunks"
        self.manifest_path = self.root / "manifest.jsonl"
        self._lock = threading.Lock()
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

    def stored_filename(self, *, filename: str, content_hash: str) -> str:
        return f"{content_hash[:12]}_{_safe_name(filename)}"

    def document_path(self, stored_filename: str) -> Path:
        return self.documents_dir / _safe_name(stored_filename)

    def chunks_path(self, document_id: str) -> Path:
        return self.chunks_dir / f"{_safe_name(document_id)}.json"

    def save_original(self, *, stored_filename: str, raw: bytes) -> Path:
        path = self.document_path(stored_filename)
        path.write_bytes(raw)
        return path

    def save_chunks(self, *, document_id: str, filename: str, chunks: list[dict]) -> Path:
        path = self.chunks_path(document_id)
        path.write_text(
            json.dumps({"document_id": document_id, "filename": Path(filename).name, "chunks": chunks}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def save(self, *, filename: str, raw: bytes, chunks: list[dict], uploaded_by: str) -> dict[str, Any]:
        upload_id = str(uuid.uuid4())
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stored_name = f"{stamp}_{upload_id[:8]}_{_safe_name(filename)}"
        document_path = self.documents_dir / stored_name
        chunks_path = self.chunks_dir / f"{upload_id}.json"
        record = {
            "upload_id": upload_id,
            "filename": Path(filename).name,
            "stored_filename": stored_name,
            "uploaded_by": uploaded_by,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "byte_size": len(raw),
            "chunk_count": len(chunks),
            "document_path": str(document_path),
            "chunks_path": str(chunks_path),
        }
        with self._lock:
            document_path.write_bytes(raw)
            chunks_path.write_text(
                json.dumps({"upload_id": upload_id, "filename": Path(filename).name, "chunks": chunks}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with self.manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def list_documents(self) -> list[dict[str, Any]]:
        if not self.manifest_path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with self.manifest_path.open(encoding="utf-8") as handle:
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
