"""MySQL-backed metadata for application-uploaded documents."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import pymysql


DOCUMENT_STATES = {
    "UPLOADING",
    "UPLOADED",
    "EXTRACTING",
    "CHUNKING",
    "EMBEDDING",
    "INDEXING",
    "INDEXED",
    "FAILED",
    "DELETED",
}

DOCUMENT_ACCESS_LEVELS = {
    "student",
    "teacher",
    "office",
}


class DuplicateDocumentError(ValueError):
    pass


class MySQLDocumentStore:
    def __init__(
        self,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        self.host = host or os.getenv("AE_RAG_DB_HOST", "localhost")
        self.port = int(port or os.getenv("AE_RAG_DB_PORT", "3306"))
        self.user = user or os.getenv("AE_RAG_DB_USER", "root")
        self.password = password if password is not None else os.getenv("AE_RAG_DB_PASSWORD", "")
        self.database = database or os.getenv("AE_RAG_DB_NAME", "adaptive_enterprise_rag")
        self.ensure_schema()
        self.ensure_access_level_column()

    def ensure_access_level_column(self) -> None:
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT COUNT(*) AS column_exists
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = DATABASE()
                    AND TABLE_NAME = 'app_documents'
                    AND COLUMN_NAME = 'access_level'
                    """
                )
                row = cursor.fetchone()

                if not row or not row["column_exists"]:
                    cursor.execute(
                        """
                        ALTER TABLE app_documents
                        ADD COLUMN access_level VARCHAR(20)
                        NOT NULL DEFAULT 'student'
                        AFTER uploaded_by
                        """
                    )

            connection.commit()
        finally:
            connection.close()

    def _connect(self):
        return pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=False,
        )

    def ensure_schema(self) -> None:
        sql = """
        CREATE TABLE IF NOT EXISTS app_documents (
            document_id CHAR(36) NOT NULL PRIMARY KEY,
            original_filename VARCHAR(255) NOT NULL,
            stored_filename VARCHAR(255) NOT NULL,
            storage_path VARCHAR(500) NOT NULL,
            file_type VARCHAR(20) NOT NULL,
            file_size BIGINT NOT NULL,
            content_hash CHAR(64) NOT NULL,
            uploaded_by CHAR(36) NOT NULL,
            access_level VARCHAR(20) NOT NULL DEFAULT 'student',
            status VARCHAR(30) NOT NULL,
            page_count INT NULL,
            extracted_word_count INT NULL,
            extracted_character_count INT NULL,
            chunk_count INT NULL,
            embedding_count INT NULL,
            chroma_record_count INT NULL,
            chroma_verified_count INT NULL,
            error_message TEXT NULL,
            metadata_json JSON NULL,
            created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
            updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
            indexed_at DATETIME(6) NULL,
            UNIQUE KEY uq_app_documents_hash (content_hash),
            INDEX idx_app_documents_status (status),
            INDEX idx_app_documents_uploaded_by (uploaded_by)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql)
            connection.commit()
        finally:
            connection.close()

    def create_pending(
        self,
        *,
        original_filename: str,
        stored_filename: str,
        storage_path: str,
        file_type: str,
        file_size: int,
        content_hash: str,
        uploaded_by: str,
        access_level: str = "student",
    ) -> dict[str, Any]:
        access_level = str(access_level or "").strip().lower()

        if access_level not in DOCUMENT_ACCESS_LEVELS:
            raise ValueError(
                "Access level must be student, teacher, or office."
            )
        existing = self.find_by_hash(content_hash)
        if existing and existing.get("status") not in {"FAILED", "DELETED"}:
            raise DuplicateDocumentError("This exact document has already been uploaded.")
        document_id = str(uuid.uuid4())
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO app_documents (
                        document_id, original_filename, stored_filename, storage_path,
                        file_type, file_size, content_hash, uploaded_by, access_level, status
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        document_id,
                        original_filename,
                        stored_filename,
                        storage_path,
                        file_type,
                        file_size,
                        content_hash,
                        uploaded_by,
                        access_level,
                        "UPLOADED",
                    ),
                )
            connection.commit()
        except pymysql.err.IntegrityError as error:
            connection.rollback()
            raise DuplicateDocumentError("This exact document has already been uploaded.") from error
        finally:
            connection.close()
        return self.get(document_id)

    def update_status(self, document_id: str, status: str, **fields) -> None:
        if status not in DOCUMENT_STATES:
            raise ValueError(f"Invalid document status: {status}")
        assignments = ["status = %s"]
        values: list[Any] = [status]
        allowed = {
            "page_count",
            "extracted_word_count",
            "extracted_character_count",
            "chunk_count",
            "embedding_count",
            "chroma_record_count",
            "chroma_verified_count",
            "error_message",
            "metadata_json",
            "content_hash",
        }
        for key, value in fields.items():
            if key not in allowed:
                continue
            assignments.append(f"{key} = %s")
            values.append(json.dumps(value, ensure_ascii=False, default=str) if key == "metadata_json" else value)
        if status == "INDEXED":
            assignments.append("indexed_at = %s")
            values.append(datetime.now(timezone.utc).replace(tzinfo=None))
        values.append(document_id)
        sql = f"UPDATE app_documents SET {', '.join(assignments)} WHERE document_id = %s"
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql, values)
            connection.commit()
        finally:
            connection.close()

    def find_by_hash(self, content_hash: str) -> Optional[dict[str, Any]]:
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM app_documents WHERE content_hash = %s LIMIT 1", (content_hash,))
                return cursor.fetchone()
        finally:
            connection.close()

    def get(self, document_id: str) -> Optional[dict[str, Any]]:
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM app_documents WHERE document_id = %s LIMIT 1", (document_id,))
                return cursor.fetchone()
        finally:
            connection.close()

    def list_documents(self) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM app_documents WHERE status <> %s ORDER BY created_at DESC",
                    ("DELETED",),
                )
                return cursor.fetchall()
        finally:
            connection.close()

    def mark_deleted(self, document_id: str) -> None:
        record = self.get(document_id) or {}
        tombstone = hashlib.sha256(f"deleted:{document_id}:{record.get('content_hash', '')}".encode("utf-8")).hexdigest()
        self.update_status(document_id, "DELETED", content_hash=tombstone)


class InMemoryDocumentStore:
    def __init__(self):
        self.rows: dict[str, dict[str, Any]] = {}

    def create_pending(self, **fields) -> dict[str, Any]:
        access_level = str(fields.get("access_level", "student")).strip().lower()

        if access_level not in DOCUMENT_ACCESS_LEVELS:
            raise ValueError(
                "Access level must be student, teacher, or office."
            )

        fields["access_level"] = access_level
        
        for row in self.rows.values():
            if row["content_hash"] == fields["content_hash"] and row["status"] not in {"FAILED", "DELETED"}:
                raise DuplicateDocumentError("This exact document has already been uploaded.")
        document_id = str(uuid.uuid4())
        row = {"document_id": document_id, "status": "UPLOADED", **fields}
        self.rows[document_id] = row
        return dict(row)

    def update_status(self, document_id: str, status: str, **fields) -> None:
        self.rows[document_id].update({"status": status, **fields})
        if status == "INDEXED":
            self.rows[document_id]["indexed_at"] = datetime.now(timezone.utc).isoformat()

    def find_by_hash(self, content_hash: str) -> Optional[dict[str, Any]]:
        for row in self.rows.values():
            if row["content_hash"] == content_hash:
                return dict(row)
        return None

    def get(self, document_id: str) -> Optional[dict[str, Any]]:
        row = self.rows.get(document_id)
        return dict(row) if row else None

    def list_documents(self) -> list[dict[str, Any]]:
        return list(reversed([dict(row) for row in self.rows.values() if row.get("status") != "DELETED"]))

    def mark_deleted(self, document_id: str) -> None:
        record = self.get(document_id) or {}
        tombstone = hashlib.sha256(f"deleted:{document_id}:{record.get('content_hash', '')}".encode("utf-8")).hexdigest()
        self.update_status(document_id, "DELETED", content_hash=tombstone)
