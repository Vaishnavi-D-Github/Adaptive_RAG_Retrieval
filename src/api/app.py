"""Session-authenticated WSGI API and static UI for Adaptive Enterprise RAG."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import secrets
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from http.cookies import SimpleCookie
from pathlib import Path
from wsgiref.simple_server import make_server

import pymysql

from adaptive.app_ingestion import (
    build_chunks,
    extract_document,
    extract_pages,
    sha256_bytes,
    validate_chunks,
    validate_embeddings,
    verify_chroma_records,
)
from adaptive.auth import AuthError, DuplicateEmailError, MySQLAuthService, SafeUser
from adaptive.config import (
    APPLICATION_CHROMA_PATH,
    APPLICATION_COLLECTION_NAME,
    FIXED_K_VALUES,
    HISTORY_ROOT,
    UPLOAD_ROOT,
    load_project_env,
)
from adaptive.document_store import DuplicateDocumentError, MySQLDocumentStore
from adaptive.history_store import FileHistoryStore
from retrieval.chroma_retriever import ABSTAIN_NO_ACCESS, ChromaRetriever
from adaptive.upload_store import UploadStore

load_project_env()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt"}
SESSION_COOKIE = "ae_rag_session"
LOG = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, str] = {}

    def create(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        self._sessions[token] = user_id
        return token

    def get_user_id(self, token: str | None) -> str | None:
        return self._sessions.get(token or "")

    def delete(self, token: str | None) -> None:
        if token:
            self._sessions.pop(token, None)


def _cookie_header(token: str | None, *, expire: bool = False) -> str:
    secure = os.getenv("AE_RAG_COOKIE_SECURE", "").strip().lower() in {"1", "true", "yes"}
    parts = [f"{SESSION_COOKIE}={token or ''}", "Path=/", "HttpOnly", "SameSite=Lax"]
    if secure:
        parts.append("Secure")
    if expire:
        parts.extend(["Max-Age=0", "Expires=Thu, 01 Jan 1970 00:00:00 GMT"])
    return "; ".join(parts)


def _session_token(environ) -> str | None:
    cookie = SimpleCookie(environ.get("HTTP_COOKIE", ""))
    morsel = cookie.get(SESSION_COOKIE)
    return morsel.value if morsel else None


def _safe_result(result):
    data = _jsonable(result)
    if isinstance(data, dict):
        telemetry = data.get("telemetry") or {}
        if isinstance(telemetry, dict):
            telemetry = dict(telemetry)
            aliases = {
                "retrieval_time_ms": "retrieval_latency_ms",
                "verification_time_ms": "verification_latency_ms",
                "optimization_time_ms": "optimization_latency_ms",
                "generation_time_ms": "generation_latency_ms",
                "predicted_k_confidence": "prediction_confidence",
            }
            for source, target in aliases.items():
                if source in telemetry and target not in telemetry:
                    telemetry[target] = telemetry[source]
                if target in telemetry and source not in telemetry:
                    telemetry[source] = telemetry[target]
        answer = data.get("generation_result") or data.get("answer") or telemetry.get("generated_answer") or ""
        if not isinstance(answer, str):
            answer = "" if answer is None else str(answer)
        sources = telemetry.get("retrieved_sources") or data.get("sources") or []
        return {"answer": answer, "sources": sources, "telemetry": telemetry, "raw": data}
    return {"answer": str(data), "sources": [], "telemetry": {}, "raw": data}


def _history_record(user: SafeUser, query: str, mode: str, result) -> dict:
    normalized = _safe_result(result)
    telemetry = normalized["telemetry"]
    return {
        "user_id": user.user_id,
        "user_name": user.full_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": query,
        "mode": mode,
        "answer": normalized["answer"],
        "sources": normalized["sources"],
        "telemetry": telemetry,
        "initial_k": telemetry.get("initial_k"),
        "selected_k": telemetry.get("selected_k"),
        "total_tokens": telemetry.get("total_tokens"),
        "total_latency_ms": telemetry.get("total_latency_ms"),
    }


def _db_config() -> dict:
    return {
        "host": os.getenv("AE_RAG_DB_HOST", "localhost"),
        "port": int(os.getenv("AE_RAG_DB_PORT", "3306")),
        "user": os.getenv("AE_RAG_DB_USER", "root"),
        "password": os.getenv("AE_RAG_DB_PASSWORD", ""),
        "database": os.getenv("AE_RAG_DB_NAME", "adaptive_enterprise_rag"),
        "charset": "utf8mb4",
        "cursorclass": pymysql.cursors.DictCursor,
    }


def _load_rag_runs(limit: int = 50) -> list[dict]:
    sql = """
    SELECT run_id, question, mode, timestamp, initial_k, predicted_k, selected_k,
           maximum_k, prediction_confidence, retrieval_iterations, retrieval_strategy,
           k_escalated, verification_performed, verification_result,
           num_retrieved_chunks, num_chunks_used, unique_documents,
           prompt_tokens, generated_tokens, total_tokens,
           retrieval_latency_ms, verification_latency_ms, optimization_latency_ms,
           generation_latency_ms, total_latency_ms, generated_answer, retrieved_sources
    FROM rag_runs
    WHERE run_type = %s
    ORDER BY timestamp DESC
    LIMIT %s
    """
    connection = pymysql.connect(**_db_config())
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, ("production", int(limit)))
            rows = cursor.fetchall()
    finally:
        connection.close()
    for row in rows:
        if isinstance(row.get("retrieved_sources"), str):
            try:
                row["retrieved_sources"] = json.loads(row["retrieved_sources"])
            except json.JSONDecodeError:
                row["retrieved_sources"] = []
    return rows


def create_app(
    adaptive_pipeline,
    fixed_pipeline,
    *,
    collection=None,
    embedding_model=None,
    auth_service=None,
    session_store=None,
    history_store=None,
    upload_root=None,
    document_store=None,
):
    auth = auth_service or MySQLAuthService()
    sessions = session_store or SessionStore()
    persistent_history = history_store if history_store is not None else FileHistoryStore(HISTORY_ROOT)
    upload_store = UploadStore(UPLOAD_ROOT) if upload_root is None else UploadStore(Path(upload_root))
    documents = document_store or MySQLDocumentStore()
    frontend_dir = PROJECT_ROOT / "frontend" / "dist"
    static_dir = frontend_dir if frontend_dir.exists() else Path(__file__).with_name("static")

    def _history_append(user_id: str, record: dict) -> None:
        if isinstance(persistent_history, dict):
            persistent_history.setdefault(user_id, []).append(record)
        else:
            persistent_history.append(user_id, record)

    def _history_list(user_id: str) -> list:
        if isinstance(persistent_history, dict):
            return list(reversed(persistent_history.get(user_id, [])))
        return persistent_history.list_for_user(user_id)

    def _has_accessible_documents(user_role: str) -> bool | None:
        if collection is None or embedding_model is None:
            if collection is not None and hasattr(collection, "count"):
                return int(collection.count()) > 0
            return None
        return ChromaRetriever(collection, embedding_model).has_accessible_documents(user_role)

    def _delete_indexed_document(document_id: str) -> None:
        if collection is not None and hasattr(collection, "delete"):
            collection.delete(where={"document_id": str(document_id)})
        chunk_file = upload_store.chunks_path(document_id)
        if chunk_file.exists():
            chunk_file.unlink()

    def respond(start_response, status, body, headers=None):
        encoded = json.dumps(_jsonable(body), default=str).encode("utf-8")
        response_headers = [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(encoded))),
        ]
        if headers:
            response_headers.extend(headers)
        start_response(status, response_headers)
        return [encoded]

    def static_response(path: Path, start_response):
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        page = path.read_bytes()
        start_response(
            "200 OK",
            [("Content-Type", content_type), ("Content-Length", str(len(page)))],
        )
        return [page]

    def resolve_static(route: str) -> Path | None:
        if route in {"/", "/index.html"}:
            return static_dir / "index.html"
        candidate = (static_dir / route.lstrip("/")).resolve()
        try:
            candidate.relative_to(static_dir.resolve())
        except ValueError:
            return None
        if candidate.is_file():
            return candidate
        if not route.startswith("/api/"):
            fallback = static_dir / "index.html"
            return fallback if fallback.exists() else None
        return None

    def payload(environ):
        try:
            length = int(environ.get("CONTENT_LENGTH") or 0)
        except ValueError as error:
            raise ValueError("Invalid request length.") from error
        if length > MAX_UPLOAD_BYTES:
            raise ValueError("Request exceeds the 20 MB upload limit.")
        raw = environ["wsgi.input"].read(length) if length else b"{}"
        data = json.loads(raw or b"{}")
        if not isinstance(data, dict):
            raise ValueError("JSON object payload required.")
        return data

    def current_user(environ) -> SafeUser | None:
        user_id = sessions.get_user_id(_session_token(environ))
        if not user_id:
            return None
        user = auth.get_user_by_id(user_id)
        if user is None or not user.is_active:
            sessions.delete(_session_token(environ))
            return None
        return user

    def require_user(environ) -> SafeUser:
        user = current_user(environ)
        if user is None:
            raise PermissionError("Authentication required.")
        return user

    def require_role(environ, role: str) -> SafeUser:
        user = require_user(environ)
        if user.role != role:
            raise PermissionError("Unauthorized for this operation.")
        return user

    def application(environ, start_response):
        route, method = environ.get("PATH_INFO", "/"), environ.get("REQUEST_METHOD")
        if method == "GET":
            static_path = resolve_static(route)
            if static_path is not None and static_path.exists():
                return static_response(static_path, start_response)
        try:
            if method == "GET" and route == "/api/health":
                return respond(
                    start_response,
                    "200 OK",
                    {"success": True, "service": "Adaptive Enterprise RAG", "status": "healthy"},
                )
            if method == "GET" and route == "/api/me":
                user = current_user(environ)
                if not user:
                    return respond(start_response, "401 Unauthorized", {"success": False, "error": "Authentication required."})
                return respond(start_response, "200 OK", {"success": True, "user": user.to_dict()})
            if method == "GET" and route == "/api/history":
                user = require_user(environ)
                history = _history_list(user.user_id)
                return respond(
                    start_response,
                    "200 OK",
                    {
                        "success": True,
                        "history": history,
                    },
                )
            if method == "GET" and route == "/api/admin/documents":
                require_role(environ, "admin")
                return respond(start_response, "200 OK", {"success": True, "documents": documents.list_documents()})

            if method != "POST":
                return respond(start_response, "405 Method Not Allowed", {"success": False, "error": "Unsupported method."})

            data = payload(environ)
            if route == "/api/signup":
                if str(data.get("password", "")) != str(data.get("confirm_password", "")):
                    raise ValueError("Password confirmation does not match.")

                signup_role = str(data.get("role", "")).strip().lower()

                if signup_role == "admin":
                    raise PermissionError(
                        "Administrator accounts cannot be created through public signup."
                    )

                if signup_role not in {"student", "teacher", "office"}:
                    raise ValueError("Role must be student, teacher, or office.")

                user = auth.create_user(
                    full_name=data.get("full_name", ""),
                    email=data.get("email", ""),
                    password=data.get("password", ""),
                    role=signup_role,
                )
                return respond(
                    start_response,
                    "201 Created",
                    {"success": True, "message": "Account created successfully. Please log in.", "user": user.to_dict()},
                )
            if route == "/api/login":
                try:
                    user = auth.authenticate_user(email=data.get("email", ""), password=data.get("password", ""))
                except AuthError as error:
                    status = "403 Forbidden" if str(error) == "Account disabled." else "401 Unauthorized"
                    public_error = "Account disabled." if str(error) == "Account disabled." else "Invalid credentials."
                    return respond(start_response, status, {"success": False, "error": public_error})
                token = sessions.create(user.user_id)
                return respond(
                    start_response,
                    "200 OK",
                    {"success": True, "user": user.to_dict()},
                    [("Set-Cookie", _cookie_header(token))],
                )
            if route == "/api/logout":
                sessions.delete(_session_token(environ))
                return respond(
                    start_response,
                    "200 OK",
                    {"success": True, "message": "Logged out."},
                    [("Set-Cookie", _cookie_header(None, expire=True))],
                )
            if route == "/api/query":
                user = require_user(environ)
                accessible = _has_accessible_documents(user.role)
                if accessible is False:
                    empty = {
                        "mode": str(data.get("mode", "adaptive")).strip().lower(),
                        "answer": ABSTAIN_NO_ACCESS,
                        "sources": [],
                        "telemetry": {
                            "application_chroma_path": str(APPLICATION_CHROMA_PATH),
                            "application_collection": APPLICATION_COLLECTION_NAME,
                            "accessible_documents": False,
                        },
                    }
                    _history_append(user.user_id, _history_record(user, str(data.get("query", "")).strip(), empty["mode"], empty))
                    return respond(start_response, "200 OK", {"success": True, **empty})
                query = str(data.get("query", "")).strip()
                if not query:
                    raise ValueError("Query is required.")
                mode = str(data.get("mode", "adaptive")).strip().lower()
                if mode == "adaptive":
                    result = adaptive_pipeline.run(query, user_role=user.role,)
                elif mode in {f"fixed_{k}" for k in FIXED_K_VALUES}:
                    result = fixed_pipeline.run(query, int(mode.split("_", 1)[1]), user_role=user.role,)
                else:
                    raise ValueError("Mode must be adaptive, fixed_3, fixed_5, or fixed_10.")
                record = _history_record(user, query, mode, result)
                _history_append(user.user_id, record)
                normalized = _safe_result(result)
                return respond(
                    start_response,
                    "200 OK",
                    {
                        "success": True,
                        "mode": mode,
                        "answer": normalized["answer"],
                        "sources": normalized["sources"],
                        "telemetry": normalized["telemetry"],
                        "result": normalized["raw"],
                        "history_entry": record,
                    },
                )
            if route == "/api/admin/upload":
                user = require_role(environ, "admin")
                access_level = str(data.get("access_level", "")).strip().lower()

                if access_level not in {"student", "teacher", "office"}:
                    raise ValueError(
                        "Access level must be student, teacher, or office."
                    )
                if collection is None or embedding_model is None:
                    raise ValueError("Upload indexing is unavailable because the vector runtime was not initialized.")
                name = Path(str(data.get("filename", ""))).name
                if not name or Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
                    raise ValueError("Only PDF, DOCX, and TXT uploads are supported.")
                try:
                    raw = base64.b64decode(str(data.get("content_base64", "")), validate=True)
                except Exception as error:
                    raise ValueError("content_base64 must be valid Base64.") from error
                if not raw:
                    raise ValueError("The uploaded file is empty.")
                if len(raw) > MAX_UPLOAD_BYTES:
                    raise ValueError("Upload exceeds the 20 MB limit.")
                content_hash = sha256_bytes(raw)
                stored_filename = upload_store.stored_filename(filename=name, content_hash=content_hash)
                stored_path = upload_store.document_path(stored_filename)
                document = documents.create_pending(
                    original_filename=name,
                    stored_filename=stored_filename,
                    storage_path=str(stored_path),
                    file_type=Path(name).suffix.lower().lstrip("."),
                    file_size=len(raw),
                    content_hash=content_hash,
                    uploaded_by=user.user_id,
                    access_level=access_level,
                )
                document_id = document["document_id"]
                try:
                    upload_store.save_original(stored_filename=stored_filename, raw=raw)
                    documents.update_status(document_id, "EXTRACTING")
                    pages, tables = extract_document(name, raw)
                    page_count = len(pages)
                    word_count = sum(len(page.text.split()) for page in pages)
                    char_count = sum(len(page.text) for page in pages)
                    documents.update_status(
                        document_id,
                        "CHUNKING",
                        page_count=page_count,
                        extracted_word_count=word_count,
                        extracted_character_count=char_count,
                    )
                    chunks = build_chunks(document_id=document_id, document_name=name, access_level=access_level, pages=pages, tables=tables)
                    validate_chunks(chunks)
                    chunk_dicts = [chunk.to_dict() for chunk in chunks]
                    chunks_path = upload_store.save_chunks(document_id=document_id, filename=name, chunks=chunk_dicts)
                    if len(chunk_dicts) != len(chunks):
                        raise ValueError("Stored chunk count does not match generated chunk count.")
                    documents.update_status(document_id, "EMBEDDING", chunk_count=len(chunks))
                    embeddings = validate_embeddings(embedding_model.encode([chunk.text for chunk in chunks]), len(chunks))
                    documents.update_status(document_id, "INDEXING", embedding_count=len(embeddings))
                    ids = [chunk.chroma_id for chunk in chunks]
                    collection.upsert(
                        ids=ids,
                        documents=[chunk.text for chunk in chunks],
                        embeddings=embeddings,
                        metadatas=[chunk.metadata() for chunk in chunks],
                    )
                    verified_count = verify_chroma_records(collection, ids)
                    documents.update_status(
                        document_id,
                        "INDEXED",
                        page_count=page_count,
                        extracted_word_count=word_count,
                        extracted_character_count=char_count,
                        chunk_count=len(chunks),
                        embedding_count=len(embeddings),
                        chroma_record_count=len(ids),
                        chroma_verified_count=verified_count,
                        metadata_json={
                            "application_chroma_path": str(APPLICATION_CHROMA_PATH),
                            "application_collection": APPLICATION_COLLECTION_NAME,
                            "access_level": access_level,
                            "sample_chunk_id": ids[0],
                            "sample_page": chunks[0].page_number,
                            "sample_text_preview": chunks[0].text[:160],
                            "table_chunk_count": sum(1 for chunk in chunks if chunk.content_type == "table"),
                            "chunks_path": str(chunks_path),
                        },
                    )
                except Exception as error:
                    documents.update_status(document_id, "FAILED", error_message=str(error))
                    raise
                return respond(
                    start_response,
                    "201 Created",
                    {
                        "success": True,
                        "filename": name,
                        "document_id": document_id,
                        "access_level": access_level,
                        "content_hash": content_hash,
                        "status": "INDEXED",
                        "page_count": page_count,
                        "pages_processed": page_count,
                        "extracted_word_count": word_count,
                        "extracted_character_count": char_count,
                        "chunk_count": len(chunks),
                        "table_chunk_count": sum(1 for chunk in chunks if chunk.content_type == "table"),
                        "chunks_stored": len(chunk_dicts),
                        "embedding_count": len(embeddings),
                        "chroma_record_count": len(ids),
                        "chroma_verified_count": verified_count,
                        "application_chroma_path": str(APPLICATION_CHROMA_PATH),
                        "application_collection": APPLICATION_COLLECTION_NAME,
                        "document_path": stored_filename,
                        "chunks_path": Path(chunks_path).name,
                        "message": "Document stored and indexed successfully.",
                    },
                )
            if route == "/api/admin/delete":
                require_role(environ, "admin")
                document_id = str(data.get("document_id", "")).strip()
                record = documents.get(document_id) if document_id else None
                if not record or record.get("status") == "DELETED":
                    raise ValueError("Document was not found.")
                _delete_indexed_document(document_id)
                stored = Path(str(record.get("storage_path") or ""))
                if stored.is_file():
                    stored.unlink()
                documents.mark_deleted(document_id)
                return respond(start_response, "200 OK", {"success": True, "document_id": document_id, "status": "DELETED", "message": "Document and indexed chunks were removed."})
            if route == "/api/admin/reindex":
                user = require_role(environ, "admin")
                document_id = str(data.get("document_id", "")).strip()
                record = documents.get(document_id) if document_id else None
                if not record or record.get("status") == "DELETED":
                    raise ValueError("Document was not found.")
                stored = Path(str(record.get("storage_path") or ""))
                if not stored.is_file():
                    raise ValueError("Original file is missing; reindex is not possible.")
                raw = stored.read_bytes()
                access_level = str(record.get("access_level") or "student")
                name = record.get("original_filename") or stored.name
                _delete_indexed_document(document_id)
                documents.update_status(document_id, "EXTRACTING")
                pages, tables = extract_document(name, raw)
                chunks = build_chunks(document_id=document_id, document_name=name, access_level=access_level, pages=pages, tables=tables)
                validate_chunks(chunks)
                upload_store.save_chunks(document_id=document_id, filename=name, chunks=[chunk.to_dict() for chunk in chunks])
                embeddings = validate_embeddings(embedding_model.encode([chunk.text for chunk in chunks]), len(chunks))
                ids = [chunk.chroma_id for chunk in chunks]
                collection.upsert(ids=ids, documents=[chunk.text for chunk in chunks], embeddings=embeddings, metadatas=[chunk.metadata() for chunk in chunks])
                verified_count = verify_chroma_records(collection, ids)
                documents.update_status(
                    document_id,
                    "INDEXED",
                    page_count=len(pages),
                    chunk_count=len(chunks),
                    embedding_count=len(embeddings),
                    chroma_record_count=len(ids),
                    chroma_verified_count=verified_count,
                    error_message=None,
                )
                return respond(start_response, "200 OK", {"success": True, "document_id": document_id, "status": "INDEXED", "chunk_count": len(chunks), "chroma_verified_count": verified_count})
            return respond(start_response, "404 Not Found", {"success": False, "error": "Unknown endpoint."})
        except DuplicateDocumentError as error:
            return respond(start_response, "409 Conflict", {"success": False, "error": str(error)})
        except DuplicateEmailError as error:
            return respond(start_response, "409 Conflict", {"success": False, "error": str(error)})
        except PermissionError as error:
            return respond(start_response, "403 Forbidden", {"success": False, "error": str(error)})
        except (AuthError, ValueError, json.JSONDecodeError) as error:
            return respond(start_response, "400 Bad Request", {"success": False, "error": str(error)})
        except Exception:
            LOG.exception("Unhandled API error on %s", route)
            return respond(start_response, "500 Internal Server Error", {"success": False, "error": "The request could not be completed."})

    return application


def main():
    logging.basicConfig(level=os.getenv("AE_RAG_LOG_LEVEL", "INFO"))
    from runtime import load_application_runtime

    adaptive, fixed, collection, embedding_model = load_application_runtime(enable_telemetry_db=True)
    try:
        MySQLAuthService().ensure_bootstrap_admin()
    except Exception:
        LOG.exception("Administrator bootstrap from environment variables was skipped.")
    host, port = os.getenv("AE_RAG_HOST", "127.0.0.1"), int(os.getenv("AE_RAG_PORT", "8000"))
    print(f"Adaptive Enterprise RAG is running at http://{host}:{port}")
    print(f"Application ChromaDB: {APPLICATION_CHROMA_PATH} / {APPLICATION_COLLECTION_NAME}")
    print("Authentication uses the existing MySQL users table and HttpOnly session cookies.")
    make_server(host, port, create_app(adaptive, fixed, collection=collection, embedding_model=embedding_model)).serve_forever()


if __name__ == "__main__":
    main()
