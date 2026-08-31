"""Session-authenticated WSGI API and static UI for Adaptive Enterprise RAG."""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from http.cookies import SimpleCookie
from pathlib import Path
from wsgiref.simple_server import make_server

import pymysql

from adaptive.auth import AuthError, DuplicateEmailError, MySQLAuthService, SafeUser
from adaptive.config import FIXED_K_VALUES

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt"}
SESSION_COOKIE = "ae_rag_session"
LOG = logging.getLogger(__name__)


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _extract_upload(name, raw):
    suffix = Path(name).suffix.lower()
    if suffix == ".txt":
        return [{"page_number": 1, "text": raw.decode("utf-8", errors="replace")}]
    if suffix == ".pdf":
        try:
            import pymupdf
        except ImportError:
            import fitz as pymupdf

        document = pymupdf.open(stream=raw, filetype="pdf")
        try:
            return [
                {"page_number": n, "text": page.get_text("text")}
                for n, page in enumerate(document, 1)
                if page.get_text("text").strip()
            ]
        finally:
            document.close()
    try:
        from docx import Document
    except ImportError as error:
        raise ValueError("DOCX uploads require python-docx. Install requirements.txt and restart.") from error
    import io

    text = "\n".join(p.text for p in Document(io.BytesIO(raw)).paragraphs if p.text.strip())
    return [{"page_number": 1, "text": text}] if text else []


def _index_upload(name, raw, collection, embedding_model):
    chunks = []
    for page in _extract_upload(name, raw):
        words, start = page["text"].split(), 0
        while start < len(words):
            text = " ".join(words[start : start + 500])
            if text:
                chunks.append({"page": page["page_number"], "text": text})
            start += 400
    if not chunks:
        raise ValueError("The uploaded file does not contain extractable text.")
    stem = "".join(c if c.isalnum() else "_" for c in Path(name).stem)
    ids = [f"upload_{stem}_p{x['page']}_c{i}" for i, x in enumerate(chunks, 1)]
    embeddings = embedding_model.encode([x["text"] for x in chunks])
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()
    collection.upsert(
        ids=ids,
        documents=[x["text"] for x in chunks],
        embeddings=embeddings,
        metadatas=[{"document": name, "page": x["page"], "chunk_id": i} for i, x in enumerate(chunks, 1)],
    )
    return len(chunks)


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
        answer = data.get("generation_result") or data.get("answer") or telemetry.get("generated_answer") or ""
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
):
    auth = auth_service or MySQLAuthService()
    sessions = session_store or SessionStore()
    user_history = history_store if history_store is not None else {}
    static_dir = Path(__file__).with_name("static")

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
        if method == "GET" and route in {"/", "/index.html"}:
            page = (static_dir / "index.html").read_bytes()
            start_response(
                "200 OK",
                [("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(page)))],
            )
            return [page]
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
                if user.role == "hr":
                    try:
                        runs = _load_rag_runs()
                    except Exception:
                        LOG.exception("Unable to load MySQL RAG history")
                        runs = []
                    return respond(
                        start_response,
                        "200 OK",
                        {
                            "success": True,
                            "history": list(reversed(user_history.get(user.user_id, []))),
                            "rag_runs": runs,
                            "schema_note": "rag_runs has no user_id column; employee history is limited to authenticated web-session activity.",
                        },
                    )
                return respond(
                    start_response,
                    "200 OK",
                    {
                        "success": True,
                        "history": list(reversed(user_history.get(user.user_id, []))),
                        "schema_note": "rag_runs has no user_id column; employee history is limited to authenticated web-session activity.",
                    },
                )

            if method != "POST":
                return respond(start_response, "405 Method Not Allowed", {"success": False, "error": "Unsupported method."})

            data = payload(environ)
            if route == "/api/signup":
                if str(data.get("password", "")) != str(data.get("confirm_password", "")):
                    raise ValueError("Password confirmation does not match.")
                user = auth.create_user(
                    full_name=data.get("full_name", ""),
                    email=data.get("email", ""),
                    password=data.get("password", ""),
                    role=data.get("role", ""),
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
                query = str(data.get("query", "")).strip()
                if not query:
                    raise ValueError("Query is required.")
                mode = str(data.get("mode", "adaptive")).strip().lower()
                if mode == "adaptive":
                    result = adaptive_pipeline.run(query)
                elif mode in {f"fixed_{k}" for k in FIXED_K_VALUES}:
                    result = fixed_pipeline.run(query, int(mode.split("_", 1)[1]))
                else:
                    raise ValueError("Mode must be adaptive, fixed_3, fixed_5, or fixed_10.")
                record = _history_record(user, query, mode, result)
                user_history.setdefault(user.user_id, []).append(record)
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
            if route == "/api/hr/upload":
                require_role(environ, "hr")
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
                return respond(
                    start_response,
                    "201 Created",
                    {
                        "success": True,
                        "filename": name,
                        "chunk_count": _index_upload(name, raw, collection, embedding_model),
                        "message": "Document indexed successfully.",
                    },
                )
            return respond(start_response, "404 Not Found", {"success": False, "error": "Unknown endpoint."})
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
    from runtime import load_runtime

    adaptive, fixed, collection, embedding_model = load_runtime(enable_telemetry_db=True)
    host, port = os.getenv("AE_RAG_HOST", "127.0.0.1"), int(os.getenv("AE_RAG_PORT", "8000"))
    print(f"Adaptive Enterprise RAG is running at http://{host}:{port}")
    print("Authentication uses the existing MySQL users table and HttpOnly session cookies.")
    make_server(host, port, create_app(adaptive, fixed, collection=collection, embedding_model=embedding_model)).serve_forever()


if __name__ == "__main__":
    main()
