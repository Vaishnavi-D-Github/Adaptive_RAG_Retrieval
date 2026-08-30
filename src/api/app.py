"""Role-enforced WSGI API and static UI for Adaptive Enterprise RAG."""
from __future__ import annotations

import base64, json, logging, os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from wsgiref.simple_server import make_server

from adaptive.config import FIXED_K_VALUES

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt"}
LOG = logging.getLogger(__name__)

def _jsonable(value):
    if is_dataclass(value): return _jsonable(asdict(value))
    if isinstance(value, dict): return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    return value

def _users():
    configured = os.getenv("APP_USERS_JSON")
    if configured:
        value = json.loads(configured)
        if not isinstance(value, dict): raise ValueError("APP_USERS_JSON must be a JSON object keyed by email.")
        return value
    return {"hr@example.com": {"password": "change-me-hr", "role": "HR"}, "employee@example.com": {"password": "change-me-employee", "role": "Employee"}}

def _extract_upload(name, raw):
    suffix = Path(name).suffix.lower()
    if suffix == ".txt": return [{"page_number": 1, "text": raw.decode("utf-8", errors="replace")}]
    if suffix == ".pdf":
        import pymupdf
        document = pymupdf.open(stream=raw, filetype="pdf")
        try: return [{"page_number": n, "text": page.get_text("text")} for n, page in enumerate(document, 1) if page.get_text("text").strip()]
        finally: document.close()
    try: from docx import Document
    except ImportError as error: raise ValueError("DOCX uploads require python-docx. Install requirements.txt and restart.") from error
    import io
    text = "\n".join(p.text for p in Document(io.BytesIO(raw)).paragraphs if p.text.strip())
    return [{"page_number": 1, "text": text}] if text else []

def _index_upload(name, raw, collection, embedding_model):
    chunks = []
    for page in _extract_upload(name, raw):
        words, start = page["text"].split(), 0
        while start < len(words):
            text = " ".join(words[start:start + 500])
            if text: chunks.append({"page": page["page_number"], "text": text})
            start += 400
    if not chunks: raise ValueError("The uploaded file does not contain extractable text.")
    stem = "".join(c if c.isalnum() else "_" for c in Path(name).stem)
    ids = [f"upload_{stem}_p{x['page']}_c{i}" for i, x in enumerate(chunks, 1)]
    embeddings = embedding_model.encode([x["text"] for x in chunks])
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()
    collection.upsert(ids=ids, documents=[x["text"] for x in chunks], embeddings=embeddings, metadatas=[{"document": name, "page": x["page"], "chunk_id": i} for i, x in enumerate(chunks, 1)])
    return len(chunks)

def create_app(adaptive_pipeline, fixed_pipeline, *, collection=None, embedding_model=None):
    users, static_dir = _users(), Path(__file__).with_name("static")
    def respond(start_response, status, body):
        encoded = json.dumps(_jsonable(body), default=str).encode("utf-8")
        start_response(status, [("Content-Type", "application/json; charset=utf-8"), ("Content-Length", str(len(encoded)))])
        return [encoded]
    def payload(environ):
        try: length = int(environ.get("CONTENT_LENGTH") or 0)
        except ValueError as error: raise ValueError("Invalid request length.") from error
        if length > MAX_UPLOAD_BYTES: raise ValueError("Request exceeds the 20 MB upload limit.")
        data = json.loads(environ["wsgi.input"].read(length) or b"{}")
        if not isinstance(data, dict): raise ValueError("JSON object payload required.")
        return data
    def authorize(data, role):
        user = users.get(str(data.get("email", "")))
        if not user or user.get("password") != data.get("password") or user.get("role") != role: raise PermissionError("Unauthorized for this operation.")
    def application(environ, start_response):
        route, method = environ.get("PATH_INFO", "/"), environ.get("REQUEST_METHOD")
        if method == "GET" and route in {"/", "/index.html"}:
            page = (static_dir / "index.html").read_bytes(); start_response("200 OK", [("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(page)))]); return [page]
        try:
            if method != "POST": return respond(start_response, "405 Method Not Allowed", {"success": False, "error": "POST required"})
            data = payload(environ)
            if route == "/api/login":
                user = users.get(str(data.get("email", "")))
                if not user or user.get("password") != data.get("password"): raise PermissionError("Invalid credentials.")
                return respond(start_response, "200 OK", {"success": True, "role": user["role"], "development_credentials": not bool(os.getenv("APP_USERS_JSON"))})
            if route == "/api/query":
                authorize(data, "Employee"); query = str(data.get("query", "")).strip()
                if not query: raise ValueError("Query is required.")
                mode = str(data.get("mode", "adaptive"))
                if mode == "adaptive": result = adaptive_pipeline.run(query)
                elif mode in {f"fixed_{k}" for k in FIXED_K_VALUES}: result = fixed_pipeline.run(query, int(mode.split("_", 1)[1]))
                else: raise ValueError("Mode must be adaptive, fixed_3, fixed_5, or fixed_10.")
                return respond(start_response, "200 OK", {"success": True, "mode": mode, "result": result})
            if route == "/api/hr/upload":
                authorize(data, "HR")
                if collection is None or embedding_model is None: raise ValueError("Upload indexing is unavailable because the vector runtime was not initialized.")
                name = Path(str(data.get("filename", ""))).name
                if not name or Path(name).suffix.lower() not in ALLOWED_SUFFIXES: raise ValueError("Only PDF, DOCX, and TXT uploads are supported.")
                try: raw = base64.b64decode(str(data.get("content_base64", "")), validate=True)
                except Exception as error: raise ValueError("content_base64 must be valid Base64.") from error
                if not raw: raise ValueError("The uploaded file is empty.")
                if len(raw) > MAX_UPLOAD_BYTES: raise ValueError("Upload exceeds the 20 MB limit.")
                return respond(start_response, "201 Created", {"success": True, "filename": name, "chunk_count": _index_upload(name, raw, collection, embedding_model), "message": "Document indexed successfully."})
            return respond(start_response, "404 Not Found", {"success": False, "error": "Unknown endpoint"})
        except PermissionError as error: return respond(start_response, "403 Forbidden", {"success": False, "error": str(error)})
        except (ValueError, json.JSONDecodeError) as error: return respond(start_response, "400 Bad Request", {"success": False, "error": str(error)})
        except Exception:
            LOG.exception("Unhandled API error on %s", route); return respond(start_response, "500 Internal Server Error", {"success": False, "error": "The request could not be completed."})
    return application

def main():
    logging.basicConfig(level=os.getenv("AE_RAG_LOG_LEVEL", "INFO"))
    from runtime import load_runtime
    adaptive, fixed, collection, embedding_model = load_runtime(
        enable_telemetry_db=True
    )
    host, port = os.getenv("AE_RAG_HOST", "127.0.0.1"), int(os.getenv("AE_RAG_PORT", "8000"))
    print(f"Adaptive Enterprise RAG is running at http://{host}:{port}")
    print("Development accounts: employee@example.com / change-me-employee; hr@example.com / change-me-hr")
    make_server(host, port, create_app(adaptive, fixed, collection=collection, embedding_model=embedding_model)).serve_forever()

if __name__ == "__main__": main()
