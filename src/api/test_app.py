import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError
from argon2.low_level import Type

from adaptive.auth import AuthError, DuplicateEmailError, SafeUser, normalize_email, validate_role
from adaptive.document_store import InMemoryDocumentStore
from adaptive.history_store import FileHistoryStore
from api.app import create_app


class _Adaptive:
    def run(self, query, **kwargs):
        return {
            "generation_result": f"adaptive: {query}",
            "telemetry": {
                "initial_k": 3,
                "selected_k": 5,
                "predicted_k": 5,
                "predicted_k_confidence": 0.82,
                "retrieval_iterations": 2,
                "num_chunks_used": 2,
                "total_tokens": 42,
                "total_latency_ms": 12.5,
                "retrieved_sources": [{"document": "guide.pdf", "page": 1, "rank": 1, "distance": 0.11}],
            },
        }


class _Fixed:
    def run(self, query, k, **kwargs):
        return {"answer": f"fixed {k}: {query}", "telemetry": {"initial_k": k, "selected_k": k}}


class _Collection:
    def __init__(self, initial_count=1):
        self.calls = []
        self.records = {}
        self.initial_count = initial_count

    def upsert(self, **kwargs):
        self.calls.append(kwargs)
        for item_id, document, metadata in zip(kwargs.get("ids", []), kwargs.get("documents", []), kwargs.get("metadatas", [])):
            self.records[item_id] = {"document": document, "metadata": metadata or {}}

    def get(self, ids=None, where=None, limit=None):
        items = list(self.records.items())
        if ids is not None:
            return {"ids": [item for item in ids if item in self.records]}
        if where:
            allowed = ((where.get("$and") or [{}])[-1].get("access_level") or {}).get("$in")
            if allowed:
                items = [(key, value) for key, value in items if value["metadata"].get("access_level") in allowed and value["metadata"].get("scope") == "application"]
        if limit is not None:
            items = items[: int(limit)]
        return {"ids": [key for key, _ in items], "metadatas": [value["metadata"] for _, value in items], "documents": [value["document"] for _, value in items]}

    def query(self, query_embeddings=None, n_results=1, where=None):
        got = self.get(where=where, limit=n_results)
        count = len(got["ids"])
        return {"documents": [got["documents"]], "metadatas": [got["metadatas"]], "distances": [[0.1] * count]}
        if ids:
            for item in ids:
                self.records.pop(item, None)
        elif where and "document_id" in where:
            target = where["document_id"]
            self.records = {key: value for key, value in self.records.items() if value["metadata"].get("document_id") != target}

    def count(self):
        return self.initial_count + len(self.records)


class _EmbeddingModel:
    def encode(self, values):
        return [[0.1, 0.2] for _ in values]


class _MemoryAuth:
    def __init__(self):
        self.hasher = PasswordHasher(type=Type.ID)
        self.users = {}
        self.next_id = 1

    def create_user(self, *, full_name, email, password, role):
        name = str(full_name or "").strip()
        normalized_email = normalize_email(email)
        normalized_role = validate_role(role)
        if not name:
            raise AuthError("Full name is required.")
        if not normalized_email:
            raise AuthError("Email is required.")
        if not str(password or ""):
            raise AuthError("Password is required.")
        if normalized_email in self.users:
            raise DuplicateEmailError("An account with this email already exists.")
        user = {
            "user_id": f"user-{self.next_id}",
            "full_name": name,
            "email": normalized_email,
            "password_hash": self.hasher.hash(str(password)),
            "role": normalized_role,
            "is_active": True,
        }
        self.next_id += 1
        self.users[normalized_email] = user
        return self._safe(user)

    def authenticate_user(self, *, email, password):
        user = self.users.get(normalize_email(email))
        if not user:
            raise AuthError("Invalid credentials.")
        if not user["is_active"]:
            raise AuthError("Account disabled.")
        try:
            verified = self.hasher.verify(user["password_hash"], str(password))
        except (VerifyMismatchError, VerificationError):
            verified = False
        if not verified:
            raise AuthError("Invalid credentials.")
        return self._safe(user)

    def get_user_by_id(self, user_id):
        for user in self.users.values():
            if user["user_id"] == user_id:
                return self._safe(user)
        return None

    @staticmethod
    def _safe(user):
        return SafeUser(user["user_id"], user["full_name"], user["email"], user["role"], user["is_active"])


def request(app, route, data=None, *, method="POST", cookie=None):
    body = json.dumps(data or {}).encode()
    status, response_headers = [], []

    def start(value, headers):
        status.append(value)
        response_headers.extend(headers)

    environ = {
        "PATH_INFO": route,
        "REQUEST_METHOD": method,
        "CONTENT_LENGTH": str(len(body)) if method == "POST" else "0",
        "wsgi.input": io.BytesIO(body if method == "POST" else b""),
    }
    if cookie:
        environ["HTTP_COOKIE"] = cookie
    result = app(environ, start)
    parsed = json.loads(b"".join(result))
    set_cookie = next((value for key, value in response_headers if key.lower() == "set-cookie"), None)
    return status[0], parsed, set_cookie


def raw_request(app, route, *, method="GET"):
    status, response_headers = [], []

    def start(value, headers):
        status.append(value)
        response_headers.extend(headers)

    result = app(
        {
            "PATH_INFO": route,
            "REQUEST_METHOD": method,
            "CONTENT_LENGTH": "0",
            "wsgi.input": io.BytesIO(b""),
        },
        start,
    )
    return status[0], response_headers, b"".join(result)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.auth = _MemoryAuth()
        self.auth.create_user(
            full_name="Student User",
            email="student@example.com",
            password="change-me-student",
            role="student",
        )
        self.auth.create_user(
            full_name="Admin User",
            email="admin@example.com",
            password="change-me-admin",
            role="admin",
        )
        self.app = create_app(_Adaptive(), _Fixed(), auth_service=self.auth, history_store={}, document_store=InMemoryDocumentStore())

    def login_cookie(self, email="student@example.com", password="change-me-student"):
        status, _, cookie = request(self.app, "/api/login", {"email": email, "password": password})
        self.assertEqual(status, "200 OK")
        return cookie

    def test_signup_student_teacher_office_and_argon2_hash(self):
        status, body, _ = request(
            self.app,
            "/api/signup",
            {
                "full_name": "New Student",
                "email": "New.Student@Example.com ",
                "password": "a@b&c!#%",
                "confirm_password": "a@b&c!#%",
                "role": "student",
            },
        )
        self.assertEqual(status, "201 Created")
        self.assertEqual(body["user"]["email"], "new.student@example.com")
        self.assertNotIn("password_hash", body["user"])
        self.assertTrue(self.auth.users["new.student@example.com"]["password_hash"].startswith("$argon2id$"))

        status, body, _ = request(
            self.app,
            "/api/signup",
            {
                "full_name": "New Teacher",
                "email": "new.teacher@example.com",
                "password": "secret",
                "confirm_password": "secret",
                "role": "teacher",
            },
        )
        self.assertEqual(status, "201 Created")
        self.assertEqual(body["user"]["role"], "teacher")

    def test_frontend_build_is_served(self):
        status, headers, body = raw_request(self.app, "/")
        self.assertEqual(status, "200 OK")
        self.assertIn(b'id="root"', body)
        self.assertTrue(any(key == "Content-Type" and "text/html" in value for key, value in headers))

        status, _, body = raw_request(self.app, "/workspace/history")
        self.assertEqual(status, "200 OK")
        self.assertIn(b'id="root"', body)

    def test_signup_validation(self):
        status, _, _ = request(
            self.app,
            "/api/signup",
            {"full_name": "Dup", "email": "student@example.com", "password": "x", "confirm_password": "x", "role": "student"},
        )
        self.assertEqual(status, "409 Conflict")
        status, _, _ = request(
            self.app,
            "/api/signup",
            {"full_name": "Bad", "email": "bad@example.com", "password": "x", "confirm_password": "x", "role": "admin"},
        )
        self.assertEqual(status, "403 Forbidden")
        status, _, _ = request(
            self.app,
            "/api/signup",
            {"full_name": "", "email": "empty@example.com", "password": "x", "confirm_password": "x", "role": "student"},
        )
        self.assertEqual(status, "400 Bad Request")
        status, _, _ = request(
            self.app,
            "/api/signup",
            {"full_name": "Mismatch", "email": "m@example.com", "password": "x", "confirm_password": "y", "role": "student"},
        )
        self.assertEqual(status, "400 Bad Request")

    def test_login_correct_incorrect_nonexistent_and_inactive(self):
        status, body, cookie = request(self.app, "/api/login", {"email": "student@example.com", "password": "change-me-student"})
        self.assertEqual(status, "200 OK")
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Lax", cookie)
        self.assertNotIn("password_hash", body["user"])

        status, _, _ = request(self.app, "/api/login", {"email": "student@example.com", "password": "wrong"})
        self.assertEqual(status, "401 Unauthorized")
        status, _, _ = request(self.app, "/api/login", {"email": "missing@example.com", "password": "wrong"})
        self.assertEqual(status, "401 Unauthorized")
        self.auth.users["student@example.com"]["is_active"] = False
        status, body, _ = request(self.app, "/api/login", {"email": "student@example.com", "password": "change-me-student"})
        self.assertEqual(status, "403 Forbidden")
        self.assertEqual(body["error"], "Account disabled.")

    def test_session_me_and_logout(self):
        status, _, _ = request(self.app, "/api/me", method="GET")
        self.assertEqual(status, "401 Unauthorized")
        cookie = self.login_cookie()
        status, body, _ = request(self.app, "/api/me", method="GET", cookie=cookie)
        self.assertEqual(status, "200 OK")
        self.assertEqual(body["user"]["role"], "student")
        status, _, expired = request(self.app, "/api/logout", {}, cookie=cookie)
        self.assertEqual(status, "200 OK")
        self.assertIn("Max-Age=0", expired)
        status, _, _ = request(self.app, "/api/me", method="GET", cookie=cookie)
        self.assertEqual(status, "401 Unauthorized")

    def test_authorization_and_query_modes(self):
        status, _, _ = request(self.app, "/api/query", {"query": "No session", "mode": "adaptive"})
        self.assertEqual(status, "403 Forbidden")
        cookie = self.login_cookie()
        status, body, _ = request(self.app, "/api/query", {"query": "What is the policy?", "mode": "adaptive"}, cookie=cookie)
        self.assertEqual(status, "200 OK")
        self.assertEqual(body["answer"], "adaptive: What is the policy?")
        status, body, _ = request(self.app, "/api/query", {"query": "What is the policy?", "mode": "fixed_3"}, cookie=cookie)
        self.assertEqual(status, "200 OK")
        self.assertEqual(body["telemetry"]["selected_k"], 3)
        status, _, _ = request(self.app, "/api/query", {"query": "What is the policy?", "mode": "fixed_5"}, cookie=cookie)
        self.assertEqual(status, "200 OK")
        status, _, _ = request(self.app, "/api/query", {"query": "What is the policy?", "mode": "fixed_10"}, cookie=cookie)
        self.assertEqual(status, "200 OK")
        status, _, _ = request(self.app, "/api/query", {"query": "Test", "mode": "fixed_4"}, cookie=cookie)
        self.assertEqual(status, "400 Bad Request")

    def test_empty_application_knowledge_base_does_not_use_research_corpus(self):
        app = create_app(
            _Adaptive(),
            _Fixed(),
            collection=_Collection(initial_count=0),
            auth_service=self.auth,
            history_store={},
            document_store=InMemoryDocumentStore(),
        )
        status, _, cookie = request(app, "/api/login", {"email": "student@example.com", "password": "change-me-student"})
        self.assertEqual(status, "200 OK")
        status, body, _ = request(app, "/api/query", {"query": "What is in the research corpus?", "mode": "adaptive"}, cookie=cookie)
        self.assertEqual(status, "200 OK")
        self.assertIn("No accessible documents", body["answer"])
        self.assertEqual(body["sources"], [])

    def test_history_requires_auth_and_returns_own_web_history(self):
        status, _, _ = request(self.app, "/api/history", method="GET")
        self.assertEqual(status, "403 Forbidden")
        cookie = self.login_cookie()
        request(self.app, "/api/query", {"query": "History question", "mode": "adaptive"}, cookie=cookie)
        status, body, _ = request(self.app, "/api/history", method="GET", cookie=cookie)
        self.assertEqual(status, "200 OK")
        self.assertEqual(len(body["history"]), 1)
        self.assertEqual(body["history"][0]["question"], "History question")

    def test_history_survives_new_session(self):
        import tempfile
        from pathlib import Path

        from adaptive.history_store import FileHistoryStore

        root = Path(tempfile.mkdtemp())
        store = FileHistoryStore(root)
        app = create_app(_Adaptive(), _Fixed(), auth_service=self.auth, history_store=store, document_store=InMemoryDocumentStore())
        status, _, cookie = request(app, "/api/login", {"email": "student@example.com", "password": "change-me-student"})
        self.assertEqual(status, "200 OK")
        request(app, "/api/query", {"query": "Persisted question", "mode": "adaptive"}, cookie=cookie)
        request(app, "/api/logout", {}, cookie=cookie)
        app2 = create_app(_Adaptive(), _Fixed(), auth_service=self.auth, history_store=FileHistoryStore(root), document_store=InMemoryDocumentStore())
        status, _, cookie2 = request(app2, "/api/login", {"email": "student@example.com", "password": "change-me-student"})
        status, body, _ = request(app2, "/api/history", method="GET", cookie=cookie2)
        self.assertEqual(status, "200 OK")
        self.assertEqual(len(body["history"]), 1)
        self.assertEqual(body["history"][0]["question"], "Persisted question")

    def test_hr_upload_authorization_and_txt_indexing(self):
        import tempfile
        from pathlib import Path

        collection = _Collection()
        upload_root = Path(tempfile.mkdtemp())
        app = create_app(
            _Adaptive(),
            _Fixed(),
            collection=collection,
            embedding_model=_EmbeddingModel(),
            auth_service=self.auth,
            upload_root=upload_root,
            document_store=InMemoryDocumentStore(),
        )
        status, _, employee_cookie = request(app, "/api/login", {"email": "student@example.com", "password": "change-me-student"})
        self.assertEqual(status, "200 OK")
        status, _, _ = request(app, "/api/admin/upload", {"filename": "guide.txt", "content_base64": "aGVsbG8=", "access_level": "student"}, cookie=employee_cookie)
        self.assertEqual(status, "403 Forbidden")
        status, _, _ = request(app, "/api/admin/documents", method="GET", cookie=employee_cookie)
        self.assertEqual(status, "403 Forbidden")

        status, _, hr_cookie = request(app, "/api/login", {"email": "admin@example.com", "password": "change-me-admin"})
        self.assertEqual(status, "200 OK")
        content = base64.b64encode(b"hello world").decode()
        status, body, _ = request(app, "/api/admin/upload", {"filename": "guide.txt", "content_base64": content, "access_level": "student"}, cookie=hr_cookie)
        self.assertEqual(status, "201 Created")
        self.assertEqual(body["chunk_count"], 1)
        self.assertEqual(collection.calls[0]["metadatas"][0]["document"], "guide.txt")
        self.assertEqual(collection.calls[0]["metadatas"][0]["scope"], "application")
        self.assertIn("document_id", collection.calls[0]["metadatas"][0])
        self.assertTrue(list((upload_root / "documents").glob("*")))
        self.assertTrue(list((upload_root / "chunks").glob("*.json")))
        status, listed, _ = request(app, "/api/admin/documents", method="GET", cookie=hr_cookie)
        self.assertEqual(status, "200 OK")
        self.assertEqual(listed["documents"][0]["original_filename"], "guide.txt")

    def test_hr_pdf_and_docx_uploads_are_indexed(self):
        collection = _Collection()
        app = create_app(
            _Adaptive(),
            _Fixed(),
            collection=collection,
            embedding_model=_EmbeddingModel(),
            auth_service=self.auth,
            history_store={},
            upload_root=Path(tempfile.mkdtemp()),
            document_store=InMemoryDocumentStore(),
        )
        status, _, hr_cookie = request(app, "/api/login", {"email": "admin@example.com", "password": "change-me-admin"})
        self.assertEqual(status, "200 OK")

        with patch("api.app.extract_document", return_value=([], [])):
            status, body, _ = request(
                app,
                "/api/admin/upload",
                {"filename": "policy.pdf", "content_base64": base64.b64encode(b"%PDF-1.7").decode(), "access_level": "student"},
                cookie=hr_cookie,
            )
            self.assertEqual(status, "400 Bad Request")

        with patch("api.app.extract_document") as mocked_extract:
            from adaptive.app_ingestion import ExtractedPage

            mocked_extract.return_value = ([ExtractedPage(1, "Policy text")], [])
            status, body, _ = request(
                app,
                "/api/admin/upload",
                {"filename": "policy.docx", "content_base64": base64.b64encode(b"docx bytes").decode(), "access_level": "teacher"},
                cookie=hr_cookie,
            )
            self.assertEqual(status, "201 Created")
            self.assertEqual(body["chunk_count"], 1)

    def test_upload_invalid_extension_and_size(self):
        status, _, hr_cookie = request(self.app, "/api/login", {"email": "admin@example.com", "password": "change-me-admin"})
        self.assertEqual(status, "200 OK")
        status, _, _ = request(self.app, "/api/admin/upload", {"filename": "bad.exe", "content_base64": "aGVsbG8=", "access_level": "student"}, cookie=hr_cookie)
        self.assertEqual(status, "400 Bad Request")
        large = base64.b64encode(b"x" * (20 * 1024 * 1024 + 1)).decode()
        status, _, _ = request(self.app, "/api/admin/upload", {"filename": "large.txt", "content_base64": large, "access_level": "student"}, cookie=hr_cookie)
        self.assertEqual(status, "400 Bad Request")

    def test_long_and_short_documents_generate_expected_counts(self):
        collection = _Collection(initial_count=0)
        documents = InMemoryDocumentStore()
        app = create_app(
            _Adaptive(),
            _Fixed(),
            collection=collection,
            embedding_model=_EmbeddingModel(),
            auth_service=self.auth,
            history_store={},
            upload_root=Path(tempfile.mkdtemp()),
            document_store=documents,
        )
        status, _, hr_cookie = request(app, "/api/login", {"email": "admin@example.com", "password": "change-me-admin"})
        self.assertEqual(status, "200 OK")

        long_text = " ".join(f"word{i}" for i in range(2000))
        status, body, _ = request(
            app,
            "/api/admin/upload",
            {"filename": "long.txt", "content_base64": base64.b64encode(long_text.encode()).decode(), "access_level": "student"},
            cookie=hr_cookie,
        )
        self.assertEqual(status, "201 Created")
        self.assertEqual(body["chunk_count"], 5)
        self.assertEqual(body["chunks_stored"], 5)
        self.assertEqual(body["embedding_count"], 5)
        self.assertEqual(body["chroma_record_count"], 5)
        self.assertEqual(body["chroma_verified_count"], 5)

        short_text = "short policy"
        status, body, _ = request(
            app,
            "/api/admin/upload",
            {"filename": "short.txt", "content_base64": base64.b64encode(short_text.encode()).decode(), "access_level": "student"},
            cookie=hr_cookie,
        )
        self.assertEqual(status, "201 Created")
        self.assertEqual(body["chunk_count"], 1)
        self.assertEqual(body["embedding_count"], 1)
        self.assertEqual(len(documents.list_documents()), 2)

    def test_duplicate_upload_is_rejected_by_content_hash(self):
        app = create_app(
            _Adaptive(),
            _Fixed(),
            collection=_Collection(initial_count=0),
            embedding_model=_EmbeddingModel(),
            auth_service=self.auth,
            history_store={},
            upload_root=Path(tempfile.mkdtemp()),
            document_store=InMemoryDocumentStore(),
        )
        status, _, hr_cookie = request(app, "/api/login", {"email": "admin@example.com", "password": "change-me-admin"})
        self.assertEqual(status, "200 OK")
        payload = {"filename": "policy.txt", "content_base64": base64.b64encode(b"same content").decode()}
        status, _, _ = request(app, "/api/admin/upload", payload, cookie=hr_cookie)
        self.assertEqual(status, "201 Created")
        status, body, _ = request(app, "/api/admin/upload", {**payload, "filename": "copy.txt"}, cookie=hr_cookie)
        self.assertEqual(status, "409 Conflict")
        self.assertIn("already been uploaded", body["error"])


if __name__ == "__main__":
    unittest.main()
