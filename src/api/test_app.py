import base64
import io
import json
import unittest

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError
from argon2.low_level import Type

from adaptive.auth import AuthError, DuplicateEmailError, SafeUser, normalize_email, validate_role
from api.app import create_app


class _Adaptive:
    def run(self, query):
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
    def run(self, query, k):
        return {"answer": f"fixed {k}: {query}", "telemetry": {"initial_k": k, "selected_k": k}}


class _Collection:
    def __init__(self):
        self.calls = []

    def upsert(self, **kwargs):
        self.calls.append(kwargs)


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


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.auth = _MemoryAuth()
        self.auth.create_user(
            full_name="Employee User",
            email="employee@example.com",
            password="change-me-employee",
            role="employee",
        )
        self.auth.create_user(
            full_name="HR User",
            email="hr@example.com",
            password="change-me-hr",
            role="hr",
        )
        self.app = create_app(_Adaptive(), _Fixed(), auth_service=self.auth)

    def login_cookie(self, email="employee@example.com", password="change-me-employee"):
        status, _, cookie = request(self.app, "/api/login", {"email": email, "password": password})
        self.assertEqual(status, "200 OK")
        return cookie

    def test_signup_employee_and_hr_and_argon2_hash(self):
        status, body, _ = request(
            self.app,
            "/api/signup",
            {
                "full_name": "New Employee",
                "email": "New.Employee@Example.com ",
                "password": "a@b&c!#%",
                "confirm_password": "a@b&c!#%",
                "role": "employee",
            },
        )
        self.assertEqual(status, "201 Created")
        self.assertEqual(body["user"]["email"], "new.employee@example.com")
        self.assertNotIn("password_hash", body["user"])
        self.assertTrue(self.auth.users["new.employee@example.com"]["password_hash"].startswith("$argon2id$"))

        status, body, _ = request(
            self.app,
            "/api/signup",
            {
                "full_name": "New HR",
                "email": "new.hr@example.com",
                "password": "secret",
                "confirm_password": "secret",
                "role": "hr",
            },
        )
        self.assertEqual(status, "201 Created")
        self.assertEqual(body["user"]["role"], "hr")

    def test_signup_validation(self):
        status, _, _ = request(
            self.app,
            "/api/signup",
            {"full_name": "Dup", "email": "employee@example.com", "password": "x", "confirm_password": "x", "role": "employee"},
        )
        self.assertEqual(status, "409 Conflict")
        status, _, _ = request(
            self.app,
            "/api/signup",
            {"full_name": "Bad", "email": "bad@example.com", "password": "x", "confirm_password": "x", "role": "admin"},
        )
        self.assertEqual(status, "400 Bad Request")
        status, _, _ = request(
            self.app,
            "/api/signup",
            {"full_name": "", "email": "empty@example.com", "password": "x", "confirm_password": "x", "role": "employee"},
        )
        self.assertEqual(status, "400 Bad Request")
        status, _, _ = request(
            self.app,
            "/api/signup",
            {"full_name": "Mismatch", "email": "m@example.com", "password": "x", "confirm_password": "y", "role": "employee"},
        )
        self.assertEqual(status, "400 Bad Request")

    def test_login_correct_incorrect_nonexistent_and_inactive(self):
        status, body, cookie = request(self.app, "/api/login", {"email": "employee@example.com", "password": "change-me-employee"})
        self.assertEqual(status, "200 OK")
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Lax", cookie)
        self.assertNotIn("password_hash", body["user"])

        status, _, _ = request(self.app, "/api/login", {"email": "employee@example.com", "password": "wrong"})
        self.assertEqual(status, "401 Unauthorized")
        status, _, _ = request(self.app, "/api/login", {"email": "missing@example.com", "password": "wrong"})
        self.assertEqual(status, "401 Unauthorized")
        self.auth.users["employee@example.com"]["is_active"] = False
        status, body, _ = request(self.app, "/api/login", {"email": "employee@example.com", "password": "change-me-employee"})
        self.assertEqual(status, "403 Forbidden")
        self.assertEqual(body["error"], "Account disabled.")

    def test_session_me_and_logout(self):
        status, _, _ = request(self.app, "/api/me", method="GET")
        self.assertEqual(status, "401 Unauthorized")
        cookie = self.login_cookie()
        status, body, _ = request(self.app, "/api/me", method="GET", cookie=cookie)
        self.assertEqual(status, "200 OK")
        self.assertEqual(body["user"]["role"], "employee")
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

    def test_history_requires_auth_and_returns_own_web_history(self):
        status, _, _ = request(self.app, "/api/history", method="GET")
        self.assertEqual(status, "403 Forbidden")
        cookie = self.login_cookie()
        request(self.app, "/api/query", {"query": "History question", "mode": "adaptive"}, cookie=cookie)
        status, body, _ = request(self.app, "/api/history", method="GET", cookie=cookie)
        self.assertEqual(status, "200 OK")
        self.assertEqual(len(body["history"]), 1)
        self.assertIn("rag_runs has no user_id column", body["schema_note"])

    def test_hr_upload_authorization_and_txt_indexing(self):
        collection = _Collection()
        app = create_app(_Adaptive(), _Fixed(), collection=collection, embedding_model=_EmbeddingModel(), auth_service=self.auth)
        status, _, employee_cookie = request(app, "/api/login", {"email": "employee@example.com", "password": "change-me-employee"})
        self.assertEqual(status, "200 OK")
        status, _, _ = request(app, "/api/hr/upload", {"filename": "guide.txt", "content_base64": "aGVsbG8="}, cookie=employee_cookie)
        self.assertEqual(status, "403 Forbidden")

        status, _, hr_cookie = request(app, "/api/login", {"email": "hr@example.com", "password": "change-me-hr"})
        self.assertEqual(status, "200 OK")
        content = base64.b64encode(b"hello world").decode()
        status, body, _ = request(app, "/api/hr/upload", {"filename": "guide.txt", "content_base64": content}, cookie=hr_cookie)
        self.assertEqual(status, "201 Created")
        self.assertEqual(body["chunk_count"], 1)
        self.assertEqual(collection.calls[0]["metadatas"][0]["document"], "guide.txt")

    def test_hr_pdf_and_docx_uploads_are_indexed(self):
        collection = _Collection()
        app = create_app(_Adaptive(), _Fixed(), collection=collection, embedding_model=_EmbeddingModel(), auth_service=self.auth)
        status, _, hr_cookie = request(app, "/api/login", {"email": "hr@example.com", "password": "change-me-hr"})
        self.assertEqual(status, "200 OK")

        try:
            import pymupdf
        except ImportError:
            import fitz as pymupdf

        pdf = pymupdf.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "PDF policy text")
        pdf_bytes = pdf.tobytes()
        pdf.close()
        status, body, _ = request(
            app,
            "/api/hr/upload",
            {"filename": "policy.pdf", "content_base64": base64.b64encode(pdf_bytes).decode()},
            cookie=hr_cookie,
        )
        self.assertEqual(status, "201 Created")
        self.assertEqual(body["chunk_count"], 1)

        from docx import Document

        doc = Document()
        doc.add_paragraph("DOCX policy text")
        doc_buffer = io.BytesIO()
        doc.save(doc_buffer)
        status, body, _ = request(
            app,
            "/api/hr/upload",
            {"filename": "policy.docx", "content_base64": base64.b64encode(doc_buffer.getvalue()).decode()},
            cookie=hr_cookie,
        )
        self.assertEqual(status, "201 Created")
        self.assertEqual(body["chunk_count"], 1)

    def test_upload_invalid_extension_and_size(self):
        status, _, hr_cookie = request(self.app, "/api/login", {"email": "hr@example.com", "password": "change-me-hr"})
        self.assertEqual(status, "200 OK")
        status, _, _ = request(self.app, "/api/hr/upload", {"filename": "bad.exe", "content_base64": "aGVsbG8="}, cookie=hr_cookie)
        self.assertEqual(status, "400 Bad Request")
        large = base64.b64encode(b"x" * (20 * 1024 * 1024 + 1)).decode()
        status, _, _ = request(self.app, "/api/hr/upload", {"filename": "large.txt", "content_base64": large}, cookie=hr_cookie)
        self.assertEqual(status, "400 Bad Request")


if __name__ == "__main__":
    unittest.main()
