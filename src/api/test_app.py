import io
import json
import unittest

from api.app import create_app


class _Adaptive:
    def run(self, query):
        return {"answer": f"adaptive: {query}", "telemetry": {"selected_k": 3}}


class _Fixed:
    def run(self, query, k):
        return {"answer": f"fixed {k}: {query}", "telemetry": {"selected_k": k}}


class _Collection:
    def __init__(self): self.calls = []
    def upsert(self, **kwargs): self.calls.append(kwargs)


class _EmbeddingModel:
    def encode(self, values): return [[0.1, 0.2] for _ in values]


def request(app, route, data):
    body = json.dumps(data).encode()
    status, response_headers = [], []
    def start(value, headers): status.append(value); response_headers.extend(headers)
    result = app({"PATH_INFO": route, "REQUEST_METHOD": "POST", "CONTENT_LENGTH": str(len(body)), "wsgi.input": io.BytesIO(body)}, start)
    return status[0], json.loads(b"".join(result))


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(_Adaptive(), _Fixed())
        self.employee = {"email": "employee@example.com", "password": "change-me-employee"}

    def test_valid_login_and_employee_query(self):
        status, login = request(self.app, "/api/login", self.employee)
        self.assertEqual(status, "200 OK")
        self.assertEqual(login["role"], "Employee")
        status, response = request(self.app, "/api/query", {**self.employee, "role": "Employee", "query": "What is the policy?", "mode": "adaptive"})
        self.assertEqual(status, "200 OK")
        self.assertTrue(response["success"])

    def test_role_isolation_and_bad_mode(self):
        status, _ = request(self.app, "/api/query", {"email": "hr@example.com", "password": "change-me-hr", "role": "Employee", "query": "No"})
        self.assertEqual(status, "403 Forbidden")
        status, _ = request(self.app, "/api/query", {**self.employee, "role": "Employee", "query": "Test", "mode": "fixed_4"})
        self.assertEqual(status, "400 Bad Request")

    def test_hr_upload_requires_runtime(self):
        status, response = request(self.app, "/api/hr/upload", {"email": "hr@example.com", "password": "change-me-hr", "role": "HR", "filename": "guide.txt", "content_base64": "aGVsbG8="})
        self.assertEqual(status, "400 Bad Request")
        self.assertFalse(response["success"])

    def test_hr_txt_upload_is_indexed(self):
        collection = _Collection()
        app = create_app(_Adaptive(), _Fixed(), collection=collection, embedding_model=_EmbeddingModel())
        status, response = request(app, "/api/hr/upload", {"email": "hr@example.com", "password": "change-me-hr", "role": "HR", "filename": "guide.txt", "content_base64": "aGVsbG8gd29ybGQ="})
        self.assertEqual(status, "201 Created")
        self.assertEqual(response["chunk_count"], 1)
        self.assertEqual(collection.calls[0]["metadatas"][0]["document"], "guide.txt")


if __name__ == "__main__":
    unittest.main()
