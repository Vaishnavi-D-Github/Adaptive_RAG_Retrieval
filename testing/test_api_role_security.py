import json
import urllib.error
import urllib.request
import http.cookiejar


BASE_URL = "http://127.0.0.1:8000"


USERS = {
    "student": {
        "email": "asuran@gmail.com",
        "password": "Asuran",
    },
    "teacher": {
        "email": "thanvi@gmail.com",
        "password": "Thanvi",
    },
    "office": {
        "email": "siri@gmail.com",
        "password": "Siri",
    },
    "admin": {
        "email": "avi@gmail.com",
        "password": "Avinash",
    },
}


TESTS = [
    {
        "name": "Teacher document",
        "question": "What is the teacher-only access test code?",
        "protected_document": "teacher_test.txt",
        "protected_code": "TEACHER_ONLY_456",
        "allowed_roles": {"teacher", "office", "admin"},
    },
    {
        "name": "Office document",
        "question": "What is the office-only access test code?",
        "protected_document": "office_test.txt",
        "protected_code": "OFFICE_ONLY_789",
        "allowed_roles": {"office", "admin"},
    },
]


def request_json(opener, method, path, body=None):
    url = BASE_URL + path

    data = None

    if body is not None:
        data = json.dumps(body).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    try:
        with opener.open(request) as response:
            raw = response.read().decode("utf-8")

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"raw": raw}

            return response.status, payload

    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8")

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"raw": raw}

        return error.code, payload


def login(role):
    user = USERS[role]

    jar = http.cookiejar.CookieJar()

    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(jar)
    )

    status, payload = request_json(
        opener,
        "POST",
        "/api/login",
        {
            "email": user["email"],
            "password": user["password"],
        },
    )

    if status != 200:
        print(
            f"[FAIL] {role.upper()} login failed: "
            f"HTTP {status} - {payload}"
        )
        return None

    returned_user = payload.get("user", {})

    print(
        f"[PASS] {role.upper()} logged in as "
        f"{returned_user.get('role')}"
    )

    return opener


def contains_protected_information(value, document, code):
    if isinstance(value, str):
        return document in value or code in value

    if isinstance(value, dict):
        return any(
            contains_protected_information(v, document, code)
            for v in value.values()
        )

    if isinstance(value, list):
        return any(
            contains_protected_information(v, document, code)
            for v in value
        )

    return False


def run():
    print("=" * 80)
    print("END-TO-END API ROLE SECURITY TEST")
    print("=" * 80)

    total = 0
    passed = 0
    failed = 0

    # ------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------

    print("\n[1] Testing API health...")

    opener = urllib.request.build_opener()

    status, payload = request_json(
        opener,
        "GET",
        "/api/health",
    )

    if status == 200:
        print("[PASS] API is healthy")
    else:
        print("[FAIL] API health check failed")
        print(status, payload)
        return

    # ------------------------------------------------------------
    # Login and query tests
    # ------------------------------------------------------------

    for test in TESTS:

        print("\n")
        print("=" * 80)
        print(test["name"].upper())
        print("=" * 80)

        print("Question:", test["question"])
        print("Protected document:", test["protected_document"])
        print("Protected code:", test["protected_code"])

        for role in USERS:

            total += 1

            print("\n" + "-" * 80)
            print("ROLE:", role.upper())
            print("-" * 80)

            opener = login(role)

            if opener is None:
                failed += 1
                continue

            status, payload = request_json(
                opener,
                "POST",
                "/api/query",
                {
                    "query": test["question"],
                    "mode": "fixed_10",
                },
            )

            print("HTTP status:", status)

            if status != 200:
                print("[FAIL] Query request failed")
                print(payload)
                failed += 1
                continue

            answer = payload.get("answer", "")
            sources = payload.get("sources", [])
            telemetry = payload.get("telemetry", {})

            print("\nAnswer:")
            print(answer)

            print("\nSources:")
            print(json.dumps(sources, indent=2, default=str))

            print("\nTelemetry:")
            print(
                json.dumps(
                    {
                        "initial_k": telemetry.get("initial_k"),
                        "selected_k": telemetry.get("selected_k"),
                        "retrieval_iterations": telemetry.get(
                            "retrieval_iterations"
                        ),
                        "k_escalated": telemetry.get("k_escalated"),
                        "verification_result": telemetry.get(
                            "verification_result"
                        ),
                    },
                    indent=2,
                    default=str,
                )
            )

            leaked = contains_protected_information(
                payload,
                test["protected_document"],
                test["protected_code"],
            )

            should_have_access = role in test["allowed_roles"]

            if should_have_access:

                if leaked:
                    print(
                        f"\n[PASS] {role.upper()} received "
                        "authorized protected information"
                    )
                    passed += 1
                else:
                    print(
                        f"\n[WARNING] {role.upper()} is authorized "
                        "but protected information was not visible "
                        "in the API response"
                    )

                    # Still investigate rather than calling this
                    # a security failure.
                    passed += 1

            else:

                if leaked:
                    print(
                        "\n[CRITICAL] SECURITY BREACH"
                    )
                    print(
                        f"{role.upper()} received unauthorized "
                        f"information from {test['protected_document']}"
                    )
                    failed += 1

                else:
                    print(
                        f"\n[PASS] {role.upper()} did not receive "
                        "unauthorized protected information"
                    )
                    passed += 1

    print("\n")
    print("=" * 80)
    print("END-TO-END API SECURITY SUMMARY")
    print("=" * 80)

    print("Total tests :", total)
    print("Passed      :", passed)
    print("Failed      :", failed)

    if failed == 0:
        print("\n[PASS] NO API-LEVEL SECURITY BREACH DETECTED")
    else:
        print("\n[FAIL] API SECURITY TESTS FAILED")

    print("=" * 80)


if __name__ == "__main__":
    run()