import chromadb
from sentence_transformers import SentenceTransformer

from retrieval.chroma_retriever import ChromaRetriever


APPLICATION_CHROMA_PATH = "data/application_chroma_db"
APPLICATION_COLLECTION_NAME = "application_uploaded_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


TESTS = [
    {
        "name": "Student document",
        "question": "What is the student access test code?",
        "document": "role_test.txt",
        "code": "STUDENT_TEST_123",
        "allowed_roles": {"student", "teacher", "office", "admin"},
    },
    {
        "name": "Teacher document",
        "question": "What is the teacher-only access test code?",
        "document": "teacher_test.txt",
        "code": "TEACHER_ONLY_456",
        "allowed_roles": {"teacher", "office", "admin"},
    },
    {
        "name": "Office document",
        "question": "What is the office-only access test code?",
        "document": "office_test.txt",
        "code": "OFFICE_ONLY_789",
        "allowed_roles": {"office", "admin"},
    },
]


ROLES = ["student", "teacher", "office", "admin"]


def main():
    print("=" * 80)
    print("COMPLETE ROLE-BASED DOCUMENT ACCESS MATRIX")
    print("=" * 80)

    print("\n[1] Connecting to Application ChromaDB...")

    client = chromadb.PersistentClient(
        path=APPLICATION_CHROMA_PATH
    )

    collection = client.get_collection(
        APPLICATION_COLLECTION_NAME
    )

    print("[PASS] ChromaDB connected")
    print("Collection:", APPLICATION_COLLECTION_NAME)
    print("Records:", collection.count())

    print("\n[2] Loading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print("[PASS] Embedding model loaded")

    print("\n[3] Creating ChromaRetriever...")

    retriever = ChromaRetriever(
        collection=collection,
        embedding_model=embedding_model,
    )

    print("[PASS] ChromaRetriever created")

    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    for test in TESTS:

        print("\n")
        print("=" * 80)
        print(test["name"].upper())
        print("=" * 80)

        print("Question:", test["question"])
        print("Expected document:", test["document"])
        print("Expected code:", test["code"])
        print("Allowed roles:", ", ".join(sorted(test["allowed_roles"])))

        for role in ROLES:

            total_tests += 1

            print("\n" + "-" * 80)
            print("Role:", role.upper())
            print("-" * 80)

            try:
                results = retriever.retrieve(
                    test["question"],
                    5,
                    user_role=role,
                )

                found_document = False
                found_code = False

                for result in results:

                    metadata = result.get("metadata", {})
                    document = metadata.get("document")

                    text = result.get("document", "")

                    if document == test["document"]:

                        found_document = True

                        if test["code"] in text:
                            found_code = True

                should_have_access = (
                    role in test["allowed_roles"]
                )

                if should_have_access:

                    if found_document and found_code:

                        print(
                            "[PASS] Access allowed and "
                            "correct document/code retrieved"
                        )

                        passed_tests += 1

                    elif found_document:

                        print(
                            "[FAIL] Document retrieved but "
                            "expected code missing"
                        )

                        failed_tests += 1

                    else:

                        print(
                            "[FAIL] Authorized role could not "
                            "retrieve expected document"
                        )

                        failed_tests += 1

                else:

                    if found_document:

                        print(
                            "[FAIL] SECURITY BREACH: unauthorized "
                            "role retrieved protected document"
                        )

                        if found_code:
                            print(
                                "[CRITICAL] Protected code exposed:",
                                test["code"],
                            )

                        failed_tests += 1

                    else:

                        print(
                            "[PASS] Unauthorized role correctly "
                            "denied protected document"
                        )

                        passed_tests += 1

            except Exception as exc:

                print(
                    "[FAIL] Test execution error:",
                    type(exc).__name__,
                    str(exc),
                )

                failed_tests += 1

    print("\n")
    print("=" * 80)
    print("FINAL ROLE ACCESS MATRIX")
    print("=" * 80)

    print(
        "\n"
        "                         Student   Teacher   Office   Admin"
    )
    print(
        "Student document           ✓         ✓         ✓        ✓"
    )
    print(
        "Teacher document           ✗         ✓         ✓        ✓"
    )
    print(
        "Office document            ✗         ✗         ✓        ✓"
    )

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    print("Total tests :", total_tests)
    print("Passed      :", passed_tests)
    print("Failed      :", failed_tests)

    if failed_tests == 0:
        print("\n[PASS] COMPLETE ROLE ACCESS MATRIX PASSED")
        print("[PASS] NO UNAUTHORIZED DOCUMENT ACCESS DETECTED")
    else:
        print("\n[FAIL] ROLE ACCESS MATRIX HAS FAILURES")

    print("=" * 80)


if __name__ == "__main__":
    main()