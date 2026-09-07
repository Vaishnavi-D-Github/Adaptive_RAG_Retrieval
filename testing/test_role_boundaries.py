import chromadb
from sentence_transformers import SentenceTransformer

from retrieval.chroma_retriever import ChromaRetriever


APPLICATION_CHROMA_PATH = "data/application_chroma_db"
APPLICATION_COLLECTION_NAME = "application_uploaded_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

QUESTION = "What is the teacher-only access test code?"

EXPECTED_DOCUMENT = "teacher_test.txt"
EXPECTED_CODE = "TEACHER_ONLY_456"


def main():
    print("=" * 70)
    print("ROLE ACCESS BOUNDARY TEST")
    print("=" * 70)

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

    print("\n" + "=" * 70)
    print("SECURITY TEST QUESTION")
    print("=" * 70)
    print(QUESTION)

    roles = ["student", "teacher", "office", "admin"]

    expected_access = {
        "student": False,
        "teacher": True,
        "office": True,
        "admin": True,
    }

    passed = 0
    failed = 0

    for role in roles:

        print("\n" + "-" * 70)
        print(f"TESTING ROLE: {role.upper()}")
        print("-" * 70)

        try:
            results = retriever.retrieve(
                QUESTION,
                5,
                user_role=role,
            )

            print("Number of retrieved results:", len(results))

            teacher_document_found = False
            teacher_code_found = False

            for i, result in enumerate(results, start=1):

                metadata = result.get("metadata", {})
                text = result.get("text", "")

                document = metadata.get("document")

                print(f"\nResult {i}")
                print("Document:", document)
                print("Access level:", metadata.get("access_level"))
                print("Page:", metadata.get("page"))
                print("Chunk:", metadata.get("chunk_id"))

                if document == EXPECTED_DOCUMENT:
                    teacher_document_found = True

                    if EXPECTED_CODE in text:
                        teacher_code_found = True

            should_have_access = expected_access[role]

            if should_have_access:

                if teacher_document_found:
                    print(
                        f"\n[PASS] {role.upper()} can retrieve "
                        "the Teacher-level document"
                    )

                    if teacher_code_found:
                        print(
                            f"[PASS] {role.upper()} retrieved the "
                            f"expected code: {EXPECTED_CODE}"
                        )
                    else:
                        print(
                            f"[WARNING] {role.upper()} retrieved the "
                            "document but the expected code was not "
                            "found in the returned text"
                        )

                    passed += 1

                else:
                    print(
                        f"[FAIL] {role.upper()} should have access "
                        "but teacher_test.txt was not retrieved"
                    )
                    failed += 1

            else:

                if teacher_document_found:
                    print(
                        f"[FAIL] SECURITY BREACH: {role.upper()} "
                        "retrieved a Teacher-level document"
                    )

                    if teacher_code_found:
                        print(
                            f"[CRITICAL] {role.upper()} received "
                            f"protected code: {EXPECTED_CODE}"
                        )

                    failed += 1

                else:
                    print(
                        f"[PASS] {role.upper()} was correctly "
                        "denied the Teacher-level document"
                    )
                    passed += 1

        except Exception as exc:

            print(
                f"[FAIL] Retrieval failed for {role}: "
                f"{type(exc).__name__}: {exc}"
            )

            failed += 1

    print("\n" + "=" * 70)
    print("ROLE ACCESS BOUNDARY TEST SUMMARY")
    print("=" * 70)

    print("Passed:", passed)
    print("Failed:", failed)

    if failed == 0:
        print("\n[PASS] ALL ROLE ACCESS BOUNDARY TESTS PASSED")
    else:
        print("\n[FAIL] ONE OR MORE ROLE ACCESS TESTS FAILED")

    print("=" * 70)


if __name__ == "__main__":
    main()