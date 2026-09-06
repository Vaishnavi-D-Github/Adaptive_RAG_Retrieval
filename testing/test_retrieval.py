import chromadb
from sentence_transformers import SentenceTransformer

from retrieval.chroma_retriever import ChromaRetriever


APPLICATION_CHROMA_PATH = "data/application_chroma_db"
APPLICATION_COLLECTION_NAME = "application_uploaded_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


QUESTION = "What is the student access test code?"


def main():
    print("=" * 70)
    print("ROLE-BASED RETRIEVAL SECURITY TEST")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Connect to application ChromaDB
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # 2. Load embedding model
    # ------------------------------------------------------------
    print("\n[2] Loading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print("[PASS] Embedding model loaded")

    # ------------------------------------------------------------
    # 3. Create retriever
    # ------------------------------------------------------------
    print("\n[3] Creating ChromaRetriever...")

    retriever = ChromaRetriever(
        collection=collection,
        embedding_model=embedding_model,
    )

    print("[PASS] ChromaRetriever created")

    # ------------------------------------------------------------
    # 4. Test every role
    # ------------------------------------------------------------
    roles = [
        "student",
        "teacher",
        "office",
        "admin",
    ]

    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(QUESTION)

    for role in roles:

        print("\n" + "-" * 70)
        print(f"ROLE: {role.upper()}")
        print("-" * 70)

        try:
            results = retriever.retrieve(
                QUESTION,
                5,
                user_role=role,
            )

            print("Retrieval executed")
            print("Number of results:", len(results))

            if not results:
                print("[FAIL] No results returned")
                continue

            found_test_document = False

            for i, result in enumerate(results, start=1):

                metadata = result.get("metadata", {})
                text = result.get("text", "")

                print(f"\nResult {i}")
                print("Document:", metadata.get("document"))
                print("Document ID:", metadata.get("document_id"))
                print("Access level:", metadata.get("access_level"))
                print("Page:", metadata.get("page"))
                print("Chunk:", metadata.get("chunk_id"))
                print("Text:", text[:300])

                if metadata.get("document") == "role_test.txt":
                    found_test_document = True

                    if "STUDENT_TEST_123" in text:
                        print(
                            "[PASS] Correct test document and "
                            "test code retrieved"
                        )

            if found_test_document:
                print(
                    f"[PASS] {role.upper()} can access "
                    "the STUDENT-level document"
                )
            else:
                print(
                    f"[FAIL] {role.upper()} did not retrieve "
                    "role_test.txt"
                )

        except Exception as exc:
            print(
                f"[FAIL] Retrieval failed for {role}: "
                f"{type(exc).__name__}: {exc}"
            )

    print("\n" + "=" * 70)
    print("ROLE RETRIEVAL TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()