from pathlib import Path

import chromadb

APPLICATION_CHROMA_PATH = "data/application_chroma_db"
APPLICATION_COLLECTION_NAME = "application_uploaded_documents"

TEST_DOCUMENT = "office_test.txt"
EXPECTED_ACCESS_LEVEL = "office"


def main():
    print("=" * 70)
    print("NEW APPLICATION DOCUMENT CHROMA TEST")
    print("=" * 70)

    client = chromadb.PersistentClient(path=APPLICATION_CHROMA_PATH)

    collection = client.get_collection(
        APPLICATION_COLLECTION_NAME
    )

    print(f"\nTotal Chroma records: {collection.count()}")

    result = collection.get(
        where={
            "$and": [
                {"scope": "application"},
                {"document": TEST_DOCUMENT},
            ]
        },
        include=["documents", "metadatas"],
    )

    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    print(f"Records for {TEST_DOCUMENT}: {len(metadatas)}")

    if not metadatas:
        print("[FAIL] No Chroma records found for test document")
        return

    print("\nMetadata:")
    for i, metadata in enumerate(metadatas, start=1):
        print(f"\nChunk {i}")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)

    required_keys = {
        "scope",
        "document",
        "document_id",
        "page",
        "chunk_id",
        "access_level",
    }

    missing = required_keys - set(metadatas[0].keys())

    if missing:
        print("[FAIL] Missing metadata:", sorted(missing))
    else:
        print("[PASS] Required metadata exists")

    access_levels = {
        metadata.get("access_level")
        for metadata in metadatas
    }

    print("Access levels found:", access_levels)

    if access_levels == {EXPECTED_ACCESS_LEVEL}:
        print(
            f"[PASS] Document is indexed with "
            f"{EXPECTED_ACCESS_LEVEL.upper()} access"
        )
    else:
        print(
            "[FAIL] Unexpected access levels. "
            f"Expected: {EXPECTED_ACCESS_LEVEL}, "
            f"Found: {access_levels}"
        )


if __name__ == "__main__":
    main()
