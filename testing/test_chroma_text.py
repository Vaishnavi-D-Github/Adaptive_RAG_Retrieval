import chromadb


APPLICATION_CHROMA_PATH = "data/application_chroma_db"
APPLICATION_COLLECTION_NAME = "application_uploaded_documents"

TEST_DOCUMENT = "role_test.txt"


def main():
    print("=" * 70)
    print("CHROMA STORED TEXT TEST")
    print("=" * 70)

    client = chromadb.PersistentClient(
        path=APPLICATION_CHROMA_PATH
    )

    collection = client.get_collection(
        APPLICATION_COLLECTION_NAME
    )

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

    print(f"\nFound records: {len(documents)}")

    if not documents:
        print("[FAIL] No stored document text found")
        return

    for i, (text, metadata) in enumerate(
        zip(documents, metadatas),
        start=1,
    ):
        print("\n" + "-" * 70)
        print(f"RECORD {i}")
        print("-" * 70)

        print("Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

        print("\nStored text:")
        print(repr(text))

        print("\nReadable text:")
        print(text)

        if "STUDENT_TEST_123" in text:
            print(
                "\n[PASS] STUDENT_TEST_123 exists "
                "inside the stored Chroma text"
            )
        else:
            print(
                "\n[FAIL] STUDENT_TEST_123 was NOT found "
                "inside the stored Chroma text"
            )

    print("\n" + "=" * 70)
    print("CHROMA STORED TEXT TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()