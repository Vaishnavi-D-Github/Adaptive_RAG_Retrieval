import sys

sys.path.insert(0, "src")

import chromadb

from adaptive.config import (
    APPLICATION_CHROMA_PATH,
    APPLICATION_COLLECTION_NAME,
)


print("=" * 70)
print("APPLICATION CHROMADB TEST")
print("=" * 70)

print("Chroma path:", APPLICATION_CHROMA_PATH)
print("Collection:", APPLICATION_COLLECTION_NAME)

try:
    client = chromadb.PersistentClient(
        path=APPLICATION_CHROMA_PATH
    )

    print("\n[PASS] ChromaDB opened successfully")

    collection = client.get_collection(
        APPLICATION_COLLECTION_NAME
    )

    print("[PASS] Application collection found")
    print("Collection name:", collection.name)

    total = collection.count()

    print("\nTotal Chroma records:", total)

    if total > 0:
        print("[PASS] Chroma contains records")
    else:
        print("[FAIL] Chroma collection is empty")

    print("\nRetrieving sample records...")

    data = collection.get(
        limit=10,
        include=[
            "documents",
            "metadatas",
        ],
    )

    print("\nSample records:")

    for i, record_id in enumerate(data["ids"]):

        print("\n" + "-" * 60)

        print("Record ID:")
        print(record_id)

        print("\nMetadata:")
        print(data["metadatas"][i])

        print("\nDocument text:")
        text = data["documents"][i]

        if text:
            print(text[:500])
        else:
            print("[EMPTY]")

    print("\n" + "=" * 70)
    print("CHROMA TEST COMPLETE")
    print("=" * 70)

except Exception as e:

    print("\n[FAIL] ChromaDB test failed")

    print("Error type:")
    print(type(e).__name__)

    print("Error:")
    print(str(e))

    sys.exit(1)
    