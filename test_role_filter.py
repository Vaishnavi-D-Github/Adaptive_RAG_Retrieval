from pathlib import Path
import sys

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from adaptive.config import APPLICATION_CHROMA_PATH, APPLICATION_COLLECTION_NAME
import chromadb


ROLE_ACCESS_LEVELS = {
    "student": {"student"},
    "teacher": {"student", "teacher"},
    "office": {"student", "teacher", "office"},
    "admin": {"student", "teacher", "office"},
}


client = chromadb.PersistentClient(path=APPLICATION_CHROMA_PATH)
collection = client.get_collection(APPLICATION_COLLECTION_NAME)

print("Application ChromaDB:")
print("Path:", APPLICATION_CHROMA_PATH)
print("Collection:", APPLICATION_COLLECTION_NAME)
print("Total records:", collection.count())
print()

for role, allowed_levels in ROLE_ACCESS_LEVELS.items():
    print(f"Testing role: {role}")
    print("Allowed levels:", sorted(allowed_levels))

    try:
        result = collection.get(
            where={
                "$and": [
                    {"scope": "application"},
                    {"access_level": {"$in": sorted(allowed_levels)}},
                ]
            },
            limit=10,
        )

        metadatas = result.get("metadatas", [])

        print("  PASS: ChromaDB accepted the filter")
        print("  Returned records:", len(metadatas))

        levels_found = sorted(
            {
                str(metadata.get("access_level"))
                for metadata in metadatas
                if metadata
            }
        )

        print("  Access levels found:", levels_found)

        unauthorized = [
            level for level in levels_found
            if level not in allowed_levels
        ]

        if unauthorized:
            print("  FAIL: Unauthorized levels returned:", unauthorized)
        else:
            print("  PASS: No unauthorized levels returned")

    except Exception as error:
        print("  FAIL: ChromaDB rejected the filter")
        print("  Error:", type(error).__name__, str(error))

    print("-" * 60)