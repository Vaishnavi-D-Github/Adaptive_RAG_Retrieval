import pprint

import chromadb
from sentence_transformers import SentenceTransformer

from retrieval.chroma_retriever import ChromaRetriever


APPLICATION_CHROMA_PATH = "data/application_chroma_db"
APPLICATION_COLLECTION_NAME = "application_uploaded_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    print("=" * 70)
    print("CHROMA RETRIEVER RETURN STRUCTURE TEST")
    print("=" * 70)

    client = chromadb.PersistentClient(
        path=APPLICATION_CHROMA_PATH
    )

    collection = client.get_collection(
        APPLICATION_COLLECTION_NAME
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    retriever = ChromaRetriever(
        collection=collection,
        embedding_model=embedding_model,
    )

    question = "What is the teacher-only access test code?"

    print("\nQuestion:")
    print(question)

    print("\nRetrieving as TEACHER...")

    results = retriever.retrieve(
        question,
        5,
        user_role="teacher",
    )

    print("\nNumber of results:", len(results))

    if not results:
        print("[FAIL] No results returned")
        return

    print("\n" + "=" * 70)
    print("RAW FIRST RESULT")
    print("=" * 70)

    pprint.pprint(results[0], width=120)

    print("\n" + "=" * 70)
    print("RESULT TYPE")
    print("=" * 70)

    print(type(results[0]))

    if isinstance(results[0], dict):
        print("\nDictionary keys:")
        print(list(results[0].keys()))

        for key, value in results[0].items():
            print("\nKEY:", repr(key))
            print("TYPE:", type(value))
            print("VALUE:", repr(value))

    print("\n" + "=" * 70)
    print("SEARCHING FOR TEACHER CODE")
    print("=" * 70)

    found = False

    for result in results:

        if isinstance(result, dict):

            for key, value in result.items():

                if isinstance(value, str):
                    if "TEACHER_ONLY_456" in value:
                        print(
                            f"[PASS] TEACHER_ONLY_456 found "
                            f"inside result field: {key}"
                        )
                        found = True

    if not found:
        print(
            "[WARNING] TEACHER_ONLY_456 was not found "
            "in any string field returned by ChromaRetriever"
        )

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()