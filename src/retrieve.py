import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "enterprise_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_model():

    print("Loading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print("Embedding model loaded.")

    return model


def connect_to_chroma():

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return collection


def retrieve_documents(
    question,
    model,
    collection,
    k=5
):

    question_embedding = model.encode(
        [question]
    )[0]

    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=k
    )

    return results


if __name__ == "__main__":

    question = (
        "How many OT cybersecurity products and services "
        "did CISA provide to critical infrastructure owners "
        "and operators between October 2018 and November 2023?"
    )

    model = load_model()

    collection = connect_to_chroma()

    print(
        f"\nQuestion:\n{question}"
    )

    results = retrieve_documents(
        question,
        model,
        collection,
        k=5
    )

    print("\n" + "=" * 80)
    print("TOP 5 RETRIEVED CHUNKS")
    print("=" * 80)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(documents)):

        print(
            f"\nRank: {i + 1}"
        )

        print(
            f"Document: {metadatas[i]['document']}"
        )

        print(
            f"Page: {metadatas[i]['page']}"
        )

        print(
            f"Chunk ID: {metadatas[i]['chunk_id']}"
        )

        print(
            f"Distance: {distances[i]:.4f}"
        )

        print("\nText:")

        print(
            documents[i][:1500]
        )

        print("\n" + "-" * 80)