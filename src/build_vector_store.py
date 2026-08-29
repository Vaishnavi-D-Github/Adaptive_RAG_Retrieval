from pathlib import Path
import json

import chromadb
from sentence_transformers import SentenceTransformer


CHUNKS_DIR = Path("data/chunks")
CHROMA_DIR = "./chroma_db"

COLLECTION_NAME = "enterprise_documents"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks():
    """
    Load all chunk JSON files from data/chunks/.
    """

    all_chunks = []

    json_files = list(CHUNKS_DIR.glob("*.json"))

    print(f"Found {len(json_files)} chunk files.")

    for json_file in json_files:

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        document_name = data["document"]

        for chunk in data["chunks"]:

            all_chunks.append({
                "document": document_name,
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "text": chunk["text"]
            })

    return all_chunks


def create_embedding_model():

    print("\nLoading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Embedding model loaded.")

    return model


def create_chroma_collection():

    print("\nConnecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Enterprise document chunks for RAG retrieval"
        }
    )

    print(f"ChromaDB collection ready: {COLLECTION_NAME}")

    return collection


def build_vector_store():

    chunks = load_chunks()

    print(f"Total chunks loaded: {len(chunks)}")

    model = create_embedding_model()

    collection = create_chroma_collection()

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:

        document_name = chunk["document"]

        document_stem = Path(document_name).stem

        page_number = chunk["page_number"]

        local_chunk_id = chunk["chunk_id"]

        unique_id = (
            f"{document_stem}"
            f"_p{page_number}"
            f"_c{local_chunk_id}"
        )

        ids.append(unique_id)

        documents.append(chunk["text"])

        metadatas.append({
            "document": document_name,
            "page": page_number,
            "chunk_id": local_chunk_id
        })

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        documents,
        show_progress_bar=True
    )

    print("Embeddings generated.")

    print("\nAdding documents to ChromaDB...")

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print("Documents successfully added to ChromaDB.")

    print(f"\nTotal records in ChromaDB: {collection.count()}")


if __name__ == "__main__":

    build_vector_store()