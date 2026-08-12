import chromadb
import ollama
from sentence_transformers import SentenceTransformer


CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "enterprise_documents"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3.2:latest"


def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


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
    embedding_model,
    collection,
    k=5
):

    question_embedding = embedding_model.encode(
        [question]
    )[0]

    results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=k
    )

    return results


def build_context(results):

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []

    for i, document in enumerate(documents):

        metadata = metadatas[i]

        context_parts.append(
            f"""
--- SOURCE {i + 1} ---
Document: {metadata['document']}
Page: {metadata['page']}

{document}

--- END SOURCE {i + 1} ---
"""
        )

    return "\n".join(context_parts)


def generate_answer(question, context):

    prompt = f"""
Answer the question using only the information in the context.

If the context does not contain sufficient evidence to answer the
question, respond exactly:

"The available documents do not contain enough information to answer this question."

Question:
{question}

Context:
{context}

Answer:
"""

    response = ollama.chat(
        model=LLM_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0,
            "num_predict": 256
        }
    )

    return response["message"]["content"]


if __name__ == "__main__":

    question = (
        "How many OT cybersecurity products and services "
        "did CISA provide to critical infrastructure owners "
        "and operators between October 2018 and October 2023?"
    )

    print("Loading embedding model...")

    embedding_model = load_embedding_model()

    print("Connecting to ChromaDB...")

    collection = connect_to_chroma()

    print("\nRetrieving documents...")

    K = 5

    results = retrieve_documents(
        question,
        embedding_model,
        collection,
        k=K
    )

    context = build_context(results)

    print("\n" + "=" * 80)
    print("RETRIEVED CONTEXT SENT TO LLM")
    print("=" * 80)
    print(context)

    print("\n" + "=" * 80)
    print("Generating answer with Ollama...")

    answer = generate_answer(
        question,
        context
    )

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(question)

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)

    print(f"\nRetrieval K = {K}")

    metadatas = results["metadatas"][0]

    for i, metadata in enumerate(metadatas):

        print(
            f"{i + 1}. "
            f"{metadata['document']} "
            f"(Page {metadata['page']})"
        )