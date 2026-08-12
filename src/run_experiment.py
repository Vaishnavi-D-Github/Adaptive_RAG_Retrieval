import csv
import json
import os
import time

import chromadb
import ollama
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "enterprise_documents"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3.2:latest"

QUESTIONS_FILE = "data/evaluation/pilot_questions_10.csv"

RESULTS_DIR = "results"
RESULTS_FILE = "results/k_experiment_results.json"

# Experiment range
MIN_K = 1
MAX_K = 10


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():

    print("=" * 80)
    print("LOADING EMBEDDING MODEL")
    print("=" * 80)

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print("Embedding model loaded.")

    return model


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

def connect_to_chroma():

    print("\n" + "=" * 80)
    print("CONNECTING TO CHROMADB")
    print("=" * 80)

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print(
        f"Records available: {collection.count()}"
    )

    return collection


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    question,
    embedding_model,
    collection,
    k
):

    # Convert question into embedding
    question_embedding = embedding_model.encode(
        [question]
    )[0]

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=k
    )

    return results


# ============================================================
# BUILD CONTEXT FOR LLM
# ============================================================

def build_context(results):

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []

    for i, document in enumerate(documents):

        metadata = metadatas[i]

        document_name = metadata.get(
            "document",
            "Unknown document"
        )

        page = metadata.get(
            "page",
            "Unknown page"
        )

        context_parts.append(
            f"""
--- SOURCE {i + 1} ---
Document: {document_name}
Page: {page}

{document}

--- END SOURCE {i + 1} ---
"""
        )

    return "\n".join(context_parts)


# ============================================================
# GENERATE ANSWER USING OLLAMA
# ============================================================

def generate_answer(
    question,
    context
):

    # IMPORTANT:
    # This is intentionally short and generic.
    # It remains the same for every question and every K.
    #
    # This allows K to remain the main experimental variable.

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

    return response["message"]["content"].strip()


# ============================================================
# LOAD PILOT QUESTIONS
# ============================================================

def load_questions():

    print("\n" + "=" * 80)
    print("LOADING PILOT QUESTIONS")
    print("=" * 80)

    questions = []

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            questions.append(row)

    print(
        f"Questions loaded: {len(questions)}"
    )

    return questions


# ============================================================
# EXTRACT RETRIEVED CHUNKS
# ============================================================

def extract_retrieved_chunks(
    results
):

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # ChromaDB normally returns distances when using query().
    distances = results.get(
        "distances",
        [[]]
    )[0]

    retrieved_chunks = []

    for index, document in enumerate(
        documents
    ):

        metadata = metadatas[index]

        # Get distance if available
        distance = None

        if index < len(distances):

            distance = distances[index]

        # Some versions/configurations may not
        # have chunk_id in metadata.
        chunk_id = metadata.get(
            "chunk_id"
        )

        retrieved_chunks.append(
            {
                "rank": index + 1,

                "document": metadata.get(
                    "document"
                ),

                "page": metadata.get(
                    "page"
                ),

                "chunk_id": chunk_id,

                "distance": distance,

                "text": document
            }
        )

    return retrieved_chunks


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_single_experiment(
    row,
    k,
    embedding_model,
    collection
):

    question_id = row["id"]
    question = row["question"]

    print("\n" + "-" * 80)

    print(
        f"Question: {question_id}"
    )

    print(
        f"K = {k}"
    )

    print(
        f"Question: {question}"
    )

    print("-" * 80)

    start_time = time.time()

    # --------------------------------------------------------
    # STEP 1: RETRIEVAL
    # --------------------------------------------------------

    print("Retrieving documents...")

    results = retrieve_documents(
        question,
        embedding_model,
        collection,
        k
    )

    # --------------------------------------------------------
    # STEP 2: EXTRACT RETRIEVED INFORMATION
    # --------------------------------------------------------

    retrieved_chunks = extract_retrieved_chunks(
        results
    )

    print(
        f"Retrieved chunks: "
        f"{len(retrieved_chunks)}"
    )

    # --------------------------------------------------------
    # STEP 3: BUILD CONTEXT
    # --------------------------------------------------------

    context = build_context(
        results
    )

    # --------------------------------------------------------
    # STEP 4: GENERATE ANSWER
    # --------------------------------------------------------

    print(
        "Generating answer with Ollama..."
    )

    answer = generate_answer(
        question,
        context
    )

    # --------------------------------------------------------
    # STEP 5: EXECUTION TIME
    # --------------------------------------------------------

    execution_time = (
        time.time() - start_time
    )

    # --------------------------------------------------------
    # STEP 6: CREATE RESULT OBJECT
    # --------------------------------------------------------

    experiment_result = {

        # ------------------------------------
        # Question information
        # ------------------------------------

        "question_id": question_id,

        "question": question,

        "primary_category": row[
            "primary_category"
        ],

        "complexity": row[
            "complexity"
        ],

        "answerable": row[
            "answerable"
        ],

        "reference_answer": row[
            "reference_answer"
        ],

        "source_document": row[
            "source_document"
        ],

        "source_page": row[
            "source_page"
        ],

        "question_rationale": row[
            "question_rationale"
        ],

        # ------------------------------------
        # Experiment configuration
        # ------------------------------------

        "k": k,

        "embedding_model": (
            EMBEDDING_MODEL_NAME
        ),

        "llm_model": (
            LLM_MODEL_NAME
        ),

        "temperature": 0,

        # ------------------------------------
        # Retrieval information
        # ------------------------------------

        "retrieved_chunks": (
            retrieved_chunks
        ),

        # Complete context sent to LLM
        "context": context,

        # ------------------------------------
        # Generation information
        # ------------------------------------

        "generated_answer": answer,

        # ------------------------------------
        # Performance information
        # ------------------------------------

        "execution_time_seconds": round(
            execution_time,
            3
        )
    }

    print(
        f"\nGenerated answer:"
    )

    print(answer)

    print(
        f"\nExecution time: "
        f"{execution_time:.2f} seconds"
    )

    return experiment_result


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 80)

    print(
        "RESULTS SAVED"
    )

    print(
        f"File: {RESULTS_FILE}"
    )

    print(
        f"Total experiment records: "
        f"{len(results)}"
    )

    print("=" * 80)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("ADAPTIVE-K RAG EXPERIMENT")
    print("=" * 80)

    print(
        f"Embedding model: "
        f"{EMBEDDING_MODEL_NAME}"
    )

    print(
        f"LLM model: "
        f"{LLM_MODEL_NAME}"
    )

    print(
        f"K range: "
        f"{MIN_K} to {MAX_K}"
    )

    print(
        f"Questions file: "
        f"{QUESTIONS_FILE}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # LOAD QUESTIONS
    # --------------------------------------------------------

    questions = load_questions()

    if len(questions) == 0:

        print(
            "ERROR: No questions were found."
        )

        return

    # --------------------------------------------------------
    # LOAD EMBEDDING MODEL
    # --------------------------------------------------------

    embedding_model = load_embedding_model()

    # --------------------------------------------------------
    # CONNECT TO CHROMADB
    # --------------------------------------------------------

    collection = connect_to_chroma()

    # --------------------------------------------------------
    # CALCULATE TOTAL RUNS
    # --------------------------------------------------------

    number_of_k_values = (
        MAX_K - MIN_K + 1
    )

    total_runs = (
        len(questions)
        * number_of_k_values
    )

    print("\n" + "=" * 80)

    print(
        f"Total questions: "
        f"{len(questions)}"
    )

    print(
        f"K values: "
        f"{number_of_k_values}"
    )

    print(
        f"Total experiment runs: "
        f"{total_runs}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # RUN EXPERIMENT
    # --------------------------------------------------------

    all_results = []

    current_run = 0

    experiment_start_time = (
        time.time()
    )

    for question_number, row in enumerate(
        questions,
        start=1
    ):

        question_id = row["id"]

        print("\n\n")

        print(
            "#" * 80
        )

        print(
            f"QUESTION "
            f"{question_number}/"
            f"{len(questions)}"
        )

        print(
            f"ID: {question_id}"
        )

        print(
            "#" * 80
        )

        for k in range(
            MIN_K,
            MAX_K + 1
        ):

            current_run += 1

            print("\n")

            print(
                f"RUN "
                f"{current_run}/"
                f"{total_runs}"
            )

            try:

                result = run_single_experiment(
                    row,
                    k,
                    embedding_model,
                    collection
                )

                all_results.append(
                    result
                )

                # ------------------------------------------------
                # SAVE AFTER EVERY RUN
                # ------------------------------------------------
                #
                # This is intentional.
                #
                # If the experiment is interrupted halfway through,
                # the results collected so far are not lost.

                save_results(
                    all_results
                )

            except Exception as error:

                print("\n" + "!" * 80)

                print(
                    f"ERROR during "
                    f"{question_id}, K={k}"
                )

                print(
                    f"Error: {error}"
                )

                print(
                    "!" * 80
                )

                # Record the failed run instead
                # of terminating the entire experiment.

                failed_result = {

                    "question_id": (
                        question_id
                    ),

                    "question": (
                        row["question"]
                    ),

                    "primary_category": (
                        row["primary_category"]
                    ),

                    "complexity": (
                        row["complexity"]
                    ),

                    "answerable": (
                        row["answerable"]
                    ),

                    "reference_answer": (
                        row["reference_answer"]
                    ),

                    "source_document": (
                        row["source_document"]
                    ),

                    "source_page": (
                        row["source_page"]
                    ),

                    "question_rationale": (
                        row["question_rationale"]
                    ),

                    "k": k,

                    "embedding_model": (
                        EMBEDDING_MODEL_NAME
                    ),

                    "llm_model": (
                        LLM_MODEL_NAME
                    ),

                    "error": str(error)
                }

                all_results.append(
                    failed_result
                )

                save_results(
                    all_results
                )

    # --------------------------------------------------------
    # EXPERIMENT FINISHED
    # --------------------------------------------------------

    total_time = (
        time.time()
        - experiment_start_time
    )

    print("\n\n")

    print("=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)

    print(
        f"Expected runs: "
        f"{total_runs}"
    )

    print(
        f"Completed records: "
        f"{len(all_results)}"
    )

    print(
        f"Total execution time: "
        f"{total_time:.2f} seconds"
    )

    print(
        f"Results file:"
    )

    print(
        RESULTS_FILE
    )

    print("=" * 80)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()