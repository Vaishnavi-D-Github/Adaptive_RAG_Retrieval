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

# IMPORTANT:
# This is a NEW result file.
# Your original experiment remains untouched.
RESULTS_FILE = (
    "results/k_experiment_results_actual_tokens.json"
)

MIN_K = 1
MAX_K = 10


# ============================================================
# LOAD QUESTIONS
# ============================================================

def load_questions():

    print("=" * 80)
    print("LOADING PILOT QUESTIONS")
    print("=" * 80)

    if not os.path.exists(QUESTIONS_FILE):

        raise FileNotFoundError(
            f"Questions file not found: {QUESTIONS_FILE}"
        )

    questions = []

    # Your CSV has the fields:
    #
    # id
    # question
    # primary_category
    # complexity
    # answerable
    # reference_answer
    # source_document
    # source_page
    # question_rationale

    import csv

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
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():

    print("\n" + "=" * 80)
    print("LOADING EMBEDDING MODEL")
    print("=" * 80)

    print(
        f"Model: {EMBEDDING_MODEL_NAME}"
    )

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
    question_embedding = (
        embedding_model.encode(
            [question]
        )[0]
    )

    # Retrieve top-k chunks
    results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=k
    )

    return results


# ============================================================
# BUILD LLM CONTEXT
# ============================================================

def build_context(results):

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    context_parts = []

    for i, document in enumerate(
        documents
    ):

        metadata = metadatas[i]

        document_name = metadata.get(
            "document",
            "Unknown document"
        )

        page = metadata.get(
            "page",
            "Unknown page"
        )

        chunk_id = metadata.get(
            "chunk_id",
            "Unknown chunk"
        )

        context_parts.append(
            f"""
--- SOURCE {i + 1} ---
Document: {document_name}
Page: {page}
Chunk ID: {chunk_id}

{document}

--- END SOURCE {i + 1} ---
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# EXTRACT RETRIEVED CHUNKS
# ============================================================

def extract_retrieved_chunks(
    results
):

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    retrieved_chunks = []

    for index, document in enumerate(
        documents
    ):

        metadata = metadatas[index]

        distance = None

        if index < len(distances):

            distance = distances[index]

        retrieved_chunks.append(
            {
                "rank": index + 1,

                "document": metadata.get(
                    "document"
                ),

                "page": metadata.get(
                    "page"
                ),

                "chunk_id": metadata.get(
                    "chunk_id"
                ),

                "distance": distance,

                "text": document
            }
        )

    return retrieved_chunks


# ============================================================
# GENERATE ANSWER + CAPTURE OLLAMA TOKEN USAGE
# ============================================================

def generate_answer(
    question,
    context
):

    # --------------------------------------------------------
    # IMPORTANT EXPERIMENTAL PROMPT
    # --------------------------------------------------------
    #
    # Keep this prompt identical for every question and
    # every K.
    #
    # This ensures K remains the main experimental variable.
    #
    # We explicitly include abstention because EQ010 is
    # intentionally unanswerable.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CALL OLLAMA
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # EXTRACT ANSWER
    # --------------------------------------------------------

    answer = response[
        "message"
    ][
        "content"
    ].strip()

    # --------------------------------------------------------
    # EXTRACT ACTUAL OLLAMA TOKEN METADATA
    # --------------------------------------------------------

    prompt_eval_count = response.get(
        "prompt_eval_count"
    )

    eval_count = response.get(
        "eval_count"
    )

    prompt_eval_duration = response.get(
        "prompt_eval_duration"
    )

    eval_duration = response.get(
        "eval_duration"
    )

    total_duration = response.get(
        "total_duration"
    )

    load_duration = response.get(
        "load_duration"
    )

    # --------------------------------------------------------
    # CALCULATE TOTAL TOKENS
    # --------------------------------------------------------

    total_tokens = None

    if (
        isinstance(
            prompt_eval_count,
            int
        )
        and isinstance(
            eval_count,
            int
        )
    ):

        total_tokens = (
            prompt_eval_count
            + eval_count
        )

    # --------------------------------------------------------
    # RETURN EVERYTHING
    # --------------------------------------------------------

    return {

        "answer": answer,

        "prompt_eval_count": (
            prompt_eval_count
        ),

        "eval_count": (
            eval_count
        ),

        "total_tokens": (
            total_tokens
        ),

        "prompt_eval_duration": (
            prompt_eval_duration
        ),

        "eval_duration": (
            eval_duration
        ),

        "total_duration": (
            total_duration
        ),

        "load_duration": (
            load_duration
        )
    }


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
        f"Question ID: {question_id}"
    )

    print(
        f"K = {k}"
    )

    print(
        f"Question: {question}"
    )

    print("-" * 80)

    experiment_start = time.time()

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    print(
        "Retrieving documents..."
    )

    results = retrieve_documents(
        question,
        embedding_model,
        collection,
        k
    )

    # --------------------------------------------------------
    # RETRIEVED CHUNKS
    # --------------------------------------------------------

    retrieved_chunks = (
        extract_retrieved_chunks(
            results
        )
    )

    print(
        f"Retrieved chunks: "
        f"{len(retrieved_chunks)}"
    )

    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context = build_context(
        results
    )

    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    print(
        "Generating answer with Ollama..."
    )

    generation = generate_answer(
        question,
        context
    )

    answer = generation[
        "answer"
    ]

    # --------------------------------------------------------
    # TOTAL EXPERIMENT TIME
    # --------------------------------------------------------

    execution_time = (
        time.time()
        - experiment_start
    )

    # --------------------------------------------------------
    # RESULT OBJECT
    # --------------------------------------------------------

    result = {

        # ================================================
        # DATASET INFORMATION
        # ================================================

        "question_id": question_id,

        "question": question,

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

        # ================================================
        # EXPERIMENT CONFIGURATION
        # ================================================

        "k": k,

        "embedding_model": (
            EMBEDDING_MODEL_NAME
        ),

        "llm_model": (
            LLM_MODEL_NAME
        ),

        "temperature": 0,

        "num_predict": 256,

        # ================================================
        # RETRIEVAL INFORMATION
        # ================================================

        "number_of_retrieved_chunks": (
            len(retrieved_chunks)
        ),

        "retrieved_chunks": (
            retrieved_chunks
        ),

        # Complete context actually given to Ollama
        "context": context,

        # ================================================
        # GENERATED ANSWER
        # ================================================

        "generated_answer": answer,

        # ================================================
        # ACTUAL OLLAMA TOKEN USAGE
        # ================================================

        "prompt_eval_count": (
            generation[
                "prompt_eval_count"
            ]
        ),

        "eval_count": (
            generation[
                "eval_count"
            ]
        ),

        "total_tokens": (
            generation[
                "total_tokens"
            ]
        ),

        # ================================================
        # OLLAMA TIMING
        # ================================================

        "prompt_eval_duration": (
            generation[
                "prompt_eval_duration"
            ]
        ),

        "eval_duration": (
            generation[
                "eval_duration"
            ]
        ),

        "total_duration": (
            generation[
                "total_duration"
            ]
        ),

        "load_duration": (
            generation[
                "load_duration"
            ]
        ),

        # ================================================
        # OVERALL PYTHON EXPERIMENT TIME
        # ================================================

        "execution_time_seconds": round(
            execution_time,
            3
        )
    }

    # --------------------------------------------------------
    # TERMINAL OUTPUT
    # --------------------------------------------------------

    print(
        "\nGenerated answer:"
    )

    print(
        answer
    )

    print(
        "\nActual Ollama token usage:"
    )

    print(
        f"Prompt tokens: "
        f"{generation['prompt_eval_count']}"
    )

    print(
        f"Generated tokens: "
        f"{generation['eval_count']}"
    )

    print(
        f"Total tokens: "
        f"{generation['total_tokens']}"
    )

    print(
        f"Execution time: "
        f"{execution_time:.2f} seconds"
    )

    return result


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

    print(
        f"\nResults saved: "
        f"{RESULTS_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 80)
    print("ACTUAL OLLAMA TOKEN K EXPERIMENT")
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

    print(
        f"Results file: "
        f"{RESULTS_FILE}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # LOAD QUESTIONS
    # --------------------------------------------------------

    questions = load_questions()

    if not questions:

        print(
            "ERROR: No questions found."
        )

        return

    # --------------------------------------------------------
    # LOAD EMBEDDING MODEL
    # --------------------------------------------------------

    embedding_model = (
        load_embedding_model()
    )

    # --------------------------------------------------------
    # CONNECT TO CHROMADB
    # --------------------------------------------------------

    collection = (
        connect_to_chroma()
    )

    # --------------------------------------------------------
    # CALCULATE TOTAL RUNS
    # --------------------------------------------------------

    total_k_values = (
        MAX_K - MIN_K + 1
    )

    total_runs = (
        len(questions)
        * total_k_values
    )

    print("\n" + "=" * 80)

    print(
        f"Questions: "
        f"{len(questions)}"
    )

    print(
        f"K values: "
        f"{total_k_values}"
    )

    print(
        f"Total runs: "
        f"{total_runs}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # RESULTS CONTAINER
    # --------------------------------------------------------

    all_results = []

    current_run = 0

    experiment_start = time.time()

    # --------------------------------------------------------
    # RUN ALL QUESTIONS
    # --------------------------------------------------------

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

                result = (
                    run_single_experiment(
                        row,
                        k,
                        embedding_model,
                        collection
                    )
                )

                all_results.append(
                    result
                )

                # ------------------------------------------------
                # SAVE AFTER EVERY RUN
                # ------------------------------------------------
                #
                # This protects your results if the experiment
                # is interrupted.
                # ------------------------------------------------

                save_results(
                    all_results
                )

            except Exception as error:

                print("\n" + "!" * 80)

                print(
                    f"ERROR: "
                    f"{question_id}, K={k}"
                )

                print(
                    f"Error: {error}"
                )

                print(
                    "!" * 80
                )

                # ------------------------------------------------
                # SAVE FAILURE AS A RESULT RECORD
                # ------------------------------------------------

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
    # FINISHED
    # --------------------------------------------------------

    total_time = (
        time.time()
        - experiment_start
    )

    print("\n\n")

    print("=" * 80)
    print("ACTUAL TOKEN EXPERIMENT COMPLETED")
    print("=" * 80)

    print(
        f"Expected runs: "
        f"{total_runs}"
    )

    print(
        f"Result records: "
        f"{len(all_results)}"
    )

    print(
        f"Total experiment time: "
        f"{total_time:.2f} seconds"
    )

    print(
        f"Results:"
    )

    print(
        RESULTS_FILE
    )

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()