import csv
import json
import os
import sys
import time
from datetime import datetime

import ollama
import pandas as pd

# ---------------------------------------------------------------------
# Allow importing rag.py when this script is executed as:
# python src/validate_model.py
# ---------------------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from rag import (
    load_embedding_model,
    connect_to_chroma,
    retrieve_documents,
    build_context,
    generate_answer,
    LLM_MODEL_NAME,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

QUESTION_FILE = "./data/evaluation/pilot_questions_10.csv"

RESULTS_DIR = "./results/validation"

CSV_OUTPUT = os.path.join(
    RESULTS_DIR,
    "model_validation_results.csv"
)

JSON_OUTPUT = os.path.join(
    RESULTS_DIR,
    "model_validation_results.json"
)

# Representative questions
VALIDATION_IDS = [
    "EQ001",
    "EQ002",
    "EQ003",
    "EQ005",
    "EQ007",
    "EQ010",
]

# K values to test
VALIDATION_K_VALUES = [1, 5, 10]


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def ensure_results_directory():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )


# ---------------------------------------------------------------------
# Load questions
# ---------------------------------------------------------------------

def load_questions():

    df = pd.read_csv(
        QUESTION_FILE,
        encoding="utf-8"
    )

    df["id"] = df["id"].astype(str).str.strip()

    selected = df[
        df["id"].isin(VALIDATION_IDS)
    ].copy()

    # Preserve our desired ordering
    order = {
        qid: i
        for i, qid in enumerate(VALIDATION_IDS)
    }

    selected["_order"] = selected["id"].map(order)

    selected = selected.sort_values(
        "_order"
    ).drop(
        columns=["_order"]
    )

    return selected


# ---------------------------------------------------------------------
# Ollama generation with usage information
# ---------------------------------------------------------------------

def generate_answer_with_usage(question, context):

    prompt = f"""
You are an enterprise document question-answering assistant.

Answer the question using ONLY the retrieved context.

Rules:

1. Use only information contained in the context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. If the context contains sufficient evidence, answer the question directly.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER:
"""

    start_time = time.perf_counter()

    response = ollama.chat(
        model=LLM_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        think=False,
        options={
            "temperature": 0,
            "num_predict": 512
        }
    )

    elapsed = time.perf_counter() - start_time

    message = response.get("message", {})

    answer = message.get(
        "content",
        ""
    )

    # Ollama usage fields
    prompt_tokens = response.get(
        "prompt_eval_count"
    )

    generated_tokens = response.get(
        "eval_count"
    )

    total_tokens = None

    if prompt_tokens is not None and generated_tokens is not None:
        total_tokens = (
            prompt_tokens +
            generated_tokens
        )

    return {
        "answer": answer,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_tokens": total_tokens,
        "latency_seconds": elapsed,
        "total_duration_ns": response.get(
            "total_duration"
        ),
        "load_duration_ns": response.get(
            "load_duration"
        ),
        "prompt_eval_duration_ns": response.get(
            "prompt_eval_duration"
        ),
        "eval_duration_ns": response.get(
            "eval_duration"
        ),
    }


# ---------------------------------------------------------------------
# Main validation experiment
# ---------------------------------------------------------------------

def run_validation():

    ensure_results_directory()

    print("=" * 80)
    print("QWEN3 MODEL VALIDATION")
    print("=" * 80)

    print(f"\nGeneration model: {LLM_MODEL_NAME}")

    print(
        f"Validation questions: "
        f"{', '.join(VALIDATION_IDS)}"
    )

    print(
        f"K values: "
        f"{VALIDATION_K_VALUES}"
    )

    print(
        f"Total runs: "
        f"{len(VALIDATION_IDS) * len(VALIDATION_K_VALUES)}"
    )

    # ---------------------------------------------------------------
    # Load questions
    # ---------------------------------------------------------------

    print("\nLoading validation questions...")

    questions_df = load_questions()

    if len(questions_df) != len(VALIDATION_IDS):

        found = set(
            questions_df["id"].tolist()
        )

        missing = [
            qid
            for qid in VALIDATION_IDS
            if qid not in found
        ]

        raise RuntimeError(
            "Missing validation questions: "
            + ", ".join(missing)
        )

    # ---------------------------------------------------------------
    # Load models / database
    # ---------------------------------------------------------------

    print("\nLoading embedding model...")

    embedding_model = load_embedding_model()

    print("Connecting to ChromaDB...")

    collection = connect_to_chroma()

    print("\nStarting validation...")

    all_results = []

    total_runs = (
        len(questions_df)
        * len(VALIDATION_K_VALUES)
    )

    completed = 0

    # ---------------------------------------------------------------
    # Run experiment
    # ---------------------------------------------------------------

    for _, row in questions_df.iterrows():

        question_id = row["id"]
        question = row["question"]

        print("\n" + "-" * 80)
        print(
            f"QUESTION {question_id}"
        )
        print("-" * 80)

        print(f"Category: {row['primary_category']}")
        print(f"Complexity: {row['complexity']}")
        print(f"Answerable: {row['answerable']}")
        print(f"Question: {question}")

        for k in VALIDATION_K_VALUES:

            completed += 1

            print(
                f"\n[{completed}/{total_runs}] "
                f"{question_id} | K={k}"
            )

            # -------------------------------------------------------
            # Retrieval
            # -------------------------------------------------------

            retrieval_start = time.perf_counter()

            results = retrieve_documents(
                question,
                embedding_model,
                collection,
                k=k
            )

            retrieval_time = (
                time.perf_counter()
                - retrieval_start
            )

            # -------------------------------------------------------
            # Context
            # -------------------------------------------------------

            context = build_context(
                results
            )

            documents = results["documents"][0]
            metadatas = results["metadatas"][0]

            sources = []

            for metadata in metadatas:

                sources.append({
                    "document": metadata.get(
                        "document"
                    ),
                    "page": metadata.get(
                        "page"
                    )
                })

            # -------------------------------------------------------
            # Generation
            # -------------------------------------------------------

            print("  Generating answer...")

            generation = generate_answer_with_usage(
                question,
                context
            )

            answer = generation["answer"]

            # -------------------------------------------------------
            # Store result
            # -------------------------------------------------------

            record = {
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

                "k": k,

                "num_retrieved_chunks": len(
                    documents
                ),

                "retrieved_sources": sources,

                "retrieved_context": context,

                "generated_answer": answer,

                "prompt_tokens": generation[
                    "prompt_tokens"
                ],

                "generated_tokens": generation[
                    "generated_tokens"
                ],

                "total_tokens": generation[
                    "total_tokens"
                ],

                "retrieval_time_seconds": retrieval_time,

                "generation_time_seconds": generation[
                    "latency_seconds"
                ],

                "total_duration_ns": generation[
                    "total_duration_ns"
                ],

                "load_duration_ns": generation[
                    "load_duration_ns"
                ],

                "prompt_eval_duration_ns": generation[
                    "prompt_eval_duration_ns"
                ],

                "eval_duration_ns": generation[
                    "eval_duration_ns"
                ],

                "timestamp": datetime.now().isoformat(),
            }

            all_results.append(record)

            # -------------------------------------------------------
            # Console summary
            # -------------------------------------------------------

            print(
                f"  Retrieved chunks: "
                f"{len(documents)}"
            )

            print(
                f"  Prompt tokens: "
                f"{generation['prompt_tokens']}"
            )

            print(
                f"  Generated tokens: "
                f"{generation['generated_tokens']}"
            )

            print(
                f"  Total tokens: "
                f"{generation['total_tokens']}"
            )

            print(
                f"  Generation time: "
                f"{generation['latency_seconds']:.2f}s"
            )

            print(
                f"  Answer: "
                f"{answer.strip()}"
            )

    # ---------------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------------

    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    with open(
        JSON_OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nDetailed JSON saved to:\n"
        f"{JSON_OUTPUT}"
    )

    # ---------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------

    csv_rows = []

    for record in all_results:

        csv_record = record.copy()

        csv_record["retrieved_sources"] = json.dumps(
            csv_record["retrieved_sources"],
            ensure_ascii=False
        )

        csv_rows.append(
            csv_record
        )

    fieldnames = list(
        csv_rows[0].keys()
    )

    with open(
        CSV_OUTPUT,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            csv_rows
        )

    print(
        f"CSV saved to:\n"
        f"{CSV_OUTPUT}"
    )

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    print(
        f"\nTotal runs: {len(all_results)}"
    )

    print(
        f"Expected runs: "
        f"{len(VALIDATION_IDS) * len(VALIDATION_K_VALUES)}"
    )

    print(
        f"\nGeneration model: "
        f"{LLM_MODEL_NAME}"
    )

    print(
        "\nValidation questions:"
    )

    for question_id in VALIDATION_IDS:

        print(
            f"  {question_id}"
        )

    print(
        "\nK values:"
    )

    for k in VALIDATION_K_VALUES:

        print(
            f"  K={k}"
        )

    print(
        "\nNext step:"
    )

    print(
        "Review the answers and retrieved evidence "
        "before running the full 100-question experiment."
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":

    run_validation()