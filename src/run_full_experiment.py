import csv
import json
import os
import time
from datetime import datetime

import ollama
import pandas as pd

# Import the existing RAG components
from rag import (
    load_embedding_model,
    connect_to_chroma,
    retrieve_documents,
    build_context,
    LLM_MODEL_NAME,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# The actual 100-question evaluation dataset
QUESTION_FILE = (
    "./data/evaluation/"
    "enterprise_query_evaluation_set_100.csv"
)

# Results directory
RESULTS_DIR = "./results"

# Detailed JSON output
JSON_OUTPUT = os.path.join(
    RESULTS_DIR,
    "full_k_experiment_qwen3.json"
)

# Tabular CSV output
CSV_OUTPUT = os.path.join(
    RESULTS_DIR,
    "full_k_experiment_qwen3.csv"
)

# K values to experiment with
K_VALUES = list(range(1, 11))


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_questions():

    print("\nLoading evaluation dataset...")

    df = pd.read_csv(
        QUESTION_FILE,
        encoding="utf-8"
    )

    # -------------------------------------------------------------------------
    # These are the ACTUAL column names in:
    #
    # enterprise_query_evaluation_set_100.csv
    # -------------------------------------------------------------------------

    required_columns = [
        "ID",
        "Question",
        "Primary_Category",
        "Secondary_Category",
        "Complexity",
        "Answerable",
        "Reference_Answer",
        "Source_Document",
        "Source_Page",
        "Question_Rationale",
    ]

    # -------------------------------------------------------------------------
    # Check that every required column exists
    # -------------------------------------------------------------------------

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nThe evaluation dataset is missing "
            "the following columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )

    # -------------------------------------------------------------------------
    # Clean ID and Question
    # -------------------------------------------------------------------------

    df["ID"] = (
        df["ID"]
        .astype(str)
        .str.strip()
    )

    df["Question"] = (
        df["Question"]
        .astype(str)
        .str.strip()
    )

    # -------------------------------------------------------------------------
    # Dataset integrity checks
    # -------------------------------------------------------------------------

    if len(df) != 100:

        raise ValueError(
            f"\nExpected exactly 100 questions, "
            f"but found {len(df)}."
        )

    if df["ID"].nunique() != 100:

        raise ValueError(
            "\nExpected 100 unique question IDs."
        )

    if df["Question"].isna().any():

        raise ValueError(
            "\nOne or more questions are missing."
        )

    if df["Reference_Answer"].isna().any():

        raise ValueError(
            "\nOne or more reference answers are missing."
        )

    # -------------------------------------------------------------------------
    # Print dataset information
    # -------------------------------------------------------------------------

    print(
        f"Questions loaded: {len(df)}"
    )

    print(
        f"Unique question IDs: "
        f"{df['ID'].nunique()}"
    )

    print(
        f"Missing questions: "
        f"{df['Question'].isna().sum()}"
    )

    print(
        f"Missing reference answers: "
        f"{df['Reference_Answer'].isna().sum()}"
    )

    return df


# =============================================================================
# OLLAMA GENERATION
# =============================================================================

def generate_answer_with_usage(
    question,
    context
):

    # -------------------------------------------------------------------------
    # Fixed prompt
    #
    # IMPORTANT:
    # Do not change this prompt during the K experiment.
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Start timing
    # -------------------------------------------------------------------------

    start_time = time.perf_counter()

    # -------------------------------------------------------------------------
    # Ollama
    # -------------------------------------------------------------------------

    response = ollama.chat(
        model=LLM_MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        # Qwen3 thinking disabled
        think=False,

        options={
            # Deterministic generation
            "temperature": 0,

            # Maximum answer length
            "num_predict": 512,
        }
    )

    # -------------------------------------------------------------------------
    # Generation time
    # -------------------------------------------------------------------------

    generation_time = (
        time.perf_counter()
        - start_time
    )

    # -------------------------------------------------------------------------
    # Extract response
    # -------------------------------------------------------------------------

    message = response.get(
        "message",
        {}
    )

    answer = message.get(
        "content",
        ""
    )

    # -------------------------------------------------------------------------
    # Ollama token usage
    # -------------------------------------------------------------------------

    prompt_tokens = response.get(
        "prompt_eval_count"
    )

    generated_tokens = response.get(
        "eval_count"
    )

    if (
        prompt_tokens is not None
        and generated_tokens is not None
    ):

        total_tokens = (
            prompt_tokens
            + generated_tokens
        )

    else:

        total_tokens = None

    # -------------------------------------------------------------------------
    # Return everything we need for the experiment
    # -------------------------------------------------------------------------

    return {

        "answer":
            answer,

        "prompt_tokens":
            prompt_tokens,

        "generated_tokens":
            generated_tokens,

        "total_tokens":
            total_tokens,

        "generation_time_seconds":
            generation_time,

        "total_duration_ns":
            response.get(
                "total_duration"
            ),

        "load_duration_ns":
            response.get(
                "load_duration"
            ),

        "prompt_eval_duration_ns":
            response.get(
                "prompt_eval_duration"
            ),

        "eval_duration_ns":
            response.get(
                "eval_duration"
            ),
    }


# =============================================================================
# RESULTS DIRECTORY
# =============================================================================

def create_results_directory():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():

    # -------------------------------------------------------------------------
    # Create output directory
    # -------------------------------------------------------------------------

    create_results_directory()

    # -------------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------------

    print("=" * 80)
    print("FULL K EXPERIMENT")
    print("=" * 80)

    print(
        f"\nGeneration model: "
        f"{LLM_MODEL_NAME}"
    )

    print(
        "K values: "
        + ", ".join(
            str(k)
            for k in K_VALUES
        )
    )

    # -------------------------------------------------------------------------
    # Load questions
    # -------------------------------------------------------------------------

    questions_df = load_questions()

    expected_runs = (
        len(questions_df)
        * len(K_VALUES)
    )

    print(
        f"Expected experiment runs: "
        f"{expected_runs}"
    )

    # -------------------------------------------------------------------------
    # Load embedding model
    # -------------------------------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    embedding_model = (
        load_embedding_model()
    )

    # -------------------------------------------------------------------------
    # Connect to ChromaDB
    # -------------------------------------------------------------------------

    print(
        "Connecting to ChromaDB..."
    )

    collection = (
        connect_to_chroma()
    )

    # -------------------------------------------------------------------------
    # Experiment timer
    # -------------------------------------------------------------------------

    experiment_start = (
        time.perf_counter()
    )

    # -------------------------------------------------------------------------
    # Results container
    # -------------------------------------------------------------------------

    all_results = []

    completed = 0

    # =========================================================================
    # QUESTION LOOP
    # =========================================================================

    for question_index, row in questions_df.iterrows():

        question_id = row["ID"]

        question = row["Question"]

        print("\n" + "=" * 80)

        print(
            f"QUESTION {question_id} "
            f"({question_index + 1}/{len(questions_df)})"
        )

        print("=" * 80)

        print(
            f"Category: "
            f"{row['Primary_Category']}"
        )

        print(
            f"Secondary Category: "
            f"{row['Secondary_Category']}"
        )

        print(
            f"Complexity: "
            f"{row['Complexity']}"
        )

        print(
            f"Answerable: "
            f"{row['Answerable']}"
        )

        print(
            f"Question: "
            f"{question}"
        )

        # =====================================================================
        # K LOOP
        # =====================================================================

        for k in K_VALUES:

            completed += 1

            print(
                "\n"
                + "-" * 80
            )

            print(
                f"[{completed}/{expected_runs}] "
                f"{question_id} | K={k}"
            )

            print(
                "-" * 80
            )

            # =================================================================
            # RETRIEVAL
            # =================================================================

            retrieval_start = (
                time.perf_counter()
            )

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

            # =================================================================
            # BUILD CONTEXT
            # =================================================================

            context = build_context(
                results
            )

            documents = results[
                "documents"
            ][0]

            metadatas = results[
                "metadatas"
            ][0]

            # =================================================================
            # SOURCE INFORMATION
            # =================================================================

            sources = []

            for metadata in metadatas:

                sources.append(
                    {
                        "document":
                            metadata.get(
                                "document"
                            ),

                        "page":
                            metadata.get(
                                "page"
                            ),
                    }
                )

            # =================================================================
            # GENERATION
            # =================================================================

            print(
                "Generating answer..."
            )

            generation = (
                generate_answer_with_usage(
                    question,
                    context
                )
            )

            # =================================================================
            # BUILD RECORD
            # =================================================================

            record = {

                # -------------------------------------------------------------
                # Question metadata
                # -------------------------------------------------------------

                "ID":
                    question_id,

                "Question":
                    question,

                "Primary_Category":
                    row[
                        "Primary_Category"
                    ],

                "Secondary_Category":
                    row[
                        "Secondary_Category"
                    ],

                "Complexity":
                    row[
                        "Complexity"
                    ],

                "Answerable":
                    row[
                        "Answerable"
                    ],

                "Reference_Answer":
                    row[
                        "Reference_Answer"
                    ],

                "Source_Document":
                    row[
                        "Source_Document"
                    ],

                "Source_Page":
                    row[
                        "Source_Page"
                    ],

                "Question_Rationale":
                    row[
                        "Question_Rationale"
                    ],

                # -------------------------------------------------------------
                # Retrieval
                # -------------------------------------------------------------

                "K":
                    k,

                "Num_Retrieved_Chunks":
                    len(documents),

                "Retrieved_Sources":
                    sources,

                "Retrieved_Context":
                    context,

                # -------------------------------------------------------------
                # Generation
                # -------------------------------------------------------------

                "Generated_Answer":
                    generation[
                        "answer"
                    ],

                # -------------------------------------------------------------
                # Token usage
                # -------------------------------------------------------------

                "Prompt_Tokens":
                    generation[
                        "prompt_tokens"
                    ],

                "Generated_Tokens":
                    generation[
                        "generated_tokens"
                    ],

                "Total_Tokens":
                    generation[
                        "total_tokens"
                    ],

                # -------------------------------------------------------------
                # Timing
                # -------------------------------------------------------------

                "Retrieval_Time_Seconds":
                    retrieval_time,

                "Generation_Time_Seconds":
                    generation[
                        "generation_time_seconds"
                    ],

                # -------------------------------------------------------------
                # Raw Ollama timing
                # -------------------------------------------------------------

                "Total_Duration_NS":
                    generation[
                        "total_duration_ns"
                    ],

                "Load_Duration_NS":
                    generation[
                        "load_duration_ns"
                    ],

                "Prompt_Eval_Duration_NS":
                    generation[
                        "prompt_eval_duration_ns"
                    ],

                "Eval_Duration_NS":
                    generation[
                        "eval_duration_ns"
                    ],

                # -------------------------------------------------------------
                # Experiment metadata
                # -------------------------------------------------------------

                "Model":
                    LLM_MODEL_NAME,

                "Timestamp":
                    datetime.now().isoformat(),
            }

            # =================================================================
            # STORE RECORD
            # =================================================================

            all_results.append(
                record
            )

            # =================================================================
            # DISPLAY RESULT
            # =================================================================

            print(
                f"Retrieved chunks: "
                f"{len(documents)}"
            )

            print(
                f"Prompt tokens: "
                f"{generation['prompt_tokens']}"
            )

            print(
                f"Generated tokens: "
                f"{generation['generated_tokens']}"
            )

            print(
                f"Total tokens: "
                f"{generation['total_tokens']}"
            )

            print(
                f"Retrieval time: "
                f"{retrieval_time:.2f}s"
            )

            print(
                f"Generation time: "
                f"{generation['generation_time_seconds']:.2f}s"
            )

            print(
                "Answer:"
            )

            print(
                generation[
                    "answer"
                ].strip()
            )

            # =================================================================
            # SAVE AFTER EVERY RUN
            #
            # This protects us from losing results if:
            #
            # - laptop crashes
            # - Ollama crashes
            # - terminal is closed
            # - power failure occurs
            #
            # =================================================================

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
                f"\nProgress saved: "
                f"{completed}/{expected_runs}"
            )

    # =========================================================================
    # EXPERIMENT COMPLETE
    # =========================================================================

    experiment_time = (
        time.perf_counter()
        - experiment_start
    )

    # =========================================================================
    # SAVE CSV
    # =========================================================================

    print(
        "\nSaving CSV..."
    )

    csv_rows = []

    for record in all_results:

        csv_record = record.copy()

        csv_record[
            "Retrieved_Sources"
        ] = json.dumps(
            csv_record[
                "Retrieved_Sources"
            ],
            ensure_ascii=False
        )

        csv_rows.append(
            csv_record
        )

    if csv_rows:

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

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print("\n" + "=" * 80)
    print("FULL K EXPERIMENT COMPLETE")
    print("=" * 80)

    print(
        f"\nQuestions: "
        f"{len(questions_df)}"
    )

    print(
        f"Expected runs: "
        f"{expected_runs}"
    )

    print(
        f"Completed runs: "
        f"{len(all_results)}"
    )

    print(
        f"Total experiment time: "
        f"{experiment_time / 60:.2f} minutes"
    )

    print(
        "\nJSON output:"
    )

    print(
        JSON_OUTPUT
    )

    print(
        "\nCSV output:"
    )

    print(
        CSV_OUTPUT
    )

    print(
        "\nThe raw K experiment is complete."
    )

    print(
        "Do NOT determine K* manually yet."
    )

    print(
        "Next stage: audit the raw results and "
        "then perform RAGAS evaluation."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    run_experiment()