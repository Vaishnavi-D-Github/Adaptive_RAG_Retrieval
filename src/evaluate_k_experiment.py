import csv
import json
import os
import statistics

import tiktoken


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_FILE = "results/k_experiment_results.json"

SUMMARY_DIR = "results"
SUMMARY_FILE = "results/k_experiment_summary.csv"

# GPT-style tokenizer used only for a consistent
# approximate token-count measurement.
#
# This is NOT claimed to be the exact Llama 3.2 tokenizer.
TOKENIZER_NAME = "cl100k_base"


# ============================================================
# LOAD TOKENIZER
# ============================================================

def load_tokenizer():

    print("=" * 80)
    print("LOADING TOKENIZER")
    print("=" * 80)

    tokenizer = tiktoken.get_encoding(
        TOKENIZER_NAME
    )

    print(
        f"Tokenizer: {TOKENIZER_NAME}"
    )

    print(
        "Note: token counts are approximate because "
        "this tokenizer is not Llama 3.2's native tokenizer."
    )

    return tokenizer


# ============================================================
# LOAD EXPERIMENT RESULTS
# ============================================================

def load_results():

    print("\n" + "=" * 80)
    print("LOADING EXPERIMENT RESULTS")
    print("=" * 80)

    if not os.path.exists(RESULTS_FILE):

        raise FileNotFoundError(
            f"Could not find: {RESULTS_FILE}"
        )

    with open(
        RESULTS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        results = json.load(file)

    print(
        f"Records loaded: {len(results)}"
    )

    return results


# ============================================================
# TOKEN COUNT
# ============================================================

def count_tokens(
    text,
    tokenizer
):

    if not text:

        return 0

    return len(
        tokenizer.encode(text)
    )


# ============================================================
# ABSTENTION DETECTION
# ============================================================

def detect_abstention(answer):

    if not answer:

        return False

    answer_lower = answer.lower().strip()

    abstention_phrases = [
        "the available documents do not contain enough information",
        "do not contain enough information",
        "not enough information to answer",
        "insufficient information",
        "cannot be answered from the available documents",
        "cannot answer this question from the available documents",
        "unable to answer from the provided context",
        "not enough evidence"
    ]

    for phrase in abstention_phrases:

        if phrase in answer_lower:

            return True

    return False


# ============================================================
# EXTRACT DISTANCE INFORMATION
# ============================================================

def extract_distance_information(
    retrieved_chunks
):

    distances = []

    for chunk in retrieved_chunks:

        distance = chunk.get(
            "distance"
        )

        if isinstance(
            distance,
            (int, float)
        ):

            distances.append(
                distance
            )

    if not distances:

        return {
            "minimum_distance": None,
            "maximum_distance": None,
            "average_distance": None
        }

    return {
        "minimum_distance": min(
            distances
        ),

        "maximum_distance": max(
            distances
        ),

        "average_distance": statistics.mean(
            distances
        )
    }


# ============================================================
# CREATE SUMMARY RECORD
# ============================================================

def create_summary_record(
    result,
    tokenizer
):

    context = result.get(
        "context",
        ""
    )

    generated_answer = result.get(
        "generated_answer",
        ""
    )

    reference_answer = result.get(
        "reference_answer",
        ""
    )

    retrieved_chunks = result.get(
        "retrieved_chunks",
        []
    )

    # --------------------------------------------------------
    # Context measurements
    # --------------------------------------------------------

    context_characters = len(
        context
    )

    context_tokens = count_tokens(
        context,
        tokenizer
    )

    # --------------------------------------------------------
    # Answer measurements
    # --------------------------------------------------------

    answer_tokens = count_tokens(
        generated_answer,
        tokenizer
    )

    reference_tokens = count_tokens(
        reference_answer,
        tokenizer
    )

    # --------------------------------------------------------
    # Retrieval measurements
    # --------------------------------------------------------

    number_of_chunks = len(
        retrieved_chunks
    )

    distance_info = (
        extract_distance_information(
            retrieved_chunks
        )
    )

    # --------------------------------------------------------
    # Abstention
    # --------------------------------------------------------

    abstention = detect_abstention(
        generated_answer
    )

    # --------------------------------------------------------
    # Error information
    # --------------------------------------------------------

    error = result.get(
        "error"
    )

    # --------------------------------------------------------
    # Create record
    # --------------------------------------------------------

    summary = {

        "question_id": result.get(
            "question_id"
        ),

        "question": result.get(
            "question"
        ),

        "primary_category": result.get(
            "primary_category"
        ),

        "complexity": result.get(
            "complexity"
        ),

        "answerable": result.get(
            "answerable"
        ),

        "k": result.get(
            "k"
        ),

        "reference_answer": reference_answer,

        "generated_answer": generated_answer,

        "abstention_detected": abstention,

        "number_of_retrieved_chunks": (
            number_of_chunks
        ),

        "context_characters": (
            context_characters
        ),

        "estimated_context_tokens": (
            context_tokens
        ),

        "estimated_answer_tokens": (
            answer_tokens
        ),

        "estimated_reference_tokens": (
            reference_tokens
        ),

        "minimum_distance": (
            distance_info[
                "minimum_distance"
            ]
        ),

        "maximum_distance": (
            distance_info[
                "maximum_distance"
            ]
        ),

        "average_distance": (
            distance_info[
                "average_distance"
            ]
        ),

        "execution_time_seconds": result.get(
            "execution_time_seconds"
        ),

        "source_document": result.get(
            "source_document"
        ),

        "source_page": result.get(
            "source_page"
        ),

        "error": error
    }

    return summary


# ============================================================
# SAVE CSV
# ============================================================

def save_summary(
    summary_records
):

    os.makedirs(
        SUMMARY_DIR,
        exist_ok=True
    )

    if not summary_records:

        print(
            "No summary records to save."
        )

        return

    fieldnames = list(
        summary_records[0].keys()
    )

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            summary_records
        )

    print("\n" + "=" * 80)

    print(
        "SUMMARY FILE CREATED"
    )

    print(
        f"File: {SUMMARY_FILE}"
    )

    print(
        f"Records: "
        f"{len(summary_records)}"
    )

    print("=" * 80)


# ============================================================
# GROUP RESULTS BY QUESTION
# ============================================================

def group_by_question(
    summary_records
):

    grouped = {}

    for record in summary_records:

        question_id = record[
            "question_id"
        ]

        if question_id not in grouped:

            grouped[question_id] = []

        grouped[question_id].append(
            record
        )

    return grouped


# ============================================================
# PRINT QUESTION-LEVEL SUMMARY
# ============================================================

def print_question_summary(
    summary_records
):

    print("\n" + "=" * 80)

    print(
        "QUESTION-LEVEL K SUMMARY"
    )

    print("=" * 80)

    grouped = group_by_question(
        summary_records
    )

    for question_id in sorted(
        grouped.keys()
    ):

        records = sorted(
            grouped[question_id],
            key=lambda x: x["k"]
        )

        first = records[0]

        print("\n" + "-" * 80)

        print(
            f"Question: {question_id}"
        )

        print(
            f"Category: "
            f"{first['primary_category']}"
        )

        print(
            f"Complexity: "
            f"{first['complexity']}"
        )

        print(
            f"Answerable: "
            f"{first['answerable']}"
        )

        print(
            "\nK | Context Tokens | "
            "Chunks | Abstain | Time(s)"
        )

        print(
            "-" * 60
        )

        for record in records:

            print(
                f"{record['k']:2} | "
                f"{record['estimated_context_tokens']:14} | "
                f"{record['number_of_retrieved_chunks']:6} | "
                f"{str(record['abstention_detected']):7} | "
                f"{record['execution_time_seconds']}"
            )


# ============================================================
# PRINT GLOBAL SUMMARY
# ============================================================

def print_global_summary(
    summary_records
):

    print("\n" + "=" * 80)

    print(
        "GLOBAL EXPERIMENT SUMMARY"
    )

    print("=" * 80)

    total = len(
        summary_records
    )

    print(
        f"Total records: {total}"
    )

    # --------------------------------------------------------
    # Successful records
    # --------------------------------------------------------

    successful = [
        r
        for r in summary_records
        if not r.get("error")
    ]

    failed = [
        r
        for r in summary_records
        if r.get("error")
    ]

    print(
        f"Successful records: "
        f"{len(successful)}"
    )

    print(
        f"Failed records: "
        f"{len(failed)}"
    )

    # --------------------------------------------------------
    # Context tokens
    # --------------------------------------------------------

    token_values = [
        r["estimated_context_tokens"]
        for r in successful
        if isinstance(
            r["estimated_context_tokens"],
            int
        )
    ]

    if token_values:

        print(
            "\nEstimated context tokens:"
        )

        print(
            f"Minimum: "
            f"{min(token_values)}"
        )

        print(
            f"Maximum: "
            f"{max(token_values)}"
        )

        print(
            f"Average: "
            f"{statistics.mean(token_values):.2f}"
        )

    # --------------------------------------------------------
    # Execution time
    # --------------------------------------------------------

    time_values = [
        r["execution_time_seconds"]
        for r in successful
        if isinstance(
            r["execution_time_seconds"],
            (int, float)
        )
    ]

    if time_values:

        print(
            "\nExecution time:"
        )

        print(
            f"Minimum: "
            f"{min(time_values):.2f} seconds"
        )

        print(
            f"Maximum: "
            f"{max(time_values):.2f} seconds"
        )

        print(
            f"Average: "
            f"{statistics.mean(time_values):.2f} seconds"
        )

    # --------------------------------------------------------
    # Abstention
    # --------------------------------------------------------

    abstentions = [
        r
        for r in successful
        if r["abstention_detected"]
    ]

    print(
        "\nDetected abstentions: "
        f"{len(abstentions)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("K EXPERIMENT EVALUATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    tokenizer = load_tokenizer()

    # --------------------------------------------------------
    # Load results
    # --------------------------------------------------------

    results = load_results()

    if not results:

        print(
            "No experiment results found."
        )

        return

    # --------------------------------------------------------
    # Process results
    # --------------------------------------------------------

    summary_records = []

    print("\n" + "=" * 80)

    print(
        "CALCULATING EXPERIMENT METRICS"
    )

    print("=" * 80)

    for index, result in enumerate(
        results,
        start=1
    ):

        summary = create_summary_record(
            result,
            tokenizer
        )

        summary_records.append(
            summary
        )

        if index % 10 == 0:

            print(
                f"Processed "
                f"{index}/{len(results)}"
            )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    save_summary(
        summary_records
    )

    # --------------------------------------------------------
    # Print summaries
    # --------------------------------------------------------

    print_question_summary(
        summary_records
    )

    print_global_summary(
        summary_records
    )

    print("\n" + "=" * 80)

    print(
        "EVALUATION COMPLETE"
    )

    print("=" * 80)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()