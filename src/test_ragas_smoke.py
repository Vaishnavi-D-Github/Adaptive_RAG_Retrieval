# ================================================================
# test_ragas_smoke.py
#
# RAGAS FAITHFULNESS SMOKE TEST
#
# Purpose:
#   Evaluate ONE complete question across K=1..10.
#
# This is ONLY a smoke test.
# It does NOT run the full 959-record evaluation.
#
# Input:
#   results/ragas/ragas_evaluation_dataset.csv
#
# Output:
#   results/ragas/smoke_test/
#       faithfulness_smoke_test.csv
#       ragas_raw_smoke_result.csv
#
# RAGAS:
#   0.3.9
#
# Evaluator:
#   Ollama llama3.2:3b
# ================================================================


# ================================================================
# IMPORTS
# ================================================================

import os
import ast

import pandas as pd

from ragas import evaluate, EvaluationDataset

from ragas.metrics import faithfulness

from langchain_ollama import ChatOllama

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)


# ================================================================
# CONFIGURATION
# ================================================================

INPUT_FILE = (
    "results/ragas/ragas_evaluation_dataset.csv"
)

OUTPUT_DIR = (
    "results/ragas/smoke_test"
)

RAW_OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "ragas_raw_smoke_result.csv"
)

FINAL_OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "faithfulness_smoke_test.csv"
)

EVALUATOR_MODEL = "qwen3:8b"

OLLAMA_BASE_URL = (
    "http://localhost:11434"
)

SMOKE_TEST_SIZE = 10


# ================================================================
# CREATE OUTPUT DIRECTORY
# ================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# HEADER
# ================================================================

print()
print("=" * 80)
print("RAGAS FAITHFULNESS SMOKE TEST")
print("=" * 80)


# ================================================================
# CHECK INPUT FILE
# ================================================================

print()
print("=" * 80)
print("CHECKING INPUT DATASET")
print("=" * 80)

print()
print(
    "Input file:"
)

print(
    INPUT_FILE
)

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"""
Input dataset not found:

{INPUT_FILE}

Make sure you are running this command
from the project root directory.
"""
    )


# ================================================================
# LOAD DATASET
# ================================================================

print()
print(
    "Loading dataset..."
)

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8"
)

print(
    "Total available records:",
    len(df)
)


# ================================================================
# DISPLAY COLUMNS
# ================================================================

print()
print(
    "Available columns:"
)

for column in df.columns:

    print(
        " -",
        column
    )


# ================================================================
# REQUIRED COLUMN VALIDATION
# ================================================================

required_columns = [
    "user_input",
    "retrieved_contexts",
    "response",
    "reference",
    "ID",
    "K",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print()
    print("=" * 80)
    print("ERROR: REQUIRED COLUMNS ARE MISSING")
    print("=" * 80)

    for column in missing_columns:

        print(
            "Missing:",
            column
        )

    raise ValueError(
        "The RAGAS dataset does not contain "
        "the required columns."
    )

print()
print(
    "RAGAS column validation: PASS"
)


# ================================================================
# BASIC CLEANING
# ================================================================

df["ID"] = (
    df["ID"]
    .astype(str)
    .str.strip()
)

df["K"] = pd.to_numeric(
    df["K"],
    errors="coerce"
)

df["user_input"] = (
    df["user_input"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["response"] = (
    df["response"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["reference"] = (
    df["reference"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ================================================================
# FIND A COMPLETE QUESTION
# ================================================================

print()
print("=" * 80)
print("FINDING COMPLETE K=1..10 QUESTION")
print("=" * 80)

question_counts = (
    df
    .groupby("ID")["K"]
    .nunique()
    .sort_values(
        ascending=False
    )
)

question_id = (
    question_counts.index[0]
)

available_k_count = (
    question_counts.iloc[0]
)

print()
print(
    "Selected question:",
    question_id
)

print(
    "Number of K values:",
    available_k_count
)


# ================================================================
# EXTRACT QUESTION
# ================================================================

smoke_df = (
    df[
        df["ID"]
        ==
        question_id
    ]
    .sort_values("K")
    .copy()
)


# ================================================================
# SELECT K=1..10
# ================================================================

smoke_df = smoke_df[
    smoke_df["K"].between(
        1,
        10
    )
].copy()

smoke_df = (
    smoke_df
    .drop_duplicates(
        subset=["ID", "K"],
        keep="first"
    )
    .head(
        SMOKE_TEST_SIZE
    )
)


# ================================================================
# DISPLAY SELECTION
# ================================================================

print()
print(
    "Smoke-test records:",
    len(smoke_df)
)

actual_k = (
    smoke_df["K"]
    .astype(int)
    .tolist()
)

print(
    "K values:",
    actual_k
)


# ================================================================
# VERIFY K COVERAGE
# ================================================================

expected_k = list(
    range(
        1,
        SMOKE_TEST_SIZE + 1
    )
)

if actual_k != expected_k:

    print()
    print(
        "Expected:",
        expected_k
    )

    print(
        "Actual:",
        actual_k
    )

    raise ValueError(
        "The selected question does not have "
        "complete K=1..10 coverage."
    )

print()
print(
    "K=1 through K=10 coverage: PASS"
)


# ================================================================
# PARSE RETRIEVED CONTEXTS
# ================================================================

def parse_contexts(value):

    """
    Convert the CSV representation of
    retrieved_contexts into a Python list.
    """

    if pd.isna(value):

        return []

    text = str(
        value
    ).strip()

    if not text:

        return []

    # ------------------------------------------------------------
    # First attempt: Python literal representation
    # ------------------------------------------------------------

    try:

        parsed = ast.literal_eval(
            text
        )

        if isinstance(
            parsed,
            list
        ):

            return [
                str(item)
                for item in parsed
            ]

    except Exception:

        pass

    # ------------------------------------------------------------
    # Fallback: treat entire value as one context
    # ------------------------------------------------------------

    return [
        text
    ]


# ================================================================
# BUILD RAGAS RECORDS
# ================================================================

print()
print("=" * 80)
print("PREPARING RAGAS RECORDS")
print("=" * 80)

ragas_rows = []


for _, row in smoke_df.iterrows():

    question = (
        row["user_input"]
    )

    answer = (
        row["response"]
    )

    reference = (
        row["reference"]
    )

    contexts = parse_contexts(
        row["retrieved_contexts"]
    )

    # ------------------------------------------------------------
    # Validate question
    # ------------------------------------------------------------

    if not question:

        raise ValueError(
            f"Empty question: "
            f"{row['ID']} K={row['K']}"
        )

    # ------------------------------------------------------------
    # Validate answer
    # ------------------------------------------------------------

    if not answer:

        raise ValueError(
            f"Empty answer: "
            f"{row['ID']} K={row['K']}"
        )

    # ------------------------------------------------------------
    # Validate reference
    # ------------------------------------------------------------

    if not reference:

        raise ValueError(
            f"Empty reference: "
            f"{row['ID']} K={row['K']}"
        )

    # ------------------------------------------------------------
    # Validate retrieved context
    # ------------------------------------------------------------

    if not contexts:

        raise ValueError(
            f"No retrieved contexts: "
            f"{row['ID']} K={row['K']}"
        )

    # ------------------------------------------------------------
    # Add RAGAS record
    # ------------------------------------------------------------

    ragas_rows.append(
        {
            "user_input":
                question,

            "retrieved_contexts":
                contexts,

            "response":
                answer,

            "reference":
                reference,
        }
    )


print()
print(
    "RAGAS records prepared:",
    len(ragas_rows)
)


# ================================================================
# PREVIEW
# ================================================================

print()
print("=" * 80)
print("FIRST RECORD PREVIEW")
print("=" * 80)

first = ragas_rows[0]

print()
print(
    "Question:"
)

print(
    first["user_input"]
)

print()
print(
    "Number of retrieved contexts:",
    len(
        first["retrieved_contexts"]
    )
)

print()
print(
    "Answer:"
)

print(
    first["response"][:500]
)

print()
print(
    "Reference:"
)

print(
    first["reference"][:500]
)


# ================================================================
# CREATE RAGAS EvaluationDataset
# ================================================================

print()
print("=" * 80)
print("CREATING RAGAS EVALUATION DATASET")
print("=" * 80)

evaluation_dataset = (
    EvaluationDataset.from_list(
        ragas_rows
    )
)

print(
    "EvaluationDataset created successfully."
)


# ================================================================
# CONNECT TO OLLAMA
# ================================================================

print()
print("=" * 80)
print("CONNECTING TO OLLAMA")
print("=" * 80)

print()
print(
    "Ollama URL:",
    OLLAMA_BASE_URL
)

print(
    "Evaluator model:",
    EVALUATOR_MODEL
)

evaluator_llm = ChatOllama(
    model=EVALUATOR_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    think=False,
)

print()
print(
    "Ollama evaluator created successfully."
)


# ================================================================
# LOAD EMBEDDING MODEL
# ================================================================

print()
print("=" * 80)
print("LOADING EMBEDDING MODEL")
print("=" * 80)

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

print()
print(
    "Embedding model:",
    EMBEDDING_MODEL
)

embedding_model = (
    HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
)

print()
print(
    "Embedding model loaded successfully."
)


# ================================================================
# RUN RAGAS
# ================================================================

print()
print("=" * 80)
print("STARTING RAGAS FAITHFULNESS EVALUATION")
print("=" * 80)

print()
print(
    "Records:",
    len(ragas_rows)
)

print(
    "Question:",
    question_id
)

print(
    "K values:",
    actual_k
)

print()
print(
    "Metric:"
)

print(
    " - Faithfulness"
)

print()
print(
    "Please wait..."
)


try:

    result = evaluate(

        dataset=evaluation_dataset,

        metrics=[
            faithfulness
        ],

        llm=evaluator_llm,

        embeddings=embedding_model,
    )


except Exception as e:

    print()
    print("=" * 80)
    print("RAGAS EVALUATION FAILED")
    print("=" * 80)

    print()
    print(
        "Error type:",
        type(e).__name__
    )

    print()
    print(
        "Error:"
    )

    print(
        str(e)
    )

    print()

    raise


# ================================================================
# CONVERT RESULT
# ================================================================

print()
print("=" * 80)
print("RAGAS EVALUATION FINISHED")
print("=" * 80)

result_df = (
    result
    .to_pandas()
)


# ================================================================
# SAVE RAW RESULT IMMEDIATELY
# ================================================================
#
# IMPORTANT:
# Save BEFORE metadata processing.
#
# This prevents a pandas/column error from causing us
# to lose the expensive RAGAS result.
# ================================================================

print()
print(
    "Saving raw RAGAS result..."
)

result_df.to_csv(
    RAW_OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(
    "Raw result saved:"
)

print(
    RAW_OUTPUT_FILE
)


# ================================================================
# ATTACH EXPERIMENT METADATA
# ================================================================

print()
print("=" * 80)
print("ATTACHING EXPERIMENT METADATA")
print("=" * 80)

metadata_columns = [
    "ID",
    "K",
    "user_input",
    "Model",
    "Prompt_Tokens",
    "Generated_Tokens",
    "Total_Tokens",
    "Retrieval_Time_Seconds",
    "Generation_Time_Seconds",
]

available_metadata_columns = [
    column
    for column in metadata_columns
    if column in smoke_df.columns
]

metadata = (
    smoke_df[
        available_metadata_columns
    ]
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# Rename user_input to Question for readability
# ------------------------------------------------------------

metadata = metadata.rename(
    columns={
        "user_input":
            "Question"
    }
)


# ================================================================
# COMBINE METADATA + RAGAS RESULT
# ================================================================

result_df = pd.concat(
    [
        metadata,
        result_df.reset_index(drop=True),
    ],
    axis=1
)


# ================================================================
# DISPLAY RESULTS
# ================================================================

print()
print("=" * 80)
print("FAITHFULNESS RESULTS")
print("=" * 80)

print()

print(
    result_df.to_string(
        index=False
    )
)


# ================================================================
# METRIC SUMMARY
# ================================================================

print()
print("=" * 80)
print("FAITHFULNESS SUMMARY")
print("=" * 80)

if "faithfulness" in result_df.columns:

    values = pd.to_numeric(
        result_df["faithfulness"],
        errors="coerce"
    )

    print()
    print(
        "Valid scores:",
        values.notna().sum(),
        "/",
        len(values)
    )

    print(
        "Mean:",
        f"{values.mean():.4f}"
    )

    print(
        "Minimum:",
        f"{values.min():.4f}"
    )

    print(
        "Maximum:",
        f"{values.max():.4f}"
    )

    print()
    print(
        "Faithfulness by K:"
    )

    k_summary = (
        result_df
        .assign(
            faithfulness_numeric=values
        )
        .groupby("K")
        ["faithfulness_numeric"]
        .mean()
        .reset_index()
    )

    print(
        k_summary.to_string(
            index=False
        )
    )

else:

    print(
        "ERROR: faithfulness column "
        "was not produced."
    )


# ================================================================
# SAVE FINAL RESULT
# ================================================================

print()
print("=" * 80)
print("SAVING FINAL SMOKE-TEST RESULT")
print("=" * 80)

result_df.to_csv(
    FINAL_OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print()
print(
    "Final result saved:"
)

print(
    FINAL_OUTPUT_FILE
)


# ================================================================
# FINAL VALIDATION
# ================================================================

print()
print("=" * 80)
print("FINAL VALIDATION")
print("=" * 80)

print()

print(
    "Expected records:",
    SMOKE_TEST_SIZE
)

print(
    "Actual records:",
    len(result_df)
)

if len(result_df) == SMOKE_TEST_SIZE:

    print(
        "PASS: 10 records evaluated."
    )

else:

    print(
        "WARNING: Record count mismatch."
    )


if "faithfulness" in result_df.columns:

    valid_scores = pd.to_numeric(
        result_df["faithfulness"],
        errors="coerce"
    ).notna().sum()

    print()
    print(
        "Valid faithfulness scores:",
        valid_scores,
        "/",
        SMOKE_TEST_SIZE
    )

    if valid_scores == SMOKE_TEST_SIZE:

        print(
            "PASS: All 10 records have "
            "faithfulness scores."
        )

    else:

        print(
            "WARNING: Some faithfulness scores "
            "are missing."
        )


# ================================================================
# COMPLETE
# ================================================================

print()
print("=" * 80)
print("RAGAS FAITHFULNESS SMOKE TEST COMPLETE")
print("=" * 80)

print()
print(
    "Raw result:"
)

print(
    RAW_OUTPUT_FILE
)

print()
print(
    "Final result:"
)

print(
    FINAL_OUTPUT_FILE
)

print()
print(
    "DO NOT START THE FULL 959-RECORD EVALUATION YET."
)

print(
    "Review the faithfulness scores first."
)

print()