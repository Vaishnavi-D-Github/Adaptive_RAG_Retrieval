# ================================================================
# prepare_ragas_dataset.py
#
# Prepare the final K experiment for RAGAS evaluation.
#
# INPUT:
#   results/final_k_experiment_qwen3.csv
#
# OUTPUT:
#   results/ragas/
#       ragas_evaluation_dataset.csv
#       ragas_evaluation_dataset.json
#       ragas_excluded_failures.csv
#       ragas_truncated_candidates.csv
#       ragas_dataset_summary.txt
#
# IMPORTANT:
#   - Does NOT modify the original experiment.
#   - Does NOT run RAGAS.
#   - Failed generations are excluded from the primary RAGAS set.
#   - Potentially truncated generations are retained in the primary
#     set but separately identified for sensitivity analysis.
# ================================================================

import os
import json
import pandas as pd


# ================================================================
# CONFIGURATION
# ================================================================

INPUT_FILE = (
    "results/final_k_experiment_qwen3.csv"
)

OUTPUT_DIR = (
    "results/ragas"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

EXPECTED_RECORDS = 1000
EXPECTED_QUESTIONS = 100
EXPECTED_K_VALUES = list(range(1, 11))

MAX_NEW_TOKENS = 256


# ================================================================
# LOAD EXPERIMENT
# ================================================================

print("=" * 80)
print("PREPARING RAGAS DATASET")
print("=" * 80)

print()
print("Input:")
print(INPUT_FILE)

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8"
)

print(
    "Records loaded:",
    len(df)
)


# ================================================================
# REQUIRED COLUMN CHECK
# ================================================================

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
    "K",
    "Num_Retrieved_Chunks",
    "Retrieved_Sources",
    "Retrieved_Context",
    "Generated_Answer",
    "Prompt_Tokens",
    "Generated_Tokens",
    "Total_Tokens",
    "Retrieval_Time_Seconds",
    "Generation_Time_Seconds",
    "Model",
    "Timestamp"
]

missing_columns = [
    c
    for c in required_columns
    if c not in df.columns
]

if missing_columns:

    print()
    print("ERROR: Missing columns:")

    for c in missing_columns:
        print("-", c)

    raise ValueError(
        "Input experiment file does not have the expected structure."
    )

print(
    "Column validation: PASS"
)


# ================================================================
# BASIC STRUCTURE VALIDATION
# ================================================================

print()
print("=" * 80)
print("STRUCTURE VALIDATION")
print("=" * 80)

if len(df) != EXPECTED_RECORDS:

    raise ValueError(
        f"Expected {EXPECTED_RECORDS} records, "
        f"found {len(df)}."
    )

print(
    "PASS: 1,000 records found."
)

unique_questions = df["ID"].nunique()

print(
    "Unique questions:",
    unique_questions
)

if unique_questions != EXPECTED_QUESTIONS:

    raise ValueError(
        f"Expected {EXPECTED_QUESTIONS} questions, "
        f"found {unique_questions}."
    )

print(
    "PASS: 100 unique questions found."
)

observed_k = sorted(
    pd.to_numeric(
        df["K"],
        errors="coerce"
    )
    .dropna()
    .astype(int)
    .unique()
)

print(
    "K values:",
    observed_k
)

if observed_k != EXPECTED_K_VALUES:

    raise ValueError(
        "K=1 through K=10 are not all present."
    )

print(
    "PASS: K=1 through K=10 present."
)


# ================================================================
# NUMERIC CLEANUP
# ================================================================

numeric_columns = [
    "K",
    "Num_Retrieved_Chunks",
    "Prompt_Tokens",
    "Generated_Tokens",
    "Total_Tokens",
    "Retrieval_Time_Seconds",
    "Generation_Time_Seconds"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ================================================================
# GENERATION FAILURE IDENTIFICATION
# ================================================================

print()
print("=" * 80)
print("IDENTIFYING GENERATION FAILURES")
print("=" * 80)

df["Generated_Answer_Clean"] = (
    df["Generated_Answer"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["Generation_Failed"] = (
    df["Generated_Answer_Clean"].eq("")
    &
    (
        (df["Prompt_Tokens"] == 0)
        |
        (df["Generation_Time_Seconds"] == 0)
    )
)

df["Generation_Succeeded"] = (
    ~df["Generation_Failed"]
)

failure_count = int(
    df["Generation_Failed"].sum()
)

success_count = int(
    df["Generation_Succeeded"].sum()
)

print(
    "Successful generations:",
    success_count
)

print(
    "Failed generations:",
    failure_count
)

print(
    "Failure rate:",
    f"{100 * failure_count / len(df):.2f}%"
)


# ================================================================
# TRUNCATION IDENTIFICATION
# ================================================================

print()
print("=" * 80)
print("IDENTIFYING POTENTIALLY TRUNCATED ANSWERS")
print("=" * 80)

df["Potentially_Truncated"] = (
    df["Generated_Tokens"]
    == MAX_NEW_TOKENS
)

truncated_count = int(
    df["Potentially_Truncated"].sum()
)

print(
    "Potentially truncated:",
    truncated_count
)


# ================================================================
# PRIMARY RAGAS DATASET
# ================================================================
#
# We exclude actual generation failures.
#
# We DO NOT exclude potentially truncated answers here.
#
# Instead, truncation is explicitly marked so that we can perform
# a sensitivity analysis later.
# ================================================================

ragas_df = df[
    df["Generation_Succeeded"]
].copy()

print()
print("=" * 80)
print("PRIMARY RAGAS DATASET")
print("=" * 80)

print(
    "Records:",
    len(ragas_df)
)

print(
    "Expected:",
    success_count
)

print(
    "Potentially truncated records:",
    ragas_df[
        "Potentially_Truncated"
    ].sum()
)


# ================================================================
# CREATE RAGAS-SPECIFIC COLUMNS
# ================================================================

ragas_df["user_input"] = (
    ragas_df["Question"]
)

ragas_df["retrieved_contexts"] = (
    ragas_df["Retrieved_Context"]
    .fillna("")
    .astype(str)
    .apply(
        lambda x:
            [x]
            if x.strip()
            else []
    )
)

ragas_df["response"] = (
    ragas_df["Generated_Answer_Clean"]
)

ragas_df["reference"] = (
    ragas_df["Reference_Answer"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ================================================================
# CHECK REQUIRED RAGAS FIELDS
# ================================================================

print()
print("=" * 80)
print("RAGAS FIELD VALIDATION")
print("=" * 80)

print(
    "Empty user inputs:",
    (
        ragas_df["user_input"]
        .str.strip()
        .eq("")
        .sum()
    )
)

print(
    "Empty retrieved contexts:",
    ragas_df["retrieved_contexts"]
    .apply(len)
    .eq(0)
    .sum()
)

print(
    "Empty responses:",
    ragas_df["response"]
    .str.strip()
    .eq("")
    .sum()
)

print(
    "Empty references:",
    ragas_df["reference"]
    .str.strip()
    .eq("")
    .sum()
)


# ================================================================
# CHECK WHETHER REFERENCE ANSWERS ARE AVAILABLE
# ================================================================

reference_missing = (
    ragas_df["reference"]
    .str.strip()
    .eq("")
)

if reference_missing.any():

    print()
    print(
        "WARNING:",
        reference_missing.sum(),
        "records have no reference answer."
    )

    print(
        "These records may not be suitable for metrics "
        "requiring a reference answer."
    )

else:

    print(
        "PASS: Reference answers available."
    )


# ================================================================
# EXCLUDED FAILURE DATASET
# ================================================================

print()
print("=" * 80)
print("CREATING FAILURE DATASET")
print("=" * 80)

failures_df = df[
    df["Generation_Failed"]
].copy()

failure_columns = [
    "ID",
    "Question",
    "Primary_Category",
    "Secondary_Category",
    "Complexity",
    "Answerable",
    "Reference_Answer",
    "Source_Document",
    "Source_Page",
    "K",
    "Prompt_Tokens",
    "Generated_Tokens",
    "Total_Tokens",
    "Retrieval_Time_Seconds",
    "Generation_Time_Seconds",
    "Model",
    "Timestamp"
]

failures_df = failures_df[
    failure_columns
]

print(
    "Failed records:",
    len(failures_df)
)


# ================================================================
# TRUNCATED DATASET
# ================================================================

print()
print("=" * 80)
print("CREATING TRUNCATION DATASET")
print("=" * 80)

truncated_df = df[
    df["Generation_Succeeded"]
    &
    df["Potentially_Truncated"]
].copy()

truncated_columns = [
    "ID",
    "Question",
    "K",
    "Complexity",
    "Answerable",
    "Generated_Answer",
    "Prompt_Tokens",
    "Generated_Tokens",
    "Total_Tokens",
    "Generation_Time_Seconds",
    "Source_Document",
    "Source_Page"
]

truncated_df = truncated_df[
    truncated_columns
]

print(
    "Potentially truncated records:",
    len(truncated_df)
)


# ================================================================
# SELECT FINAL CSV COLUMNS
# ================================================================

final_columns = [
    # RAGAS fields
    "user_input",
    "retrieved_contexts",
    "response",
    "reference",

    # Experiment identifiers
    "ID",
    "K",

    # Question metadata
    "Primary_Category",
    "Secondary_Category",
    "Complexity",
    "Answerable",
    "Question_Rationale",

    # Ground-truth source metadata
    "Source_Document",
    "Source_Page",

    # Original retrieval/generation information
    "Retrieved_Sources",
    "Retrieved_Context",
    "Generated_Answer",

    # Experiment measurements
    "Num_Retrieved_Chunks",
    "Prompt_Tokens",
    "Generated_Tokens",
    "Total_Tokens",
    "Retrieval_Time_Seconds",
    "Generation_Time_Seconds",

    # Audit flags
    "Generation_Succeeded",
    "Potentially_Truncated",

    # Model
    "Model",
    "Timestamp"
]

ragas_output_df = ragas_df[
    final_columns
].copy()


# ================================================================
# SAVE CSV
# ================================================================

ragas_csv = os.path.join(
    OUTPUT_DIR,
    "ragas_evaluation_dataset.csv"
)

ragas_output_df.to_csv(
    ragas_csv,
    index=False,
    encoding="utf-8-sig"
)

print()
print(
    "Saved:",
    ragas_csv
)


# ================================================================
# SAVE JSON
# ================================================================

ragas_json = os.path.join(
    OUTPUT_DIR,
    "ragas_evaluation_dataset.json"
)

json_records = []

for _, row in ragas_output_df.iterrows():

    record = row.to_dict()

    # Convert NaN to None for valid JSON
    for key, value in record.items():

        if pd.isna(value):

            record[key] = None

    json_records.append(
        record
    )

with open(
    ragas_json,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_records,
        f,
        ensure_ascii=False,
        indent=2
    )

print(
    "Saved:",
    ragas_json
)


# ================================================================
# SAVE FAILURES
# ================================================================

failure_output = os.path.join(
    OUTPUT_DIR,
    "ragas_excluded_failures.csv"
)

failures_df.to_csv(
    failure_output,
    index=False,
    encoding="utf-8-sig"
)

print(
    "Saved:",
    failure_output
)


# ================================================================
# SAVE TRUNCATED CANDIDATES
# ================================================================

truncated_output = os.path.join(
    OUTPUT_DIR,
    "ragas_truncated_candidates.csv"
)

truncated_df.to_csv(
    truncated_output,
    index=False,
    encoding="utf-8-sig"
)

print(
    "Saved:",
    truncated_output
)


# ================================================================
# K DISTRIBUTION
# ================================================================

print()
print("=" * 80)
print("RAGAS DATASET DISTRIBUTION BY K")
print("=" * 80)

k_distribution = (
    ragas_df
    .groupby("K")
    .agg(
        Total_Experiment_Runs=("ID", "count"),

        Potentially_Truncated=(
            "Potentially_Truncated",
            "sum"
        )
    )
    .reset_index()
)

print(
    k_distribution.to_string(
        index=False
    )
)


# ================================================================
# ANSWERABLE DISTRIBUTION
# ================================================================

print()
print("=" * 80)
print("RAGAS DATASET — ANSWERABILITY")
print("=" * 80)

answerability = (
    ragas_df
    .groupby("Answerable")
    .size()
    .reset_index(
        name="Records"
    )
)

print(
    answerability.to_string(
        index=False
    )
)


# ================================================================
# COMPLEXITY DISTRIBUTION
# ================================================================

print()
print("=" * 80)
print("RAGAS DATASET — COMPLEXITY")
print("=" * 80)

complexity = (
    ragas_df
    .groupby("Complexity")
    .size()
    .reset_index(
        name="Records"
    )
)

print(
    complexity.to_string(
        index=False
    )
)


# ================================================================
# CREATE SUMMARY REPORT
# ================================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "ragas_dataset_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "RAGAS DATASET PREPARATION SUMMARY\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        f"Original records: {len(df)}\n"
    )

    f.write(
        f"Unique questions: {df['ID'].nunique()}\n"
    )

    f.write(
        f"Expected records: {EXPECTED_RECORDS}\n"
    )

    f.write(
        f"Successful generations: {success_count}\n"
    )

    f.write(
        f"Failed generations: {failure_count}\n"
    )

    f.write(
        f"Potentially truncated: {truncated_count}\n"
    )

    f.write(
        f"Primary RAGAS records: {len(ragas_df)}\n"
    )

    f.write(
        "\n"
    )

    f.write(
        "K DISTRIBUTION\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        k_distribution.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "ANSWERABILITY\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        answerability.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "COMPLEXITY\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        complexity.to_string(
            index=False
        )
    )


# ================================================================
# FINAL VALIDATION
# ================================================================

print()
print("=" * 80)
print("FINAL VALIDATION")
print("=" * 80)

print(
    "Original records:",
    len(df)
)

print(
    "Primary RAGAS records:",
    len(ragas_df)
)

print(
    "Excluded failures:",
    len(failures_df)
)

print(
    "Truncated candidates:",
    len(truncated_df)
)

print(
    "Primary + failures:",
    len(ragas_df) + len(failures_df)
)

if (
    len(ragas_df)
    +
    len(failures_df)
    ==
    len(df)
):

    print(
        "PASS: Every original record accounted for."
    )

else:

    print(
        "WARNING: Record accounting mismatch."
    )


print()
print("=" * 80)
print("FILES CREATED")
print("=" * 80)

print()
print(
    "Primary RAGAS dataset:"
)
print(
    ragas_csv
)

print()
print(
    "Primary RAGAS JSON:"
)
print(
    ragas_json
)

print()
print(
    "Excluded failures:"
)
print(
    failure_output
)

print()
print(
    "Potentially truncated:"
)
print(
    truncated_output
)

print()
print(
    "Summary:"
)
print(
    summary_file
)

print()
print("=" * 80)
print("RAGAS DATASET PREPARATION COMPLETE")
print("=" * 80)

print()
print(
    "DO NOT RUN RAGAS YET."
)

print(
    "Review the summary and failure/truncation files first."
)