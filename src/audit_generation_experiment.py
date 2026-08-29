# ================================================================
# audit_generation_experiment.py
#
# Audit the final 1,000-run Qwen3-8B experiment
#
# Input:
#   results/final_k_experiment_qwen3.csv
#
# Outputs:
#   results/audit_generation/
#       generation_audit_records.csv
#       generation_audit_summary.csv
#       generation_failures.csv
#       generation_audit_report.txt
#
# Purpose:
#   1. Verify the 100 x 10 experimental structure
#   2. Identify failed generations
#   3. Separate failures from normal answers
#   4. Detect possible truncation
#   5. Analyze failures by K
#   6. Analyze latency and token usage
#   7. Analyze long-context behavior
#   8. Produce a clean audit before RAGAS
#
# IMPORTANT:
#   This script DOES NOT run RAGAS.
# ================================================================

import os
import ast
import json
import numpy as np
import pandas as pd


# ================================================================
# CONFIGURATION
# ================================================================

INPUT_FILE = (
    "results/final_k_experiment_qwen3.csv"
)

OUTPUT_DIR = (
    "results/audit_generation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

K_VALUES = list(range(1, 11))

EXPECTED_QUESTIONS = 100

EXPECTED_RECORDS = (
    EXPECTED_QUESTIONS * len(K_VALUES)
)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def safe_numeric(series):
    """
    Convert a pandas series to numeric values.
    Invalid values become NaN.
    """
    return pd.to_numeric(
        series,
        errors="coerce"
    )


def is_empty_answer(value):
    """
    True if the generated answer is empty.
    """
    if pd.isna(value):
        return True

    return str(value).strip() == ""


def parse_sources(value):
    """
    Safely parse Retrieved_Sources.

    The CSV normally contains a Python/JSON-like list.
    """
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    text = str(value).strip()

    if not text:
        return []

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception:
        return []


def count_sources(value):
    """
    Count retrieved sources.
    """
    sources = parse_sources(value)
    return len(sources)


def max_source_distance(value):
    """
    Return the largest retrieval distance among
    retrieved sources.
    """
    sources = parse_sources(value)

    distances = []

    for source in sources:

        try:

            distance = float(
                source.get(
                    "distance"
                )
            )

            if np.isfinite(distance):
                distances.append(distance)

        except Exception:
            continue

    if not distances:
        return np.nan

    return max(distances)


def min_source_distance(value):
    """
    Return the closest retrieval distance.
    """
    sources = parse_sources(value)

    distances = []

    for source in sources:

        try:

            distance = float(
                source.get(
                    "distance"
                )
            )

            if np.isfinite(distance):
                distances.append(distance)

        except Exception:
            continue

    if not distances:
        return np.nan

    return min(distances)


# ================================================================
# LOAD DATA
# ================================================================

print("=" * 80)
print("GENERATION EXPERIMENT AUDIT")
print("=" * 80)

print()
print("Loading:")
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
# COLUMN VALIDATION
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
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print()
    print(
        "ERROR: Missing columns:"
    )

    for column in missing_columns:
        print(
            " -",
            column
        )

    raise ValueError(
        "Required columns are missing."
    )

print(
    "Column validation: PASS"
)


# ================================================================
# BASIC TYPE CLEANUP
# ================================================================

df["K"] = safe_numeric(
    df["K"]
)

df["Num_Retrieved_Chunks"] = safe_numeric(
    df["Num_Retrieved_Chunks"]
)

df["Prompt_Tokens"] = safe_numeric(
    df["Prompt_Tokens"]
)

df["Generated_Tokens"] = safe_numeric(
    df["Generated_Tokens"]
)

df["Total_Tokens"] = safe_numeric(
    df["Total_Tokens"]
)

df["Retrieval_Time_Seconds"] = safe_numeric(
    df["Retrieval_Time_Seconds"]
)

df["Generation_Time_Seconds"] = safe_numeric(
    df["Generation_Time_Seconds"]
)


# ================================================================
# STRUCTURAL AUDIT
# ================================================================

print()
print("=" * 80)
print("1. STRUCTURE AUDIT")
print("=" * 80)

print(
    "Expected records:",
    EXPECTED_RECORDS
)

print(
    "Actual records:",
    len(df)
)

if len(df) == EXPECTED_RECORDS:

    print(
        "PASS: Record count is correct."
    )

else:

    print(
        "WARNING: Record count differs."
    )


unique_questions = df["ID"].nunique()

print()
print(
    "Unique question IDs:",
    unique_questions
)

if unique_questions == EXPECTED_QUESTIONS:

    print(
        "PASS: 100 unique questions found."
    )

else:

    print(
        "WARNING: Expected 100 unique questions."
    )


observed_k = sorted(
    df["K"]
    .dropna()
    .astype(int)
    .unique()
)

print()
print(
    "K values:",
    observed_k
)

if observed_k == K_VALUES:

    print(
        "PASS: K=1 through K=10 present."
    )

else:

    print(
        "WARNING: K coverage is incomplete."
    )


models = sorted(
    df["Model"]
    .dropna()
    .astype(str)
    .unique()
)

print()
print(
    "Models:",
    models
)


# ================================================================
# QUESTION × K COVERAGE
# ================================================================

print()
print("=" * 80)
print("2. QUESTION × K COVERAGE")
print("=" * 80)

pair_counts = (
    df
    .groupby(
        ["ID", "K"]
    )
    .size()
)

duplicate_pairs = (
    pair_counts[
        pair_counts > 1
    ]
)

missing_pairs = []

question_ids = sorted(
    df["ID"].dropna().astype(str).unique()
)

for question_id in question_ids:

    observed = set(
        df.loc[
            df["ID"].astype(str)
            == question_id,
            "K"
        ]
        .dropna()
        .astype(int)
    )

    missing = (
        set(K_VALUES)
        - observed
    )

    for k in sorted(missing):

        missing_pairs.append(
            {
                "ID": question_id,
                "K": k
            }
        )

print(
    "Duplicate (ID,K) pairs:",
    len(duplicate_pairs)
)

if len(duplicate_pairs) == 0:

    print(
        "PASS: No duplicate experiment pairs."
    )

else:

    print(
        "WARNING: Duplicate pairs found."
    )


print(
    "Missing (ID,K) pairs:",
    len(missing_pairs)
)

if len(missing_pairs) == 0:

    print(
        "PASS: Every question has K=1..10."
    )

else:

    print(
        "WARNING: Missing experiment pairs."
    )


# ================================================================
# GENERATION STATUS
# ================================================================

print()
print("=" * 80)
print("3. GENERATION STATUS AUDIT")
print("=" * 80)

df["Answer_Empty"] = (
    df["Generated_Answer"]
    .apply(is_empty_answer)
)

df["Generation_Failed"] = (
    df["Answer_Empty"]
    & (
        (df["Prompt_Tokens"] == 0)
        | (df["Generation_Time_Seconds"] == 0)
    )
)

df["Generation_Succeeded"] = (
    ~df["Generation_Failed"]
)


total_failures = int(
    df["Generation_Failed"].sum()
)

total_successes = int(
    df["Generation_Succeeded"].sum()
)

print(
    "Successful generations:",
    total_successes
)

print(
    "Failed generations:",
    total_failures
)

print(
    "Failure rate:",
    f"{100 * total_failures / len(df):.2f}%"
)


# ================================================================
# FAILURE BY K
# ================================================================

print()
print("=" * 80)
print("4. GENERATION FAILURES BY K")
print("=" * 80)

failure_by_k = (
    df
    .groupby("K")
    .agg(
        Total_Runs=("ID", "count"),
        Failures=("Generation_Failed", "sum")
    )
    .reset_index()
)

failure_by_k["Successful"] = (
    failure_by_k["Total_Runs"]
    - failure_by_k["Failures"]
)

failure_by_k["Failure_Rate_Percent"] = (
    100
    * failure_by_k["Failures"]
    / failure_by_k["Total_Runs"]
)

print(
    failure_by_k.to_string(
        index=False
    )
)


# ================================================================
# FAILURE BY QUESTION
# ================================================================

print()
print("=" * 80)
print("5. FAILURE DISTRIBUTION BY QUESTION")
print("=" * 80)

failure_by_question = (
    df
    .groupby("ID")
    .agg(
        Total_Runs=("K", "count"),
        Failures=("Generation_Failed", "sum")
    )
    .reset_index()
)

failure_by_question["Failure_Rate_Percent"] = (
    100
    * failure_by_question["Failures"]
    / failure_by_question["Total_Runs"]
)

failure_by_question = (
    failure_by_question
    .sort_values(
        "Failures",
        ascending=False
    )
)

print(
    failure_by_question[
        failure_by_question["Failures"] > 0
    ].to_string(
        index=False
    )
)


# ================================================================
# TOKEN AUDIT
# ================================================================

print()
print("=" * 80)
print("6. TOKEN AUDIT")
print("=" * 80)

invalid_total_tokens = df[
    df["Total_Tokens"]
    !=
    (
        df["Prompt_Tokens"]
        +
        df["Generated_Tokens"]
    )
]

print(
    "Invalid total-token calculations:",
    len(invalid_total_tokens)
)

if len(invalid_total_tokens) == 0:

    print(
        "PASS: Total token calculations are valid."
    )

else:

    print(
        "WARNING: Invalid token calculations detected."
    )


print()
print("All-run token statistics:")

print(
    df[
        [
            "Prompt_Tokens",
            "Generated_Tokens",
            "Total_Tokens"
        ]
    ].describe().to_string()
)


# ================================================================
# SUCCESSFUL-RUN TOKEN STATISTICS
# ================================================================

successful = df[
    df["Generation_Succeeded"]
].copy()

print()
print(
    "Successful-run token statistics:"
)

if len(successful) > 0:

    print(
        successful[
            [
                "Prompt_Tokens",
                "Generated_Tokens",
                "Total_Tokens"
            ]
        ].describe().to_string()
    )


# ================================================================
# TRUNCATION AUDIT
# ================================================================

print()
print("=" * 80)
print("7. POSSIBLE TRUNCATION AUDIT")
print("=" * 80)

# The experiment used max_new_tokens=256.
# A generation reaching exactly 256 tokens is treated
# as potentially truncated.

MAX_NEW_TOKENS = 256

df["Potentially_Truncated"] = (
    df["Generated_Tokens"]
    == MAX_NEW_TOKENS
)

truncated_count = int(
    df["Potentially_Truncated"].sum()
)

print(
    "Potentially truncated generations:",
    truncated_count
)

if truncated_count > 0:

    print()
    print(
        "Potential truncation by K:"
    )

    truncation_by_k = (
        df
        .groupby("K")
        .agg(
            Runs=("ID", "count"),
            Potentially_Truncated=(
                "Potentially_Truncated",
                "sum"
            )
        )
        .reset_index()
    )

    truncation_by_k[
        "Truncation_Rate_Percent"
    ] = (
        100
        * truncation_by_k[
            "Potentially_Truncated"
        ]
        / truncation_by_k["Runs"]
    )

    print(
        truncation_by_k.to_string(
            index=False
        )
    )


# ================================================================
# VERY LONG ANSWERS
# ================================================================

print()
print("=" * 80)
print("8. ANSWER LENGTH AUDIT")
print("=" * 80)

df["Answer_Character_Length"] = (
    df["Generated_Answer"]
    .fillna("")
    .astype(str)
    .str.len()
)

print(
    "Maximum answer characters:",
    df["Answer_Character_Length"].max()
)

print(
    "Answers > 3000 characters:",
    (
        df["Answer_Character_Length"] > 3000
    ).sum()
)

print(
    "Answers > 2000 characters:",
    (
        df["Answer_Character_Length"] > 2000
    ).sum()
)

print(
    "Answers < 20 characters:",
    (
        (
            df["Answer_Character_Length"] < 20
        )
        &
        df["Generation_Succeeded"]
    ).sum()
)


# ================================================================
# RETRIEVAL CONSISTENCY AUDIT
# ================================================================

print()
print("=" * 80)
print("9. RETRIEVAL CONSISTENCY AUDIT")
print("=" * 80)

df["Parsed_Retrieved_Source_Count"] = (
    df["Retrieved_Sources"]
    .apply(count_sources)
)

df["Retrieval_Count_Matches_K"] = (
    df["Parsed_Retrieved_Source_Count"]
    ==
    df["K"]
)

retrieval_mismatches = (
    ~df["Retrieval_Count_Matches_K"]
)

print(
    "Retrieval count mismatches:",
    retrieval_mismatches.sum()
)

if retrieval_mismatches.sum() == 0:

    print(
        "PASS: Retrieved source count equals K."
    )

else:

    print(
        "WARNING: Retrieval count mismatches found."
    )


# ================================================================
# RETRIEVAL DISTANCE STATISTICS
# ================================================================

df["Minimum_Retrieval_Distance"] = (
    df["Retrieved_Sources"]
    .apply(min_source_distance)
)

df["Maximum_Retrieval_Distance"] = (
    df["Retrieved_Sources"]
    .apply(max_source_distance)
)


# ================================================================
# LATENCY AUDIT
# ================================================================

print()
print("=" * 80)
print("10. LATENCY AUDIT")
print("=" * 80)

print(
    "Generation time statistics:"
)

print(
    df[
        "Generation_Time_Seconds"
    ].describe().to_string()
)

print()
print(
    "Retrieval time statistics:"
)

print(
    df[
        "Retrieval_Time_Seconds"
    ].describe().to_string()
)

print()
print(
    "Successful generations taking >5 minutes:",
    (
        (
            df["Generation_Time_Seconds"]
            > 300
        )
        &
        df["Generation_Succeeded"]
    ).sum()
)


# ================================================================
# K-WISE PERFORMANCE
# ================================================================

print()
print("=" * 80)
print("11. K-WISE PERFORMANCE")
print("=" * 80)

k_performance = (
    df
    .groupby("K")
    .agg(

        Runs=("ID", "count"),

        Successful=(
            "Generation_Succeeded",
            "sum"
        ),

        Failed=(
            "Generation_Failed",
            "sum"
        ),

        Failure_Rate_Percent=(
            "Generation_Failed",
            lambda x:
                100 * x.mean()
        ),

        Avg_Prompt_Tokens=(
            "Prompt_Tokens",
            "mean"
        ),

        Avg_Generated_Tokens=(
            "Generated_Tokens",
            "mean"
        ),

        Avg_Total_Tokens=(
            "Total_Tokens",
            "mean"
        ),

        Avg_Generation_Time=(
            "Generation_Time_Seconds",
            "mean"
        ),

        Median_Generation_Time=(
            "Generation_Time_Seconds",
            "median"
        ),

        Max_Generation_Time=(
            "Generation_Time_Seconds",
            "max"
        ),

        Potentially_Truncated=(
            "Potentially_Truncated",
            "sum"
        )
    )
    .reset_index()
)

print(
    k_performance.to_string(
        index=False
    )
)


# ================================================================
# SUCCESSFUL-RUN K PERFORMANCE
# ================================================================

print()
print("=" * 80)
print("12. K-WISE PERFORMANCE — SUCCESSFUL RUNS ONLY")
print("=" * 80)

successful_k = (
    df[
        df["Generation_Succeeded"]
    ]
    .groupby("K")
    .agg(

        Successful_Runs=(
            "ID",
            "count"
        ),

        Avg_Prompt_Tokens=(
            "Prompt_Tokens",
            "mean"
        ),

        Avg_Generated_Tokens=(
            "Generated_Tokens",
            "mean"
        ),

        Avg_Total_Tokens=(
            "Total_Tokens",
            "mean"
        ),

        Avg_Generation_Time=(
            "Generation_Time_Seconds",
            "mean"
        ),

        Median_Generation_Time=(
            "Generation_Time_Seconds",
            "median"
        )
    )
    .reset_index()
)

print(
    successful_k.to_string(
        index=False
    )
)


# ================================================================
# LONG-CONTEXT ANALYSIS
# ================================================================

print()
print("=" * 80)
print("13. LONG-CONTEXT ANALYSIS")
print("=" * 80)

# Prompt-token bins
df["Prompt_Length_Bin"] = pd.cut(
    df["Prompt_Tokens"],
    bins=[
        -1,
        1000,
        2000,
        3000,
        4000,
        5000,
        10000
    ],
    labels=[
        "<=1000",
        "1001-2000",
        "2001-3000",
        "3001-4000",
        "4001-5000",
        ">5000"
    ]
)

context_analysis = (
    df
    .groupby(
        "Prompt_Length_Bin",
        observed=False
    )
    .agg(
        Runs=("ID", "count"),

        Failures=(
            "Generation_Failed",
            "sum"
        ),

        Failure_Rate_Percent=(
            "Generation_Failed",
            lambda x:
                100 * x.mean()
        ),

        Avg_Generation_Time=(
            "Generation_Time_Seconds",
            "mean"
        ),

        Avg_Generated_Tokens=(
            "Generated_Tokens",
            "mean"
        )
    )
    .reset_index()
)

print(
    context_analysis.to_string(
        index=False
    )
)


# ================================================================
# ANSWERABLE VS UNANSWERABLE
# ================================================================

print()
print("=" * 80)
print("14. ANSWERABLE / UNANSWERABLE AUDIT")
print("=" * 80)

answerability_analysis = (
    df
    .groupby("Answerable")
    .agg(
        Runs=("ID", "count"),

        Failures=(
            "Generation_Failed",
            "sum"
        ),

        Failure_Rate_Percent=(
            "Generation_Failed",
            lambda x:
                100 * x.mean()
        ),

        Avg_Prompt_Tokens=(
            "Prompt_Tokens",
            "mean"
        ),

        Avg_Generation_Time=(
            "Generation_Time_Seconds",
            "mean"
        )
    )
    .reset_index()
)

print(
    answerability_analysis.to_string(
        index=False
    )
)


# ================================================================
# COMPLEXITY ANALYSIS
# ================================================================

print()
print("=" * 80)
print("15. COMPLEXITY AUDIT")
print("=" * 80)

complexity_analysis = (
    df
    .groupby("Complexity")
    .agg(
        Runs=("ID", "count"),

        Failures=(
            "Generation_Failed",
            "sum"
        ),

        Failure_Rate_Percent=(
            "Generation_Failed",
            lambda x:
                100 * x.mean()
        ),

        Avg_Prompt_Tokens=(
            "Prompt_Tokens",
            "mean"
        ),

        Avg_Generation_Time=(
            "Generation_Time_Seconds",
            "mean"
        )
    )
    .reset_index()
)

print(
    complexity_analysis.to_string(
        index=False
    )
)


# ================================================================
# PRIMARY CATEGORY ANALYSIS
# ================================================================

print()
print("=" * 80)
print("16. PRIMARY CATEGORY AUDIT")
print("=" * 80)

category_analysis = (
    df
    .groupby("Primary_Category")
    .agg(
        Runs=("ID", "count"),

        Failures=(
            "Generation_Failed",
            "sum"
        ),

        Failure_Rate_Percent=(
            "Generation_Failed",
            lambda x:
                100 * x.mean()
        ),

        Avg_Prompt_Tokens=(
            "Prompt_Tokens",
            "mean"
        ),

        Avg_Generation_Time=(
            "Generation_Time_Seconds",
            "mean"
        )
    )
    .reset_index()
    .sort_values(
        "Failure_Rate_Percent",
        ascending=False
    )
)

print(
    category_analysis.to_string(
        index=False
    )
)


# ================================================================
# FAILED GENERATION RECORDS
# ================================================================

print()
print("=" * 80)
print("17. FAILED GENERATIONS")
print("=" * 80)

failures = df[
    df["Generation_Failed"]
].copy()

print(
    "Failed records:",
    len(failures)
)

if len(failures) > 0:

    failure_columns = [
        "ID",
        "K",
        "Question",
        "Complexity",
        "Answerable",
        "Source_Document",
        "Source_Page",
        "Prompt_Tokens",
        "Generated_Tokens",
        "Total_Tokens",
        "Generation_Time_Seconds",
        "Retrieved_Sources"
    ]

    print(
        failures[
            failure_columns
        ].to_string(
            index=False
        )
    )


# ================================================================
# CREATE CLEAN AUDIT DATASET
# ================================================================

audit_columns = [
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
    "Num_Retrieved_Chunks",
    "Generated_Answer",
    "Prompt_Tokens",
    "Generated_Tokens",
    "Total_Tokens",
    "Retrieval_Time_Seconds",
    "Generation_Time_Seconds",
    "Model",
    "Timestamp",
    "Answer_Empty",
    "Generation_Failed",
    "Generation_Succeeded",
    "Potentially_Truncated",
    "Answer_Character_Length",
    "Parsed_Retrieved_Source_Count",
    "Retrieval_Count_Matches_K",
    "Minimum_Retrieval_Distance",
    "Maximum_Retrieval_Distance"
]

audit_df = df[
    audit_columns
].copy()


# ================================================================
# SAVE AUDIT RECORDS
# ================================================================

audit_records_file = os.path.join(
    OUTPUT_DIR,
    "generation_audit_records.csv"
)

audit_df.to_csv(
    audit_records_file,
    index=False,
    encoding="utf-8-sig"
)


# ================================================================
# SAVE FAILURE FILE
# ================================================================

failure_file = os.path.join(
    OUTPUT_DIR,
    "generation_failures.csv"
)

failures.to_csv(
    failure_file,
    index=False,
    encoding="utf-8-sig"
)


# ================================================================
# CREATE SUMMARY TABLE
# ================================================================

summary_rows = []

for _, row in k_performance.iterrows():

    summary_rows.append(
        {
            "K":
                int(row["K"]),

            "Runs":
                int(row["Runs"]),

            "Successful":
                int(row["Successful"]),

            "Failed":
                int(row["Failed"]),

            "Failure_Rate_Percent":
                float(
                    row[
                        "Failure_Rate_Percent"
                    ]
                ),

            "Avg_Prompt_Tokens":
                float(
                    row[
                        "Avg_Prompt_Tokens"
                    ]
                ),

            "Avg_Generated_Tokens":
                float(
                    row[
                        "Avg_Generated_Tokens"
                    ]
                ),

            "Avg_Total_Tokens":
                float(
                    row[
                        "Avg_Total_Tokens"
                    ]
                ),

            "Avg_Generation_Time":
                float(
                    row[
                        "Avg_Generation_Time"
                    ]
                ),

            "Median_Generation_Time":
                float(
                    row[
                        "Median_Generation_Time"
                    ]
                ),

            "Max_Generation_Time":
                float(
                    row[
                        "Max_Generation_Time"
                    ]
                ),

            "Potentially_Truncated":
                int(
                    row[
                        "Potentially_Truncated"
                    ]
                )
        }
    )

summary_df = pd.DataFrame(
    summary_rows
)

summary_file = os.path.join(
    OUTPUT_DIR,
    "generation_audit_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False,
    encoding="utf-8-sig"
)


# ================================================================
# TEXT REPORT
# ================================================================

report_file = os.path.join(
    OUTPUT_DIR,
    "generation_audit_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "GENERATION EXPERIMENT AUDIT REPORT\n"
    )

    report.write(
        "=" * 80 + "\n\n"
    )

    report.write(
        f"Input file: {INPUT_FILE}\n"
    )

    report.write(
        f"Total records: {len(df)}\n"
    )

    report.write(
        f"Unique questions: {unique_questions}\n"
    )

    report.write(
        f"K values: {observed_k}\n"
    )

    report.write(
        f"Models: {models}\n\n"
    )

    report.write(
        "GENERATION STATUS\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    report.write(
        f"Successful generations: "
        f"{total_successes}\n"
    )

    report.write(
        f"Failed generations: "
        f"{total_failures}\n"
    )

    report.write(
        f"Failure rate: "
        f"{100 * total_failures / len(df):.2f}%\n\n"
    )

    report.write(
        "FAILURES BY K\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    report.write(
        failure_by_k.to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        "K-WISE PERFORMANCE\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    report.write(
        k_performance.to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        "TRUNCATION\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    report.write(
        f"Potentially truncated: "
        f"{truncated_count}\n"
    )

    report.write(
        "\n"
    )

    report.write(
        "RETRIEVAL CONSISTENCY\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    report.write(
        f"Retrieval count mismatches: "
        f"{retrieval_mismatches.sum()}\n"
    )

    report.write(
        "\n"
    )

    report.write(
        "ANSWERABLE / UNANSWERABLE\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    report.write(
        answerability_analysis.to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        "COMPLEXITY\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    report.write(
        complexity_analysis.to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        "PRIMARY CATEGORY\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    report.write(
        category_analysis.to_string(
            index=False
        )
    )


# ================================================================
# FINAL OUTPUT
# ================================================================

print()
print("=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)

print()
print(
    "Audit records:"
)

print(
    audit_records_file
)

print()
print(
    "Audit summary:"
)

print(
    summary_file
)

print()
print(
    "Failed generations:"
)

print(
    failure_file
)

print()
print(
    "Audit report:"
)

print(
    report_file
)

print()
print("=" * 80)
print("IMPORTANT")
print("=" * 80)

print(
    "Do NOT run RAGAS until the audit has been reviewed."
)