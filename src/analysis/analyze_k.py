"""
FAST ADAPTIVE-K QUALITY ANALYSIS

Uses the existing RAG dataset only.
NO LLM calls.
NO RAGAS calls.

For every question and K, calculates:

1. Reference concept coverage in retrieved context
2. Reference-answer similarity
3. Evidence quality
4. Token cost
5. Latency

Then identifies a CANDIDATE optimal K.

IMPORTANT:
-----------
This is a deterministic proxy for answer/evidence quality.
It is NOT RAGAS.

RAGAS/Qwen evaluation will be used later as validation.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "results/ragas/ragas_evaluation_dataset.csv"
)

OUTPUT_DIR = Path(
    "results/adaptive"
)

# Quality threshold.
#
# Example:
# if the best quality for a question is 0.95,
# a K producing >= 0.93 is considered sufficient.
#
# We use 0.02 tolerance.
QUALITY_TOLERANCE = 0.02


# ============================================================
# TEXT NORMALIZATION
# ============================================================

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "from",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "as",
    "at",
    "that",
    "this",
    "these",
    "those",
    "they",
    "their",
    "them",
    "it",
    "its",
    "such",
    "also",
    "into",
    "than",
    "then",
    "there",
    "which",
    "what",
    "how",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "will",
    "may",
    "might",
}


def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Normalize common PDF artefacts.
    text = text.replace("’", "'")
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    # Remove excessive punctuation.
    text = re.sub(
        r"[^a-z0-9\s\-]",
        " ",
        text,
    )

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def content_tokens(text):

    text = normalize_text(text)

    tokens = text.split()

    return [
        token
        for token in tokens
        if (
            len(token) >= 3
            and token not in STOPWORDS
        )
    ]


# ============================================================
# REFERENCE CONCEPT COVERAGE
# ============================================================

def reference_coverage(
    reference,
    context,
):
    """
    Measures how much of the reference answer's
    meaningful vocabulary appears in the retrieved context.

    This is deliberately recall-oriented.

    Additional retrieved context does NOT penalize the score.
    """

    ref_tokens = set(
        content_tokens(reference)
    )

    context_tokens = set(
        content_tokens(context)
    )

    if not ref_tokens:
        return np.nan

    matched = (
        ref_tokens
        &
        context_tokens
    )

    return len(matched) / len(ref_tokens)


# ============================================================
# REFERENCE -> ANSWER SIMILARITY
# ============================================================

def answer_similarity(
    reference,
    answer,
):
    """
    TF-IDF cosine similarity between the reference answer
    and generated answer.

    This is a cheap deterministic proxy for answer quality.
    """

    reference = normalize_text(
        reference
    )

    answer = normalize_text(
        answer
    )

    if not reference or not answer:
        return np.nan

    try:

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
        )

        matrix = vectorizer.fit_transform(
            [
                reference,
                answer,
            ]
        )

        score = cosine_similarity(
            matrix[0:1],
            matrix[1:2],
        )[0][0]

        return float(score)

    except Exception:

        return np.nan


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{INPUT_FILE.resolve()}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print("=" * 80)
    print("FAST ADAPTIVE-K QUALITY ANALYSIS")
    print("=" * 80)

    print(
        f"\nRows: {len(df)}"
    )

    print(
        f"Unique questions: "
        f"{df['ID'].nunique()}"
    )

    print(
        "\nK distribution:"
    )

    print(
        df["K"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return df


# ============================================================
# PREPARE COST DATA
# ============================================================

def prepare_costs(df):

    df = df.copy()

    numeric_columns = [
        "K",
        "Prompt_Tokens",
        "Generated_Tokens",
        "Total_Tokens",
        "Retrieval_Time_Seconds",
        "Generation_Time_Seconds",
        "Num_Retrieved_Chunks",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # Reconstruct total tokens where necessary.

    if (
        "Total_Tokens" in df.columns
        and "Prompt_Tokens" in df.columns
        and "Generated_Tokens" in df.columns
    ):

        missing = (
            df["Total_Tokens"].isna()
        )

        df.loc[
            missing,
            "Total_Tokens"
        ] = (
            df.loc[
                missing,
                "Prompt_Tokens"
            ].fillna(0)
            +
            df.loc[
                missing,
                "Generated_Tokens"
            ].fillna(0)
        )

    # Total latency.

    df["Total_Latency_Seconds"] = (
        df["Retrieval_Time_Seconds"].fillna(0)
        +
        df["Generation_Time_Seconds"].fillna(0)
    )

    return df


# ============================================================
# CALCULATE QUALITY
# ============================================================

def calculate_quality(df):

    records = []

    for _, row in df.iterrows():

        reference = row.get(
            "reference",
            "",
        )

        context = row.get(
            "Retrieved_Context",
            "",
        )

        answer = row.get(
            "Generated_Answer",
            "",
        )

        coverage = reference_coverage(
            reference,
            context,
        )

        similarity = answer_similarity(
            reference,
            answer,
        )

        # ----------------------------------------------------
        # Combined deterministic quality score.
        #
        # Evidence coverage gets slightly more weight because
        # our purpose is determining retrieval depth.
        # ----------------------------------------------------

        valid_scores = [
            x
            for x in [
                coverage,
                similarity,
            ]
            if pd.notna(x)
        ]

        if valid_scores:

            if (
                pd.notna(coverage)
                and pd.notna(similarity)
            ):

                quality = (
                    0.60 * coverage
                    +
                    0.40 * similarity
                )

            else:

                quality = (
                    float(
                        np.mean(
                            valid_scores
                        )
                    )
                )

        else:

            quality = np.nan

        records.append({

            "ID": row["ID"],

            "K": row["K"],

            "Question": row.get(
                "user_input",
                "",
            ),

            "Reference": reference,

            "Generated_Answer": answer,

            "Evidence_Coverage": coverage,

            "Answer_Similarity": similarity,

            "Quality_Score": quality,

            "Prompt_Tokens": row.get(
                "Prompt_Tokens",
                np.nan,
            ),

            "Generated_Tokens": row.get(
                "Generated_Tokens",
                np.nan,
            ),

            "Total_Tokens": row.get(
                "Total_Tokens",
                np.nan,
            ),

            "Retrieval_Time_Seconds": row.get(
                "Retrieval_Time_Seconds",
                np.nan,
            ),

            "Generation_Time_Seconds": row.get(
                "Generation_Time_Seconds",
                np.nan,
            ),

            "Total_Latency_Seconds": row.get(
                "Total_Latency_Seconds",
                np.nan,
            ),

            "Num_Retrieved_Chunks": row.get(
                "Num_Retrieved_Chunks",
                np.nan,
            ),

            "Primary_Category": row.get(
                "Primary_Category",
                "",
            ),

            "Secondary_Category": row.get(
                "Secondary_Category",
                "",
            ),

            "Complexity": row.get(
                "Complexity",
                "",
            ),

            "Answerable": row.get(
                "Answerable",
                "",
            ),
        })

    return pd.DataFrame(
        records
    )


# ============================================================
# FIND CANDIDATE OPTIMAL K
# ============================================================

def find_optimal_k(
    quality_df,
):

    results = []

    for question_id, group in quality_df.groupby(
        "ID",
        sort=True,
    ):

        group = group.sort_values(
            "K"
        ).copy()

        valid = group[
            group["Quality_Score"].notna()
        ]

        if valid.empty:
            continue

        best_quality = (
            valid["Quality_Score"].max()
        )

        threshold = (
            best_quality
            - QUALITY_TOLERANCE
        )

        eligible = valid[
            valid["Quality_Score"]
            >= threshold
        ]

        if eligible.empty:
            continue

        candidate = (
            eligible
            .sort_values("K")
            .iloc[0]
        )

        candidate_k = int(
            candidate["K"]
        )

        max_k_row = (
            valid
            .sort_values("K")
            .iloc[-1]
        )

        max_k = int(
            max_k_row["K"]
        )

        candidate_tokens = (
            candidate["Total_Tokens"]
        )

        max_tokens = (
            max_k_row["Total_Tokens"]
        )

        if (
            pd.notna(candidate_tokens)
            and pd.notna(max_tokens)
            and max_tokens > 0
        ):

            token_savings = (
                1
                -
                candidate_tokens
                / max_tokens
            )

        else:

            token_savings = np.nan

        candidate_latency = (
            candidate[
                "Total_Latency_Seconds"
            ]
        )

        max_latency = (
            max_k_row[
                "Total_Latency_Seconds"
            ]
        )

        if (
            pd.notna(candidate_latency)
            and pd.notna(max_latency)
            and max_latency > 0
        ):

            latency_savings = (
                1
                -
                candidate_latency
                / max_latency
            )

        else:

            latency_savings = np.nan

        results.append({

            "ID": question_id,

            "Question": candidate[
                "Question"
            ],

            "Primary_Category": candidate[
                "Primary_Category"
            ],

            "Secondary_Category": candidate[
                "Secondary_Category"
            ],

            "Complexity": candidate[
                "Complexity"
            ],

            "Answerable": candidate[
                "Answerable"
            ],

            "Candidate_Optimal_K": candidate_k,

            "Best_K_Quality": best_quality,

            "Quality_Threshold": threshold,

            "Candidate_Quality": candidate[
                "Quality_Score"
            ],

            "Candidate_Evidence_Coverage": candidate[
                "Evidence_Coverage"
            ],

            "Candidate_Answer_Similarity": candidate[
                "Answer_Similarity"
            ],

            "Candidate_Total_Tokens": candidate[
                "Total_Tokens"
            ],

            "Candidate_Latency": candidate_latency,

            "Maximum_Available_K": max_k,

            "Maximum_K_Quality": max_k_row[
                "Quality_Score"
            ],

            "Maximum_K_Tokens": max_tokens,

            "Maximum_K_Latency": max_latency,

            "Estimated_Token_Savings": token_savings,

            "Estimated_Latency_Savings": latency_savings,
        })

    return pd.DataFrame(
        results
    )


# ============================================================
# SUMMARY BY K
# ============================================================

def create_k_summary(
    quality_df,
):

    summary = (
        quality_df
        .groupby("K")
        .agg(

            Questions=(
                "ID",
                "nunique",
            ),

            Records=(
                "ID",
                "size",
            ),

            Mean_Evidence_Coverage=(
                "Evidence_Coverage",
                "mean",
            ),

            Mean_Answer_Similarity=(
                "Answer_Similarity",
                "mean",
            ),

            Mean_Quality_Score=(
                "Quality_Score",
                "mean",
            ),

            Mean_Prompt_Tokens=(
                "Prompt_Tokens",
                "mean",
            ),

            Mean_Generated_Tokens=(
                "Generated_Tokens",
                "mean",
            ),

            Mean_Total_Tokens=(
                "Total_Tokens",
                "mean",
            ),

            Mean_Latency=(
                "Total_Latency_Seconds",
                "mean",
            ),

            Mean_Num_Chunks=(
                "Num_Retrieved_Chunks",
                "mean",
            ),
        )
        .reset_index()
    )

    return summary


# ============================================================
# SAVE
# ============================================================

def save_results(
    quality_df,
    optimal_df,
    summary_df,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    quality_path = (
        OUTPUT_DIR
        / "quality_vs_k.csv"
    )

    optimal_path = (
        OUTPUT_DIR
        / "optimal_k_candidates.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "k_summary.csv"
    )

    quality_df.to_csv(
        quality_path,
        index=False,
    )

    optimal_df.to_csv(
        optimal_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("FILES CREATED")
    print("=" * 80)

    print(
        quality_path
    )

    print(
        optimal_path
    )

    print(
        summary_path
    )


# ============================================================
# REPORT
# ============================================================

def report(
    quality_df,
    optimal_df,
    summary_df,
):

    print("\n" + "=" * 80)
    print("QUALITY VS K")
    print("=" * 80)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n" + "=" * 80)
    print("CANDIDATE OPTIMAL-K DISTRIBUTION")
    print("=" * 80)

    if not optimal_df.empty:

        print(
            optimal_df[
                "Candidate_Optimal_K"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print(
            "\nMean candidate K:",
            round(
                optimal_df[
                    "Candidate_Optimal_K"
                ].mean(),
                2,
            ),
        )

        token_savings = (
            optimal_df[
                "Estimated_Token_Savings"
            ]
            .dropna()
        )

        if not token_savings.empty:

            print(
                "Mean estimated token savings "
                "vs maximum K:",
                f"{token_savings.mean() * 100:.2f}%",
            )

        latency_savings = (
            optimal_df[
                "Estimated_Latency_Savings"
            ]
            .dropna()
        )

        if not latency_savings.empty:

            print(
                "Mean estimated latency savings "
                "vs maximum K:",
                f"{latency_savings.mean() * 100:.2f}%",
            )

    else:

        print(
            "No candidate K values generated."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    df = prepare_costs(
        df
    )

    quality_df = calculate_quality(
        df
    )

    optimal_df = find_optimal_k(
        quality_df
    )

    summary_df = create_k_summary(
        quality_df
    )

    save_results(
        quality_df,
        optimal_df,
        summary_df,
    )

    report(
        quality_df,
        optimal_df,
        summary_df,
    )

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

    print(
        "\nNo LLM was used."
    )

    print(
        "No RAGAS evaluation was used."
    )

    print(
        "These are deterministic candidate "
        "Optimal-K labels."
    )

    print(
        "\nThese labels will later be validated "
        "with Qwen/RAGAS on a smaller evaluation set."
    )


if __name__ == "__main__":
    main()