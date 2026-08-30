import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    EVALUATION_DATASET,
    CANDIDATE_K_DATASET,
    OUTPUT_DIR,
    ID_COLUMN,
    QUESTION_COLUMN,
    TARGET_COLUMN,
    MIN_K,
    MAX_K,
)


def clean_text(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def count_sentences(text: str) -> int:
    if not text:
        return 0

    parts = re.split(r"[.!?]+", text)
    return sum(bool(x.strip()) for x in parts)


def extract_basic_features(question: str) -> dict:
    question = clean_text(question)

    words = re.findall(r"\b[\w'-]+\b", question)
    lower_words = [w.lower() for w in words]

    unique_words = set(lower_words)

    if words:
        avg_word_length = float(
            sum(len(w) for w in words) / len(words)
        )
    else:
        avg_word_length = 0.0

    first_word = lower_words[0] if lower_words else ""

    features = {
        "query_char_count": len(question),
        "query_word_count": len(words),
        "query_unique_word_count": len(unique_words),
        "query_avg_word_length": avg_word_length,
        "query_sentence_count": count_sentences(question),
        "question_mark_indicator": int("?" in question),
        "number_indicator": int(
            bool(re.search(r"\b\d+\b", question))
        ),
        "uppercase_token_count": sum(
            1 for w in words if w.isupper() and len(w) > 1
        ),
        "starts_what": int(first_word == "what"),
        "starts_why": int(first_word == "why"),
        "starts_how": int(first_word == "how"),
        "starts_when": int(first_word == "when"),
        "starts_where": int(first_word == "where"),
        "starts_who": int(first_word == "who"),
        "starts_which": int(first_word == "which"),
        "entity_count": 0,
        "question_count": int("?" in question),
        "intent": "",
        "comparison": 0,
        "procedural": 0,
        "policy": 0,
        "summary": 0,
        "definition": 0,
        "multi_hop": 0,
        "evidence_breadth": 0,
    }

    return features


def add_semantic_keyword_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple pre-retrieval semantic/structural indicators.

    These are deliberately deterministic and cheap.
    """

    result = df.copy()

    questions = result[QUESTION_COLUMN].fillna("").astype(str)

    result["comparison"] = questions.str.lower().str.contains(
        r"\b(compare|difference|versus|vs\.?|better|similar)\b",
        regex=True,
    ).astype(int)

    result["procedural"] = questions.str.lower().str.contains(
        r"\b(how|steps|process|procedure|method)\b",
        regex=True,
    ).astype(int)

    result["policy"] = questions.str.lower().str.contains(
        r"\b(policy|requirement|regulation|rule|standard)\b",
        regex=True,
    ).astype(int)

    result["summary"] = questions.str.lower().str.contains(
        r"\b(summarize|summary|overview|main points|key points)\b",
        regex=True,
    ).astype(int)

    result["definition"] = questions.str.lower().str.contains(
        r"\b(what is|define|definition|meaning of)\b",
        regex=True,
    ).astype(int)

    result["multi_hop"] = questions.str.lower().str.contains(
        r"\b(and|also|relationship|related|together|multiple)\b",
        regex=True,
    ).astype(int)

    return result


def build_dataset() -> pd.DataFrame:
    if not EVALUATION_DATASET.exists():
        raise FileNotFoundError(
            f"Missing evaluation dataset: {EVALUATION_DATASET}"
        )

    if not CANDIDATE_K_DATASET.exists():
        raise FileNotFoundError(
            f"Missing candidate K dataset: {CANDIDATE_K_DATASET}"
        )

    evaluation = pd.read_csv(EVALUATION_DATASET)
    candidates = pd.read_csv(CANDIDATE_K_DATASET)

    required_eval = {
        ID_COLUMN,
        QUESTION_COLUMN,
    }

    missing = required_eval - set(evaluation.columns)

    if missing:
        raise ValueError(
            f"Evaluation dataset missing columns: {sorted(missing)}"
        )

    candidate_id = ID_COLUMN

    if candidate_id not in candidates.columns:
        raise ValueError(
            f"Candidate dataset missing '{candidate_id}'."
        )

    if TARGET_COLUMN not in candidates.columns:
        raise ValueError(
            f"Candidate dataset missing '{TARGET_COLUMN}'."
        )

    # One question-level record.
    questions = (
        evaluation[
            [ID_COLUMN, QUESTION_COLUMN]
        ]
        .drop_duplicates(subset=[ID_COLUMN])
        .copy()
    )

    # Candidate K must be exactly one target per question.
    candidates = candidates[
        [ID_COLUMN, TARGET_COLUMN]
    ].drop_duplicates()

    duplicate_target_ids = (
        candidates.groupby(ID_COLUMN)
        .size()
        .loc[lambda x: x > 1]
    )

    if not duplicate_target_ids.empty:
        raise ValueError(
            "Multiple candidate K targets found for questions: "
            f"{duplicate_target_ids.index.tolist()}"
        )

    dataset = questions.merge(
        candidates,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    if dataset.empty:
        raise ValueError(
            "No questions could be matched to candidate K targets."
        )

    dataset[TARGET_COLUMN] = pd.to_numeric(
        dataset[TARGET_COLUMN],
        errors="coerce",
    )

    dataset = dataset.dropna(
        subset=[TARGET_COLUMN]
    ).copy()

    dataset[TARGET_COLUMN] = dataset[TARGET_COLUMN].astype(int)

    invalid_k = dataset[
        ~dataset[TARGET_COLUMN].between(MIN_K, MAX_K)
    ]

    if not invalid_k.empty:
        raise ValueError(
            "Invalid candidate K values found:\n"
            f"{invalid_k[[ID_COLUMN, TARGET_COLUMN]]}"
        )

    feature_rows = []

    for _, row in dataset.iterrows():
        features = extract_basic_features(
            row[QUESTION_COLUMN]
        )

        features[ID_COLUMN] = row[ID_COLUMN]

        feature_rows.append(features)

    feature_df = pd.DataFrame(feature_rows)

    dataset = dataset.drop(
        columns=[
            c for c in feature_df.columns
            if c != ID_COLUMN and c in dataset.columns
        ],
        errors="ignore",
    )

    dataset = dataset.merge(
        feature_df,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    dataset = add_semantic_keyword_features(dataset)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "dataset.csv"

    dataset.to_csv(
        output_path,
        index=False,
    )

    print("=" * 80)
    print("ADAPTIVE-K BOOSTING EXPERIMENT DATASET")
    print("=" * 80)
    print(f"Questions: {len(dataset)}")
    print(f"Output: {output_path}")
    print()
    print("Target distribution:")
    print(
        dataset[TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return dataset


if __name__ == "__main__":
    build_dataset()