import re

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from .config import (
    EVALUATION_DATASET,
    CANDIDATE_K_DATASET,
    OUTPUT_DIR,
    QUESTION_COLUMN,
    ID_COLUMN,
    TARGET_COLUMN,
    MIN_K,
    MAX_K,
    EMBEDDING_MODEL,
    ENGINEERED_FEATURES,
)


def clean_text(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def count_sentences(text: str) -> int:
    if not text:
        return 0

    parts = re.split(
        r"[.!?]+",
        text,
    )

    return sum(
        bool(part.strip())
        for part in parts
    )


def extract_basic_features(
    question: str,
) -> dict:

    question = clean_text(question)

    words = re.findall(
        r"\b[\w'-]+\b",
        question,
    )

    lower_words = [
        word.lower()
        for word in words
    ]

    unique_words = set(
        lower_words
    )

    avg_word_length = (
        sum(len(word) for word in words)
        / len(words)
        if words
        else 0.0
    )

    first_word = (
        lower_words[0]
        if lower_words
        else ""
    )

    return {
        "query_char_count": len(question),
        "query_word_count": len(words),
        "query_unique_word_count": len(unique_words),
        "query_avg_word_length": avg_word_length,
        "query_sentence_count": count_sentences(
            question
        ),
        "question_mark_indicator": int(
            "?" in question
        ),
        "number_indicator": int(
            bool(
                re.search(
                    r"\b\d+\b",
                    question,
                )
            )
        ),
        "uppercase_token_count": sum(
            1
            for word in words
            if word.isupper()
            and len(word) > 1
        ),
        "starts_what": int(
            first_word == "what"
        ),
        "starts_why": int(
            first_word == "why"
        ),
        "starts_how": int(
            first_word == "how"
        ),
        "starts_when": int(
            first_word == "when"
        ),
        "starts_where": int(
            first_word == "where"
        ),
        "starts_who": int(
            first_word == "who"
        ),
        "starts_which": int(
            first_word == "which"
        ),
        "entity_count": 0,
        "question_count": int(
            "?" in question
        ),
        "comparison": 0,
        "procedural": 0,
        "policy": 0,
        "summary": 0,
        "definition": 0,
        "multi_hop": 0,
        "evidence_breadth": 0,
    }


def add_semantic_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    questions = (
        result[QUESTION_COLUMN]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    result["comparison"] = (
        questions.str.contains(
            r"\b(?:compare|difference|versus|vs\.?|better|similar)\b",
            regex=True,
            na=False,
        )
        .astype(int)
    )

    result["procedural"] = (
        questions.str.contains(
            r"\b(?:how|steps|process|procedure|method)\b",
            regex=True,
            na=False,
        )
        .astype(int)
    )

    result["policy"] = (
        questions.str.contains(
            r"\b(?:policy|requirement|regulation|rule|standard)\b",
            regex=True,
            na=False,
        )
        .astype(int)
    )

    result["summary"] = (
        questions.str.contains(
            r"\b(?:summarize|summary|overview|main points|key points)\b",
            regex=True,
            na=False,
        )
        .astype(int)
    )

    result["definition"] = (
        questions.str.contains(
            r"\b(?:what is|define|definition|meaning of)\b",
            regex=True,
            na=False,
        )
        .astype(int)
    )

    result["multi_hop"] = (
        questions.str.contains(
            r"\b(?:and|also|relationship|related|together|multiple)\b",
            regex=True,
            na=False,
        )
        .astype(int)
    )

    return result


def build_dataset():

    if not EVALUATION_DATASET.exists():
        raise FileNotFoundError(
            f"Missing evaluation dataset:\n"
            f"{EVALUATION_DATASET}"
        )

    if not CANDIDATE_K_DATASET.exists():
        raise FileNotFoundError(
            f"Missing candidate K dataset:\n"
            f"{CANDIDATE_K_DATASET}"
        )

    evaluation = pd.read_csv(
        EVALUATION_DATASET
    )

    candidates = pd.read_csv(
        CANDIDATE_K_DATASET
    )

    required_columns = {
        ID_COLUMN,
        QUESTION_COLUMN,
    }

    missing = (
        required_columns
        - set(evaluation.columns)
    )

    if missing:
        raise ValueError(
            "Evaluation dataset is missing: "
            f"{sorted(missing)}"
        )

    if TARGET_COLUMN not in candidates.columns:
        raise ValueError(
            f"Candidate dataset does not contain "
            f"{TARGET_COLUMN}"
        )

    # IMPORTANT:
    # Collapse K=1...10 into ONE row per question.
    questions = (
        evaluation[
            [
                ID_COLUMN,
                QUESTION_COLUMN,
            ]
        ]
        .drop_duplicates(
            subset=[ID_COLUMN]
        )
        .copy()
    )

    candidate_targets = (
        candidates[
            [
                ID_COLUMN,
                TARGET_COLUMN,
            ]
        ]
        .drop_duplicates()
    )

    duplicate_targets = (
        candidate_targets
        .groupby(ID_COLUMN)
        .size()
        .loc[lambda x: x > 1]
    )

    if not duplicate_targets.empty:
        raise ValueError(
            "Multiple candidate K values found "
            "for questions: "
            f"{duplicate_targets.index.tolist()}"
        )

    dataset = questions.merge(
        candidate_targets,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    dataset[TARGET_COLUMN] = pd.to_numeric(
        dataset[TARGET_COLUMN],
        errors="coerce",
    )

    dataset = dataset.dropna(
        subset=[TARGET_COLUMN]
    ).copy()

    dataset[TARGET_COLUMN] = (
        dataset[TARGET_COLUMN]
        .astype(int)
    )

    invalid = dataset[
        ~dataset[TARGET_COLUMN].between(
            MIN_K,
            MAX_K,
        )
    ]

    if not invalid.empty:
        raise ValueError(
            "Invalid K values detected."
        )

    # Engineered features.
    rows = []

    for _, row in dataset.iterrows():

        features = extract_basic_features(
            row[QUESTION_COLUMN]
        )

        features[ID_COLUMN] = row[ID_COLUMN]

        rows.append(features)

    feature_df = pd.DataFrame(rows)

    dataset = dataset.merge(
        feature_df,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    dataset = add_semantic_features(
        dataset
    )

    # ------------------------------------------------------------------
    # QUERY EMBEDDINGS
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING SENTENCE TRANSFORMER")
    print("=" * 80)

    print(
        f"Model: {EMBEDDING_MODEL}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    questions_text = (
        dataset[QUESTION_COLUMN]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print(
        f"Generating embeddings for "
        f"{len(questions_text)} questions..."
    )

    embeddings = model.encode(
        questions_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    # Save embeddings separately.
    embedding_path = (
        OUTPUT_DIR
        / "query_embeddings.npy"
    )

    np.save(
        embedding_path,
        embeddings,
    )

    # Add embedding columns to dataset.
    for i in range(
        embeddings.shape[1]
    ):
        dataset[
            f"embedding_{i}"
        ] = embeddings[:, i]

    dataset_path = (
        OUTPUT_DIR
        / "dataset.csv"
    )

    dataset.to_csv(
        dataset_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("DATASET CREATED")
    print("=" * 80)

    print(
        f"Questions: {len(dataset)}"
    )

    print(
        f"Embedding dimensions: "
        f"{embeddings.shape[1]}"
    )

    print(
        f"Dataset: {dataset_path}"
    )

    print(
        f"Embeddings: {embedding_path}"
    )

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