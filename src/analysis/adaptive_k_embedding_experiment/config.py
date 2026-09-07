from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

EVALUATION_DATASET = (
    PROJECT_ROOT
    / "results"
    / "ragas"
    / "ragas_evaluation_dataset.csv"
)

CANDIDATE_K_DATASET = (
    PROJECT_ROOT
    / "results"
    / "adaptive"
    / "optimal_k_candidates.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "adaptive"
    / "embedding_xgboost_experiment"
)

RANDOM_STATE = 42
CV_FOLDS = 5

MIN_K = 1
MAX_K = 10

ID_COLUMN = "ID"
QUESTION_COLUMN = "user_input"
TARGET_COLUMN = "Candidate_Optimal_K"

# Sentence Transformer model.
#
# This generates the embedding ONLY from the query.
# It does not see retrieved documents.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Existing engineered pre-retrieval features.
ENGINEERED_FEATURES = [
    "query_char_count",
    "query_word_count",
    "query_unique_word_count",
    "query_avg_word_length",
    "query_sentence_count",
    "question_mark_indicator",
    "number_indicator",
    "uppercase_token_count",
    "starts_what",
    "starts_why",
    "starts_how",
    "starts_when",
    "starts_where",
    "starts_who",
    "starts_which",
    "entity_count",
    "question_count",
    "comparison",
    "procedural",
    "policy",
    "summary",
    "definition",
    "multi_hop",
    "evidence_breadth",
]


def ensure_output_dir():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )