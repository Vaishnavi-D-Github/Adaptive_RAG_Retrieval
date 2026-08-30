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
    / "boosting_experiment"
)

RANDOM_STATE = 42

# Because there are only about 100 unique questions, keep this modest.
CV_FOLDS = 5

MIN_K = 1
MAX_K = 10

TARGET_COLUMN = "Candidate_Optimal_K"
ID_COLUMN = "ID"
QUESTION_COLUMN = "user_input"


# IMPORTANT:
# These must be obtainable before retrieval.
#
# Do not add:
# - retrieved context
# - generated answer
# - RAGAS scores
# - token counts
# - latency
# - retrieval distance
# - number of retrieved chunks
#
# Those would leak post-retrieval information.
BASE_FEATURES = [
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
    "intent",
    "comparison",
    "procedural",
    "policy",
    "summary",
    "definition",
    "multi_hop",
    "evidence_breadth",
]


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)