import os
import sys
import ast
import json
import re
import time
import math

import pandas as pd
from langchain_ollama import ChatOllama


# ================================================================
# CONFIG
# ================================================================

INPUT_FILE = "results/ragas/ragas_evaluation_dataset.csv"

OUTPUT_DIR = "results/evaluation/parallel"

MODEL = "qwen3:8b"
OLLAMA_BASE_URL = "http://localhost:11434"

TEMPERATURE = 0
THINK = False
NUM_PREDICT = 100

MAX_RETRIES = 2
RETRY_DELAY = 2

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# WORKER ARGUMENT
# ================================================================

if len(sys.argv) != 4:

    print(
        "Usage:"
    )

    print(
        "python src/evaluation_worker.py "
        "<worker_id> <start_index> <end_index>"
    )

    sys.exit(1)


worker_id = int(sys.argv[1])

start_index = int(sys.argv[2])

end_index = int(sys.argv[3])


CHECKPOINT_FILE = os.path.join(
    OUTPUT_DIR,
    f"worker_{worker_id}_checkpoint.csv"
)


print()
print("=" * 80)
print(f"WORKER {worker_id}")
print("=" * 80)

print(
    "Range:",
    start_index,
    "to",
    end_index
)

print(
    "Checkpoint:",
    CHECKPOINT_FILE
)


# ================================================================
# LOAD DATA
# ================================================================

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8"
)

df["Generation_Succeeded"] = (
    df["Generation_Succeeded"]
    .astype(str)
    .str.lower()
    .isin(["true", "1", "yes"])
)

df = df[
    df["Generation_Succeeded"]
].copy()

df["K"] = pd.to_numeric(
    df["K"],
    errors="coerce"
)

df = (
    df
    .sort_values(["ID", "K"])
    .reset_index(drop=True)
)


# ================================================================
# EXCLUDE THE FIVE ALREADY COMPLETED RECORDS
# ================================================================

existing_checkpoint = (
    "results/evaluation/evaluation_checkpoint.csv"
)

already_completed = set()

if os.path.exists(existing_checkpoint):

    old = pd.read_csv(
        existing_checkpoint,
        encoding="utf-8-sig"
    )

    for _, row in old.iterrows():

        already_completed.add(
            (
                str(row["ID"]),
                int(row["K"])
            )
        )

print(
    "Existing completed records:",
    len(already_completed)
)


# ================================================================
# PARTITION
# ================================================================

worker_df = df.iloc[
    start_index:end_index
].copy()

print(
    "Worker records:",
    len(worker_df)
)


# ================================================================
# LOAD WORKER CHECKPOINT
# ================================================================

if os.path.exists(
    CHECKPOINT_FILE
):

    checkpoint = pd.read_csv(
        CHECKPOINT_FILE,
        encoding="utf-8-sig"
    )

    completed = set()

    for _, row in checkpoint.iterrows():

        completed.add(
            (
                str(row["ID"]),
                int(row["K"])
            )
        )

    print(
        "Worker checkpoint records:",
        len(checkpoint)
    )

else:

    checkpoint = pd.DataFrame()

    completed = set()

    print(
        "No worker checkpoint."
    )


# ================================================================
# PARSING
# ================================================================

def extract_expected_documents(value):

    if pd.isna(value):
        return []

    return [
        x.strip()
        for x in str(value).split(";")
        if x.strip()
    ]


def extract_expected_pages(value):

    if pd.isna(value):
        return []

    return [
        int(x)
        for x in re.findall(
            r"\d+",
            str(value)
        )
    ]


def normalize_document(name):

    if name is None:
        return ""

    text = os.path.basename(
        str(name).strip().lower()
    )

    text = text.replace(
        ".pdf",
        ""
    )

    text = re.sub(
        r"[^a-z0-9]",
        "",
        text
    )

    return text


def parse_sources(value):

    if pd.isna(value):
        return []

    text = str(value).strip()

    try:

        return json.loads(text)

    except Exception:

        try:

            return ast.literal_eval(text)

        except Exception:

            return []


# ================================================================
# RETRIEVAL METRICS
# ================================================================

def retrieval_metrics(row):

    expected_docs = (
        extract_expected_documents(
            row["Source_Document"]
        )
    )

    expected_pages = (
        extract_expected_pages(
            row["Source_Page"]
        )
    )

    expected_docs = [
        normalize_document(x)
        for x in expected_docs
    ]

    sources = parse_sources(
        row["Retrieved_Sources"]
    )

    retrieved_docs = []

    retrieved_pages = []

    for source in sources:

        retrieved_docs.append(
            normalize_document(
                source.get(
                    "document",
                    ""
                )
            )
        )

        try:

            retrieved_pages.append(
                int(
                    source.get(
                        "page"
                    )
                )
            )

        except Exception:

            pass


    # ------------------------------------------------------------
    # Document coverage
    # ------------------------------------------------------------

    matched_docs = 0

    for expected in expected_docs:

        if any(
            expected in retrieved
            or retrieved in expected
            for retrieved in retrieved_docs
        ):

            matched_docs += 1


    if expected_docs:

        document_coverage = (
            matched_docs
            /
            len(expected_docs)
        )

    else:

        document_coverage = float("nan")


    # ------------------------------------------------------------
    # Page coverage
    # ------------------------------------------------------------

    matched_pages = sum(
        1
        for page in expected_pages
        if page in retrieved_pages
    )

    if expected_pages:

        page_coverage = (
            matched_pages
            /
            len(expected_pages)
        )

    else:

        page_coverage = float("nan")


    # ------------------------------------------------------------
    # Source hit
    # ------------------------------------------------------------

    source_hit = (
        1
        if matched_docs > 0
        else 0
    )


    # ------------------------------------------------------------
    # MRR
    # ------------------------------------------------------------

    mrr = 0.0

    first_rank = None

    for rank, retrieved in enumerate(
        retrieved_docs,
        start=1
    ):

        relevant = any(
            expected in retrieved
            or retrieved in expected
            for expected in expected_docs
        )

        if relevant:

            first_rank = rank

            mrr = 1.0 / rank

            break


    # ------------------------------------------------------------
    # nDCG
    # ------------------------------------------------------------

    relevance = []

    for retrieved in retrieved_docs:

        rel = 0

        for expected in expected_docs:

            if (
                expected in retrieved
                or retrieved in expected
            ):

                rel = 1
                break

        relevance.append(rel)


    def dcg(values):

        score = 0.0

        for rank, rel in enumerate(
            values,
            start=1
        ):

            score += (
                rel /
                math.log2(rank + 1)
            )

        return score


    actual = dcg(
        relevance
    )

    ideal = dcg(
        sorted(
            relevance,
            reverse=True
        )
    )

    ndcg = (
        actual / ideal
        if ideal > 0
        else 0.0
    )


    return {

        "Expected_Document_Count":
            len(expected_docs),

        "Expected_Page_Count":
            len(expected_pages),

        "Matched_Document_Count":
            matched_docs,

        "Matched_Page_Count":
            matched_pages,

        "Document_Coverage":
            document_coverage,

        "Page_Coverage":
            page_coverage,

        "Source_Hit":
            source_hit,

        "First_Relevant_Rank":
            first_rank,

        "MRR":
            mrr,

        "nDCG":
            ndcg,
    }


# ================================================================
# EVALUATION PROMPT
# ================================================================

def make_prompt(
    question,
    context,
    answer
):

    return f"""
You are evaluating an enterprise RAG system.

Use ONLY the supplied information.

Return ONLY valid JSON.

Do not provide explanations.

Required format:

{{
  "faithfulness": 0.0,
  "answer_relevancy": 0.0,
  "context_relevance": 0.0
}}

All values must be numbers from 0.0 to 1.0.

Faithfulness:
How well is the answer supported by the retrieved context?

Answer relevancy:
How directly does the answer address the question?

Context relevance:
How relevant is the retrieved context to the question?

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Return ONLY JSON.
"""


# ================================================================
# JSON PARSER
# ================================================================

def parse_evaluator_output(text):

    text = str(
        text
    ).strip()

    try:

        obj = json.loads(
            text
        )

    except Exception:

        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL
        )

        if not match:
            return None

        try:

            obj = json.loads(
                match.group(0)
            )

        except Exception:

            return None


    required = [
        "faithfulness",
        "answer_relevancy",
        "context_relevance",
    ]

    result = {}

    for metric in required:

        if metric not in obj:
            return None

        try:

            value = float(
                obj[metric]
            )

        except Exception:

            return None

        if not 0 <= value <= 1:
            return None

        result[metric] = value

    return result


# ================================================================
# OLLAMA
# ================================================================

evaluator = ChatOllama(
    model=MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=TEMPERATURE,
    think=THINK,
    num_predict=NUM_PREDICT,
)


# ================================================================
# EVALUATE RECORD
# ================================================================

def evaluate(row):

    prompt = make_prompt(
        str(row["user_input"]),
        str(row["Retrieved_Context"]),
        str(row["response"])
    )

    last_raw = ""

    last_error = ""

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            start = time.time()

            response = evaluator.invoke(
                prompt
            )

            elapsed = (
                time.time()
                -
                start
            )

            raw = str(
                response.content
            )

            last_raw = raw

            scores = parse_evaluator_output(
                raw
            )

            if scores is None:

                raise ValueError(
                    "Invalid JSON evaluator output"
                )

            return {

                **scores,

                "Evaluation_Time_Seconds":
                    elapsed,

                "Evaluation_Status":
                    "success",

                "Evaluation_Attempts":
                    attempt,

                "Evaluation_Error":
                    "",

                "Raw_Evaluator_Response":
                    raw,
            }


        except Exception as e:

            last_error = (
                f"{type(e).__name__}: "
                f"{e}"
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_DELAY
                )


    return {

        "faithfulness":
            float("nan"),

        "answer_relevancy":
            float("nan"),

        "context_relevance":
            float("nan"),

        "Evaluation_Time_Seconds":
            float("nan"),

        "Evaluation_Status":
            "failed",

        "Evaluation_Attempts":
            MAX_RETRIES,

        "Evaluation_Error":
            last_error,

        "Raw_Evaluator_Response":
            last_raw,
    }


# ================================================================
# MAIN LOOP
# ================================================================

for local_index, (_, row) in enumerate(
    worker_df.iterrows()
):

    global_index = (
        start_index
        +
        local_index
    )

    record_id = str(
        row["ID"]
    )

    k = int(
        row["K"]
    )

    key = (
        record_id,
        k
    )


    # Already completed by original run
    if key in already_completed:

        print(
            f"[W{worker_id}] SKIP existing "
            f"{record_id} K={k}"
        )

        continue


    # Already completed by this worker
    if key in completed:

        print(
            f"[W{worker_id}] SKIP checkpoint "
            f"{record_id} K={k}"
        )

        continue


    print()
    print(
        f"[W{worker_id}] "
        f"{global_index + 1}/{len(df)} "
        f"{record_id} K={k}"
    )


    metrics = retrieval_metrics(
        row
    )

    evaluation = evaluate(
        row
    )


    result = {

        "ID":
            record_id,

        "K":
            k,

        "Question":
            row["user_input"],

        "Source_Document":
            row["Source_Document"],

        "Source_Page":
            row["Source_Page"],

        "Model":
            row["Model"],

        "Num_Retrieved_Chunks":
            row["Num_Retrieved_Chunks"],

        **metrics,

        "Faithfulness":
            evaluation[
                "faithfulness"
            ],

        "Answer_Relevancy":
            evaluation[
                "answer_relevancy"
            ],

        "Context_Relevance":
            evaluation[
                "context_relevance"
            ],

        "Prompt_Tokens":
            row["Prompt_Tokens"],

        "Generated_Tokens":
            row["Generated_Tokens"],

        "Total_Tokens":
            row["Total_Tokens"],

        "Retrieval_Time_Seconds":
            row["Retrieval_Time_Seconds"],

        "Generation_Time_Seconds":
            row["Generation_Time_Seconds"],

        "Evaluation_Time_Seconds":
            evaluation[
                "Evaluation_Time_Seconds"
            ],

        "Evaluation_Status":
            evaluation[
                "Evaluation_Status"
            ],

        "Evaluation_Attempts":
            evaluation[
                "Evaluation_Attempts"
            ],

        "Evaluation_Error":
            evaluation[
                "Evaluation_Error"
            ],

        "Raw_Evaluator_Response":
            evaluation[
                "Raw_Evaluator_Response"
            ],
    }


    # ------------------------------------------------------------
    # Save immediately
    # ------------------------------------------------------------

    one = pd.DataFrame(
        [result]
    )

    if checkpoint.empty:

        checkpoint = one

    else:

        checkpoint = pd.concat(
            [
                checkpoint,
                one
            ],
            ignore_index=True
        )


    checkpoint.to_csv(
        CHECKPOINT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    completed.add(
        key
    )


    print(
        f"[W{worker_id}] "
        f"Status={result['Evaluation_Status']} "
        f"Faithfulness={result['Faithfulness']} "
        f"Time={result['Evaluation_Time_Seconds']}"
    )

    print(
        f"[W{worker_id}] CHECKPOINT SAVED "
        f"({len(checkpoint)} records)"
    )


print()
print("=" * 80)
print(f"WORKER {worker_id} COMPLETE")
print("=" * 80)

print(
    "Records:",
    len(checkpoint)
)

print(
    "File:",
    CHECKPOINT_FILE
)