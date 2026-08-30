# ================================================================
# evaluate_ragas_style.py
#
# FINAL RAG EVALUATION
#
# Evaluates the 959 successful Qwen3:8B experiment records.
#
# LLM-based metrics:
#   - Faithfulness
#   - Answer Relevancy
#   - Context Relevance
#
# Retrieval-grounded metrics:
#   - Document Coverage
#   - Page Coverage
#   - Source Hit
#   - MRR
#   - nDCG
#
# Engineering features:
#   - Qwen3:8B evaluator
#   - Thinking disabled
#   - Temperature = 0
#   - Strict JSON output
#   - Retry handling
#   - Checkpoint after EVERY record
#   - Automatic resume
#   - Raw evaluator response logging
#   - Failure logging
#
# INPUT:
#   results/ragas/ragas_evaluation_dataset.csv
#
# OUTPUT:
#   results/evaluation/
# ================================================================


import os
import ast
import json
import re
import time
import math

import pandas as pd

from langchain_ollama import ChatOllama


# ================================================================
# CONFIGURATION
# ================================================================

INPUT_FILE = (
    "results/ragas/ragas_evaluation_dataset.csv"
)

OUTPUT_DIR = (
    "results/evaluation"
)

CHECKPOINT_FILE = os.path.join(
    OUTPUT_DIR,
    "evaluation_checkpoint.csv"
)

FINAL_FILE = os.path.join(
    OUTPUT_DIR,
    "evaluation_final.csv"
)

FAILURE_FILE = os.path.join(
    OUTPUT_DIR,
    "evaluation_failures.csv"
)

RAW_RESPONSE_FILE = os.path.join(
    OUTPUT_DIR,
    "evaluator_raw_responses.jsonl"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "evaluation_summary.csv"
)


# ------------------------------------------------
# Evaluator
# ------------------------------------------------

EVALUATOR_MODEL = "qwen3:8b"

OLLAMA_BASE_URL = (
    "http://localhost:11434"
)

TEMPERATURE = 0

THINK = False


# ------------------------------------------------
# Retry configuration
# ------------------------------------------------

MAX_RETRIES = 3

RETRY_DELAY_SECONDS = 3


# ------------------------------------------------
# Checkpoint frequency
# ------------------------------------------------

CHECKPOINT_EVERY = 1


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
print("FINAL RAG EVALUATION")
print("=" * 80)

print()
print(
    "Evaluator:",
    EVALUATOR_MODEL
)

print(
    "Thinking:",
    THINK
)

print(
    "Temperature:",
    TEMPERATURE
)


# ================================================================
# LOAD DATA
# ================================================================

print()
print("=" * 80)
print("LOADING DATASET")
print("=" * 80)

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Dataset not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8"
)

print()
print(
    "Total records:",
    len(df)
)


# ================================================================
# REQUIRED COLUMNS
# ================================================================

required_columns = [
    "ID",
    "K",
    "user_input",
    "retrieved_contexts",
    "response",
    "reference",
    "Source_Document",
    "Source_Page",
    "Retrieved_Sources",
    "Retrieved_Context",
    "Generated_Answer",
    "Num_Retrieved_Chunks",
    "Prompt_Tokens",
    "Generated_Tokens",
    "Total_Tokens",
    "Retrieval_Time_Seconds",
    "Generation_Time_Seconds",
    "Generation_Succeeded",
    "Model",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:

    print()
    print(
        "Missing required columns:"
    )

    for column in missing:
        print(
            " -",
            column
        )

    raise ValueError(
        "Dataset schema does not match expected schema."
    )

print()
print(
    "Dataset schema: PASS"
)


# ================================================================
# KEEP ONLY SUCCESSFUL GENERATIONS
# ================================================================

df["Generation_Succeeded"] = (
    df["Generation_Succeeded"]
    .astype(str)
    .str.lower()
    .isin(
        [
            "true",
            "1",
            "yes"
        ]
    )
)

df = df[
    df["Generation_Succeeded"]
].copy()

print()
print(
    "Successful records:",
    len(df)
)


# ================================================================
# SORT EXPERIMENT
# ================================================================

df["K"] = pd.to_numeric(
    df["K"],
    errors="coerce"
)

df = df.sort_values(
    ["ID", "K"]
).reset_index(
    drop=True
)


# ================================================================
# PARSING FUNCTIONS
# ================================================================

def parse_python_list(value):

    """
    Parse a CSV cell containing a Python-style list.
    """

    if pd.isna(value):

        return []

    text = str(
        value
    ).strip()

    if not text:

        return []

    try:

        parsed = ast.literal_eval(
            text
        )

        if isinstance(
            parsed,
            list
        ):

            return parsed

    except Exception:

        pass

    return []


def normalize_document_name(name):

    """
    Convert document identifiers to a canonical GAO identifier.

    Examples:
        GAO-24-105658 -> GAO-24-105658
        d24105658.pdf -> GAO-24-105658
        gao-24-105658.pdf -> GAO-24-105658

    This prevents the retrieval metrics from incorrectly reporting
    zero coverage when the source metadata uses a different filename
    convention from the ground-truth document identifier.
    """

    if name is None or pd.isna(name):
        return ""

    value = os.path.basename(str(name).strip()).lower()

    # Canonical GAO identifier already present.
    m = re.search(r"gao[-_ ]?(\d{2})[-_ ]?(\d{3,6})", value)
    if m:
        return f"GAO-{m.group(1)}-{m.group(2)}".upper()

    # Chunk/PDF convention such as d24105658.pdf:
    # first two digits are year, remaining six are report number.
    m = re.fullmatch(r"d(\d{2})(\d{6})", value.replace(".pdf", ""))
    if m:
        return f"GAO-{m.group(1)}-{m.group(2)}".upper()

    # Handle common gao-04-321-highlights style names.
    m = re.search(r"gao[-_ ]?(\d{2})[-_ ]?(\d{3,6})", value)
    if m:
        return f"GAO-{m.group(1)}-{m.group(2)}".upper()

    return re.sub(r"[^a-z0-9]", "", value)


def extract_expected_documents(value):

    """
    Source_Document may contain multiple documents:

        GAO-24-107733; GAO-24-107231
    """

    if pd.isna(value):

        return []

    return [
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    ]


def extract_expected_pages(value):

    """
    Extract page numbers from strings such as:

        PDF p. 2
        PDF pp. 2-3
        PDF pp. 2, 4
        PDF p. 2; PDF pp. 1-2
    """

    if pd.isna(value):

        return []

    text = str(
        value
    )

    numbers = re.findall(
        r"\d+",
        text
    )

    return [
        int(x)
        for x in numbers
    ]


def parse_retrieved_sources(value):

    """
    Parse Retrieved_Sources JSON/list representation.

    Expected structure:

        [
          {
            "rank": 1,
            "document": "...",
            "page": 2,
            "distance": 0.5
          }
        ]
    """

    if pd.isna(value):

        return []

    text = str(
        value
    ).strip()

    if not text:

        return []

    try:

        parsed = json.loads(
            text
        )

        if isinstance(
            parsed,
            list
        ):

            return parsed

    except Exception:

        pass

    try:

        parsed = ast.literal_eval(
            text
        )

        if isinstance(
            parsed,
            list
        ):

            return parsed

    except Exception:

        pass

    return []


# ================================================================
# RETRIEVAL METRICS
# ================================================================

def calculate_retrieval_metrics(row):

    expected_documents = (
        extract_expected_documents(
            row["Source_Document"]
        )
    )

    expected_pages = (
        extract_expected_pages(
            row["Source_Page"]
        )
    )

    retrieved_sources = (
        parse_retrieved_sources(
            row["Retrieved_Sources"]
        )
    )

    expected_doc_norm = [
        normalize_document_name(
            x
        )
        for x in expected_documents
    ]

    retrieved_docs = []

    retrieved_pages = []

    for source in retrieved_sources:

        document = source.get(
            "document",
            ""
        )

        page = source.get(
            "page",
            None
        )

        retrieved_docs.append(
            normalize_document_name(
                document
            )
        )

        try:

            retrieved_pages.append(
                int(page)
            )

        except Exception:

            pass


    # ------------------------------------------------------------
    # Document Coverage
    # ------------------------------------------------------------

    if expected_doc_norm:

        matched_documents = sum(
            1
            for expected in expected_doc_norm
            if any(
                expected in retrieved
                or retrieved in expected
                for retrieved in retrieved_docs
            )
        )

        document_coverage = (
            matched_documents
            /
            len(expected_doc_norm)
        )

    else:

        matched_documents = 0

        document_coverage = float(
            "nan"
        )


    # ------------------------------------------------------------
    # Page Coverage
    # ------------------------------------------------------------
    #
    # IMPORTANT:
    # Source_Page in the dataset may refer to the report's printed
    # page / source-page annotation, while Retrieved_Sources["page"]
    # may be the physical PDF page. These namespaces are not assumed
    # to be identical.
    #
    # Therefore this evaluator does NOT manufacture a page-coverage
    # score from incomparable page numbers. It preserves the expected
    # pages and retrieved physical pages separately.
    #
    # A reliable page mapping can be added later once the ingestion
    # metadata explicitly records both physical_pdf_page and
    # report_page.

    matched_pages = None
    page_coverage = float("nan")
    page_coverage_status = (
        "not_computed_pages_not_proven_same_namespace"
    )


    # ------------------------------------------------------------
    # Source Hit
    # ------------------------------------------------------------

    source_hit = (
        1
        if matched_documents > 0
        else 0
    )


    # ------------------------------------------------------------
    # Reciprocal Rank
    # ------------------------------------------------------------

    reciprocal_rank = 0.0

    first_relevant_rank = None

    for index, retrieved_doc in enumerate(
        retrieved_docs,
        start=1
    ):

        relevant = any(
            expected in retrieved_doc
            or retrieved_doc in expected
            for expected in expected_doc_norm
        )

        if relevant:

            first_relevant_rank = index

            reciprocal_rank = (
                1.0
                /
                index
            )

            break


    # ------------------------------------------------------------
    # nDCG
    # ------------------------------------------------------------

    relevances = []

    for retrieved_doc in retrieved_docs:

        relevance = 0

        for expected in expected_doc_norm:

            if (
                expected in retrieved_doc
                or
                retrieved_doc in expected
            ):

                relevance = 1

                break

        relevances.append(
            relevance
        )


    def dcg(values):

        score = 0.0

        for rank, relevance in enumerate(
            values,
            start=1
        ):

            score += (
                relevance
                /
                math.log2(
                    rank + 1
                )
            )

        return score


    actual_dcg = dcg(
        relevances
    )

    ideal_relevances = sorted(
        relevances,
        reverse=True
    )

    ideal_dcg = dcg(
        ideal_relevances
    )

    if ideal_dcg > 0:

        ndcg = (
            actual_dcg
            /
            ideal_dcg
        )

    else:

        ndcg = 0.0


    return {

        "Expected_Document_Count":
            len(expected_documents),

        "Expected_Page_Count":
            len(expected_pages),

        "Matched_Document_Count":
            matched_documents,

        "Matched_Page_Count":
            matched_pages,

        "Document_Coverage":
            document_coverage,

        "Page_Coverage":
            page_coverage,

        "Page_Coverage_Status":
            page_coverage_status,

        "Retrieved_Physical_Pages":
            retrieved_pages,

        "Expected_Source_Pages":
            expected_pages,

        "Source_Hit":
            source_hit,

        "First_Relevant_Rank":
            first_relevant_rank,

        "MRR":
            reciprocal_rank,

        "nDCG":
            ndcg,
    }


# ================================================================
# EVALUATOR
# ================================================================

print()
print("=" * 80)
print("INITIALIZING QWEN3:8B EVALUATOR")
print("=" * 80)

evaluator = ChatOllama(
    model=EVALUATOR_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=TEMPERATURE,
    think=THINK,
)

print()
print(
    "Evaluator initialized."
)

print()
print(
    "NOTE: This script uses Qwen3:8B as a deterministic LLM judge "
    "with a RAGAS-inspired three-metric rubric. It does NOT call "
    "the ragas Python package directly."
)


# ================================================================
# EVALUATOR PROMPT
# ================================================================

def build_evaluation_prompt(
    question,
    context,
    answer
):

    return f"""
You are evaluating an enterprise Retrieval-Augmented Generation (RAG) system.

You MUST evaluate the answer ONLY using the supplied question,
retrieved context, and generated answer.

Do not use outside knowledge.

Return ONLY valid JSON.
Do not use Markdown.
Do not write explanations outside the JSON.

Use EXACTLY this structure:

{{
  "faithfulness": 0.0,
  "answer_relevancy": 0.0,
  "context_relevance": 0.0
}}

All three values MUST be numbers between 0.0 and 1.0.

Definitions:

FAITHFULNESS:
How well is the generated answer supported by the retrieved context?
1.0 = all substantive claims are supported.
0.0 = the answer is completely unsupported.

ANSWER_RELEVANCY:
How directly and completely does the generated answer address the question?
1.0 = directly answers the question.
0.0 = does not answer the question.

CONTEXT_RELEVANCE:
How relevant is the retrieved context to answering the question?
1.0 = the context is highly relevant.
0.0 = the context is irrelevant.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Return ONLY the JSON object.
"""


# ================================================================
# JSON EXTRACTION
# ================================================================

def extract_json(text):

    """
    Extract JSON from evaluator output.

    Handles accidental surrounding text.
    """

    if text is None:

        return None

    text = str(
        text
    ).strip()

    # ------------------------------------------------------------
    # Direct JSON
    # ------------------------------------------------------------

    try:

        obj = json.loads(
            text
        )

        if isinstance(
            obj,
            dict
        ):

            return obj

    except Exception:

        pass

    # ------------------------------------------------------------
    # JSON embedded in text
    # ------------------------------------------------------------

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if match:

        try:

            obj = json.loads(
                match.group(0)
            )

            if isinstance(
                obj,
                dict
            ):

                return obj

        except Exception:

            pass

    return None


# ================================================================
# VALIDATE SCORES
# ================================================================

def validate_scores(obj):

    if not isinstance(
        obj,
        dict
    ):

        return None

    required = [
        "faithfulness",
        "answer_relevancy",
        "context_relevance",
    ]

    scores = {}

    for metric in required:

        if metric not in obj:

            return None

        try:

            value = float(
                obj[metric]
            )

        except Exception:

            return None

        if not (
            0.0
            <=
            value
            <=
            1.0
        ):

            return None

        scores[metric] = value

    return scores


# ================================================================
# EVALUATE ONE RECORD
# ================================================================

def evaluate_record(row):

    question = str(
        row["user_input"]
    ).strip()

    answer = str(
        row["response"]
    ).strip()

    context = str(
        row["Retrieved_Context"]
    ).strip()

    prompt = build_evaluation_prompt(
        question,
        context,
        answer
    )

    last_error = None

    last_raw_response = ""

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            start_time = time.time()

            response = evaluator.invoke(
                prompt
            )

            evaluation_time = (
                time.time()
                -
                start_time
            )

            raw = str(
                response.content
            )

            last_raw_response = raw

            parsed = extract_json(
                raw
            )

            scores = validate_scores(
                parsed
            )

            if scores is None:

                raise ValueError(
                    "Evaluator response "
                    "could not be parsed "
                    "into valid scores."
                )

            return {

                "faithfulness":
                    scores[
                        "faithfulness"
                    ],

                "answer_relevancy":
                    scores[
                        "answer_relevancy"
                    ],

                "context_relevance":
                    scores[
                        "context_relevance"
                    ],

                "Evaluation_Time_Seconds":
                    evaluation_time,

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
                f"{str(e)}"
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_DELAY_SECONDS
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
            last_raw_response,
    }


# ================================================================
# LOAD CHECKPOINT
# ================================================================

print()
print("=" * 80)
print("CHECKPOINT / RESUME")
print("=" * 80)

if os.path.exists(
    CHECKPOINT_FILE
):

    checkpoint = pd.read_csv(
        CHECKPOINT_FILE,
        encoding="utf-8-sig"
    )

    print()
    print(
        "Existing checkpoint found."
    )

    print(
        "Completed records:",
        len(checkpoint)
    )

else:

    checkpoint = pd.DataFrame()

    print()
    print(
        "No existing checkpoint."
    )

    print(
        "Starting from record 1."
    )


# ================================================================
# BUILD COMPLETED KEY SET
# ================================================================

completed_keys = set()

if not checkpoint.empty:

    for _, row in checkpoint.iterrows():

        key = (
            str(row["ID"]),
            int(row["K"])
        )

        completed_keys.add(
            key
        )


# ================================================================
# MAIN EVALUATION LOOP
# ================================================================

print()
print("=" * 80)
print("STARTING EVALUATION")
print("=" * 80)

print()
print(
    "Total successful records:",
    len(df)
)

print(
    "Already completed:",
    len(completed_keys)
)

print(
    "Remaining:",
    len(df) - len(completed_keys)
)


new_results = []

run_start = time.time()

for index, row in df.iterrows():

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

    # ------------------------------------------------------------
    # Skip completed record
    # ------------------------------------------------------------

    if key in completed_keys:

        continue


    overall_number = (
        index + 1
    )

    print()
    print(
        "-" * 80
    )

    print(
        f"RECORD {overall_number}/{len(df)}"
    )

    print(
        "ID:",
        record_id
    )

    print(
        "K:",
        k
    )


    # ============================================================
    # RETRIEVAL METRICS
    # ============================================================

    retrieval_metrics = (
        calculate_retrieval_metrics(
            row
        )
    )


    # ============================================================
    # LLM EVALUATION
    # ============================================================

    evaluation_metrics = (
        evaluate_record(
            row
        )
    )


    # ============================================================
    # BUILD RESULT
    # ============================================================

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

        "Expected_Source_Pages":
            retrieval_metrics["Expected_Source_Pages"],

        "Retrieved_Physical_Pages":
            retrieval_metrics["Retrieved_Physical_Pages"],

        "Page_Coverage_Status":
            retrieval_metrics["Page_Coverage_Status"],

        "Model":
            row["Model"],

        "Num_Retrieved_Chunks":
            row["Num_Retrieved_Chunks"],

        # --------------------------------------------------------
        # Retrieval metrics
        # --------------------------------------------------------

        **retrieval_metrics,

        # --------------------------------------------------------
        # LLM metrics
        # --------------------------------------------------------

        "Faithfulness":
            evaluation_metrics[
                "faithfulness"
            ],

        "Answer_Relevancy":
            evaluation_metrics[
                "answer_relevancy"
            ],

        "Context_Relevance":
            evaluation_metrics[
                "context_relevance"
            ],

        # --------------------------------------------------------
        # Performance
        # --------------------------------------------------------

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
            evaluation_metrics[
                "Evaluation_Time_Seconds"
            ],

        # --------------------------------------------------------
        # Evaluation status
        # --------------------------------------------------------

        "Evaluation_Status":
            evaluation_metrics[
                "Evaluation_Status"
            ],

        "Evaluation_Attempts":
            evaluation_metrics[
                "Evaluation_Attempts"
            ],

        "Evaluation_Error":
            evaluation_metrics[
                "Evaluation_Error"
            ],

        "Raw_Evaluator_Response":
            evaluation_metrics[
                "Raw_Evaluator_Response"
            ],
    }


    new_results.append(
        result
    )


    # ============================================================
    # PRINT CURRENT RESULT
    # ============================================================

    status = result[
        "Evaluation_Status"
    ]

    print()

    print(
        "Faithfulness:",
        result["Faithfulness"]
    )

    print(
        "Answer Relevancy:",
        result["Answer_Relevancy"]
    )

    print(
        "Context Relevance:",
        result["Context_Relevance"]
    )

    print(
        "Document Coverage:",
        result["Document_Coverage"]
    )

    print(
        "Page Coverage:",
        result["Page_Coverage"]
    )

    print(
        "MRR:",
        result["MRR"]
    )

    print(
        "nDCG:",
        result["nDCG"]
    )

    print(
        "Status:",
        status
    )


    # ============================================================
    # CHECKPOINT
    # ============================================================

    if (
        len(new_results)
        %
        CHECKPOINT_EVERY
        ==
        0
    ):

        new_df = pd.DataFrame(
            new_results
        )

        if checkpoint.empty:

            checkpoint = new_df

        else:

            checkpoint = pd.concat(
                [
                    checkpoint,
                    new_df,
                ],
                ignore_index=True
            )

        checkpoint.to_csv(
            CHECKPOINT_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        # --------------------------------------------------------
        # Update completed keys
        # --------------------------------------------------------

        completed_keys.add(
            key
        )

        print()
        print(
            "CHECKPOINT SAVED"
        )

        print(
            "Completed:",
            len(checkpoint),
            "/",
            len(df)
        )

        # --------------------------------------------------------
        # Clear new-results buffer
        # --------------------------------------------------------

        new_results = []


# ================================================================
# FINAL LOAD
# ================================================================

print()
print("=" * 80)
print("FINALIZING RESULTS")
print("=" * 80)

if os.path.exists(
    CHECKPOINT_FILE
):

    final_df = pd.read_csv(
        CHECKPOINT_FILE,
        encoding="utf-8-sig"
    )

else:

    final_df = pd.DataFrame()


# ================================================================
# SORT FINAL RESULTS
# ================================================================

if not final_df.empty:

    final_df["K"] = pd.to_numeric(
        final_df["K"],
        errors="coerce"
    )

    final_df = (
        final_df
        .sort_values(
            ["ID", "K"]
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# SAVE FINAL
# ================================================================

final_df.to_csv(
    FINAL_FILE,
    index=False,
    encoding="utf-8-sig"
)

print()
print(
    "Final evaluation saved:"
)

print(
    FINAL_FILE
)


# ================================================================
# SAVE FAILURES
# ================================================================

if not final_df.empty:

    failures = final_df[
        final_df[
            "Evaluation_Status"
        ]
        !=
        "success"
    ].copy()

else:

    failures = pd.DataFrame()


failures.to_csv(
    FAILURE_FILE,
    index=False,
    encoding="utf-8-sig"
)

print()
print(
    "Failure file saved:"
)

print(
    FAILURE_FILE
)


# ================================================================
# SUMMARY BY K
# ================================================================

if not final_df.empty:

    print()
    print("=" * 80)
    print("SUMMARY BY K")
    print("=" * 80)


    numeric_columns = [
        "Faithfulness",
        "Answer_Relevancy",
        "Context_Relevance",
        "Document_Coverage",
        "Page_Coverage",
        "Source_Hit",
        "MRR",
        "nDCG",
        "Prompt_Tokens",
        "Generated_Tokens",
        "Total_Tokens",
        "Retrieval_Time_Seconds",
        "Generation_Time_Seconds",
        "Evaluation_Time_Seconds",
    ]


    existing_numeric = [
        column
        for column in numeric_columns
        if column in final_df.columns
    ]


    summary = (
        final_df
        .groupby("K")[
            existing_numeric
        ]
        .mean(
            numeric_only=True
        )
        .reset_index()
    )


    # ------------------------------------------------------------
    # Add record count
    # ------------------------------------------------------------

    counts = (
        final_df
        .groupby("K")
        .size()
        .reset_index(
            name="Record_Count"
        )
    )

    summary = summary.merge(
        counts,
        on="K",
        how="left"
    )


    # ------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    print()

    print(
        summary.to_string(
            index=False
        )
    )


    print()
    print(
        "Summary saved:"
    )

    print(
        SUMMARY_FILE
    )


# ================================================================
# FINAL STATISTICS
# ================================================================

print()
print("=" * 80)
print("FINAL EVALUATION STATISTICS")
print("=" * 80)

print()

print(
    "Input successful records:",
    len(df)
)

print(
    "Evaluated records:",
    len(final_df)
)

if not final_df.empty:
    successful_evaluations = (
        final_df["Evaluation_Status"] == "success"
    ).sum()
else:
    successful_evaluations = 0

print(
    "Successful evaluations:",
    successful_evaluations
)

print(
    "Failed evaluations:",
    len(failures)
)

print()

if not final_df.empty:

    valid_faithfulness = (
        pd.to_numeric(
            final_df["Faithfulness"],
            errors="coerce"
        )
        .notna()
        .sum()
    )

    valid_relevancy = (
        pd.to_numeric(
            final_df["Answer_Relevancy"],
            errors="coerce"
        )
        .notna()
        .sum()
    )

    valid_context = (
        pd.to_numeric(
            final_df["Context_Relevance"],
            errors="coerce"
        )
        .notna()
        .sum()
    )

    print(
        "Valid Faithfulness:",
        valid_faithfulness
    )

    print(
        "Valid Answer Relevancy:",
        valid_relevancy
    )

    print(
        "Valid Context Relevance:",
        valid_context
    )


# ================================================================
# COMPLETE
# ================================================================

elapsed = (
    time.time()
    -
    run_start
)

print()
print("=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)

print()
print(
    "Methodology: Qwen3:8B custom LLM judge; "
    "RAGAS Python package is not invoked by this script."
)

print(
    "Document-level retrieval metrics use canonical GAO identifier matching."
)

print(
    "Page Coverage is intentionally left NaN until physical PDF pages "
    "and report/source pages are explicitly mapped."
)

print()

print(
    "Elapsed time:",
    f"{elapsed / 3600:.2f} hours"
)

print()

print(
    "Checkpoint:"
)

print(
    CHECKPOINT_FILE
)

print()

print(
    "Final:"
)

print(
    FINAL_FILE
)

print()

print(
    "Failures:"
)

print(
    FAILURE_FILE
)

print()

print(
    "Summary:"
)

print(
    SUMMARY_FILE
)

print()
print("=" * 80)