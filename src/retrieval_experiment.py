import json
import os
import re
import time

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "enterprise_documents"

QUESTIONS_FILE = (
    "data/evaluation/enterprise_query_evaluation_set_100.csv"
)

OUTPUT_DIR = "./results/retrieval_v2"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

K_VALUES = list(range(1, 11))


# ============================================================
# DOCUMENT NAME MAPPING
# ============================================================

DOCUMENT_MAPPING = {
    "GAO-04-321": "gao-04-321-highlights.pdf",
    "GAO-16-885T": "gao-16-885t.pdf",
    "GAO-22-104467": "gao-22-104467-highlights.pdf",
    "GAO-23-105480": "gao-23-105480.pdf",
    "GAO-24-105658": "d24105658.pdf",
    "GAO-24-106576": "d24106576.pdf",
    "GAO-24-106916": "d24106916.pdf",
    "GAO-24-107231": "gao-24-107231.pdf",
    "GAO-24-107733": "gao-24-107733.pdf",
    "GAO-25-108440": "gao-25-108440.pdf",
    "GAO-26-107529": "gao-26-107529.pdf",
    "GAO-26-107974": "gao-26-107974.pdf",
    "GAO-26-108011": "gao-26-108011-highlights.pdf",
    "GAO-26-108615": "gao-26-108615.pdf",
    "GAO-26-900644": "gao-26-900644.pdf",
}


# ============================================================
# HELPERS
# ============================================================

def normalize_document_name(name):

    name = str(name).strip().lower()

    if name.endswith(".pdf"):
        name = name[:-4]

    return name


def normalize_page(page):

    page = str(page).strip().lower()

    # Remove common prefixes
    page = page.replace("pdf", "")
    page = page.replace("pages", "")
    page = page.replace("page", "")
    page = page.replace("pp.", "")
    page = page.replace("pp", "")
    page = page.replace("p.", "")
    page = page.replace("p", "")

    page = page.strip()

    return page


def expand_page_expression(expression):

    """
    Examples:

        "1-3" -> {1,2,3}
        "2"   -> {2}
        "1, 3" -> {1,3}
    """

    expression = str(expression).strip()

    pages = set()

    # Split comma-separated pages
    parts = expression.split(",")

    for part in parts:

        part = part.strip()

        if not part:
            continue

        # Range
        if "-" in part:

            pieces = part.split("-")

            if len(pieces) == 2:

                try:

                    start = int(
                        pieces[0].strip()
                    )

                    end = int(
                        pieces[1].strip()
                    )

                    for p in range(
                        start,
                        end + 1
                    ):

                        pages.add(p)

                except ValueError:
                    pass

        else:

            try:

                pages.add(
                    int(part)
                )

            except ValueError:
                pass

    return pages


# ============================================================
# PARSE EXPECTED DOCUMENTS + PAGES
# ============================================================

def parse_expected_sources(
    source_document,
    source_page
):

    documents = [
        x.strip()
        for x in str(source_document).split(";")
    ]

    page_text = str(source_page).strip()

    expected = []

    # --------------------------------------------------------
    # Multi-document with semicolon-separated pages
    #
    # Example:
    # PDF pp. 2; 1-2
    # --------------------------------------------------------

    if ";" in page_text:

        page_parts = [
            x.strip()
            for x in page_text.split(";")
        ]

    else:

        # ----------------------------------------------------
        # If documents are multiple but page text is shared:
        #
        # PDF pp. 1-3
        # ----------------------------------------------------

        page_clean = page_text

        page_clean = re.sub(
            r"^pdf\s*",
            "",
            page_clean,
            flags=re.IGNORECASE
        )

        page_clean = re.sub(
            r"^pp?\.\s*",
            "",
            page_clean,
            flags=re.IGNORECASE
        )

        page_parts = [
            page_clean
        ] * len(documents)

    # --------------------------------------------------------
    # Make sure there is one page expression per document
    # --------------------------------------------------------

    if len(page_parts) < len(documents):

        page_parts.extend(
            [page_parts[-1]]
            * (
                len(documents)
                - len(page_parts)
            )
        )

    # --------------------------------------------------------
    # Build expected source list
    # --------------------------------------------------------

    for i, document in enumerate(documents):

        document = document.strip()

        if not document:
            continue

        filename = DOCUMENT_MAPPING.get(
            document
        )

        if filename is None:

            filename = document

        page_expression = page_parts[i]

        page_expression = normalize_page(
            page_expression
        )

        pages = expand_page_expression(
            page_expression
        )

        expected.append(
            {
                "evaluation_document":
                    document,

                "indexed_document":
                    filename,

                "expected_pages":
                    sorted(pages)
            }
        )

    return expected


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    question,
    embedding_model,
    collection,
    k
):

    start = time.perf_counter()

    embedding = embedding_model.encode(
        [question]
    )[0]

    results = collection.query(
        query_embeddings=[
            embedding.tolist()
        ],
        n_results=k
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    retrieved_sources = []

    for i, metadata in enumerate(
        metadatas
    ):

        retrieved_sources.append(
            {
                "rank": i + 1,

                "document":
                    metadata.get(
                        "document"
                    ),

                "page":
                    metadata.get(
                        "page"
                    ),

                "distance":
                    (
                        distances[i]
                        if i < len(distances)
                        else None
                    )
            }
        )

    return (
        retrieved_sources,
        elapsed
    )


# ============================================================
# SOURCE COVERAGE
# ============================================================

def calculate_source_coverage(
    expected_sources,
    retrieved_sources
):

    retrieved_documents = set()

    retrieved_pages = {}

    for source in retrieved_sources:

        document = normalize_document_name(
            source["document"]
        )

        try:
            page = int(
                normalize_page(
                    source["page"]
                )
            )
        except:
            page = None

        retrieved_documents.add(
            document
        )

        if document not in retrieved_pages:
            retrieved_pages[document] = set()

        if page is not None:
            retrieved_pages[
                document
            ].add(page)

    document_results = []

    page_results = []

    for expected in expected_sources:

        expected_document = (
            normalize_document_name(
                expected[
                    "indexed_document"
                ]
            )
        )

        expected_pages = set(
            expected[
                "expected_pages"
            ]
        )

        document_found = (
            expected_document
            in retrieved_documents
        )

        retrieved_page_set = (
            retrieved_pages.get(
                expected_document,
                set()
            )
        )

        matched_pages = (
            expected_pages
            & retrieved_page_set
        )

        page_found = (
            len(matched_pages) > 0
        )

        document_results.append(
            document_found
        )

        page_results.append(
            page_found
        )

    if expected_sources:

        document_coverage = (
            sum(document_results)
            / len(document_results)
        )

        page_coverage = (
            sum(page_results)
            / len(page_results)
        )

    else:

        document_coverage = 0.0
        page_coverage = 0.0

    # For multi-document questions,
    # require every expected source.
    all_documents_found = all(
        document_results
    ) if document_results else False

    all_pages_found = all(
        page_results
    ) if page_results else False

    return {
        "document_coverage":
            document_coverage,

        "page_coverage":
            page_coverage,

        "all_documents_found":
            all_documents_found,

        "all_pages_found":
            all_pages_found
    }


# ============================================================
# MAIN
# ============================================================

def run_experiment():

    print("=" * 80)
    print("RETRIEVAL EXPERIMENT V2")
    print("=" * 80)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load questions
    # --------------------------------------------------------

    print("\nLoading evaluation dataset...")

    df = pd.read_csv(
        QUESTIONS_FILE,
        encoding="utf-8"
    )

    print(
        f"Questions loaded: {len(df)}"
    )

    required_columns = [
        "ID",
        "Question",
        "Reference_Answer",
        "Source_Document",
        "Source_Page"
    ]

    missing = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    embedding_model = (
        SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )
    )

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    # --------------------------------------------------------
    # Experiment
    # --------------------------------------------------------

    total_runs = (
        len(df)
        * len(K_VALUES)
    )

    print(
        f"\nExpected retrieval runs: "
        f"{total_runs}"
    )

    results = []

    completed = 0

    for _, row in df.iterrows():

        question_id = row["ID"]

        question = row["Question"]

        expected_sources = (
            parse_expected_sources(
                row["Source_Document"],
                row["Source_Page"]
            )
        )

        for k in K_VALUES:

            retrieved_sources, elapsed = (
                retrieve(
                    question,
                    embedding_model,
                    collection,
                    k
                )
            )

            coverage = (
                calculate_source_coverage(
                    expected_sources,
                    retrieved_sources
                )
            )

            completed += 1

            print(
                f"{completed:4d}/{total_runs} "
                f"{question_id} "
                f"K={k:2d} "
                f"DocCov="
                f"{coverage['document_coverage']:.2f} "
                f"PageCov="
                f"{coverage['page_coverage']:.2f}"
            )

            results.append(
                {
                    "ID":
                        question_id,

                    "Question":
                        question,

                    "Reference_Answer":
                        row[
                            "Reference_Answer"
                        ],

                    "Expected_Source_Document":
                        row[
                            "Source_Document"
                        ],

                    "Expected_Source_Page":
                        row[
                            "Source_Page"
                        ],

                    "K":
                        k,

                    "Expected_Sources":
                        json.dumps(
                            expected_sources,
                            ensure_ascii=False
                        ),

                    "Retrieved_Sources":
                        json.dumps(
                            retrieved_sources,
                            ensure_ascii=False
                        ),

                    "Document_Coverage":
                        coverage[
                            "document_coverage"
                        ],

                    "Page_Coverage":
                        coverage[
                            "page_coverage"
                        ],

                    "All_Documents_Found":
                        coverage[
                            "all_documents_found"
                        ],

                    "All_Pages_Found":
                        coverage[
                            "all_pages_found"
                        ],

                    "Retrieval_Time_Seconds":
                        elapsed
                }
            )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    output_csv = os.path.join(
        OUTPUT_DIR,
        "retrieval_experiment_v2.csv"
    )

    results_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig"
    )

    output_json = os.path.join(
        OUTPUT_DIR,
        "retrieval_experiment_v2.json"
    )

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Determine K_evidence
    #
    # Here we define retrieval success as:
    #
    # all expected documents retrieved.
    #
    # We keep page coverage separately.
    # --------------------------------------------------------

    k_rows = []

    for question_id, group in (
        results_df.groupby("ID")
    ):

        group = group.sort_values(
            "K"
        )

        successful = group[
            group[
                "All_Documents_Found"
            ] == True
        ]

        if len(successful) > 0:

            k_evidence = int(
                successful.iloc[0]["K"]
            )

        else:

            k_evidence = None

        first = group.iloc[0]

        k_rows.append(
            {
                "ID":
                    question_id,

                "Question":
                    first["Question"],

                "Reference_Answer":
                    first[
                        "Reference_Answer"
                    ],

                "Source_Document":
                    first[
                        "Expected_Source_Document"
                    ],

                "Source_Page":
                    first[
                        "Expected_Source_Page"
                    ],

                "K_evidence":
                    k_evidence
            }
        )

    k_df = pd.DataFrame(
        k_rows
    )

    k_file = os.path.join(
        OUTPUT_DIR,
        "k_evidence.csv"
    )

    k_df.to_csv(
        k_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("RETRIEVAL EXPERIMENT V2 COMPLETE")
    print("=" * 80)

    print(
        f"Total retrieval runs: "
        f"{len(results_df)}"
    )

    print(
        f"Questions: "
        f"{len(k_df)}"
    )

    found = (
        k_df["K_evidence"]
        .notna()
        .sum()
    )

    not_found = (
        k_df["K_evidence"]
        .isna()
        .sum()
    )

    print(
        f"Questions with all expected "
        f"documents retrieved within K=10: "
        f"{found}"
    )

    print(
        f"Questions without all expected "
        f"documents within K=10: "
        f"{not_found}"
    )

    print("\nK_evidence distribution:")

    print(
        k_df[
            "K_evidence"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    print("\nFiles:")

    print(output_csv)
    print(output_json)
    print(k_file)


if __name__ == "__main__":
    run_experiment()