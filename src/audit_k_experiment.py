"""
audit_k_experiment.py

Purpose
-------
Audit the pilot K=1..10 experiment WITHOUT changing the original results.

The audit separates:
1. Retrieval evidence presence
2. LLM generation behaviour
3. Abstention behaviour
4. Token/latency measurements

IMPORTANT:
- This script is a PILOT audit aid, not a final ground-truth evaluator.
- "evidence_present_heuristic" is deliberately labelled heuristic.
- Complex questions should be manually verified before being used to
  determine final optimal K.
- The original JSON is never modified.

Inputs
------
results/k_experiment_results_actual_tokens.json

Optional:
data/evaluation/pilot_questions_10.csv

Outputs
-------
results/k_experiments/pilot_audit_results.csv
results/k_experiments/pilot_audit_summary.csv
results/k_experiments/pilot_manual_review.csv
"""

import csv
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

INPUT_JSON = Path("results/k_experiment_results_actual_tokens.json")
OUTPUT_DIR = Path("results/k_experiments")

AUDIT_CSV = OUTPUT_DIR / "pilot_audit_results.csv"
SUMMARY_CSV = OUTPUT_DIR / "pilot_audit_summary.csv"
MANUAL_CSV = OUTPUT_DIR / "pilot_manual_review.csv"


# ---------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
    "with", "by", "from", "as", "at", "is", "are", "was", "were",
    "be", "been", "being", "this", "that", "these", "those", "it",
    "its", "their", "they", "them", "what", "which", "who", "how",
    "did", "does", "do", "according", "than", "under", "between",
    "into", "about", "during", "have", "has", "had", "will", "would",
    "can", "could", "should", "must", "also", "both", "each",
}


def normalize(text):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9%.\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def content_tokens(text):
    words = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?%?", normalize(text))
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def numbers(text):
    """
    Extract numeric evidence. Handles values such as:
    13, 1,303, 77,183, 2023, 19.1, 19.6, 14%, etc.
    """
    text = text or ""
    return {
        x.replace(",", "")
        for x in re.findall(r"\b\d[\d,]*(?:\.\d+)?%?", text)
    }


def is_abstention(answer):
    text = normalize(answer)
    phrases = [
        "available documents do not contain enough information",
        "does not contain enough information",
        "do not contain enough information",
        "not enough information to answer",
        "cannot be answered from the available documents",
        "cannot answer this question",
        "unable to answer",
        "insufficient information",
    ]
    return any(p in text for p in phrases)


def answer_overlap(reference, answer):
    """
    Lightweight lexical overlap used only as a diagnostic.

    It is NOT treated as definitive correctness.
    """
    ref = set(content_tokens(reference))
    ans = set(content_tokens(answer))

    if not ref:
        return 0.0

    return len(ref & ans) / len(ref)


def answer_correctness_heuristic(reference, answer, answerable):
    """
    Conservative heuristic.

    For answerable questions:
      - abstention => incorrect
      - numeric reference => require reference number in answer
      - otherwise require substantial lexical overlap

    For unanswerable questions:
      - abstention => correct abstention

    Returns:
      correct / incorrect / manual_review
    """
    abstain = is_abstention(answer)

    if str(answerable).strip().lower() == "no":
        return "correct_abstention" if abstain else "incorrect_non_abstention"

    if abstain:
        return "incorrect_abstention"

    ref_numbers = numbers(reference)

    if ref_numbers:
        # If the reference contains numeric values, require at least one
        # reference number to appear in the generated answer.
        ans_numbers = numbers(answer)
        if ref_numbers & ans_numbers:
            return "likely_correct"
        return "likely_incorrect"

    overlap = answer_overlap(reference, answer)

    if overlap >= 0.35:
        return "likely_correct"

    if overlap < 0.10:
        return "likely_incorrect"

    return "manual_review"


def evidence_heuristic(reference, context):
    """
    Conservative retrieval-evidence heuristic.

    Strong signal:
      - a reference number appears in the retrieved context.

    General signal:
      - meaningful reference terms appear in context.

    This does NOT prove semantic sufficiency. Complex definition,
    comparison, policy and summary questions should be manually checked.
    """
    ref_numbers = numbers(reference)
    ctx_numbers = numbers(context)

    numeric_match = bool(ref_numbers & ctx_numbers)

    ref_terms = set(content_tokens(reference))
    ctx_terms = set(content_tokens(context))

    if ref_terms:
        overlap = len(ref_terms & ctx_terms) / len(ref_terms)
    else:
        overlap = 0.0

    if numeric_match:
        return "likely_present_numeric", overlap

    if overlap >= 0.50:
        return "likely_present_lexical", overlap

    if overlap >= 0.25:
        return "possible_present_manual_review", overlap

    return "not_detected_manual_review", overlap


def failure_type(answerable, evidence_status, abstained, correctness):
    """
    Classify the observed behaviour without pretending that the heuristic
    evidence test is perfect.
    """
    answerable = str(answerable).strip().lower() == "yes"

    if not answerable:
        if abstained:
            return "correct_abstention"
        return "unanswerable_non_abstention"

    evidence_present = evidence_status.startswith("likely_present")

    if evidence_present and abstained:
        return "generation_failure_with_evidence"

    if evidence_present and correctness == "likely_correct":
        return "success"

    if not evidence_present and abstained:
        return "retrieval_or_evidence_failure"

    if not evidence_present and correctness == "likely_correct":
        return "answer_without_detected_evidence_manual_review"

    if correctness == "likely_incorrect":
        return "incorrect_answer"

    return "manual_review"


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------

def load_results():
    if not INPUT_JSON.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_JSON}\n"
            "Run this script from the project root."
        )

    with INPUT_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected the JSON root to be a list.")

    return data


# ---------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------

def audit_records(records):
    audited = []

    for r in records:
        reference = r.get("reference_answer", "")
        context = r.get("context", "")
        answer = r.get("generated_answer", "")
        answerable = r.get("answerable", "")

        evidence_status, evidence_overlap = evidence_heuristic(
            reference,
            context
        )

        abstained = is_abstention(answer)

        correctness = answer_correctness_heuristic(
            reference,
            answer,
            answerable
        )

        failure = failure_type(
            answerable,
            evidence_status,
            abstained,
            correctness
        )

        retrieved = r.get("retrieved_chunks", [])

        top_distance = ""
        if retrieved:
            top_distance = retrieved[0].get("distance", "")

        source_pages = sorted({
            str(x.get("page", ""))
            for x in retrieved
            if x.get("page") is not None
        })

        source_documents = sorted({
            str(x.get("document", ""))
            for x in retrieved
            if x.get("document") is not None
        })

        audited.append({
            "question_id": r.get("question_id", ""),
            "question": r.get("question", ""),
            "primary_category": r.get("primary_category", ""),
            "complexity": r.get("complexity", ""),
            "answerable": r.get("answerable", ""),
            "reference_answer": reference,

            "k": r.get("k", ""),
            "number_of_retrieved_chunks": r.get(
                "number_of_retrieved_chunks", ""
            ),

            "retrieved_source_documents": "; ".join(source_documents),
            "retrieved_source_pages": "; ".join(source_pages),
            "top_retrieval_distance": top_distance,

            "evidence_status_heuristic": evidence_status,
            "evidence_reference_overlap": round(evidence_overlap, 4),

            "generated_answer": answer,
            "abstained": abstained,
            "answer_correctness_heuristic": correctness,
            "failure_type": failure,

            "context_tokens_approx_or_recorded": (
                r.get("context_tokens", "")
                or r.get("prompt_eval_count", "")
            ),
            "prompt_tokens": r.get("prompt_eval_count", ""),
            "generated_tokens": r.get("eval_count", ""),
            "total_tokens": r.get("total_tokens", ""),
            "execution_time_seconds": r.get(
                "execution_time_seconds", ""
            ),

            "embedding_model": r.get("embedding_model", ""),
            "llm_model": r.get("llm_model", ""),
            "temperature": r.get("temperature", ""),
            "num_predict": r.get("num_predict", ""),

            # Manual-review flag is intentionally broad.
            "manual_review_required": (
                "YES"
                if (
                    "manual_review" in evidence_status
                    or correctness == "manual_review"
                    or failure == "answer_without_detected_evidence_manual_review"
                )
                else "NO"
            ),
        })

    return audited


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def make_summary(audited):
    by_k = defaultdict(list)
    by_question = defaultdict(list)
    by_category = defaultdict(list)
    by_complexity = defaultdict(list)

    for r in audited:
        by_k[r["k"]].append(r)
        by_question[r["question_id"]].append(r)
        by_category[r["primary_category"]].append(r)
        by_complexity[r["complexity"]].append(r)

    rows = []

    def add_group(group_name, group_value, records):
        n = len(records)

        abstentions = sum(
            1 for r in records if r["abstained"]
        )

        evidence_likely = sum(
            1 for r in records
            if r["evidence_status_heuristic"].startswith("likely_present")
        )

        likely_correct = sum(
            1 for r in records
            if r["answer_correctness_heuristic"] == "likely_correct"
        )

        generation_failures = sum(
            1 for r in records
            if r["failure_type"] == "generation_failure_with_evidence"
        )

        retrieval_failures = sum(
            1 for r in records
            if r["failure_type"] == "retrieval_or_evidence_failure"
        )

        manual = sum(
            1 for r in records
            if r["manual_review_required"] == "YES"
        )

        token_values = []
        latency_values = []

        for r in records:
            try:
                token_values.append(float(r["total_tokens"]))
            except (TypeError, ValueError):
                pass

            try:
                latency_values.append(float(
                    r["execution_time_seconds"]
                ))
            except (TypeError, ValueError):
                pass

        rows.append({
            "group": group_name,
            "value": group_value,
            "records": n,
            "abstentions": abstentions,
            "abstention_rate": round(abstentions / n, 4) if n else "",
            "likely_evidence_present": evidence_likely,
            "likely_evidence_rate": (
                round(evidence_likely / n, 4) if n else ""
            ),
            "likely_correct_answers": likely_correct,
            "likely_correct_rate": (
                round(likely_correct / n, 4) if n else ""
            ),
            "generation_failures_with_evidence": generation_failures,
            "retrieval_or_evidence_failures": retrieval_failures,
            "manual_review_required": manual,
            "average_total_tokens": (
                round(sum(token_values) / len(token_values), 2)
                if token_values else ""
            ),
            "average_latency_seconds": (
                round(sum(latency_values) / len(latency_values), 3)
                if latency_values else ""
            ),
        })

    # K-level summary
    for k in sorted(by_k, key=lambda x: int(x)):
        add_group("K", k, by_k[k])

    # Category-level summary
    for category in sorted(by_category):
        add_group("category", category, by_category[category])

    # Complexity-level summary
    for complexity in sorted(by_complexity):
        add_group("complexity", complexity, by_complexity[complexity])

    return rows


# ---------------------------------------------------------------------
# Manual review queue
# ---------------------------------------------------------------------

def make_manual_review(audited):
    """
    Produces a compact queue for manual inspection.

    This is especially important for:
      Definition
      Policy
      Procedural
      Comparison
      Summary
      Multi-document

    where lexical matching cannot reliably prove evidence sufficiency.
    """
    rows = []

    for r in audited:
        if r["manual_review_required"] != "YES":
            continue

        rows.append({
            "question_id": r["question_id"],
            "k": r["k"],
            "primary_category": r["primary_category"],
            "complexity": r["complexity"],
            "question": r["question"],
            "reference_answer": r["reference_answer"],
            "evidence_status_heuristic": (
                r["evidence_status_heuristic"]
            ),
            "evidence_reference_overlap": (
                r["evidence_reference_overlap"]
            ),
            "generated_answer": r["generated_answer"],
            "abstained": r["abstained"],
            "answer_correctness_heuristic": (
                r["answer_correctness_heuristic"]
            ),
            "failure_type": r["failure_type"],
            "retrieved_source_documents": (
                r["retrieved_source_documents"]
            ),
            "retrieved_source_pages": r["retrieved_source_pages"],
            "total_tokens": r["total_tokens"],
        })

    return rows


# ---------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"No rows to write: {path}")
        return

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------

def print_report(audited, summary, manual):
    print("\n" + "=" * 80)
    print("K EXPERIMENT PILOT AUDIT")
    print("=" * 80)

    questions = sorted({
        r["question_id"] for r in audited
    })

    ks = sorted({
        int(r["k"]) for r in audited
    })

    print(f"\nExperiment records: {len(audited)}")
    print(f"Unique questions:   {len(questions)}")
    print(f"K values:            {ks}")

    print("\n" + "-" * 80)
    print("FAILURE / BEHAVIOUR COUNTS")
    print("-" * 80)

    counts = Counter(r["failure_type"] for r in audited)

    for name, count in counts.most_common():
        print(f"{name:45s} {count}")

    print("\n" + "-" * 80)
    print("ABSTENTION BY K")
    print("-" * 80)

    k_records = defaultdict(list)

    for r in audited:
        k_records[int(r["k"])].append(r)

    for k in sorted(k_records):
        rows = k_records[k]
        abstain = sum(1 for r in rows if r["abstained"])
        evidence = sum(
            1 for r in rows
            if r["evidence_status_heuristic"].startswith(
                "likely_present"
            )
        )
        gen_fail = sum(
            1 for r in rows
            if r["failure_type"] == "generation_failure_with_evidence"
        )

        print(
            f"K={k:2d} | "
            f"abstain={abstain:2d}/{len(rows):2d} | "
            f"likely evidence={evidence:2d} | "
            f"generation failures with evidence={gen_fail:2d}"
        )

    print("\n" + "-" * 80)
    print("MANUAL REVIEW")
    print("-" * 80)
    print(
        f"{len(manual)} records require manual review "
        f"before final K* analysis."
    )

    print("\n" + "=" * 80)
    print("IMPORTANT")
    print("=" * 80)
    print(
        "The heuristic evidence/correctness fields are diagnostic only. "
        "Do NOT use them as final ground truth for the 100-question "
        "optimal-K experiment until the pilot methodology has been "
        "validated."
    )

    print("\nOutput files:")
    print(f"  {AUDIT_CSV}")
    print(f"  {SUMMARY_CSV}")
    print(f"  {MANUAL_CSV}")


def main():
    records = load_results()

    print(f"Loaded {len(records)} experiment records.")

    audited = audit_records(records)
    summary = make_summary(audited)
    manual = make_manual_review(audited)

    write_csv(AUDIT_CSV, audited)
    write_csv(SUMMARY_CSV, summary)
    write_csv(MANUAL_CSV, manual)

    print_report(audited, summary, manual)


if __name__ == "__main__":
    main()