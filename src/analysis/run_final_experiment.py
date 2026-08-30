"""Run fixed K=3/5/10 and Adaptive-K over one controlled question set."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

from adaptive.config import FIXED_K_VALUES, ROOT
from runtime import load_runtime


OUTPUT_COLUMNS = [
    "ID", "Question", "Mode", "Initial_K", "Final_K", "Iterations", "K_Escalated",
    "Verification_Performed", "Verification_Result", "Query_Features", "Prediction_Confidence",
    "Chunks_Retrieved", "Chunks_Used", "Prompt_Tokens", "Generated_Tokens", "Total_Tokens",
    "Retrieval_Latency", "Verification_Latency", "Optimization_Latency", "Generation_Latency",
    "Total_Latency", "Generated_Answer", "Retrieved_Sources", "Faithfulness", "Answer_Relevancy",
    "Context_Relevance", "Evaluation_Status",
]


def _serialise(value):
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, ensure_ascii=False, default=str)


def _row(question_id: str, question: str, mode: str, result) -> dict:
    payload = result.telemetry if hasattr(result, "telemetry") else result["telemetry"]
    answer = result.generation_result if hasattr(result, "generation_result") else result["answer"]
    sources = payload.get("retrieved_sources", [])
    return {
        "ID": question_id, "Question": question, "Mode": mode,
        "Initial_K": payload["initial_k"], "Final_K": payload["selected_k"],
        "Iterations": payload["retrieval_iterations"], "K_Escalated": payload["k_escalated"],
        "Verification_Performed": payload["verification_performed"], "Verification_Result": payload["verification_result"],
        "Query_Features": _serialise(payload["query_features"]), "Prediction_Confidence": payload["predicted_k_confidence"],
        "Chunks_Retrieved": payload["num_retrieved_chunks"], "Chunks_Used": payload["num_chunks_used"],
        "Prompt_Tokens": payload["prompt_tokens"], "Generated_Tokens": payload["generated_tokens"], "Total_Tokens": payload["total_tokens"],
        "Retrieval_Latency": payload["retrieval_time_ms"], "Verification_Latency": payload["verification_time_ms"],
        "Optimization_Latency": payload["optimization_time_ms"], "Generation_Latency": payload["generation_time_ms"],
        "Total_Latency": payload["total_latency_ms"], "Generated_Answer": answer,
        "Retrieved_Sources": _serialise(sources), "Faithfulness": None, "Answer_Relevancy": None,
        "Context_Relevance": None, "Evaluation_Status": "pending",
    }


def _questions(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            question = (row.get("Question") or row.get("question") or row.get("user_input") or "").strip()
            if question:
                yield str(row.get("ID") or row.get("id") or index), question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=ROOT / "data" / "evaluation" / "enterprise_query_evaluation_set_100.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "final")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    adaptive, fixed, _, _ = load_runtime()
    rows = []
    for index, (question_id, question) in enumerate(_questions(args.questions), start=1):
        if args.limit and index > args.limit:
            break
        for k in FIXED_K_VALUES:
            rows.append(_row(question_id, question, f"fixed_{k}", fixed.run(question, k)))
        rows.append(_row(question_id, question, "adaptive", adaptive.run(question)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "final_experiment_results.csv"
    json_path = args.output_dir / "final_experiment_results.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader(); writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Saved {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
