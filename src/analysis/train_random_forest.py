"""Train the leakage-safe, query-characteristic K predictor.

Candidate_Optimal_K is deliberately labelled as a deterministic proxy target.
The model never consumes retrieved context, answers, RAGAS scores, tokens, or
latency when making its initial runtime prediction.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from adaptive.k_model import build_k_features


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = ROOT / "results" / "adaptive" / "optimal_k_candidates.csv"
DEFAULT_DATASET = ROOT / "results" / "ragas" / "ragas_evaluation_dataset.csv"
DEFAULT_OUTPUT = ROOT / "models"
RANDOM_SEED = 42


def _safe_splits(labels: pd.Series, requested: int) -> int:
    minimum_class_count = int(labels.value_counts().min())
    return min(requested, minimum_class_count)


def train(labels_path: Path, dataset_path: Path, output_dir: Path, folds: int = 5) -> dict:
    labels = pd.read_csv(labels_path)
    # The larger dataset is read only to validate the documented experimental source.
    dataset = pd.read_csv(dataset_path, usecols=lambda name: name in {"ID", "user_input"})
    required = {"ID", "Question", "Candidate_Optimal_K"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"Candidate label file is missing columns: {sorted(missing)}")
    if labels["ID"].duplicated().any():
        raise ValueError("Candidate label file must contain one row per question ID.")
    if not set(labels["ID"]).issubset(set(dataset["ID"])):
        raise ValueError("Candidate labels contain IDs absent from the evaluation dataset.")

    frame = labels.dropna(subset=["Question", "Candidate_Optimal_K"]).copy()
    frame["Candidate_Optimal_K"] = frame["Candidate_Optimal_K"].astype(int)
    X = build_k_features(frame["Question"].tolist())
    y = frame["Candidate_Optimal_K"]
    splits = _safe_splits(y, folds)
    if splits < 2:
        raise ValueError("At least two examples are required in every K class for cross-validation.")

    model = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_SEED)
    predictions = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
    model.fit(X, y)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"adaptive_k_random_forest_{stamp}.joblib"
    report_path = output_dir / f"adaptive_k_random_forest_{stamp}.json"
    joblib.dump({"model": model, "feature_columns": list(X.columns), "version": stamp}, model_path)

    report = {
        "model_version": stamp,
        "target": "Candidate_Optimal_K",
        "target_definition": "Smallest K within 0.02 of the best deterministic proxy Quality_Score for a question.",
        "target_status": "experimental proxy label; not RAGAS validation",
        "feature_policy": "Only query-characteristic features available before retrieval. No context, answers, scores, tokens, or latency.",
        "random_seed": RANDOM_SEED,
        "cross_validation": {"method": "StratifiedKFold", "folds": splits},
        "records": int(len(frame)),
        "accuracy": float(accuracy_score(y, predictions)),
        "mae_in_k": float(mean_absolute_error(y, predictions)),
        "classification_report": classification_report(y, predictions, output_dict=True, zero_division=0),
        "confusion_matrix": {
            "labels": sorted(int(value) for value in y.unique()),
            "values": confusion_matrix(y, predictions, labels=sorted(y.unique())).tolist(),
        },
        "feature_importance": dict(
            sorted(zip(X.columns, model.feature_importances_), key=lambda item: item[1], reverse=True)
        ),
        "inputs": {"candidate_labels": str(labels_path), "evaluation_dataset": str(dataset_path)},
        "model_artifact": str(model_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"model_path": model_path, "report_path": report_path, **report}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    result = train(args.labels, args.dataset, args.output_dir, args.folds)
    print(json.dumps({key: str(value) for key, value in result.items() if key in {"model_path", "report_path", "accuracy", "mae_in_k"}}, indent=2))


if __name__ == "__main__":
    main()
