"""Leakage-safe diagnostic and controlled K-target experiments.

This is analysis only.  It does not alter the runtime Adaptive K architecture
or replace the current exact-K Random Forest artifact.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from adaptive.feature_extractor import extract_features
from adaptive.k_model import build_k_features
from analysis.analyze_k import calculate_quality, prepare_costs

ROOT = Path(__file__).resolve().parents[2]
SEED = 42


def k_band(k: int) -> str:
    return "K1-3" if k <= 3 else "K4-6" if k <= 6 else "K7-10"


def safe_splits(y: pd.Series, requested: int = 5) -> int:
    return min(requested, int(y.value_counts().min()))


def experiment(X: pd.DataFrame, y: pd.Series, *, name: str, mae_labels: pd.Series | None = None) -> dict:
    folds = safe_splits(y)
    model = RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced", n_jobs=-1)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    predicted = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
    probabilities = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=1)
    model.fit(X, y)
    confidence = probabilities.max(axis=1)
    labels = sorted(y.unique())
    output = {
        "name": name,
        "folds": folds,
        "class_distribution": {str(key): int(value) for key, value in y.value_counts().sort_index().items()},
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, predicted, average="weighted", zero_division=0)),
        "classification_report": classification_report(y, predicted, output_dict=True, zero_division=0),
        "confusion_matrix": {"labels": [str(label) for label in labels], "values": confusion_matrix(y, predicted, labels=labels).tolist()},
        "oof_probability_confidence": {
            "mean": float(confidence.mean()),
            "median": float(np.median(confidence)),
            "correct_prediction_mean": float(confidence[predicted == y.to_numpy()].mean()) if any(predicted == y.to_numpy()) else None,
            "incorrect_prediction_mean": float(confidence[predicted != y.to_numpy()].mean()) if any(predicted != y.to_numpy()) else None,
        },
        "feature_importance": dict(sorted(zip(X.columns, model.feature_importances_), key=lambda pair: pair[1], reverse=True)),
        "out_of_fold_predictions": [
            {
                "actual": str(actual), "predicted": str(guess), "confidence": float(score),
                "probabilities": {str(label): float(value) for label, value in zip(model.classes_, row)},
            }
            for actual, guess, score, row in zip(y, predicted, confidence, probabilities)
        ],
    }
    if mae_labels is not None:
        output["mae_in_exact_k"] = float(mean_absolute_error(mae_labels, predicted))
    return output


def main() -> None:
    labels = pd.read_csv(ROOT / "results/adaptive/optimal_k_candidates.csv")
    dataset = pd.read_csv(ROOT / "results/ragas/ragas_evaluation_dataset.csv")
    labels = labels.dropna(subset=["Question", "Candidate_Optimal_K"]).copy()
    labels["Candidate_Optimal_K"] = labels["Candidate_Optimal_K"].astype(int)
    features = pd.DataFrame([extract_features(question).__dict__ for question in labels["Question"]])
    diagnostic = labels[["ID", "Candidate_Optimal_K", "Complexity", "Primary_Category", "Secondary_Category", "Answerable"]].copy()
    diagnostic = pd.concat([diagnostic.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    diagnostic["candidate_k_band"] = diagnostic["Candidate_Optimal_K"].map(k_band)

    quality = calculate_quality(prepare_costs(dataset))
    curves = []
    for question_id, group in quality.groupby("ID"):
        group = group.dropna(subset=["Quality_Score"]).sort_values("K")
        if group.empty:
            continue
        scores = group["Quality_Score"].to_numpy()
        curves.append({
            "ID": question_id, "available_k": int(len(group)), "best_k": int(group.loc[group["Quality_Score"].idxmax(), "K"]),
            "quality_range": float(scores.max() - scores.min()),
            "non_decreasing": bool(np.all(np.diff(scores) >= 0)),
            "candidate_k": int(labels.loc[labels.ID == question_id, "Candidate_Optimal_K"].iloc[0]) if any(labels.ID == question_id) else None,
        })
    curves = pd.DataFrame(curves)
    diagnostic = diagnostic.merge(curves, on="ID", how="left", suffixes=("", "_curve"))
    X = build_k_features(labels["Question"].tolist())
    exact = experiment(X, labels["Candidate_Optimal_K"], name="A: exact 10-class Candidate_Optimal_K", mae_labels=labels["Candidate_Optimal_K"])
    band_labels = labels["Candidate_Optimal_K"].map(k_band)
    bands = experiment(X, band_labels, name="B: grouped K bands (1-3, 4-6, 7-10)")

    majority_exact = labels["Candidate_Optimal_K"].value_counts(normalize=True).max()
    majority_band = band_labels.value_counts(normalize=True).max()
    recommendation = (
        "Neither target shows a defensible pre-retrieval signal above its majority-class baseline; retain exact K as the research target and use the trained model only as an experimental diagnostic until labels/features are validated."
        if exact["accuracy"] <= majority_exact and bands["accuracy"] <= majority_band
        else "Use grouped K bands as a reported sensitivity experiment only; preserve exact K as the research objective until an independently validated held-out result supports a runtime target change."
    )
    numeric_features = [column for column in X.columns if pd.api.types.is_numeric_dtype(X[column])]
    categorical_cross_tabs = {
        column: pd.crosstab(diagnostic[column].astype(str), diagnostic["candidate_k_band"], normalize="index").round(4).to_dict("index")
        for column in ["Complexity", "Primary_Category", "Secondary_Category", "Answerable"] + [column for column in X.columns if X[column].nunique() <= 2]
        if column in diagnostic
    }
    length_bins = {
        column: pd.crosstab(pd.qcut(diagnostic[column], q=4, duplicates="drop").astype(str), diagnostic["candidate_k_band"], normalize="index").round(4).to_dict("index")
        for column in ["query_length", "word_count", "unique_word_count", "average_word_length"]
    }
    report = {
        "purpose": "Diagnostic before any target change; runtime architecture unchanged.",
        "feature_policy": "Initial-K models use query-only, pre-retrieval characteristics. Category/complexity are descriptive diagnostics only because dataset annotations may not exist at runtime.",
        "candidate_label_definition": "Smallest K within 0.02 of the best deterministic proxy Quality_Score.",
        "diagnostics": {
            "candidate_k_distribution": {str(key): int(value) for key, value in labels.Candidate_Optimal_K.value_counts().sort_index().items()},
            "candidate_k_by_complexity": diagnostic.groupby("Complexity").Candidate_Optimal_K.agg(["count", "mean", "median"]).to_dict("index"),
            "candidate_k_by_primary_category": diagnostic.groupby("Primary_Category").Candidate_Optimal_K.agg(["count", "mean", "median"]).to_dict("index"),
            "spearman_candidate_k_vs_every_numeric_query_feature": diagnostic[["Candidate_Optimal_K"] + numeric_features].corr(method="spearman").loc["Candidate_Optimal_K"].drop("Candidate_Optimal_K").to_dict(),
            "candidate_k_band_cross_tabs": categorical_cross_tabs,
            "candidate_k_band_by_query_feature_quartile": length_bins,
            "quality_vs_k_curves": {"questions": int(len(curves)), "mean_quality_range": float(curves.quality_range.mean()), "non_decreasing_share": float(curves.non_decreasing.mean()), "candidate_matches_best_k_share": float((curves.candidate_k == curves.best_k).mean())},
            "candidate_k_vs_quality_curve": diagnostic[["Candidate_Optimal_K", "available_k", "best_k", "quality_range", "non_decreasing"]].corr(method="spearman").loc["Candidate_Optimal_K"].drop("Candidate_Optimal_K").to_dict(),
            "majority_baseline_accuracy": {"exact": float(majority_exact), "band": float(majority_band)},
        },
        "experiments": {"exact_k": exact, "k_bands": bands},
        "recommendation": recommendation,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out = ROOT / "results/adaptive/diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidate_k_diagnostic_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    diagnostic.to_csv(out / "candidate_k_vs_query_characteristics.csv", index=False)
    curves.to_csv(out / "per_question_quality_vs_k_curve_summary.csv", index=False)
    print(json.dumps({"report": str(out / "candidate_k_diagnostic_report.json"), "exact_accuracy": exact["accuracy"], "band_accuracy": bands["accuracy"], "recommendation": recommendation}, indent=2))


if __name__ == "__main__":
    main()
