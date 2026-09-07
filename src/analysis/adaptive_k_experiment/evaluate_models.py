import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)


from .config import (
    OUTPUT_DIR,
    TARGET_COLUMN,
    ID_COLUMN,
    RANDOM_STATE,
)


def evaluate():
    prediction_path = (
        OUTPUT_DIR / "predictions_oof.csv"
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            "Run train_boosting_k.py first."
        )

    predictions = pd.read_csv(
        prediction_path
    )

    report = {}

    all_models = sorted(
        predictions["Model"].unique()
    )

    for model_name in all_models:
        df = predictions[
            predictions["Model"] == model_name
        ].copy()

        y_true = df["Actual_K"].astype(int)
        y_pred = df["Predicted_K"].astype(int)

        exact = float(
            np.mean(y_true == y_pred)
        )

        within_one = float(
            np.mean(
                np.abs(y_true - y_pred) <= 1
            )
        )

        mae = float(
            np.mean(
                np.abs(y_true - y_pred)
            )
        )

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(1, 11)),
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                f"Actual_K_{k}"
                for k in range(1, 11)
            ],
            columns=[
                f"Predicted_K_{k}"
                for k in range(1, 11)
            ],
        )

        cm_df.to_csv(
            OUTPUT_DIR
            / f"confusion_matrix_{model_name}.csv"
        )

        classification = classification_report(
            y_true,
            y_pred,
            labels=list(range(1, 11)),
            zero_division=0,
            output_dict=True,
        )

        report[model_name] = {
            "exact_k_accuracy": exact,
            "within_plus_minus_1_accuracy": within_one,
            "mae": mae,
            "classification_report": classification,
        }

    payload = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "random_state": RANDOM_STATE,
        "target": TARGET_COLUMN,
        "evaluation_unit": "question",
        "models": report,
    }

    with open(
        OUTPUT_DIR / "experiment_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

    print("=" * 80)
    print("DETAILED ADAPTIVE-K EXPERIMENT REPORT")
    print("=" * 80)

    for model_name, values in report.items():
        print()
        print(model_name)
        print(
            f"Exact K accuracy: "
            f"{values['exact_k_accuracy']:.4f}"
        )
        print(
            f"Within ±1 K: "
            f"{values['within_plus_minus_1_accuracy']:.4f}"
        )
        print(
            f"MAE: "
            f"{values['mae']:.4f}"
        )

    print()
    print(
        f"Report: "
        f"{OUTPUT_DIR / 'experiment_report.json'}"
    )


if __name__ == "__main__":
    evaluate()