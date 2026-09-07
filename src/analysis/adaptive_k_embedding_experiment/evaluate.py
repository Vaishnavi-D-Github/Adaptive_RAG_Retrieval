import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

from .config import (
    OUTPUT_DIR,
    RANDOM_STATE,
)


def main():

    path = (
        OUTPUT_DIR
        / "predictions_oof.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            "predictions_oof.csv not found."
        )

    df = pd.read_csv(path)

    report = {}

    for model_name in sorted(
        df["Model"].unique()
    ):

        current = df[
            df["Model"] == model_name
        ].copy()

        actual = (
            current["Actual_K"]
            .astype(int)
            .to_numpy()
        )

        predicted = (
            current["Predicted_K"]
            .astype(int)
            .to_numpy()
        )

        exact = float(
            np.mean(
                actual == predicted
            )
        )

        within_one = float(
            np.mean(
                np.abs(
                    actual - predicted
                ) <= 1
            )
        )

        within_two = float(
            np.mean(
                np.abs(
                    actual - predicted
                ) <= 2
            )
        )

        mae = float(
            np.mean(
                np.abs(
                    actual - predicted
                )
            )
        )

        cm = confusion_matrix(
            actual,
            predicted,
            labels=list(
                range(1, 11)
            ),
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

        report[
            model_name
        ] = {
            "exact_k_accuracy": exact,
            "within_plus_minus_1_accuracy": within_one,
            "within_plus_minus_2_accuracy": within_two,
            "mae": mae,
        }

    payload = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "random_state": RANDOM_STATE,
        "evaluation_unit": "unique_question",
        "models": report,
        "important_note": (
            "All metrics are out-of-fold metrics. "
            "No post-retrieval information is used "
            "as an initial-K feature."
        ),
    }

    report_path = (
        OUTPUT_DIR
        / "experiment_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
        )

    print()
    print("=" * 80)
    print("EMBEDDING EXPERIMENT REPORT")
    print("=" * 80)

    for model, values in report.items():

        print()
        print(model)

        print(
            "Exact K accuracy: "
            f"{values['exact_k_accuracy']:.4f}"
        )

        print(
            "Within ±1 K: "
            f"{values['within_plus_minus_1_accuracy']:.4f}"
        )

        print(
            "Within ±2 K: "
            f"{values['within_plus_minus_2_accuracy']:.4f}"
        )

        print(
            "MAE: "
            f"{values['mae']:.4f}"
        )

    print()
    print(
        f"Report:\n{report_path}"
    )


if __name__ == "__main__":
    main()