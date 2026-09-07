import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from sklearn.model_selection import (
    KFold,
    cross_val_predict,
)

from .config import (
    OUTPUT_DIR,
    RANDOM_STATE,
    CV_FOLDS,
    TARGET_COLUMN,
    ID_COLUMN,
    QUESTION_COLUMN,
    BASE_FEATURES,
)


def prepare_features(df):
    feature_columns = [
        c for c in BASE_FEATURES
        if c in df.columns
    ]

    if not feature_columns:
        raise ValueError("No usable feature columns found.")

    X = df[feature_columns].copy()

    # Convert categorical-ish fields if any slipped through.
    for column in X.columns:
        if X[column].dtype == object:
            X[column] = (
                X[column]
                .fillna("")
                .astype("category")
                .cat.codes
            )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0)

    return X, feature_columns


def evaluate_regressor(
    name,
    model,
    X,
    y,
    ids,
):
    n_splits = min(CV_FOLDS, len(X))

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
    )

    predictions = np.clip(
        np.rint(predictions),
        1,
        10,
    ).astype(int)

    mae = mean_absolute_error(
        y,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions,
        )
    )

    exact_accuracy = float(
        np.mean(predictions == y)
    )

    within_one = float(
        np.mean(np.abs(predictions - y) <= 1)
    )

    output = pd.DataFrame(
        {
            ID_COLUMN: ids,
            "Actual_K": y,
            "Predicted_K": predictions,
            "Absolute_K_Error": np.abs(
                predictions - y
            ),
            "Exact": predictions == y,
            "Within_Plus_Minus_1": (
                np.abs(predictions - y) <= 1
            ),
            "Model": name,
        }
    )

    return {
        "name": name,
        "model": model,
        "predictions": output,
        "mae": float(mae),
        "rmse": float(rmse),
        "exact_accuracy": exact_accuracy,
        "within_plus_minus_1": within_one,
    }


def build_models():
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=2,
            max_features="sqrt",
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except ImportError:
        print("XGBoost not installed; skipping.")

    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=15,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
    except ImportError:
        print("LightGBM not installed; skipping.")

    return models


def main():
    dataset_path = OUTPUT_DIR / "dataset.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(
            "Run build_dataset.py first."
        )

    df = pd.read_csv(dataset_path)

    X, feature_columns = prepare_features(df)

    y = df[TARGET_COLUMN].astype(int).to_numpy()

    ids = df[ID_COLUMN].to_numpy()

    models = build_models()

    results = []
    prediction_frames = []

    for name, model in models.items():
        print()
        print("=" * 80)
        print(f"TRAINING: {name}")
        print("=" * 80)

        result = evaluate_regressor(
            name,
            model,
            X,
            y,
            ids,
        )

        results.append(
            {
                "Model": name,
                "Exact_K_Accuracy": result[
                    "exact_accuracy"
                ],
                "Within_Plus_Minus_1": result[
                    "within_plus_minus_1"
                ],
                "MAE": result["mae"],
                "RMSE": result["rmse"],
            }
        )

        prediction_frames.append(
            result["predictions"]
        )

        print(
            f"Exact K accuracy: "
            f"{result['exact_accuracy']:.4f}"
        )

        print(
            f"Within ±1 K: "
            f"{result['within_plus_minus_1']:.4f}"
        )

        print(
            f"MAE: {result['mae']:.4f}"
        )

        print(
            f"RMSE: {result['rmse']:.4f}"
        )

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        by=["Within_Plus_Minus_1", "MAE"],
        ascending=[False, True],
    )

    comparison.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    predictions.to_csv(
        OUTPUT_DIR / "predictions_oof.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(
        comparison.to_string(index=False)
    )

    print()
    print(
        "IMPORTANT: These are out-of-fold predictions. "
        "They are not training-set scores."
    )


if __name__ == "__main__":
    main()