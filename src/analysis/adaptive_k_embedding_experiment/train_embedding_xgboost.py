import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from .config import (
    OUTPUT_DIR,
    RANDOM_STATE,
    CV_FOLDS,
    TARGET_COLUMN,
    ID_COLUMN,
    QUESTION_COLUMN,
    ENGINEERED_FEATURES,
)


def prepare_engineered_features(
    df: pd.DataFrame,
):

    columns = [
        column
        for column in ENGINEERED_FEATURES
        if column in df.columns
    ]

    if not columns:
        raise ValueError(
            "No engineered features found."
        )

    X = df[columns].copy()

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

    return X, columns


def prepare_embedding_features(
    df: pd.DataFrame,
):

    embedding_columns = [
        column
        for column in df.columns
        if column.startswith(
            "embedding_"
        )
    ]

    if not embedding_columns:
        raise ValueError(
            "No embedding columns found."
        )

    X = df[
        embedding_columns
    ].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0)

    return X, embedding_columns


def build_xgb():

    return XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.75,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def run_oof_experiment(
    model_name,
    X,
    y,
    ids,
):

    cv = KFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    predictions = np.zeros(
        len(y),
        dtype=float,
    )

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        cv.split(X),
        start=1,
    ):

        model = build_xgb()

        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_train = y[train_idx]

        model.fit(
            X_train,
            y_train,
        )

        predictions[
            test_idx
        ] = model.predict(
            X_test
        )

        print(
            f"{model_name} "
            f"fold {fold}/{CV_FOLDS} complete"
        )

    rounded = np.clip(
        np.rint(predictions),
        1,
        10,
    ).astype(int)

    exact_accuracy = float(
        np.mean(
            rounded == y
        )
    )

    within_one = float(
        np.mean(
            np.abs(
                rounded - y
            ) <= 1
        )
    )

    within_two = float(
        np.mean(
            np.abs(
                rounded - y
            ) <= 2
        )
    )

    mae = float(
        mean_absolute_error(
            y,
            rounded,
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y,
                rounded,
            )
        )
    )

    frame = pd.DataFrame(
        {
            ID_COLUMN: ids,
            "Actual_K": y,
            "Predicted_K_Raw": predictions,
            "Predicted_K": rounded,
            "Absolute_K_Error": np.abs(
                rounded - y
            ),
            "Exact": (
                rounded == y
            ),
            "Within_Plus_Minus_1": (
                np.abs(
                    rounded - y
                ) <= 1
            ),
            "Within_Plus_Minus_2": (
                np.abs(
                    rounded - y
                ) <= 2
            ),
            "Model": model_name,
        }
    )

    return {
        "model": model_name,
        "exact_accuracy": exact_accuracy,
        "within_one": within_one,
        "within_two": within_two,
        "mae": mae,
        "rmse": rmse,
        "predictions": frame,
    }


def main():

    dataset_path = (
        OUTPUT_DIR
        / "dataset.csv"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            "Dataset not found. "
            "Run build_dataset.py first."
        )

    df = pd.read_csv(
        dataset_path
    )

    y = (
        df[TARGET_COLUMN]
        .astype(int)
        .to_numpy()
    )

    ids = df[
        ID_COLUMN
    ].to_numpy()

    # --------------------------------------------------------------
    # MODEL A — ENGINEERED FEATURES ONLY
    # --------------------------------------------------------------

    X_engineered, engineered_columns = (
        prepare_engineered_features(df)
    )

    print()
    print("=" * 80)
    print("MODEL A — ENGINEERED FEATURES + XGBOOST")
    print("=" * 80)

    print(
        f"Features: {len(engineered_columns)}"
    )

    result_a = run_oof_experiment(
        "XGBoost_Engineered",
        X_engineered,
        y,
        ids,
    )

    # --------------------------------------------------------------
    # MODEL B — ENGINEERED + EMBEDDINGS
    # --------------------------------------------------------------

    X_embeddings, embedding_columns = (
        prepare_embedding_features(df)
    )

    X_combined = pd.concat(
        [
            X_engineered.reset_index(
                drop=True
            ),
            X_embeddings.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    print()
    print("=" * 80)
    print(
        "MODEL B — ENGINEERED FEATURES + "
        "QUERY EMBEDDING + XGBOOST"
    )
    print("=" * 80)

    print(
        f"Engineered features: "
        f"{len(engineered_columns)}"
    )

    print(
        f"Embedding dimensions: "
        f"{len(embedding_columns)}"
    )

    print(
        f"Total features: "
        f"{X_combined.shape[1]}"
    )

    result_b = run_oof_experiment(
        "XGBoost_Engineered_Embedding",
        X_combined,
        y,
        ids,
    )

    # --------------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "Model": result_a["model"],
                "Exact_K_Accuracy": result_a[
                    "exact_accuracy"
                ],
                "Within_Plus_Minus_1": result_a[
                    "within_one"
                ],
                "Within_Plus_Minus_2": result_a[
                    "within_two"
                ],
                "MAE": result_a["mae"],
                "RMSE": result_a["rmse"],
            },
            {
                "Model": result_b["model"],
                "Exact_K_Accuracy": result_b[
                    "exact_accuracy"
                ],
                "Within_Plus_Minus_1": result_b[
                    "within_one"
                ],
                "Within_Plus_Minus_2": result_b[
                    "within_two"
                ],
                "MAE": result_b["mae"],
                "RMSE": result_b["rmse"],
            },
        ]
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "model_comparison.csv",
        index=False,
    )

    predictions = pd.concat(
        [
            result_a["predictions"],
            result_b["predictions"],
        ],
        ignore_index=True,
    )

    predictions.to_csv(
        OUTPUT_DIR
        / "predictions_oof.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print(
        f"Results directory:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()