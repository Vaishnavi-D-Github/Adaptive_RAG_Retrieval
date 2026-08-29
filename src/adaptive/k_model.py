from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from .feature_extractor import extract_features


K_FEATURE_COLUMNS = [
    "query_length",
    "word_count",
    "entity_count",
    "question_count",
    "is_comparison",
    "is_procedural",
    "is_policy",
    "is_summary",
    "is_definition",
    "is_multi_hop",
    "requires_multiple_sources",
]


def build_k_features(
    queries,
    complexity_labels=None,
):

    rows = []

    for index, query in enumerate(
        queries
    ):

        features = extract_features(
            query
        )

        row = {
            column: getattr(
                features,
                column,
            )
            for column in K_FEATURE_COLUMNS
        }

        if complexity_labels is not None:

            row["predicted_complexity"] = (
                complexity_labels[index]
            )

        rows.append(row)

    df = pd.DataFrame(rows)

    if "predicted_complexity" in df:

        df = pd.get_dummies(
            df,
            columns=[
                "predicted_complexity"
            ],
            dtype=int,
        )

    return df


class KModel:

    def __init__(
        self,
        n_estimators=300,
        random_state=42,
    ):

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )

        self.feature_columns = None

        self.trained = False


    def train(
        self,
        queries,
        optimal_k,
        complexity_labels=None,
    ):

        X = build_k_features(
            queries,
            complexity_labels,
        )

        y = pd.Series(
            optimal_k
        )

        self.model.fit(
            X,
            y,
        )

        self.feature_columns = list(
            X.columns
        )

        self.trained = True

        return self


    def predict(
        self,
        query,
        complexity=None,
    ):

        if not self.trained:
            raise RuntimeError(
                "K model is not trained."
            )

        X = build_k_features(
            [query],
            (
                [complexity]
                if complexity is not None
                else None
            ),
        )

        X = X.reindex(
            columns=self.feature_columns,
            fill_value=0,
        )

        prediction = int(
            self.model.predict(X)[0]
        )

        probabilities = (
            self.model.predict_proba(X)[0]
        )

        confidence = float(
            probabilities.max()
        )

        return prediction, confidence


    def feature_importance(self):

        if not self.trained:
            raise RuntimeError(
                "K model is not trained."
            )

        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_columns,
        ).sort_values(
            ascending=False
        )


    def save(self, path):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            {
                "model": self.model,
                "feature_columns": self.feature_columns,
            },
            path,
        )


    def load(self, path):

        data = joblib.load(
            path
        )

        self.model = data["model"]

        self.feature_columns = (
            data["feature_columns"]
        )

        self.trained = True

        return self