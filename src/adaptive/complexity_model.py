from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold, cross_val_score

from .feature_extractor import extract_features


FEATURE_COLUMNS = [
    "query_length",
    "word_count",
    "unique_word_count",
    "average_word_length",
    "entity_count",
    "question_count",
    "question_mark_indicator",
    "number_indicator",
    "uppercase_token_count",
    "starts_with_what",
    "starts_with_why",
    "starts_with_how",
    "starts_with_when",
    "starts_with_where",
    "starts_with_who",
    "starts_with_which",
    "is_comparison",
    "is_procedural",
    "is_policy",
    "is_summary",
    "is_definition",
    "is_multi_hop",
    "requires_multiple_sources",
]


def _feature_dict(query):

    features = extract_features(
        query
    )

    return {
        column: getattr(
            features,
            column,
        )
        for column in FEATURE_COLUMNS
    }


def build_feature_matrix(
    queries,
):

    return pd.DataFrame(
        [
            _feature_dict(query)
            for query in queries
        ]
    )


class ComplexityModel:

    def __init__(
        self,
        n_estimators: int = 300,
        random_state: int = 42,
        max_depth: Optional[int] = None,
    ):

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=max_depth,
            class_weight="balanced",
            n_jobs=-1,
        )

        self.trained = False


    def train(
        self,
        queries,
        labels,
    ):

        X = build_feature_matrix(
            queries
        )

        y = pd.Series(labels)

        self.model.fit(
            X,
            y,
        )

        self.trained = True

        return self


    def predict(
        self,
        query,
    ):

        if not self.trained:
            raise RuntimeError(
                "Complexity model is not trained."
            )

        X = build_feature_matrix(
            [query]
        )

        prediction = self.model.predict(
            X
        )[0]

        probabilities = (
            self.model.predict_proba(X)[0]
        )

        confidence = float(
            probabilities.max()
        )

        return str(prediction), confidence


    def feature_importance(self):

        if not self.trained:
            raise RuntimeError(
                "Model is not trained."
            )

        return pd.Series(
            self.model.feature_importances_,
            index=FEATURE_COLUMNS,
        ).sort_values(
            ascending=False
        )


    def save(
        self,
        path,
    ):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self.model,
            path,
        )


    def load(
        self,
        path,
    ):

        self.model = joblib.load(
            path
        )

        self.trained = True

        return self
