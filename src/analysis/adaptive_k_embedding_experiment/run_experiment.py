from .build_dataset import build_dataset
from .train_embedding_xgboost import main as train_models
from .evaluate import main as evaluate


def main():

    print()
    print("=" * 80)
    print("QUERY EMBEDDING + XGBOOST ADAPTIVE-K EXPERIMENT")
    print("=" * 80)

    print()
    print("STEP 1/3 — Build dataset and embeddings")

    build_dataset()

    print()
    print("STEP 2/3 — Train and cross-validate XGBoost models")

    train_models()

    print()
    print("STEP 3/3 — Generate evaluation report")

    evaluate()

    print()
    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()