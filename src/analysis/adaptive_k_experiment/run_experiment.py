from .build_dataset import build_dataset
from .train_boosting_k import main as train_models
from .evaluate_models import evaluate


def main():
    print()
    print("=" * 80)
    print("ADAPTIVE-K MODEL EXPERIMENT")
    print("=" * 80)

    print()
    print("STEP 1/3 — Building question-level dataset")
    build_dataset()

    print()
    print("STEP 2/3 — Training and cross-validating models")
    train_models()

    print()
    print("STEP 3/3 — Generating detailed evaluation")
    evaluate()

    print()
    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()