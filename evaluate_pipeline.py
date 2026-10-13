import json
import hashlib
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "document_classifier.joblib"
HOLDOUT_PATH = MODEL_DIR / "evaluation_holdout.csv"
TRAIN_HASHES_PATH = MODEL_DIR / "train_text_hashes.json"
REPORT_PATH = ROOT / "evaluation_report.json"
ERRORS_PATH = ROOT / "misclassified_documents.csv"
DATA_QUALITY_PATH = ROOT / "dataset_quality_report.json"
LABELS = ("Invoice", "Receipt", "Bill")


def write_unavailable_report(missing):
    data_quality = {}
    if DATA_QUALITY_PATH.is_file():
        data_quality = json.loads(DATA_QUALITY_PATH.read_text(encoding="utf-8"))
    if data_quality.get("verified_for_training", 0) == 0:
        reason = (
            "No documents from a permission-cleared invoice/receipt source have been "
            "verified. Collect the public datasets and rebuild the OCR dataset first."
        )
    else:
        reason = f"Evaluation artifacts are missing: {', '.join(missing)}"
    results = {
        "status": "not_evaluated",
        "reason": reason,
        "missing_artifacts": missing,
        "dataset_quality": data_quality,
    }
    REPORT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


def main():
    required_files = (MODEL_PATH, HOLDOUT_PATH, TRAIN_HASHES_PATH)
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        write_unavailable_report(missing)
        return

    model = joblib.load(MODEL_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)
    holdout = holdout.dropna(subset=["document_text", "category"]).copy()
    train_hashes = set(json.loads(TRAIN_HASHES_PATH.read_text(encoding="utf-8")))
    texts = [str(text) for text in holdout["document_text"].tolist()]
    true_labels = [str(label) for label in holdout["category"].tolist()]
    holdout_hashes = {
        hashlib.sha256(
            re.sub(r"\s+", " ", text).strip().casefold().encode("utf-8")
        ).hexdigest()
        for text in texts
    }
    overlap = train_hashes & holdout_hashes
    if overlap:
        raise ValueError(
            f"Evaluation leakage detected: {len(overlap)} holdout texts also occur in training."
        )

    predictions = model.predict(texts)
    per_class = {}
    for label in LABELS:
        true_positive = sum(
            actual == label and predicted == label
            for actual, predicted in zip(true_labels, predictions)
        )
        predicted_count = sum(predicted == label for predicted in predictions)
        actual_count = sum(actual == label for actual in true_labels)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / actual_count if actual_count else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
            "support": actual_count,
        }
    matrix = confusion_matrix(true_labels, predictions, labels=LABELS)
    accuracy = accuracy_score(true_labels, predictions)
    balanced_accuracy = balanced_accuracy_score(true_labels, predictions)
    macro_f1 = sum(metrics["f1-score"] for metrics in per_class.values()) / len(LABELS)
    total_support = len(true_labels)
    weighted_f1 = sum(
        metrics["f1-score"] * metrics["support"] for metrics in per_class.values()
    ) / total_support

    incorrect = [actual != predicted for actual, predicted in zip(true_labels, predictions)]
    misclassified = holdout.loc[incorrect].copy()
    misclassified["predicted_category"] = [
        predicted for predicted, is_incorrect in zip(predictions, incorrect) if is_incorrect
    ]
    misclassified.to_csv(ERRORS_PATH, index=False)
    results = {
        "status": "complete",
        "evaluation_split": (
            "stratified 20% holdout; licensed public Invoice/Receipt image corpora plus "
            "synthetic-derived electricity-bill text, random_state=42"
        ),
        "test_documents": int(len(holdout)),
        "training_documents": int(len(train_hashes)),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix": matrix.tolist(),
        "misclassified_documents": int(len(misclassified)),
        "label_quality_caveat": (
            "Invoice/Receipt labels come from the cited public datasets. Bill examples "
            "are synthetic text rendered from Kaggle's synthetic electricity billing "
            "records, not scanned/OCR bill documents. Holdout metrics do not establish "
            "generalization to real utility bills, unfamiliar layouts, or languages."
        ),
    }
    REPORT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nDetailed results saved to {REPORT_PATH}")
    print(f"Misclassified holdout documents saved to {ERRORS_PATH}")


if __name__ == "__main__":
    main()
