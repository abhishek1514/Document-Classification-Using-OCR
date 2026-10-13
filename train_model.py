import hashlib
import json
import re
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dataset.csv"
SYNTHETIC_BILLS_PATH = ROOT / "dataset_synthetic_bills.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "document_classifier.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
HOLDOUT_PATH = MODEL_DIR / "evaluation_holdout.csv"
TRAIN_HASHES_PATH = MODEL_DIR / "train_text_hashes.json"
REQUIRED_CATEGORIES = ("Invoice", "Receipt", "Bill")
VERIFIED_CATEGORIES = ("Invoice", "Receipt")
SYNTHETIC_BILL_SOURCE = "rahibvk/new-york-household-electricity-bills"
SYNTHETIC_BILL_LICENSE = "CC BY-SA 4.0"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_EXAMPLES_PER_CLASS = 20


def normalized_hash(text):
    normalized = re.sub(r"\s+", " ", str(text)).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_training_data():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}. Run create_dataset.py first.")

    frame = pd.read_csv(DATASET_PATH)
    required_columns = {"document_text", "category"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    frame = frame.dropna(subset=["document_text", "category"]).copy()
    frame["document_text"] = frame["document_text"].astype(str).str.strip()
    frame["category"] = frame["category"].astype(str).str.strip()
    frame = frame[frame["document_text"].str.len() >= 20]
    frame = frame[frame["category"].isin(VERIFIED_CATEGORIES)]

    weak_labels = "--weak-labels" in sys.argv
    if weak_labels:
        if "source_path" not in frame:
            raise ValueError(
                "The dataset has no source_path column, so category folders cannot be "
                "used as provisional training labels. Rebuild it with create_dataset.py."
            )
        normalized_paths = frame["source_path"].astype(str).str.replace("\\", "/", regex=False)
        frame = frame[
            frame["category"].isin(VERIFIED_CATEGORIES)
            & normalized_paths.str.startswith(("data/Invoice/", "data/Receipt/"))
        ].copy()
        frame["_label_source"] = "original Invoice/Receipt folder names (unverified)"
    else:
        if "label_status" not in frame:
            raise ValueError(
                "Dataset rows do not include verified labels. Run "
                "create_dataset.py and review its label-status field."
            )
        frame = frame[
            (frame["label_status"] == "verified")
            & frame["category"].isin(VERIFIED_CATEGORIES)
        ].copy()

    if not SYNTHETIC_BILLS_PATH.is_file():
        raise FileNotFoundError(
            f"Synthetic bill training data not found: {SYNTHETIC_BILLS_PATH}. "
            "Run prepare_bill_dataset.py first."
        )
    bills = pd.read_csv(SYNTHETIC_BILLS_PATH)
    required_bill_columns = {
        "document_text",
        "category",
        "label_status",
        "source_dataset",
        "source_license",
    }
    missing_bill_columns = required_bill_columns - set(bills.columns)
    if missing_bill_columns:
        raise ValueError(
            f"Synthetic bill dataset is missing columns: {sorted(missing_bill_columns)}"
        )
    bills = bills.dropna(
        subset=["document_text", "category", "label_status", "source_dataset", "source_license"]
    ).copy()
    valid_bills = (
        (bills["category"] == "Bill")
        & (bills["label_status"] == "synthetic-derived")
        & (bills["source_dataset"] == SYNTHETIC_BILL_SOURCE)
        & (bills["source_license"] == SYNTHETIC_BILL_LICENSE)
    )
    if not valid_bills.all():
        raise ValueError(
            "Bill training data must contain only Bill examples derived from the "
            "pinned synthetic Kaggle source and marked synthetic-derived."
        )
    bills["document_text"] = bills["document_text"].astype(str).str.strip()
    bills = bills[bills["document_text"].str.len() >= 20].copy()
    bills["_label_source"] = "synthetic utility bill records rendered as text"
    frame = pd.concat([frame, bills], ignore_index=True, sort=False)

    if frame.empty:
        if weak_labels:
            raise ValueError(
                "No Invoice/Receipt documents were found under data/Invoice/ and "
                "data/Receipt/. Rebuild the dataset with create_dataset.py."
            )
        raise ValueError(
            "No verified Invoice/Receipt examples or synthetic Bill records are available."
        )

    frame["_text_hash"] = frame["document_text"].map(normalized_hash)
    frame = frame.drop_duplicates(subset="_text_hash", keep="first")
    counts = frame["category"].value_counts()
    missing_categories = [label for label in REQUIRED_CATEGORIES if counts.get(label, 0) == 0]
    if missing_categories:
        raise ValueError(
            "Cannot train the three-class document model. No usable examples remain for "
            f"{', '.join(missing_categories)}. Check the public dataset manifest and "
            "re-run create_dataset.py."
        )
    too_small = {label: int(counts.get(label, 0)) for label in REQUIRED_CATEGORIES
                 if counts.get(label, 0) < MIN_EXAMPLES_PER_CLASS}
    if too_small:
        raise ValueError(
            f"At least {MIN_EXAMPLES_PER_CLASS} unique documents per class are "
            f"required for a stratified holdout; found {too_small}."
        )
    return frame


def main():
    frame = load_training_data()
    weak_labels = "--weak-labels" in sys.argv
    print(
        "Training from folder labels for Invoice/Receipt plus synthetic-derived Bill text."
        if weak_labels
        else (
            "Training from verified public Invoice/Receipt images and "
            "synthetic-derived electricity-bill text:"
        )
    )
    print(frame["category"].value_counts().reindex(REQUIRED_CATEGORIES, fill_value=0))

    train_frame, test_frame = train_test_split(
        frame,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=frame["category"],
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    max_features=20_000,
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(train_frame["document_text"], train_frame["category"])

    predictions = model.predict(test_frame["document_text"])
    accuracy = float(accuracy_score(test_frame["category"], predictions))
    report = classification_report(
        test_frame["category"],
        predictions,
        labels=REQUIRED_CATEGORIES,
        output_dict=True,
        zero_division=0,
    )
    print(f"\nHoldout accuracy against source labels: {accuracy:.4f}")
    print(
        classification_report(
            test_frame["category"],
            predictions,
            labels=REQUIRED_CATEGORIES,
            zero_division=0,
        )
    )
    print("Holdout confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(test_frame["category"], predictions, labels=REQUIRED_CATEGORIES))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    test_frame.drop(columns="_text_hash").to_csv(HOLDOUT_PATH, index=False)
    train_hashes = sorted(train_frame["_text_hash"].unique().tolist())
    TRAIN_HASHES_PATH.write_text(json.dumps(train_hashes, indent=2), encoding="utf-8")
    metadata = {
        "model_type": "TF-IDF + LogisticRegression",
        "classes": list(REQUIRED_CATEGORIES),
        "label_quality": (
            "Invoice/Receipt: weak folder labels; Bill: synthetic-derived text"
            if weak_labels
            else (
                "Invoice/Receipt: verified public image labels; "
                "Bill: synthetic-derived text, not OCR document images"
            )
        ),
        "bill_training_source": SYNTHETIC_BILL_SOURCE,
        "bill_training_license": SYNTHETIC_BILL_LICENSE,
        "bill_training_examples": int((frame["category"] == "Bill").sum()),
        "bill_class_supported": True,
        "bill_input_domain": "synthetic-derived text; real bill-image accuracy unvalidated",
        "train_counts": {
            label: int((train_frame["category"] == label).sum())
            for label in REQUIRED_CATEGORIES
        },
        "holdout_counts": {
            label: int((test_frame["category"] == label).sum())
            for label in REQUIRED_CATEGORIES
        },
        "holdout_accuracy_against_source_labels": accuracy,
        "holdout_classification_report": report,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved classifier: {MODEL_PATH}")
    print(f"Saved model metadata: {METADATA_PATH}")
    print(f"Saved independent holdout: {HOLDOUT_PATH}")
    print("Run evaluate_pipeline.py to save the detailed evaluation report.")


if __name__ == "__main__":
    main()
