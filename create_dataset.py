import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from rapidocr_onnxruntime import RapidOCR

from ocr_utils import extract_document_text


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PUBLIC_DATA_DIR = DATA_DIR / "public"
PUBLIC_MANIFEST_PATH = DATA_DIR / "public_dataset_manifest.csv"
OUTPUT_CSV = ROOT / "dataset.csv"
REVIEW_CSV = ROOT / "dataset_review.csv"
ERRORS_CSV = ROOT / "dataset_processing_errors.csv"
REPORT_JSON = ROOT / "dataset_quality_report.json"
CATEGORIES = ("Invoice", "Receipt")
SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
FIELDNAMES = [
    "document_text", "category", "filename", "source_path",
    "suggested_category", "label_status", "sha256", "source_dataset",
    "source_url", "source_license", "source_revision",
]


def suggest_category(text):
    normalized = re.sub(r"\s+", " ", text.lower())
    receipt_markers = (
        "receipt",
        "cash bill",
        "cashier",
        "please come again",
        "change",
        "amount tendered",
    )
    utility_markers = (
        "utility bill",
        "electricity",
        "electric bill",
        "water bill",
        "gas bill",
        "meter reading",
        "billing period",
        "bill period",
        "kwh",
        "kilowatt",
        "tariff",
    )
    invoice_markers = (
        "invoice number",
        "invoice no",
        "invoice date",
        "tax invoice",
        "bill to",
        "due date",
        "seller:",
        "client:",
    )

    receipt_hits = sum(marker in normalized for marker in receipt_markers)
    utility_hits = sum(marker in normalized for marker in utility_markers)
    invoice_hits = sum(marker in normalized for marker in invoice_markers)

    if receipt_hits >= 2 or "receipt" in normalized or "cash bill" in normalized:
        return "Receipt"
    if utility_hits >= 2:
        return "Bill"
    if invoice_hits >= 2 or "invoice number" in normalized or "invoice no" in normalized:
        return "Invoice"
    return None


def refresh_existing_labels():
    if not OUTPUT_CSV.is_file():
        raise FileNotFoundError(f"Dataset not found: {OUTPUT_CSV}. Run create_dataset.py first.")

    with OUTPUT_CSV.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    public_manifest = load_public_manifest()
    previous_report = {}
    if REPORT_JSON.is_file():
        previous_report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))

    review_rows = []
    for row in rows:
        suggested = suggest_category(row["document_text"])
        row["suggested_category"] = suggested or ""
        source = public_manifest.get(row["source_path"])
        row["label_status"] = (
            "verified"
            if source and source["category"] == row["category"] and source["sha256"] == row["sha256"]
            else "unverified" if suggested == row["category"] else "label_mismatch"
        )
        if source:
            row.update(
                {
                    "source_dataset": source["source_dataset"],
                    "source_url": source["source_url"],
                    "source_license": source["source_license"],
                    "source_revision": source["source_revision"],
                }
            )
        review_rows.append(row)

    for output_path, output_rows in ((OUTPUT_CSV, rows), (REVIEW_CSV, review_rows)):
        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(output_rows)

    source_counts = Counter(row["category"] for row in rows)
    suggested_counts = Counter(
        row["suggested_category"] for row in rows if row["suggested_category"]
    )
    verified_count = sum(row["label_status"] == "verified" for row in rows)
    matching_count = sum(row["suggested_category"] == row["category"] for row in rows)
    report = {
        "source_directory": previous_report.get(
            "source_directory", "data/public/<category>/"
        ),
        "ocr_engine": previous_report.get(
            "ocr_engine", "RapidOCR ONNX Runtime (CPU)"
        ),
        "documents_written": len(rows),
        "verified_for_training": verified_count,
        "unverified_documents": len(rows) - verified_count,
        "heuristic_label_matches": matching_count,
        "potential_label_mismatches": len(rows) - matching_count,
        "processing_errors": previous_report.get("processing_errors", 0),
        "duplicate_files_skipped": previous_report.get("duplicate_files_skipped", 0),
        "duplicate_texts_skipped": previous_report.get("duplicate_texts_skipped", 0),
        "source_category_counts": dict(source_counts),
        "suggested_category_counts": dict(suggested_counts),
        "warning": (
            "Only rows recorded in the pinned, licensed public-dataset manifest are "
            "verified for training. OCR keyword checks are review hints, not labels."
        ),
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Verified public examples available for training: {verified_count}")


def main():
    if "--refresh-labels" in sys.argv:
        refresh_existing_labels()
        return

    missing = [
        category for category in CATEGORIES
        if not (PUBLIC_DATA_DIR / category).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing public image folders. Run collect_public_dataset.py first; "
            f"missing: {', '.join(missing)}"
        )

    print("Loading RapidOCR CPU engine...")
    ocr_engine = RapidOCR()
    public_manifest = load_public_manifest()
    rows = []
    review_rows = []
    error_rows = []
    seen_files = set()
    seen_texts = set()
    skipped_duplicate_files = 0
    skipped_duplicate_texts = 0

    for category in CATEGORIES:
        category_dir = PUBLIC_DATA_DIR / category
        files = sorted(
            path
            for path in category_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        print(f"{category}: processing {len(files)} documents")

        for path in files:
            try:
                content = path.read_bytes()
                file_hash = hashlib.sha256(content).hexdigest()
                if file_hash in seen_files:
                    skipped_duplicate_files += 1
                    continue
                seen_files.add(file_hash)

                text = extract_document_text(content, path.name, ocr_engine)
                if len(text) < 20:
                    error_rows.append(
                        {"source_path": str(path.relative_to(ROOT)), "error": "No usable text extracted"}
                    )
                    continue

                text_hash = hashlib.sha256(
                    re.sub(r"\s+", " ", text).strip().casefold().encode("utf-8")
                ).hexdigest()
                if text_hash in seen_texts:
                    skipped_duplicate_texts += 1
                    continue
                seen_texts.add(text_hash)

                suggested = suggest_category(text)
                source_path = path.relative_to(ROOT).as_posix()
                source = public_manifest.get(source_path)
                source_verified = bool(
                    source
                    and source["category"] == category
                    and source["sha256"] == file_hash
                    and source["source_dataset"]
                    and source["source_license"]
                    and source["source_revision"]
                )
                label_status = (
                    "verified"
                    if source_verified
                    else "unverified" if suggested == category else "label_mismatch"
                )
                row = {
                    "document_text": text,
                    "category": category,
                    "filename": path.name,
                    "source_path": source_path,
                    "suggested_category": suggested or "",
                    "label_status": label_status,
                    "sha256": file_hash,
                    "source_dataset": source["source_dataset"] if source_verified else "",
                    "source_url": source["source_url"] if source_verified else "",
                    "source_license": source["source_license"] if source_verified else "",
                    "source_revision": source["source_revision"] if source_verified else "",
                }
                rows.append(row)
                review_rows.append(row)
            except Exception as exc:
                error_rows.append(
                    {"source_path": str(path.relative_to(ROOT)), "error": str(exc)}
                )

    for output_path, output_rows, fields in (
        (OUTPUT_CSV, rows, FIELDNAMES),
        (REVIEW_CSV, review_rows, FIELDNAMES),
        (ERRORS_CSV, error_rows, ["source_path", "error"]),
    ):
        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output_rows)

    counts = Counter(row["category"] for row in rows)
    suggested_counts = Counter(
        row["suggested_category"] for row in rows if row["suggested_category"]
    )
    verified_count = sum(row["label_status"] == "verified" for row in rows)
    matching_count = sum(row["suggested_category"] == row["category"] for row in rows)
    report = {
        "source_directory": "data/public/<category>/",
        "ocr_engine": "RapidOCR ONNX Runtime (CPU)",
        "documents_written": len(rows),
        "verified_for_training": verified_count,
        "unverified_documents": len(rows) - verified_count,
        "heuristic_label_matches": matching_count,
        "potential_label_mismatches": len(rows) - matching_count,
        "processing_errors": len(error_rows),
        "duplicate_files_skipped": skipped_duplicate_files,
        "duplicate_texts_skipped": skipped_duplicate_texts,
        "source_category_counts": dict(counts),
        "suggested_category_counts": dict(suggested_counts),
        "warning": (
            "Only documents present in the pinned, licensed public-dataset manifest are "
            "verified for training. Legacy folders remain review-only; OCR keyword checks "
            "are not treated as ground-truth labels."
        ),
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Dataset: {OUTPUT_CSV}")
    print(f"Review suggestions: {REVIEW_CSV}")
    print(f"Verified public documents: {verified_count}")


def load_public_manifest():
    if not PUBLIC_MANIFEST_PATH.is_file():
        return {}
    with PUBLIC_MANIFEST_PATH.open(newline="", encoding="utf-8") as input_file:
        rows = csv.DictReader(input_file)
        return {row["source_path"]: row for row in rows if row.get("source_path")}


if __name__ == "__main__":
    main()
