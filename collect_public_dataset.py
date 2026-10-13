import argparse
import csv
import hashlib
import io
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from PIL import Image


ROOT = Path(__file__).resolve().parent
PUBLIC_DATA_DIR = ROOT / "data" / "public"
MANIFEST_PATH = ROOT / "data" / "public_dataset_manifest.csv"
DOWNLOAD_ERRORS_PATH = ROOT / "data" / "public_download_errors.csv"
PAGE_SIZE = 100
USER_AGENT = "OCR-project-public-dataset-collector/1.0"
DATASETS = {
    "Invoice": {
        "id": "katanaml-org/invoices-donut-data-v1",
        "revision": "d2cde298e79c94fb05bc320999deb4b7889b0464",
        "license": "mit",
        "page": "https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1",
        "split": "train",
    },
    "Receipt": {
        "id": "Voxel51/scanned_receipts",
        "revision": "3ddf1beda2f4d1cd56efa11503c0ab87edb2de80",
        "license": "cc-by-4.0",
        "page": "https://huggingface.co/datasets/Voxel51/scanned_receipts",
    },
}
MANIFEST_FIELDS = (
    "category",
    "filename",
    "source_path",
    "source_dataset",
    "source_url",
    "source_license",
    "source_revision",
    "sha256",
)


def get_bytes(url):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    last_error = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=90) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"Could not download {url}: {last_error}") from last_error


def get_json(url):
    return json.loads(get_bytes(url))


def evenly_spaced_indices(total, count):
    if count > total:
        raise ValueError(f"Requested {count} documents, but the source has only {total}.")
    if count == 1:
        return [total // 2]
    return [round(index * (total - 1) / (count - 1)) for index in range(count)]


def verify_dataset_metadata(dataset):
    metadata = get_json(f"https://huggingface.co/api/datasets/{dataset['id']}")
    actual_license = (metadata.get("cardData") or {}).get("license", "").casefold()
    if metadata.get("sha") != dataset["revision"]:
        raise RuntimeError(
            f"{dataset['id']} changed from the reviewed revision "
            f"{dataset['revision']} to {metadata.get('sha')}; review its card before updating."
        )
    if actual_license != dataset["license"]:
        raise RuntimeError(
            f"{dataset['id']} license changed from {dataset['license']} to "
            f"{actual_license or 'undeclared'}; do not download until reviewed."
        )
    return metadata


def invoice_image_urls(count):
    dataset = DATASETS["Invoice"]
    metadata = verify_dataset_metadata(dataset)
    total = next(
        split["num_examples"]
        for split in metadata["cardData"]["dataset_info"]["splits"]
        if split["name"] == dataset["split"]
    )
    wanted = set(evenly_spaced_indices(total, count))
    records = {}
    for offset in range(0, total, PAGE_SIZE):
        query = urlencode(
            {
                "dataset": dataset["id"],
                "config": "default",
                "split": dataset["split"],
                "offset": offset,
                "length": PAGE_SIZE,
            }
        )
        page = get_json(f"https://datasets-server.huggingface.co/rows?{query}")
        for item in page.get("rows", []):
            row_index = item.get("row_idx")
            if row_index not in wanted:
                continue
            image = item.get("row", {}).get("image", {})
            if image.get("src"):
                records[row_index] = image["src"]

    missing = wanted - records.keys()
    if missing:
        raise RuntimeError(
            f"The invoice dataset viewer did not return selected rows: {sorted(missing)}"
        )
    return [(index, records[index]) for index in sorted(wanted)]


def receipt_image_urls(count):
    dataset = DATASETS["Receipt"]
    metadata = verify_dataset_metadata(dataset)
    filenames = sorted(
        item["rfilename"]
        for item in metadata.get("siblings", [])
        if item.get("rfilename", "").lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    )
    indices = evenly_spaced_indices(len(filenames), count)
    return [
        (
            index,
            "https://huggingface.co/datasets/"
            f"{dataset['id']}/resolve/{dataset['revision']}/{quote(filenames[index], safe='/')}",
        )
        for index in indices
    ]


def download_category(category, image_sources):
    dataset = DATASETS[category]
    output_dir = PUBLIC_DATA_DIR / category
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    errors = []
    for source_index, image_url in image_sources:
        suffix = ".jpg" if category == "Invoice" else Path(image_url.split("?", 1)[0]).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            suffix = ".jpg"
        filename = f"public_{category.lower()}_{source_index:04d}{suffix}"
        output_path = output_dir / filename
        try:
            if not output_path.is_file() or output_path.stat().st_size == 0:
                content = get_bytes(image_url)
                with Image.open(io.BytesIO(content)) as image:
                    image.verify()
                output_path.write_bytes(content)
            with Image.open(output_path) as image:
                image.verify()
            digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
            manifest_rows.append(
                {
                    "category": category,
                    "filename": filename,
                    "source_path": output_path.relative_to(ROOT).as_posix(),
                    "source_dataset": dataset["id"],
                    "source_url": dataset["page"],
                    "source_license": dataset["license"],
                    "source_revision": dataset["revision"],
                    "sha256": digest,
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "category": category,
                    "filename": filename,
                    "source_index": source_index,
                    "source_url": dataset["page"],
                    "error": str(exc),
                }
            )
    return manifest_rows, errors


def main():
    parser = argparse.ArgumentParser(
        description="Download licensed invoice and receipt images for OCR classification."
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=300,
        help="Number of documents to collect from each dataset (default: 300).",
    )
    args = parser.parse_args()
    if args.per_class < 20:
        parser.error("--per-class must be at least 20 to support a useful holdout evaluation.")

    sources = {
        "Invoice": invoice_image_urls(args.per_class),
        "Receipt": receipt_image_urls(args.per_class),
    }
    manifest_rows = []
    errors = []
    for category, image_urls in sources.items():
        rows, category_errors = download_category(category, image_urls)
        manifest_rows.extend(rows)
        errors.extend(category_errors)
        print(f"{category}: downloaded or verified {len(rows)} of {len(image_urls)} images")

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    with DOWNLOAD_ERRORS_PATH.open("w", newline="", encoding="utf-8") as output_file:
        fields = ("category", "filename", "source_index", "source_url", "error")
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(errors)

    counts = {category: sum(row["category"] == category for row in manifest_rows) for category in DATASETS}
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Download errors: {len(errors)} ({DOWNLOAD_ERRORS_PATH})")
    if errors or min(counts.values()) != args.per_class:
        raise RuntimeError(
            f"Collection incomplete. Successful counts: {counts}; expected "
            f"{args.per_class} per class. Resolve errors before training."
        )
    print(f"Ready to build the OCR dataset with {counts}.")


if __name__ == "__main__":
    main()
