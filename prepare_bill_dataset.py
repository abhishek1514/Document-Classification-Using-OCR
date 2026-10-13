import csv
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "data" / "external_bill_source" / "download" / "sample_1000_rows.csv"
SOURCE_ARCHIVE_PATH = ROOT / "data" / "external_bill_source" / "kaggle_bills.zip"
SOURCE_ARCHIVE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "rahibvk/new-york-household-electricity-bills"
)
OUTPUT_PATH = ROOT / "dataset_synthetic_bills.csv"
SOURCE_ID = "rahibvk/new-york-household-electricity-bills"
SOURCE_URL = "https://www.kaggle.com/datasets/rahibvk/new-york-household-electricity-bills"
SOURCE_LICENSE = "CC BY-SA 4.0"
SOURCE_REVISION = "Kaggle snapshot 2026-06-25"

TEMPLATES = (
    (
        "ELECTRICITY BILL\nBilling month: {month}\nEnergy used: {kwh} kWh\n"
        "Supply charge: ${supply}\nDelivery charge: ${delivery}\n"
        "Taxes and fees: ${taxes}\nTotal amount due: ${total}\nStatus: {status}"
    ),
    (
        "ELECTRIC SERVICE STATEMENT\nStatement period: {month}\n"
        "Metered electricity consumption: {kwh} kilowatt-hours\n"
        "Electric supply: ${supply}\nElectric delivery: ${delivery}\n"
        "Tax/fee total: ${taxes}\nBILL TOTAL: ${total}\nPayment status: {status}"
    ),
    (
        "MONTHLY POWER BILL\nService month: {month}\nUsage this period: {kwh} kWh\n"
        "Energy supply cost ${supply}\nGrid delivery cost ${delivery}\n"
        "Taxes and fees ${taxes}\nAmount billed ${total}\nAccount status: {status}"
    ),
    (
        "UTILITY BILL - ELECTRIC\nBilling date: {month}\n"
        "Electricity consumed: {kwh} kWh\nSupply amount ${supply}\n"
        "Delivery amount ${delivery}\nFees and taxes ${taxes}\n"
        "TOTAL DUE ${total}\nBill status: {status}"
    ),
)


def main():
    if not SOURCE_PATH.is_file():
        SOURCE_ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not SOURCE_ARCHIVE_PATH.is_file():
            with urlopen(SOURCE_ARCHIVE_URL, timeout=60) as response:
                SOURCE_ARCHIVE_PATH.write_bytes(response.read())
        with ZipFile(SOURCE_ARCHIVE_PATH) as archive:
            try:
                with archive.open("sample_1000_rows.csv") as source, SOURCE_PATH.open(
                    "wb"
                ) as destination:
                    destination.write(source.read())
            except KeyError as exc:
                raise ValueError(
                    "The downloaded Kaggle dataset archive does not contain "
                    "sample_1000_rows.csv."
                ) from exc

    rows = []
    with SOURCE_PATH.open(newline="", encoding="utf-8-sig") as source_file:
        reader = csv.DictReader(source_file)
        required = {
            "bill_month",
            "kwh_usage",
            "supply_charge",
            "delivery_charge",
            "taxes_and_fees",
            "total_bill_amount",
            "bill_status",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Bill source is missing fields: {sorted(missing)}")

        for index, record in enumerate(reader):
            values = {
                "month": record["bill_month"],
                "kwh": record["kwh_usage"],
                "supply": record["supply_charge"],
                "delivery": record["delivery_charge"],
                "taxes": record["taxes_and_fees"],
                "total": record["total_bill_amount"],
                "status": record["bill_status"],
            }
            text = TEMPLATES[index % len(TEMPLATES)].format(**values)
            rows.append(
                {
                    "document_text": text,
                    "category": "Bill",
                    "filename": f"synthetic_electric_bill_{index + 1:04d}",
                    "source_path": (
                        "data/external_bill_source/download/sample_1000_rows.csv"
                    ),
                    "source_record": index + 1,
                    "label_status": "synthetic-derived",
                    "source_dataset": SOURCE_ID,
                    "source_url": SOURCE_URL,
                    "source_license": SOURCE_LICENSE,
                    "source_revision": SOURCE_REVISION,
                }
            )

    if not rows:
        raise ValueError("The downloaded bill sample contains no records.")

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} synthetic-derived Bill examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
