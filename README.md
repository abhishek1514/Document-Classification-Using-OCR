# PDF Document Classification with OCR

A Streamlit application that extracts text from PDFs and images with RapidOCR
and predicts Invoice, Receipt, or Bill with a TF-IDF + Logistic Regression
classifier.

## Important limitation

Invoice and Receipt training examples come from the licensed public image
datasets listed in [DATASET_SOURCES.md](DATASET_SOURCES.md). The Bill class
uses synthetic electricity-billing records rendered into text templates. It
is not trained or validated on real utility-bill scans. A high confidence or
holdout score does not establish accuracy on real bill images.

The old `data/Bill/` folder and `dataset_bills.csv` are deliberately excluded:
the folder contains invoice-like images, and that CSV contains unrelated SQL
snippets mislabeled as bills.

## Run the application

Python 3.11+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The trained model and metadata are included under `models/` so the app can
make predictions without first downloading the training corpora.

## Rebuild the training data and model

The public Invoice/Receipt images are not committed. Download them and rebuild
their OCR text dataset, then prepare synthetic Bill text and train/evaluate:

```powershell
.\.venv\Scripts\python.exe collect_public_dataset.py
.\.venv\Scripts\python.exe create_dataset.py
.\.venv\Scripts\python.exe prepare_bill_dataset.py
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe evaluate_pipeline.py
```

The bill dataset is available at
[Kaggle: New York Household Electricity Bills 2023-2025](https://www.kaggle.com/datasets/rahibvk/new-york-household-electricity-bills).
The dataset page declares CC BY-SA 4.0, while its bundled README suggests
CC0/Public Domain. This project records that discrepancy and follows the
stricter declared CC BY-SA 4.0 condition.

## Current holdout report

The included model's stratified 20% source-split holdout reported 100%
accuracy and macro-F1 across 319 examples (60 Invoice, 59 Receipt, 200
synthetic-derived Bill text examples). These examples share source-corpus
characteristics, and the Bill test examples are templated text rather than
OCR from scanned bills. See [evaluation_report.json](evaluation_report.json)
and [DATASET_SOURCES.md](DATASET_SOURCES.md) for source details and caveats.
