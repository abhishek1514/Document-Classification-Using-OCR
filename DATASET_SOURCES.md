# Dataset, licensing, and model scope

## Training corpora

The reproducible collection script downloads 300 images per class from pinned
Hugging Face revisions and records the repository, revision, license, source
page, and image SHA-256 in `data/public_dataset_manifest.csv`.

| Class | Dataset | Contents | Declared license |
| --- | --- | --- | --- |
| Invoice | [KatanaML Sparrow invoices](https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1) | 500 invoice documents annotated by Katana ML; the training split contains 425 images. The card cites the original [Mendeley invoice samples](https://data.mendeley.com/datasets/tnj49gpmtz/2). | MIT on the Hugging Face dataset card |
| Receipt | [FiftyOne scanned receipts](https://huggingface.co/datasets/Voxel51/scanned_receipts) | 712 scanned receipt images from the ICDAR SROIE training corpus. | CC BY 4.0 on the Hugging Face dataset card |
| Bill | [New York Household Electricity Bills 2023-2025](https://www.kaggle.com/datasets/rahibvk/new-york-household-electricity-bills) | 1,000 synthetic monthly electricity billing records from the published sample CSV. `prepare_bill_dataset.py` renders each record as one of four clearly identified text templates; these are not scanned bills or OCR output. | Kaggle declares CC BY-SA 4.0; the bundled README says “Suggested Kaggle license: CC0/Public Domain.” This conflict is recorded here, and CC BY-SA 4.0 is treated as the stricter reuse condition. |

The collector checks that each dataset still has the reviewed revision and
license before downloading. It selects documents at evenly spaced positions
in each source, checks image readability, and stops rather than silently
training if downloads fail. Keep the citations and attribution requirements
from the dataset cards when redistributing collected images or results.

Run the pipeline from the project root:

```powershell
.\.venv\Scripts\python.exe collect_public_dataset.py
.\.venv\Scripts\python.exe create_dataset.py
.\.venv\Scripts\python.exe prepare_bill_dataset.py
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe evaluate_pipeline.py
```

`create_dataset.py` OCRs the collected images with CPU RapidOCR and writes
`dataset.csv`, `dataset_review.csv`, `dataset_processing_errors.csv`, and
`dataset_quality_report.json`. Only files whose SHA-256 and class match the
pinned public-dataset manifest are accepted by `train_model.py`; keyword
suggestions are review hints and are never treated as model labels.

## Bill-class limitation

The legacy `data/Bill/` directory is not used: OCR spot checks found invoice
headings and invoice numbers in files named `Real_Utility_Bill_*.png`.
Three-class support now includes an explicitly synthetic Bill class derived
from the Kaggle dataset above. Each tabular record is rendered as text with
electricity usage, charges, and a total; it is not an original document image.
The model can now return Bill, but this synthetic-text class does not establish
reliable recognition of real scanned utility bills. Real utility-bill images
must be collected and independently evaluated before making that claim.

The source archive and extracted files are kept under
`data/external_bill_source/`. Rebuild the synthetic text rows with:

```powershell
.\.venv\Scripts\python.exe prepare_bill_dataset.py
```

## OCR and classification approach

PDFs with a usable embedded text layer are extracted directly. Scanned PDFs
and images use RapidOCR with ONNX Runtime on CPU, avoiding a GPU-only or
hosted OCR dependency. The classifier is a scikit-learn pipeline of word
TF-IDF features and class-balanced Logistic Regression. This is a compact,
interpretable small-data baseline whose class probabilities support an
inconclusive result.

The trainer deduplicates normalized text, requires verified Invoice/Receipt
examples and synthetic-derived Bill examples, and reserves a stratified 20%
holdout. The evaluator checks for text
hash overlap and reports accuracy, balanced accuracy, macro/weighted F1,
per-class precision/recall/F1, a confusion matrix, and misclassified holdout
rows in `evaluation_report.json` and `misclassified_documents.csv`.
Holdout results measure performance within the cited corpora and synthetic
Bill text; they are not a guarantee for unseen languages, layouts, or real
utility-bill images.

## Legacy files

The pre-existing files under `data/Invoice/`, `data/Bill/`, and `data/Receipt/`
have no source manifest in this workspace. They remain untouched for review
but are not used as verified training examples. This prevents repeated,
synthetic, or incorrectly named files from silently becoming ground truth.
