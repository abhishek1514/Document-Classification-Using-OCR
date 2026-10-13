from io import BytesIO
import json
from pathlib import Path

import joblib
import pymupdf as fitz
import streamlit as st
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from ocr_utils import extract_document_text


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "document_classifier.joblib"
MODEL_METADATA_PATH = ROOT / "models" / "model_metadata.json"
MIN_CONFIDENCE = 0.55

st.set_page_config(page_title="Document classifier", page_icon="📄", layout="wide")
st.title("PDF document classifier")
st.caption(
    "Extract text and classify invoices, receipts, and electricity bills with a "
    "TF-IDF machine-learning model. Bill examples are synthetic text, not scanned bills."
)


@st.cache_resource
def load_classifier(model_version):
    classifier = joblib.load(MODEL_PATH) if MODEL_PATH.is_file() else None
    metadata = (
        json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        if MODEL_METADATA_PATH.is_file()
        else {}
    )
    return classifier, metadata


try:
    model_version = MODEL_PATH.stat().st_mtime_ns if MODEL_PATH.is_file() else 0
    classifier, model_metadata = load_classifier(model_version)
except Exception as exc:
    classifier, model_metadata = None, {}
    st.warning(f"The machine-learning model could not be loaded: {exc}")

@st.cache_resource
def load_ocr_engine():
    return RapidOCR()


try:
    ocr_engine = load_ocr_engine()
except Exception as exc:
    st.error(f"Could not initialize OCR: {exc}")
    st.stop()

if classifier is None:
    st.info(
        "The machine-learning model has not been trained yet. Add a verified "
        "Invoice/Receipt dataset and synthetic bill dataset, then run train_model.py."
    )
elif "weak folder labels" in model_metadata.get("label_quality", ""):
    st.warning(
        "Experimental ML model loaded. Invoice/Receipt labels come from unverified "
        "folder names and Bill examples are synthetic text; review predictions."
    )
else:
    st.success(
        "Invoice/Receipt/Electricity Bill ML classifier is ready. Bill examples are "
        "synthetic-derived text; verify bill predictions against the original document."
    )

uploaded_file = st.file_uploader(
    "Upload a PDF or image",
    type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"],
)

if uploaded_file is not None:
    content = uploaded_file.getvalue()
    preview_column, result_column = st.columns(2)
    try:
        if Path(uploaded_file.name).suffix.lower() == ".pdf":
            with fitz.open(stream=content, filetype="pdf") as document:
                if document.page_count == 0:
                    raise ValueError("The PDF contains no pages.")
                pixmap = document[0].get_pixmap(dpi=140, alpha=False)
                preview = Image.open(BytesIO(pixmap.tobytes("png"))).convert("RGB")
        else:
            preview = Image.open(BytesIO(content)).convert("RGB")

        with preview_column:
            st.subheader("Document preview")
            st.image(preview, caption=uploaded_file.name)

        with st.spinner("Extracting and classifying document text..."):
            extracted_text = extract_document_text(content, uploaded_file.name, ocr_engine)

        with result_column:
            st.subheader("Classification")
            if len(extracted_text) < 20:
                st.warning("Not enough readable text was found to classify this document.")
            elif classifier is None:
                st.warning(
                    "OCR text was extracted, but no ML prediction is available until "
                    "the classifier model is trained."
                )
            else:
                probabilities = classifier.predict_proba([extracted_text])[0]
                best_index = int(probabilities.argmax())
                predicted_category = classifier.classes_[best_index]
                confidence = float(probabilities[best_index])
                if confidence < MIN_CONFIDENCE:
                    st.warning(
                        f"Prediction is inconclusive (best match: {predicted_category}, "
                        f"{confidence:.1%} model confidence). Please verify it manually."
                    )
                else:
                    st.success(
                        f"ML prediction: **{predicted_category}** "
                        f"({confidence:.1%} model confidence)"
                    )
                st.dataframe(
                    {
                        "Category": classifier.classes_,
                        "Score": [float(value) for value in probabilities],
                    },
                    hide_index=True,
                    width="stretch",
                )

        st.subheader("Extracted text")
        st.text_area(
            "OCR output",
            value=extracted_text or "No readable text found.",
            height=300,
            label_visibility="collapsed",
        )
    except Exception as exc:
        with result_column:
            st.error(f"Document processing failed: {exc}")
else:
    st.info("Choose a document to see its extracted text and predicted category.")
