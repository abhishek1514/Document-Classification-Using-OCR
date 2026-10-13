import io
import re
from pathlib import Path

import numpy as np
import pymupdf as fitz
from PIL import Image


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ocr_image(image, ocr_engine):
    result, _ = ocr_engine(np.asarray(image.convert("RGB")))
    if not result:
        return ""
    return "\n".join(line[1] for line in result if len(line) > 1 and line[1].strip())


def extract_document_text(content, filename, ocr_engine, max_pages=20):
    suffix = Path(filename).suffix.lower()
    page_texts = []

    if suffix == ".pdf":
        with fitz.open(stream=content, filetype="pdf") as document:
            if document.is_encrypted:
                raise ValueError("The PDF is password-protected.")
            if document.page_count == 0:
                raise ValueError("The PDF contains no pages.")
            if document.page_count > max_pages:
                raise ValueError(f"The PDF exceeds the {max_pages}-page processing limit.")

            for page in document:
                text = clean_text(page.get_text("text"))
                if len(text) < 40:
                    pixmap = page.get_pixmap(dpi=220, alpha=False)
                    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                    text = clean_text(_ocr_image(image, ocr_engine))
                if text:
                    page_texts.append(text)
    else:
        try:
            image = Image.open(io.BytesIO(content))
            image.verify()
            image = Image.open(io.BytesIO(content))
        except Exception as exc:
            raise ValueError("The uploaded file is not a readable image.") from exc
        text = clean_text(_ocr_image(image, ocr_engine))
        if text:
            page_texts.append(text)

    return clean_text("\n".join(page_texts))
