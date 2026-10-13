raise SystemExit(
    "This downloader treated cropped receipt text lines as document-level samples and "
    "produced incomplete receipts. Add whole, permission-cleared receipt documents to "
    "data/Receipt/ and run create_dataset.py."
)
