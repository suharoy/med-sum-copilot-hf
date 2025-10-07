# Medical Research Summarizer — HF-only (CPU / Free)

## Quickstart
```
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt

# Put PDFs into data/pdfs/
python -m scripts.ingest_pdf

# UI
streamlit run apps/web/app.py  # http://localhost:8501
```
