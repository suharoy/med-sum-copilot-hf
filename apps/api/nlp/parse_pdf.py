from pathlib import Path
import fitz  # PyMuPDF
def parse_pdf_to_sections(pdf_path: Path) -> dict:
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    return {"paper_id": pdf_path.stem, "title": pdf_path.stem,
            "sections": [{"name":"full_text","text":text,"pages":[0,len(doc)-1]}]}
