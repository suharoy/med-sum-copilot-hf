from fastapi import APIRouter
from pathlib import Path
import json
from ..core.config import settings
from ..nlp.parse_pdf import parse_pdf_to_sections
router = APIRouter()
@router.post("")
def ingest_all():
    pdf_dir = Path(settings.STORAGE_DIR) / "pdfs"
    out_dir = Path(settings.STORAGE_DIR) / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for pdf in pdf_dir.glob("*.pdf"):
        parsed = parse_pdf_to_sections(pdf)
        with open(out_dir / f"{pdf.stem}.json", "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False)
        count += 1
    return {"ok": True, "parsed": count}
