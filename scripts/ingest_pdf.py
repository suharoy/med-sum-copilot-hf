from pathlib import Path, PurePath
import json
from apps.api.nlp.parse_pdf import parse_pdf_to_sections
SRC=Path("data/pdfs"); OUT=Path("data/parsed"); OUT.mkdir(parents=True, exist_ok=True)
for pdf in SRC.glob("*.pdf"):
    parsed = parse_pdf_to_sections(pdf)
    (OUT / f"{pdf.stem}.json").write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
    print("parsed:", pdf.name)
