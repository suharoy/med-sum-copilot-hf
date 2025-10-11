from pathlib import Path
import re
import fitz  # PyMuPDF

# ---- Config ----
SECTION_HINTS = [
    "abstract", "introduction", "background", "methods", "materials",
    "results", "discussion", "conclusion", "limitations", "summary"
]

BOILERPLATE_PATTERNS = [
    r"\bindian journal of\b",
    r"\bscientific scholar\b",
    r"\bdepartment of\b",
    r"\buniversity\b",
    r"\binstitute\b",
    r"\bcorresponding author\b",
    r"\bdoi:\b",
    r"\bkeywords?:\b",
    r"\bconflicts? of interest\b",
    r"\bfunding\b",
]
_BOILER_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)

# ---- Helpers ----
def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    low = s.lower()
    if low in SECTION_HINTS:
        return True
    if any(low.startswith(h) for h in SECTION_HINTS):
        return True
    # ALL CAPS short line → likely a heading
    if len(s) <= 60 and s.isupper():
        return True
    return False

def _clean_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    if _BOILER_RE.search(line):
        return ""
    # collapse whitespace
    line = re.sub(r"\s+", " ", line)
    return line

# ---- Main ----
def parse_pdf_to_sections(pdf_path: Path) -> dict:
    doc = fitz.open(pdf_path)
    lines = []
    for page in doc:
        # get_text() returns a string with newlines; split into lines
        lines.extend(page.get_text().splitlines())

    sections = []
    current_name = "unknown"
    current_buf = []

    for raw in lines:
        line = _clean_line(raw)
        if not line:
            continue

        if _looks_like_heading(line):
            # flush previous
            if current_buf:
                sections.append({"name": current_name, "text": " ".join(current_buf).strip()})
                current_buf = []
            current_name = line.strip().lower()
        else:
            current_buf.append(line)

    if current_buf:
        sections.append({"name": current_name, "text": " ".join(current_buf).strip()})

    # Fallback to a single section if nothing segmented
    if not sections:
        cleaned = " ".join(_clean_line(l) for l in lines if _clean_line(l))
        sections = [{"name": "full_text", "text": cleaned}]

    return {
        "paper_id": pdf_path.stem,
        "title": pdf_path.stem,
        "sections": sections
    }