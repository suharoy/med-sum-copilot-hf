from fastapi import APIRouter, UploadFile, File
from pathlib import Path
from ..core.config import settings
router = APIRouter()
@router.post("")
async def upload_pdf(file: UploadFile = File(...)):
    dest = Path(settings.STORAGE_DIR) / "pdfs" / file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        f.write(await file.read())
    return {"ok": True, "stored_as": str(dest)}
