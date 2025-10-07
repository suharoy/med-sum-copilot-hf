from fastapi import APIRouter
from pydantic import BaseModel
from pathlib import Path
import json
from ..core.config import settings
from ..rag.generate import answer_from_context
router = APIRouter()
class SumReq(BaseModel):
    paper_id: str
    mode: str = "expert"
@router.post("")
def summarize(req: SumReq):
    parsed_file = Path(settings.STORAGE_DIR) / "parsed" / f"{req.paper_id}.json"
    if not parsed_file.exists():
        return {"error":"paper not found"}
    j = json.loads(parsed_file.read_text(encoding="utf-8"))
    sections = j.get("sections", [])
    ctxs = [{"text": s["text"], "meta": {"source": parsed_file.name, "section": s["name"]}} for s in sections]
    q = f"Summarize this paper for a {'researcher' if req.mode=='expert' else 'patient'}."
    answer = answer_from_context(q, ctxs)
    return {"summary": answer, "paper_id": req.paper_id, "mode": req.mode}
