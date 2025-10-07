from fastapi import APIRouter
from pydantic import BaseModel
from ..rag.retrieve import demo_store
from ..rag.evidence import select_evidence
from ..rag.generate import answer_from_context
router = APIRouter()
class AskReq(BaseModel):
    query: str
    top_k: int = 6
@router.post("")
def ask(req: AskReq):
    store = demo_store()
    contexts = store.search(req.query, top_k=req.top_k)
    if not contexts:
        return {"answer": "No relevant evidence found.", "citations": [], "evidence": []}
    evidence = select_evidence(contexts)
    answer = answer_from_context(req.query, contexts)
    citations = [{"source": c["meta"]["source"], "section": c["meta"]["section"]} for c in contexts]
    return {"answer": answer, "citations": citations, "evidence": evidence}
