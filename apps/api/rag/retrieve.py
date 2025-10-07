from pathlib import Path
import json, numpy as np
from typing import List, Dict
from .embed import EmbeddingModel
def _cosine_scores(matrix: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    return matrix @ query_vec
class SimpleStore:
    def __init__(self, parsed_dir: Path):
        self.chunks: List[str] = []
        self.meta: List[Dict] = []
        for jf in parsed_dir.glob("*.json"):
            j = json.loads(jf.read_text(encoding="utf-8"))
            pid = j.get("paper_id", jf.stem)
            for sec in j.get("sections", []):
                txt = (sec.get("text") or "").strip()
                if txt:
                    self.chunks.append(txt)
                    self.meta.append({"paper_id": pid, "section": sec.get("name",""), "source": jf.name})
        self.emb_model = EmbeddingModel()
        self.embs = self.emb_model.encode(self.chunks) if self.chunks else None
    def search(self, query: str, top_k: int = 6):
        if self.embs is None or not self.chunks:
            return []
        q = self.emb_model.encode([query])[0]
        sims = _cosine_scores(self.embs, q)
        idx = np.argsort(-sims)[:top_k]
        return [{"text": self.chunks[i], "meta": self.meta[i], "score": float(sims[i])} for i in idx]
def demo_store():
    return SimpleStore(Path("data/parsed"))
