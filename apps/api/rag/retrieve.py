from pathlib import Path
import json
from typing import List, Dict
import numpy as np
from rank_bm25 import BM25Okapi

from .embed import EmbeddingModel


TOPN_SHORTLIST = 250          # how many sentences to shortlist from BM25 before dense scoring
BM25_WEIGHT = 0.7             # lexical weight in hybrid score
COSINE_WEIGHT = 0.3           # semantic weight in hybrid score
MMR_LAMBDA = 0.6              # 0..1 (higher = more relevance, lower = more diversity)
MAX_SENT_PER_SECTION = 12     # cap sentences per section to keep index light

# Prefer results & conclusions for most clinical questions
SECTION_WEIGHTS = {
    "abstract":   1.15,
    "results":    1.30,
    "conclusion": 1.35,
    "discussion": 1.10,
}

# ======= helpers =======
def _sent_split(text: str) -> List[str]:
    """Tiny sentence splitter that keeps deps minimal."""
    parts: List[str] = []
    for para in text.split("\n"):
        p = para.strip()
        if not p:
            continue
        start = 0
        for i in range(len(p) - 1):
            if p[i] == "." and p[i + 1] == " ":
                parts.append(p[start : i + 1])
                start = i + 2
        last = p[start:].strip()
        if last:
            parts.append(last)
    if not parts:
        parts = [text.strip()]
    # hard cap a very long sentence
    return [s.strip()[:1200] for s in parts if s.strip()]

def _tokenize(s: str) -> List[str]:
    s = s.lower().strip()
    return [t for t in s.split() if any(ch.isalpha() for ch in t)]

def _l2n(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return v / n

def mmr(cands: np.ndarray, q: np.ndarray, k: int, lambda_: float = 0.6) -> List[int]:
    """Maximal Marginal Relevance over L2-normalized vectors."""
    n = cands.shape[0]
    if n == 0:
        return []
    rel = (cands @ q)  # (n,)
    chosen = [int(np.argmax(rel))]
    remaining = set(range(n)) - set(chosen)
    while len(chosen) < min(k, n) and remaining:
        best_i = None
        best_score = -1e9
        for i in remaining:
            relevance = float(rel[i])
            diversity = max(float(cands[i] @ cands[j]) for j in chosen)
            score = lambda_ * relevance - (1.0 - lambda_) * diversity
            if score > best_score:
                best_score = score
                best_i = i
        chosen.append(int(best_i))
        remaining.remove(int(best_i))
    return chosen


# ======= store =======
class SimpleStore:
    """
    Sentence-level store.
    Pipeline:
      1) BM25 shortlist (TOPN_SHORTLIST)
      2) Dense encode shortlist only (MiniLM on CPU)
      3) Hybrid score (BM25 + cosine) + section weights
      4) MMR to pick diverse top-k sentences
    """
    def __init__(self, parsed_dir: Path):
        self.sentences: List[str] = []
        self.meta: List[Dict] = []

        # Load sections and turn into sentences
        for jf in parsed_dir.glob("*.json"):
            j = json.loads(jf.read_text(encoding="utf-8"))
            pid = j.get("paper_id", jf.stem)
            for sec in j.get("sections", []):
                sec_name = str(sec.get("name", "")).lower()
                sents = _sent_split(sec.get("text", ""))[:MAX_SENT_PER_SECTION]
                for s in sents:
                    self.sentences.append(s)
                    self.meta.append({"paper_id": pid, "section": sec_name, "source": jf.name})

        # BM25 over tokenized sentences (fast and memory-light)
        self.corpus_tokens: List[List[str]] = [_tokenize(s) for s in self.sentences]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.sentences else None

        # Dense encoder (created once, used on shortlist per query)
        self.emb = EmbeddingModel()

    def search(self, query: str, top_k: int = 6) -> List[Dict]:
        if not self.sentences or self.bm25 is None:
            return []

        # 1) BM25 scores for all sentences
        q_tokens = _tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(q_tokens), dtype=np.float32)
        if float(bm25_scores.max()) > 0:
            bm25_scores = bm25_scores / (float(bm25_scores.max()) + 1e-12)

        # shortlist the top-N by BM25
        N = min(TOPN_SHORTLIST, len(self.sentences))
        short_idx = np.argsort(-bm25_scores)[:N]
        short_texts = [self.sentences[i] for i in short_idx]

        # 2) Dense encode ONLY shortlist + query
        q_vec = self.emb.encode([query])[0]    # (d,)
        q_vec = _l2n(q_vec)
        cand_vecs = self.emb.encode(short_texts)
        cand_vecs = _l2n(cand_vecs)            # (N, d)

        cos_scores = (cand_vecs @ q_vec).astype(np.float32)

        # 3) Hybrid score + section weighting
        hybrid = BM25_WEIGHT * bm25_scores[short_idx] + COSINE_WEIGHT * cos_scores

        # section boosts
        boosts = np.ones_like(hybrid)
        for i, idx in enumerate(short_idx):
            sec = self.meta[int(idx)]["section"]
            boosts[i] = SECTION_WEIGHTS.get(sec, 1.0)
        hybrid = hybrid * boosts

        # 4) MMR to diversify final k
        chosen_local = mmr(cand_vecs, q_vec, k=int(top_k), lambda_=MMR_LAMBDA)
        final_idx = [int(short_idx[i]) for i in chosen_local]

        out: List[Dict] = []
        for i in final_idx:
            out.append({
                "text": self.sentences[i],
                "meta": self.meta[i],
                "score": float(hybrid[np.where(short_idx == i)[0][0]])
            })
        return out


def demo_store() -> SimpleStore:
    return SimpleStore(Path("data/parsed"))