from typing import List, Dict, Tuple
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from ..core.config import settings

# ---- lazy singletons ----
_summarizer = None
_embedder = None

def _get_summarizer():
    """Small, instructionless summarizer. Summarize only evidence text."""
    global _summarizer
    if _summarizer is None:
        # Tip: set HF_SUMMARIZER_MODEL=t5-base in .env if BART still echoes
        _summarizer = pipeline("summarization", model=settings.HF_SUMMARIZER_MODEL
        )
    return _summarizer

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.HF_EMBED_MODEL)
    return _embedder

# ---- utilities ----
def _split_sentences(text: str) -> List[str]:
    parts = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        sents = []
        start = 0
        for i in range(len(para)-1):
            if para[i] == "." and para[i+1] == " ":
                sents.append(para[start:i+1])
                start = i+2
        last = para[start:].strip()
        if last:
            sents.append(last)
        if not sents:
            sents = [para]
        parts.extend([s.strip() for s in sents if s.strip()])
    return parts

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n

def _extract_top_sentences(question: str, contexts: List[Dict], max_sentences: int = 6) -> List[Tuple[str, int, Dict]]:
    """Return (sentence, evidence_idx, meta) sorted by similarity to the question."""
    emb = _get_embedder()
    qv = _normalize(emb.encode([question], convert_to_numpy=True))[0]

    sent_texts: List[str] = []
    sent_tags: List[Tuple[int, Dict]] = []   # (ev_idx, meta)

    for i, c in enumerate(contexts, start=1):
        sents = _split_sentences(c["text"])[:12]  # keep small
        for s in sents:
            sent_texts.append(s)
            sent_tags.append((i, c["meta"]))

    if not sent_texts:
        return []

    sims_all = []
    batch = 64
    for j in range(0, len(sent_texts), batch):
        chunk = sent_texts[j:j+batch]
        sv = _normalize(emb.encode(chunk, convert_to_numpy=True))
        sims_all.append(sv @ qv)
    sims = np.concatenate(sims_all, axis=0)

    idx = np.argsort(-sims)[:max_sentences]
    out, seen = [], set()
    for k in idx:
        s = sent_texts[k].strip()
        if s in seen:
            continue
        seen.add(s)
        ev_idx, meta = sent_tags[k]
        out.append((s, ev_idx, meta))
    return out

def _extractive_bullets(picked: List[Tuple[str, int, Dict]]) -> str:
    return "\n".join(f"• {s} [{ev}]" for (s, ev, _m) in picked)

# ---- main ----
def answer_from_context(question: str, contexts: List[Dict]) -> str:
    """
    1) Select top evidence sentences by similarity to the question.
    2) Summarize ONLY those sentences (no instructions).
    3) If anything fails, fall back to bullet points with citations.
    """
    picked = _extract_top_sentences(question, contexts, max_sentences=6)
    if not picked:
        return "No relevant evidence found."

    # Build a compact evidence text that the model will actually summarize.
    # We include the [#] tags inline so we can keep citations if model preserves them,
    # but we DO NOT add any instruction words.
    evidence_lines = [f"[{ev}] {s}" for (s, ev, _m) in picked]
    evidence_text = " ".join(evidence_lines)

    # Keep inputs well under model limits
    words = evidence_text.split()
    if len(words) > 900:
        words = words[:900]
    safe_input = " ".join(words)

    summarizer = _get_summarizer()

    try:
        # Summarize the evidence text ONLY (no “QUESTION/ANSWER” boilerplate).
        # For T5 models, they also work with plain text (prefix optional).
        result = summarizer(
            safe_input,
            max_length=220,
            min_length=60,
            do_sample=False,
            truncation=True
        )[0]["summary_text"].strip()

        # If the model outputs almost nothing or echoes, use extractive fallback.
        if len(result) < 40 or "QUESTION:" in result or "CONTEXT:" in result:
            return _extractive_bullets(picked)
        return result
    except Exception:
        # Robust fallback that always works.
        return _extractive_bullets(picked)