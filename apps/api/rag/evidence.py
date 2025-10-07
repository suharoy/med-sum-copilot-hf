def select_evidence(chunks, per_chunk:int=1):
    out = []
    for c in chunks:
        first = c["text"].split(". ")[0].strip()
        out.append({"paper_id": c["meta"]["paper_id"], "sentence": first, "score": c.get("score", 0.0)})
    return out
