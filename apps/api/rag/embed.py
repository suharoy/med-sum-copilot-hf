import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from ..core.config import settings
def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms
class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(settings.HF_EMBED_MODEL)
    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        return _normalize(vecs.astype("float32"))
