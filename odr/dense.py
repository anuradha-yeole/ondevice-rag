"""Dense retrieval: transformer bi-encoder (mean pooling) + FAISS inner-product index."""
from __future__ import annotations

import os

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class Encoder:
    def __init__(self, name_or_model=None, tokenizer=None,
                 max_len: int = 256, device: str | None = None):
        if name_or_model is None:
            name_or_model = os.environ.get("ODR_ENCODER", "sentence-transformers/all-MiniLM-L6-v2")
        if isinstance(name_or_model, str):
            self.tok = AutoTokenizer.from_pretrained(name_or_model)
            self.model = AutoModel.from_pretrained(name_or_model)
        else:
            self.tok, self.model = tokenizer, name_or_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.max_len = max_len

    @torch.inference_mode()
    def encode(self, texts: list[str], bs: int = 128) -> np.ndarray:
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], padding=True, truncation=True,
                           max_length=self.max_len, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.model.config.hidden_size), np.float32)


class DenseIndex:
    def __init__(self, emb: np.ndarray):
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(np.ascontiguousarray(emb, dtype=np.float32))

    def search(self, q: np.ndarray, k: int = 100):
        s, i = self.index.search(np.ascontiguousarray(q, dtype=np.float32), k)
        return i, s


def rrf(rankings: list[list[int]], k: int = 60, top: int = 100) -> list[int]:
    """Reciprocal rank fusion of several ranked lists of doc indices."""
    score: dict[int, float] = {}
    for ranks in rankings:
        for r, d in enumerate(ranks):
            score[d] = score.get(d, 0.0) + 1.0 / (k + r + 1)
    return [d for d, _ in sorted(score.items(), key=lambda x: -x[1])[:top]]
