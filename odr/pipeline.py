"""First-stage retrieval: BM25 (+RM3 query expansion), dense, and hybrid (RRF)."""
from __future__ import annotations

import os

import numpy as np

from .dense import DenseIndex, Encoder, rrf
from .lexical import BM25, QueryProcessor


class Retriever:
    def __init__(self, corpus: dict[str, str], encoder: Encoder | None = None,
                 cache: str | None = None):
        self.doc_ids = list(corpus)
        self.docs = [corpus[d] for d in self.doc_ids]
        self.bm25 = BM25(self.docs)
        self.qp = QueryProcessor(self.bm25)
        self.encoder = encoder
        self.dense = None
        if encoder is not None:
            if cache and os.path.exists(cache):
                emb = np.load(cache)
            else:
                emb = encoder.encode(self.docs)
                if cache:
                    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
                    np.save(cache, emb)
            self.dense = DenseIndex(emb)

    def bm25_ranks(self, query: str, k: int = 100, expand: bool = True) -> list[int]:
        q = self.qp.expand(query) if expand else self.qp.parse(query)
        return self.bm25.search(q, k)[0].tolist() if q else []

    def dense_ranks(self, queries: list[str], k: int = 100) -> list[list[int]]:
        idx, _ = self.dense.search(self.encoder.encode(queries), k)
        return [row.tolist() for row in idx]

    def run(self, queries: dict[str, str], k: int = 100) -> dict[str, dict[str, list[int]]]:
        """All first-stage variants for a batch of queries: {variant: {qid: [doc_idx]}}."""
        qids = list(queries)
        out = {"bm25": {}, "bm25_rm3": {}}
        for q in qids:
            out["bm25"][q] = self.bm25_ranks(queries[q], k, expand=False)
            out["bm25_rm3"][q] = self.bm25_ranks(queries[q], k, expand=True)
        if self.dense is not None:
            dense = self.dense_ranks([queries[q] for q in qids], k)
            out["dense"] = dict(zip(qids, dense))
            out["hybrid"] = {q: rrf([out["bm25_rm3"][q], out["dense"][q]], top=k) for q in qids}
        return out

    def ids(self, ranks: list[int]) -> list[str]:
        return [self.doc_ids[i] for i in ranks]
