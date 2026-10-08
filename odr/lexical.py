"""Sparse retrieval: BM25 over a scipy CSR matrix, plus query understanding
(normalisation, stopword removal and RM3 pseudo-relevance-feedback expansion)."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
from scipy import sparse

STOP = set("""a an and are as at be but by for from has have how i if in into is it its
of on or so such than that the their then there these they this to was what when where
which who why will with you your do does did can could should would my me we our""".split())
_TOK = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    toks = _TOK.findall(text.lower())
    return [t[:-2] if t.endswith("'s") else t for t in toks if t not in STOP and len(t) > 1]


class BM25:
    def __init__(self, docs: list[str], k1: float = 0.9, b: float = 0.4):
        self.vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        dl = np.zeros(len(docs), dtype=np.float32)
        for i, d in enumerate(docs):
            tf = Counter(tokenize(d))
            dl[i] = sum(tf.values())
            for t, c in tf.items():
                rows.append(i)
                cols.append(self.vocab.setdefault(t, len(self.vocab)))
                vals.append(c)
        tf = sparse.csr_matrix((np.array(vals, np.float32), (rows, cols)),
                               shape=(len(docs), len(self.vocab)))
        df = np.bincount(tf.indices, minlength=len(self.vocab))
        self.idf = np.log(1 + (len(docs) - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm = k1 * (1 - b + b * dl / max(dl.mean(), 1e-9))
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + norm[tf.row]) * self.idf[tf.col]
        self.W = sparse.csc_matrix((w, (tf.row, tf.col)), shape=tf.shape)   # column access per term
        self.tf = sparse.csr_matrix((tf.data, (tf.row, tf.col)), shape=tf.shape)
        self.dl = dl

    def score(self, weighted_terms: dict[str, float]) -> np.ndarray:
        q = np.zeros(len(self.vocab), dtype=np.float32)
        for t, w in weighted_terms.items():
            j = self.vocab.get(t)
            if j is not None:
                q[j] += w
        return np.asarray(self.W @ q).ravel()

    def search(self, weighted_terms: dict[str, float], k: int = 100):
        s = self.score(weighted_terms)
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        return top, s[top]


class QueryProcessor:
    """Query understanding for lexical retrieval.

    1. normalise + tokenise, drop stopwords;
    2. RM3: run the query, take the top ``fb_docs`` documents, and add the
       ``fb_terms`` terms most associated with them (weighted by document score),
       interpolated with the original query by ``orig_weight``.
    """

    def __init__(self, bm25: BM25, fb_docs: int = 10, fb_terms: int = 10, orig_weight: float = 0.7):
        self.bm25, self.fb_docs, self.fb_terms, self.w0 = bm25, fb_docs, fb_terms, orig_weight
        self.inv_vocab = {i: t for t, i in bm25.vocab.items()}

    def parse(self, query: str) -> dict[str, float]:
        c = Counter(tokenize(query))
        n = sum(c.values()) or 1
        return {t: v / n for t, v in c.items()}

    def expand(self, query: str) -> dict[str, float]:
        q = self.parse(query)
        if not q or self.fb_terms == 0:
            return q
        top, s = self.bm25.search(q, self.fb_docs)
        p_doc = np.exp(s - s.max())
        p_doc /= p_doc.sum()
        rel = self.bm25.tf[top].multiply(1.0 / np.maximum(self.bm25.dl[top], 1)[:, None])
        term_w = np.asarray(rel.T @ p_doc).ravel()
        cand = np.argsort(-term_w)[: self.fb_terms + len(q)]
        fb = {self.inv_vocab[j]: float(term_w[j]) for j in cand if term_w[j] > 0}
        z = sum(fb.values()) or 1.0
        out = {t: self.w0 * w for t, w in q.items()}
        for t, w in fb.items():
            out[t] = out.get(t, 0.0) + (1 - self.w0) * w / z
        return out
