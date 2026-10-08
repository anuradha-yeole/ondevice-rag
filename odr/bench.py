"""On-device latency benchmark: rerank the top-k candidates for one query on a
single CPU thread, teacher (fp32) vs. compressed student (INT8).

    python -m odr.bench --k 20 --n 200

Appends sizes from compress.py's size report and writes results/latency.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .data import load
from .rerank import TEACHER, OnnxReranker


def time_reranker(rr, pairs, warmup=10):
    for q, docs in pairs[:warmup]:
        rr.score(q, docs)
    lat = []
    for q, docs in pairs:
        t0 = time.perf_counter()
        rr.score(q, docs, bs=len(docs))
        lat.append((time.perf_counter() - t0) * 1e3)
    lat = np.array(lat)
    return {"p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
            "mean_ms": float(lat.mean())}


def main():
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fiqa")
    ap.add_argument("--onnx", default="artifacts/onnx")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)

    ds = load(a.dataset, "test")
    docs = list(ds.corpus.values())
    rng = np.random.default_rng(0)
    qs = list(ds.queries.values())[: a.n]
    pairs = [(q, [docs[i] for i in rng.integers(len(docs), size=a.k)]) for q in qs]

    t_tok = AutoTokenizer.from_pretrained(TEACHER)
    s_tok = AutoTokenizer.from_pretrained(a.onnx)
    out = {"k": a.k, "n_queries": len(pairs), "threads": 1,
           "teacher_fp32": time_reranker(OnnxReranker(os.path.join(a.onnx, "teacher_fp32.onnx"), t_tok), pairs),
           "student_int8": time_reranker(OnnxReranker(os.path.join(a.onnx, "student_int8.onnx"), s_tok), pairs)}
    out["speedup_p50_x"] = out["teacher_fp32"]["p50_ms"] / out["student_int8"]["p50_ms"]
    rep = os.path.join(a.onnx, "size_report.json")
    if os.path.exists(rep):
        out["sizes"] = json.load(open(rep))
    json.dump(out, open(os.path.join(a.out, "latency.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
