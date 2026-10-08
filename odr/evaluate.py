"""Full evaluation on the BEIR test split; writes results/results.json and RESULTS.md.

    python -m odr.evaluate --rerank-depth 50

Variants: BM25, BM25+RM3 query expansion, dense (MiniLM bi-encoder), hybrid
(RRF), and hybrid reranked by the teacher, the distilled student (fp32 ONNX)
and the compressed student (pruned + INT8 ONNX).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .data import load
from .dense import Encoder
from .metrics import mean, ndcg_at_k, recall_at_k
from .pipeline import Retriever
from .rerank import TEACHER, OnnxReranker, TorchReranker, load_cross_encoder


def rerank_run(reranker, queries, first_stage, retriever, depth):
    out = {}
    for q, ranks in first_stage.items():
        cand = ranks[:depth]
        s = reranker.score(queries[q], [retriever.docs[i] for i in cand])
        order = np.argsort(-s, kind="stable")
        out[q] = [cand[i] for i in order] + ranks[depth:]
    return out


def score_run(run, retriever, qrels):
    ids = {q: retriever.ids(r) for q, r in run.items()}
    return {"ndcg@10": mean(ndcg_at_k(ids[q], qrels[q], 10) for q in ids),
            "recall@100": mean(recall_at_k(ids[q], qrels[q], 100) for q in ids)}


def bootstrap_ci(a: list[float], b: list[float], n=2000, seed=0):
    """95% CI of mean(b - a) over queries."""
    d = np.array(b) - np.array(a)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(len(d), size=(n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fiqa")
    ap.add_argument("--rerank-depth", type=int, default=50)
    ap.add_argument("--onnx", default="artifacts/onnx")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ds = load(a.dataset, "test")
    retriever = Retriever(ds.corpus, Encoder(), cache=f"artifacts/{a.dataset}_doc_emb.npy")
    runs = retriever.run(ds.queries, k=100)

    t_tok, t_model = load_cross_encoder(TEACHER)
    rerankers = {"hybrid+teacher": TorchReranker(t_model, t_tok)}
    if os.path.exists(os.path.join(a.onnx, "student_int8.onnx")):
        from transformers import AutoTokenizer
        s_tok = AutoTokenizer.from_pretrained(a.onnx)
        rerankers["hybrid+student_fp32"] = OnnxReranker(os.path.join(a.onnx, "student_fp32.onnx"), s_tok, threads=os.cpu_count())
        rerankers["hybrid+student_int8"] = OnnxReranker(os.path.join(a.onnx, "student_int8.onnx"), s_tok, threads=os.cpu_count())
    for name, rr in rerankers.items():
        t0 = time.time()
        runs[name] = rerank_run(rr, ds.queries, runs["hybrid"], retriever, a.rerank_depth)
        print(f"{name}: reranked {len(ds.queries)} queries in {time.time() - t0:.0f}s")

    results = {name: score_run(run, retriever, ds.qrels) for name, run in runs.items()}
    per_q = {name: [ndcg_at_k(retriever.ids(run[q]), ds.qrels[q], 10) for q in ds.queries]
             for name, run in runs.items()}
    base = results["bm25"]["ndcg@10"]
    for name in results:
        results[name]["ndcg@10_vs_bm25_pct"] = (100 * (results[name]["ndcg@10"] / base - 1)
                                                if base > 0 else float("nan"))
        results[name]["ndcg@10_diff_ci95"] = bootstrap_ci(per_q["bm25"], per_q[name])
    if "hybrid+student_int8" in results and results["hybrid+teacher"]["ndcg@10"] > 0:
        results["student_int8_quality_retained_pct"] = (
            100 * results["hybrid+student_int8"]["ndcg@10"] / results["hybrid+teacher"]["ndcg@10"])
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=2)

    lines = [f"# Results: BEIR {a.dataset} test ({len(ds.queries)} queries, "
             f"{len(ds.corpus):,} passages, rerank depth {a.rerank_depth})", "",
             "| Pipeline | NDCG@10 | vs. BM25 | Recall@100 |", "|---|---|---|---|"]
    for name, r in results.items():
        if isinstance(r, dict):
            lines.append(f"| {name} | {r['ndcg@10']:.4f} | {r['ndcg@10_vs_bm25_pct']:+.1f}% | {r['recall@100']:.4f} |")
    if "student_int8_quality_retained_pct" in results:
        lines += ["", f"Compressed student keeps **{results['student_int8_quality_retained_pct']:.1f}%** "
                      "of the teacher's NDCG@10."]
    open(os.path.join(a.out, "RESULTS.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
