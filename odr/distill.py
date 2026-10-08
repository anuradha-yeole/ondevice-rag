"""Knowledge distillation: 12-layer cross-encoder teacher -> 2-layer student.

    python -m odr.distill --out artifacts/student

For each FiQA *train* query we retrieve hybrid candidates, score them with the
teacher, and train the student to match the teacher's ranking distribution
over each candidate list (listwise KL) plus its raw logits (MSE).
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from .data import load
from .dense import Encoder
from .pipeline import Retriever
from .rerank import STUDENT_INIT, TEACHER, TorchReranker, load_cross_encoder


def build_lists(retriever: Retriever, ds, n_queries: int, n_cand: int, seed: int):
    rng = random.Random(seed)
    qids = list(ds.queries)
    rng.shuffle(qids)
    qids = qids[:n_queries]
    runs = retriever.run({q: ds.queries[q] for q in qids}, k=n_cand)["hybrid"]
    known = set(retriever.doc_ids)
    lists = []
    for q in qids:
        docs = retriever.ids(runs[q])
        # make sure labelled positives are in the list so the student sees them;
        # they replace the lowest-ranked candidates
        missing = [p for p in ds.qrels.get(q, {}) if p in known and p not in docs]
        for j, pos in enumerate(missing[: len(docs) // 2]):
            docs[-1 - j] = pos
        lists.append((ds.queries[q], docs))           # (query text, [doc_id])
    return lists


def distill(teacher: TorchReranker, student, tok, lists, corpus, epochs=2, lr=5e-5, tau=2.0,
            alpha=0.1, accum=8, max_len=256, seed=0, log=print):
    torch.manual_seed(seed)
    # teacher.score runs under inference_mode and returns numpy, so these are plain tensors
    t_scores = [torch.tensor(teacher.score(q, [corpus[d] for d in docs])) for q, docs in lists]
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(lists) // accum
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.1)
    order = list(range(len(lists)))
    hist = []
    student.train()
    step = 0
    for ep in range(epochs):
        random.Random(seed + ep).shuffle(order)
        tot = 0.0
        for i, li in enumerate(order):
            q, docs = lists[li]
            enc = tok([q] * len(docs), [corpus[d] for d in docs], padding=True,
                      truncation="only_second", max_length=max_len, return_tensors="pt")
            s = student(**enc).logits[:, 0]
            t = t_scores[li]
            kl = F.kl_div(F.log_softmax(s / tau, -1), F.softmax(t / tau, -1), reduction="sum") * tau ** 2
            loss = (kl + alpha * F.mse_loss(s, t)) / accum
            loss.backward()
            tot += loss.item() * accum
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                if step < steps - 1:
                    sched.step()
                step += 1
        hist.append(tot / len(order))
        log(f"  epoch {ep + 1}: distill loss {hist[-1]:.4f}")
    student.eval()
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fiqa")
    ap.add_argument("--out", default="artifacts/student")
    ap.add_argument("--n-queries", type=int, default=4000)
    ap.add_argument("--n-cand", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ds = load(a.dataset, "train")
    retriever = Retriever(ds.corpus, Encoder(), cache=f"artifacts/{a.dataset}_doc_emb.npy")
    lists = build_lists(retriever, ds, a.n_queries, a.n_cand, a.seed)
    print(f"{len(lists)} training lists x {a.n_cand} candidates")
    t_tok, t_model = load_cross_encoder(TEACHER)
    s_tok, s_model = load_cross_encoder(STUDENT_INIT)
    hist = distill(TorchReranker(t_model, t_tok), s_model, s_tok, lists, ds.corpus,
                   epochs=a.epochs, seed=a.seed)
    os.makedirs(a.out, exist_ok=True)
    s_model.save_pretrained(a.out)
    s_tok.save_pretrained(a.out)
    json.dump({"loss": hist, **vars(a)}, open(os.path.join(a.out, "distill_log.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
