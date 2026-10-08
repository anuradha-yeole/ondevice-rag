"""Answer a question with retrieved, reranked passages (the "G" in RAG).

    python -m odr.rag "Is it better to pay off my credit card or invest?"

Retrieval and reranking run locally; generation uses any small instruction-tuned
model from the Hugging Face Hub (default Qwen2.5-0.5B-Instruct, small enough to
run on a laptop CPU).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .data import load
from .dense import Encoder
from .pipeline import Retriever
from .rerank import OnnxReranker

PROMPT = """Answer the question using only the passages below. Cite passages as [1], [2], ...
If the passages do not contain the answer, say so.

{context}

Question: {question}
Answer:"""


def build_prompt(question: str, passages: list[str], max_chars: int = 800) -> str:
    ctx = "\n\n".join(f"[{i + 1}] {p[:max_chars]}" for i, p in enumerate(passages))
    return PROMPT.format(context=ctx, question=question)


def main():
    from transformers import AutoTokenizer, pipeline
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--dataset", default="fiqa")
    ap.add_argument("--onnx", default="artifacts/onnx")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--llm", default="Qwen/Qwen2.5-0.5B-Instruct")
    a = ap.parse_args()

    ds = load(a.dataset, "test")
    r = Retriever(ds.corpus, Encoder(), cache=f"artifacts/{a.dataset}_doc_emb.npy")
    cand = r.run({"q": a.question}, k=50)["hybrid"]["q"]
    rr = OnnxReranker(os.path.join(a.onnx, "student_int8.onnx"), AutoTokenizer.from_pretrained(a.onnx))
    s = rr.score(a.question, [r.docs[i] for i in cand])
    passages = [r.docs[cand[i]] for i in np.argsort(-s)[: a.top]]
    gen = pipeline("text-generation", model=a.llm)
    msgs = [{"role": "user", "content": build_prompt(a.question, passages)}]
    print(gen(msgs, max_new_tokens=256, do_sample=False)[0]["generated_text"][-1]["content"])


if __name__ == "__main__":
    main()
