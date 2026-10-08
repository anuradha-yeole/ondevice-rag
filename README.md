# On-Device RAG

**A distilled retriever and neural reranker small enough to run on a laptop or phone CPU.**

[![tests](https://github.com/anuradha-yeole/ondevice-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/anuradha-yeole/ondevice-rag/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A retrieval-augmented search pipeline evaluated on **BEIR FiQA-2018** — 57k
financial-advice passages and 648 test questions.

> **The question this project answers:** how much ranking quality can you keep
> when you shrink a cross-encoder reranker enough to run on-device?

The approach: build a strong hybrid retrieval stack, rerank with a 12-layer
cross-encoder teacher, then distil that teacher into a **2-layer student** and
compress it (prune → ONNX → INT8) until it fits on a phone. Every stage is
measured against the same test split, so the quality you trade for size is
visible rather than assumed.

```
query ─▶ query understanding ─▶ BM25 + RM3 ──┐
          (normalise, stopwords,              ├─ RRF fusion ─▶ top-50 ─▶ reranker ─▶ top-k ─▶ LLM answer
           pseudo-relevance expansion)        │                          (teacher 12-layer, or
                        dense bi-encoder ─────┘                           distilled 2-layer INT8)
                        (MiniLM + FAISS)
```

---

## Contents

- [Components](#components)
- [How it is evaluated](#how-it-is-evaluated)
- [Results](#results)
- [Run it](#run-it)
- [Design notes](#design-notes)
- [Repo layout](#repo-layout)

---

## Components

| Stage | Implementation |
|---|---|
| Query understanding | normalisation and stopword removal, then **RM3** pseudo-relevance-feedback expansion — top 10 feedback docs, 10 expansion terms, original query weighted 0.7 ([`odr/lexical.py`](odr/lexical.py)) |
| Lexical retrieval | BM25 (`k1=0.9`, `b=0.4`) over a scipy sparse matrix, fully vectorised |
| Dense retrieval | `all-MiniLM-L6-v2` bi-encoder, mean pooling, FAISS inner-product index |
| Fusion | reciprocal rank fusion (`k=60`) over BM25+RM3 and dense |
| Teacher reranker | `cross-encoder/ms-marco-MiniLM-L-12-v2` (12 layers, 33M params) |
| Student reranker | `ms-marco-TinyBERT-L-2-v2` (2 layers, 4M params), **distilled on FiQA** from the teacher with listwise KL plus logit MSE ([`odr/distill.py`](odr/distill.py)) |
| Compression | global magnitude **pruning** (30% of Linear weights), ONNX export, **INT8** dynamic quantization, optional **Core ML** export ([`odr/compress.py`](odr/compress.py)) |
| Generation | top-3 passages in a cited prompt to a small local LLM — `Qwen2.5-0.5B-Instruct` by default ([`odr/rag.py`](odr/rag.py)) |

Model choices are overridable by environment variable (`ODR_TEACHER`,
`ODR_STUDENT`, `ODR_ENCODER`) without touching code.

## How it is evaluated

[`odr/evaluate.py`](odr/evaluate.py) scores seven pipeline variants on the FiQA
test split, so each addition can be attributed:

`bm25` → `bm25_rm3` → `dense` → `hybrid` → `hybrid+teacher` →
`hybrid+student_fp32` → `hybrid+student_int8`

- **Metrics.** NDCG@10 and Recall@100, using graded NDCG with a log2 discount
  as in BEIR/`trec_eval` ([`odr/metrics.py`](odr/metrics.py)).
- **Significance.** Every variant's NDCG@10 gain over the BM25 baseline gets a
  **query-level bootstrap 95% CI** (2,000 resamples), because a mean gain over
  648 queries can easily be noise.
- **Headline number.** `student_int8_quality_retained_pct` — the compressed
  student's NDCG@10 as a percentage of the teacher's. That single figure is the
  answer to the question at the top of this README.
- **Latency.** [`odr/bench.py`](odr/bench.py) times reranking the top-20
  candidates for one query on **one CPU thread** (`torch.set_num_threads(1)`),
  which approximates a phone or laptop core. It reports p50/p95/mean for teacher
  fp32 vs. student INT8 and the resulting speedup, with 10 warmup queries
  discarded.
- **Size.** [`odr/compress.py`](odr/compress.py) reports parameter counts, raw
  and gzipped on-disk size for teacher fp32 / student fp32 / student INT8, and
  the end-to-end `size_reduction_x`.

## Results

**Not yet committed.** This repo contains the full pipeline and evaluation
harness, but the measured numbers are produced by running it — nothing is
checked in, and no figures are quoted here that haven't been generated.

Running the commands below writes:

| File | Contents |
|---|---|
| `results/RESULTS.md` | NDCG@10, % vs. BM25 and Recall@100 for all seven variants, plus the teacher-retention figure |
| `results/results.json` | the same, with bootstrap 95% CIs per variant |
| `results/latency.json` | single-thread p50/p95/mean per reranker, speedup, and the size report |
| `artifacts/onnx/size_report.json` | parameter counts, sparsity, raw and gzipped sizes |

`results/` is deliberately **not** in `.gitignore`, so once generated these can
be committed alongside this README. `artifacts/` and `data/` are ignored — the
models, ONNX exports and document embeddings are regenerable and too large to
version.

## Run it

Requires Python 3.10+. The first run downloads FiQA (~17 MB) and the models from
the Hugging Face Hub. A GPU is used automatically if present.

```bash
pip install -r requirements.txt
```

```bash
python -m odr.distill      # distil teacher -> student on the FiQA train split
python -m odr.compress     # prune, export to ONNX, quantize to INT8
python -m odr.evaluate     # all variants on the test split -> results/
python -m odr.bench        # single-thread latency, teacher vs. compressed student
```

Ask a question end to end:

```bash
python -m odr.rag "Should I pay off my credit card before investing?"
```

Run the offline test suite — tiny randomly-initialised BERTs, no downloads:

```bash
pytest -q
```

On a laptop CPU, distillation takes about 30–60 min and evaluation about
20–40 min, mostly spent scoring candidates with the teacher. Export to Core ML
with `python -m odr.compress --coreml` (macOS, needs `coremltools`).

### Key flags

| Module | Purpose | Flags |
|---|---|---|
| `odr.distill` | distil teacher into student | `--dataset`, `--out`, `--n-queries` (4000), `--n-cand` (30), `--epochs` (2), `--seed` |
| `odr.compress` | prune, ONNX export, INT8 quantize | `--student`, `--out`, `--prune` (0.3), `--coreml` |
| `odr.evaluate` | score all pipeline variants | `--dataset`, `--rerank-depth` (50), `--onnx`, `--out` |
| `odr.bench` | single-thread latency benchmark | `--dataset`, `--onnx`, `--k` (20), `--n` (200), `--out` |
| `odr.rag` | answer one question with citations | `--dataset`, `--onnx`, `--top` (3), `--llm` |

`odr.evaluate` degrades gracefully: if `student_int8.onnx` is absent it scores
the first-stage and teacher variants only, so you can evaluate retrieval before
committing to a distillation run.

## Design notes

**Why distil on FiQA instead of using TinyBERT as is?** The off-the-shelf
student was distilled on MS MARCO web search. Financial-advice questions are
longer and use different vocabulary, so distilling again on in-domain teacher
scores is what lets the student close the gap.

**Why listwise KL?** At inference time only the *order* of the candidates
matters, not their absolute scores. Matching the teacher's softmax over each
candidate list (temperature 2.0) transfers that order directly. A small MSE term
(α=0.1) keeps the logits on a stable scale for thresholding.

**What pruning actually buys.** Unstructured sparsity does not shrink a dense
ONNX file. It shows up in the compressed download size (`gzip_mb` in
`size_report.json`) and lets sparse kernels skip work. INT8 quantization does
most of the size and latency reduction — pruning is the cheaper second win.

**Why rerank only the top 50.** Reranking is the expensive stage; first-stage
recall caps what it can recover. Reporting Recall@100 alongside NDCG@10 makes
that ceiling explicit, so the rerank depth is a visible knob rather than a
hidden constant.

## Repo layout

```
odr/
├── data.py       BEIR download + load (corpus, queries, qrels)
├── lexical.py    tokeniser, BM25 over sparse matrices, RM3 query expansion
├── dense.py      MiniLM bi-encoder, FAISS index, reciprocal rank fusion
├── pipeline.py   first-stage retrieval: bm25 / bm25_rm3 / dense / hybrid
├── rerank.py     Torch and ONNX cross-encoder rerankers
├── distill.py    listwise KL + MSE distillation, teacher -> student
├── compress.py   magnitude pruning, ONNX export, INT8 quantize, Core ML
├── metrics.py    graded NDCG@k, Recall@k
├── evaluate.py   score all variants, bootstrap CIs, write results/
├── bench.py      single-thread on-device latency benchmark
└── rag.py        retrieve -> rerank -> cited answer from a local LLM
tests/
└── test_pipeline.py   offline, tiny random BERTs, no network
```

## Limitations

- **One dataset.** FiQA is a single English, financial-advice domain. The
  distillation gain is in-domain by construction and would need re-running per
  domain.
- **Latency is measured on random candidates.** The benchmark pairs each query
  with `k` randomly drawn passages, which is right for timing but means the
  inputs are not the real reranking distribution.
- **The generation stage is unevaluated.** `odr.rag` produces cited answers, but
  there is no answer-quality metric here — the measured claims are all about
  retrieval and ranking.

## License

MIT — see [LICENSE](LICENSE).
