"""Offline tests: tiny randomly initialised BERTs, so no downloads are needed."""
import math

import numpy as np
import pytest
import torch
from transformers import (BertConfig, BertForSequenceClassification, BertModel,
                          BertTokenizerFast)

from odr.compress import export_onnx, prune_linear, quantize_int8
from odr.dense import Encoder, rrf
from odr.distill import distill
from odr.lexical import BM25, QueryProcessor, tokenize
from odr.metrics import ndcg_at_k, recall_at_k
from odr.pipeline import Retriever
from odr.rerank import OnnxReranker, TorchReranker

CORPUS = {
    "d0": "How to pay off credit card debt fast with the avalanche method",
    "d1": "Index funds versus actively managed mutual funds for retirement",
    "d2": "Credit card interest rates and how APR compounds monthly",
    "d3": "Filing taxes as a freelancer: quarterly estimated payments",
    "d4": "Should I invest or pay down my student loan debt first",
    "d5": "Roth IRA contribution limits and income phase-outs",
}


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    words = sorted({w for t in CORPUS.values() for w in t.lower().replace(":", "").split()}
                   | {"credit", "card", "debt", "invest", "example", "query", "passage", "text", "q", "p"})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
    tok = BertTokenizerFast(vocab={w: i for i, w in enumerate(vocab)})
    assert "[UNK]" not in tok.tokenize("credit card debt")
    return tok


def tiny_cfg(tok, layers=2, hidden=32):
    return BertConfig(vocab_size=tok.vocab_size, hidden_size=hidden, num_hidden_layers=layers,
                      num_attention_heads=2, intermediate_size=2 * hidden, num_labels=1,
                      max_position_embeddings=128, initializer_range=0.2)


def test_tokenize_drops_stopwords():
    assert tokenize("How do I pay off my credit card's debt?") == ["pay", "off", "credit", "card", "debt"]


def test_bm25_ranks_exact_match_first():
    bm = BM25(list(CORPUS.values()))
    qp = QueryProcessor(bm, fb_docs=2, fb_terms=3)
    top, _ = bm.search(qp.parse("Roth IRA contribution limits"), 3)
    assert top[0] == 5
    expanded = qp.expand("credit card")
    assert {"credit", "card"} <= set(expanded) and len(expanded) > 2   # RM3 added feedback terms
    assert math.isclose(sum(expanded.values()), 1.0, rel_tol=1e-5)


def test_metrics():
    rel = {"a": 1, "b": 1}
    assert ndcg_at_k(["a", "b", "c"], rel) == pytest.approx(1.0)
    assert ndcg_at_k(["c", "a"], rel) < 1.0
    assert recall_at_k(["c", "a"], rel, 2) == 0.5
    assert rrf([[1, 2, 3], [3, 1]])[:2] == [1, 3]


def test_dense_retrieves_identical_text_first(tok):
    torch.manual_seed(0)
    enc = Encoder(BertModel(tiny_cfg(tok)), tok, device="cpu")
    docs = list(CORPUS.values())
    from odr.dense import DenseIndex
    idx, _ = DenseIndex(enc.encode(docs)).search(enc.encode(docs), 1)
    assert idx[:, 0].tolist() == list(range(len(docs)))


def test_retriever_hybrid(tok):
    enc = Encoder(BertModel(tiny_cfg(tok)), tok, device="cpu")
    r = Retriever(CORPUS, enc)
    runs = r.run({"q1": "credit card debt"}, k=4)
    assert set(runs) == {"bm25", "bm25_rm3", "dense", "hybrid"}
    assert len(runs["hybrid"]["q1"]) == 4
    assert r.ids(runs["bm25"]["q1"])[0] in {"d0", "d2"}


def test_distillation_moves_student_toward_teacher(tok):
    torch.manual_seed(0)
    teacher = BertForSequenceClassification(tiny_cfg(tok, layers=4, hidden=64)).eval()
    student = BertForSequenceClassification(tiny_cfg(tok, layers=1, hidden=32))
    lists = [("credit card debt", list(CORPUS)), ("invest retirement", list(CORPUS))]
    hist = distill(TorchReranker(teacher, tok), student, tok, lists, CORPUS,
                   epochs=30, lr=3e-3, accum=1, log=lambda *_: None)
    assert hist[-1] < 0.5 * hist[0]


def test_compress_and_onnx_parity(tok, tmp_path):
    torch.manual_seed(0)
    m = BertForSequenceClassification(tiny_cfg(tok)).eval()
    assert prune_linear(m, 0.3) == pytest.approx(0.3, abs=0.01)
    fp32, int8 = str(tmp_path / "m.onnx"), str(tmp_path / "m8.onnx")
    export_onnx(m, tok, fp32)
    quantize_int8(fp32, int8)
    docs = list(CORPUS.values())
    ref = TorchReranker(m, tok).score("credit card debt", docs)
    np.testing.assert_allclose(OnnxReranker(fp32, tok).score("credit card debt", docs), ref, atol=1e-4)
    q8 = OnnxReranker(int8, tok).score("credit card debt", docs)
    assert np.corrcoef(q8, ref)[0, 1] > 0.9          # INT8 keeps the ranking signal
