"""Cross-encoder rerankers: PyTorch and ONNX Runtime backends behind one interface."""
from __future__ import annotations

import os

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# override with env vars to try other models
TEACHER = os.environ.get("ODR_TEACHER", "cross-encoder/ms-marco-MiniLM-L-12-v2")         # 12 layers
STUDENT_INIT = os.environ.get("ODR_STUDENT", "cross-encoder/ms-marco-TinyBERT-L-2-v2")  # 2 layers


def load_cross_encoder(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).eval()
    return tok, model


class TorchReranker:
    def __init__(self, model, tok, max_len: int = 256):
        self.model, self.tok, self.max_len = model.eval(), tok, max_len

    def _enc(self, query: str, docs: list[str]):
        return self.tok([query] * len(docs), docs, padding=True, truncation="only_second",
                        max_length=self.max_len, return_tensors="pt")

    @torch.inference_mode()
    def score(self, query: str, docs: list[str], bs: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(docs), bs):
            out.append(self.model(**self._enc(query, docs[i:i + bs])).logits[:, 0].float().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)


class OnnxReranker:
    def __init__(self, path: str, tok, max_len: int = 256, threads: int = 1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.tok, self.max_len = tok, max_len

    def score(self, query: str, docs: list[str], bs: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(docs), bs):
            enc = self.tok([query] * len(docs[i:i + bs]), docs[i:i + bs], padding=True,
                           truncation="only_second", max_length=self.max_len, return_tensors="np")
            feed = {k: v.astype(np.int64) for k, v in enc.items() if k in self.inputs}
            out.append(self.sess.run(None, feed)[0][:, 0])
        return np.concatenate(out) if out else np.zeros(0, np.float32)
