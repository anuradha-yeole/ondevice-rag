"""Compress the distilled reranker for on-device inference.

    python -m odr.compress --student artifacts/student --out artifacts/onnx

1. Global magnitude pruning of the encoder's Linear weights (default 30%).
2. Export to ONNX (fp32), then INT8 dynamic quantization with ONNX Runtime.
3. Optional Core ML export (``--coreml``; needs ``coremltools``).
Writes a size report comparing teacher, student fp32 and student INT8.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os

import torch
from torch.nn.utils import prune

from .rerank import TEACHER


def prune_linear(model: torch.nn.Module, amount: float) -> float:
    """Global L1 unstructured pruning over encoder Linear layers; returns sparsity."""
    params = [(m, "weight") for n, m in model.named_modules()
              if isinstance(m, torch.nn.Linear) and "classifier" not in n]
    if amount > 0:
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
        for m, name in params:
            prune.remove(m, name)
    zeros = sum(int((m.weight == 0).sum()) for m, _ in params)
    total = sum(m.weight.numel() for m, _ in params)
    return zeros / max(total, 1)


class _Logits(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.m(input_ids=input_ids, attention_mask=attention_mask,
                      token_type_ids=token_type_ids).logits


def export_onnx(model, tok, path: str, max_len: int = 256) -> None:
    enc = tok(["example query"], ["example passage text"], padding="max_length",
              max_length=32, truncation=True, return_tensors="pt")
    if "token_type_ids" not in enc:
        enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])
    axes = {0: "batch", 1: "seq"}
    torch.onnx.export(_Logits(model.eval()),
                      (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]), path,
                      input_names=["input_ids", "attention_mask", "token_type_ids"],
                      output_names=["logits"], opset_version=17, dynamo=False,
                      dynamic_axes={"input_ids": axes, "attention_mask": axes,
                                    "token_type_ids": axes, "logits": {0: "batch"}})


def quantize_int8(src: str, dst: str) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8)


def size_mb(path: str) -> dict:
    raw = open(path, "rb").read()
    return {"mb": len(raw) / 2 ** 20, "gzip_mb": len(gzip.compress(raw, 6)) / 2 ** 20}


def fp32_param_mb(model) -> float:
    return sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20


def export_coreml(model, tok, path: str) -> None:
    import coremltools as ct
    enc = tok(["q"], ["p"], padding="max_length", max_length=128, return_tensors="pt")
    traced = torch.jit.trace(_Logits(model.eval()),
                             (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]))
    shape = ct.Shape(shape=(1, ct.RangeDim(8, 256)))
    ml = ct.convert(traced, inputs=[ct.TensorType(name=n, shape=shape, dtype=int)
                                    for n in ["input_ids", "attention_mask", "token_type_ids"]],
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS16)
    ml.save(path)


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="artifacts/student")
    ap.add_argument("--out", default="artifacts/onnx")
    ap.add_argument("--prune", type=float, default=0.3)
    ap.add_argument("--coreml", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.student)
    student = AutoModelForSequenceClassification.from_pretrained(a.student).eval()
    sparsity = prune_linear(student, a.prune)
    fp32 = os.path.join(a.out, "student_fp32.onnx")
    int8 = os.path.join(a.out, "student_int8.onnx")
    export_onnx(student, tok, fp32)
    quantize_int8(fp32, int8)
    tok.save_pretrained(a.out)

    teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER)
    t_tok = AutoTokenizer.from_pretrained(TEACHER)
    t_path = os.path.join(a.out, "teacher_fp32.onnx")
    export_onnx(teacher, t_tok, t_path)
    report = {
        "prune_amount": a.prune, "linear_sparsity": sparsity,
        "teacher_params_m": sum(p.numel() for p in teacher.parameters()) / 1e6,
        "student_params_m": sum(p.numel() for p in student.parameters()) / 1e6,
        "teacher_fp32": size_mb(t_path), "student_fp32": size_mb(fp32), "student_int8": size_mb(int8),
    }
    report["size_reduction_x"] = report["teacher_fp32"]["mb"] / report["student_int8"]["mb"]
    if a.coreml:
        export_coreml(student, tok, os.path.join(a.out, "student.mlpackage"))
    json.dump(report, open(os.path.join(a.out, "size_report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
