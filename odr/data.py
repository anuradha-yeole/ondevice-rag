"""BEIR dataset loader (default: FiQA-2018 financial question answering)."""
from __future__ import annotations

import csv
import json
import os
import urllib.request
import zipfile
from dataclasses import dataclass

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


@dataclass
class Dataset:
    corpus: dict[str, str]                 # doc_id -> "title. text"
    queries: dict[str, str]                # query_id -> text
    qrels: dict[str, dict[str, int]]       # query_id -> {doc_id: relevance}


def download(name: str = "fiqa", root: str = "data") -> str:
    path = os.path.join(root, name)
    if os.path.isdir(path):
        return path
    os.makedirs(root, exist_ok=True)
    zpath = os.path.join(root, f"{name}.zip")
    print(f"downloading {name} ...")
    urllib.request.urlretrieve(BEIR_URL.format(name=name), zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(root)
    os.remove(zpath)
    return path


def load(name: str = "fiqa", split: str = "test", root: str = "data") -> Dataset:
    path = download(name, root)
    corpus = {}
    with open(os.path.join(path, "corpus.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            title = d.get("title") or ""
            corpus[d["_id"]] = f"{title}. {d['text']}" if title else d["text"]
    qrels: dict[str, dict[str, int]] = {}
    with open(os.path.join(path, "qrels", f"{split}.tsv")) as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for qid, did, score in r:
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    queries = {}
    with open(os.path.join(path, "queries.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if d["_id"] in qrels:
                queries[d["_id"]] = d["text"]
    return Dataset(corpus, queries, qrels)
