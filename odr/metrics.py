"""IR metrics (graded NDCG with log2 discount, as in BEIR/trec_eval)."""
from __future__ import annotations

import math


def ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], rel: dict[str, int], k: int = 100) -> float:
    if not rel:
        return 0.0
    return len(set(ranked[:k]) & set(rel)) / len(rel)


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0
