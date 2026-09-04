"""Ranking metrics for link prediction.

Both answer the same question a citation ranking asks: are the real citations
above the imagined ones? Neither needs a threshold, and neither needs a
dependency beyond torch.
"""

from __future__ import annotations

import torch
from torch import Tensor


def roc_auc(scores: Tensor, labels: Tensor) -> float:
    """The chance a random true citation outranks a random sampled non-citation."""
    positives = labels == 1
    hits, misses = int(positives.sum()), int((~positives).sum())
    if not hits or not misses:
        return float("nan")

    # Ranks by position, so equal scores are ordered arbitrarily rather than tied.
    ranks = torch.empty_like(scores, dtype=torch.float)
    ranks[scores.argsort()] = torch.arange(1, scores.numel() + 1, dtype=torch.float)
    return float((ranks[positives].sum() - hits * (hits + 1) / 2) / (hits * misses))


def average_precision(scores: Tensor, labels: Tensor) -> float:
    """Mean precision at every rank a true citation occupies."""
    positives = labels[scores.argsort(descending=True)] == 1
    if not int(positives.sum()):
        return float("nan")

    found = positives.cumsum(0)
    precision = found / torch.arange(1, positives.numel() + 1)
    return float(precision[positives].mean())
