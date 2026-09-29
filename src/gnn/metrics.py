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

    # Equal scores share the mean of the ranks they span, so a tie counts as half
    # a win. Heuristic scores tie constantly, and ordering them by position would
    # let the order the labels were concatenated in decide the metric.
    _, group, sizes = torch.unique(scores, return_inverse=True, return_counts=True)
    ranks = (sizes.cumsum(0) - (sizes - 1) / 2)[group]
    return float((ranks[positives].sum() - hits * (hits + 1) / 2) / (hits * misses))


def average_precision(scores: Tensor, labels: Tensor) -> float:
    """Mean precision at every rank a true citation occupies."""
    positives = labels[scores.argsort(descending=True)] == 1
    if not int(positives.sum()):
        return float("nan")

    found = positives.cumsum(0)
    precision = found / torch.arange(1, positives.numel() + 1)
    return float(precision[positives].mean())
