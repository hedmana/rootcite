"""Ranking metrics for link prediction.

Each answers the same question a citation ranking asks: are the real citations
above the imagined ones? None needs a threshold, and none needs a dependency
beyond torch.
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


def positive_ranks(positive: Tensor, negative: Tensor, owner: Tensor) -> Tensor:
    """Where each true citation places among the non-citations drawn for it, 1 being first.

    `owner` names, for every negative, the positive it was drawn against. A tie
    counts as half a loss, so a scorer that cannot tell the two apart lands
    mid-table rather than on top.
    """
    rival = positive[owner]
    lost = (negative > rival).float() + (negative == rival).float() / 2
    return torch.zeros(positive.numel()).scatter_add_(0, owner, lost) + 1
