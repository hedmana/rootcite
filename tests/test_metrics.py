import math

import pytest
import torch

from gnn.metrics import average_precision, positive_ranks, roc_auc


def test_a_perfect_ranking_scores_one():
    scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])

    assert roc_auc(scores, labels) == 1.0
    assert average_precision(scores, labels) == 1.0


def test_an_inverted_ranking_scores_zero():
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])

    assert roc_auc(scores, labels) == 0.0


def test_a_mixed_ranking_scores_between():
    scores = torch.tensor([3.0, 2.0, 1.0])
    labels = torch.tensor([1.0, 0.0, 1.0])

    assert roc_auc(scores, labels) == 0.5
    assert average_precision(scores, labels) == pytest.approx(5 / 6)


def test_a_tie_counts_as_half_whichever_label_comes_first():
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

    assert roc_auc(torch.ones(4), labels) == 0.5
    assert roc_auc(torch.tensor([2.0, 1.0, 1.0, 0.0]), labels) == 0.875


def test_one_sided_labels_have_no_ranking_to_measure():
    scores = torch.tensor([0.4, 0.6])

    assert math.isnan(roc_auc(scores, torch.ones(2)))
    assert math.isnan(roc_auc(scores, torch.zeros(2)))
    assert math.isnan(average_precision(scores, torch.zeros(2)))


def test_each_positive_is_ranked_against_its_own_negatives_alone():
    positive = torch.tensor([3.0, 1.0])
    negative = torch.tensor([4.0, 2.0, 3.0, 0.0, 1.0])
    owner = torch.tensor([0, 0, 0, 1, 1])

    assert positive_ranks(positive, negative, owner).tolist() == [2.5, 1.5]
