import networkx as nx
import pytest

from gnn.content import Content


def works(**texts):
    graph = nx.DiGraph()
    for node, (title, abstract) in texts.items():
        graph.add_node(node, title=title, abstract=abstract)
    return graph


def test_the_same_words_are_the_same_subject_and_none_shared_is_none():
    content = Content(
        works(
            A=("Graph attention networks", "Attention over neighbours."),
            B=("Graph attention networks", "Attention over neighbours."),
            C=("Stochastic optimisation", "Adaptive moment estimates."),
        )
    )

    assert content.similarity("A", "B") == pytest.approx(1.0)
    assert content.similarity("A", "C") == 0.0


def test_case_and_common_english_carry_no_subject():
    content = Content(
        works(
            A=("THE Attention", "With the attention."),
            B=("attention for the", None),
            C=("Kernels", "Unrelated."),
        )
    )

    assert content.similarity("A", "B") == pytest.approx(1.0)


def test_a_work_with_nothing_on_record_shares_nothing():
    content = Content(works(A=("Attention", "Attention."), B=(None, float("nan"))))

    assert content.similarity("A", "B") == 0.0
    assert content.similarity("A", "NOT-IN-GRAPH") == 0.0
