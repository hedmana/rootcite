"""Baseline originator scores: four ways to rank a paper's ancestors, none learned.

They exist to be beaten. An originator score that cannot outrank in-degree is
not measuring anything a citation count does not already say, so every scorer
here is a control arm for the learned scorer as much as it is a candidate.

Edges point from citing to cited, so a paper's ancestors are its *descendants*
in the graph's own direction.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import networkx as nx

from graph.build import is_missing, latest_snapshot, load_snapshot
from graph.crawler import DATA_DIR

logger = logging.getLogger(__name__)

Scores = dict[str, float]
Scorer = Callable[[nx.DiGraph, str], Scores]


def ancestry(graph: nx.DiGraph, target: str) -> nx.DiGraph:
    """Everything `target` can reach, target included."""
    if target not in graph:
        raise KeyError(f"{target!r} is not in the graph")
    return graph.subgraph({target} | nx.descendants(graph, target))


def year_gap(graph: nx.DiGraph, source: str, target: str) -> float:
    """Years between a citing work and the work it cites. Negative is impossible."""
    citing = graph.nodes[source].get("publication_year")
    cited = graph.nodes[target].get("publication_year")
    if is_missing(citing) or is_missing(cited):
        return 0.0
    return float(citing) - float(cited)


def acyclic(graph: nx.DiGraph) -> nx.DiGraph:
    """Break cycles, which arrive via preprints revised after being cited.

    The edge dropped from each cycle is the one that most disagrees with
    publication order, so the surviving graph is the chronologically coherent
    reading of the citations.
    """
    if nx.is_directed_acyclic_graph(graph):
        return graph

    graph = nx.DiGraph(graph)
    while True:
        try:
            cycle = nx.find_cycle(graph)
        except nx.NetworkXNoCycle:
            return graph
        graph.remove_edge(*min(cycle, key=lambda edge: year_gap(graph, *edge[:2]))[:2])


def skeleton(lineage: nx.DiGraph, shown: list[str]) -> list[tuple[str, str, bool]]:
    """Which of `shown` leads to which, as (citing, cited, cites it outright).

    A ranking's members rarely cite each other directly: the paths between them
    run through works outside it. So an edge stands for reachability in the
    acyclic `lineage`, and any edge two others already imply is left out. Each
    node's reach is a bitmask over `shown`, settled in one pass from the oldest
    works up rather than by a traversal per shown node.
    """
    bit = {node: 1 << order for order, node in enumerate(shown)}
    below: dict[str, int] = {}
    for node in reversed(list(nx.topological_sort(lineage))):
        mask = 0
        for cited in lineage.successors(node):
            mask |= below[cited] | bit.get(cited, 0)
        below[node] = mask

    reach = nx.DiGraph()
    reach.add_nodes_from(shown)
    reach.add_edges_from((a, b) for a in shown for b in shown if below[a] & bit[b])
    return [(a, b, lineage.has_edge(a, b)) for a, b in nx.transitive_reduction(reach).edges]


def in_degree(graph: nx.DiGraph, target: str) -> Scores:
    """How often the lineage cites a work. The control: local popularity."""
    subgraph = ancestry(graph, target)
    return {node: float(count) for node, count in subgraph.in_degree() if node != target}


def gateway(graph: nx.DiGraph, target: str) -> Scores:
    """How much prior literature is reachable *only* through a work.

    The dominator tree rooted at the target: a work scores the size of the
    subtree it dominates, which is the number of ancestors that drop out of
    reach when it is removed. This is 'structurally load-bearing' read
    literally, and it is the definition the learned scorer has to improve on.
    """
    subgraph = acyclic(ancestry(graph, target))
    dominators = nx.immediate_dominators(subgraph, target)
    tree = nx.DiGraph((parent, child) for child, parent in dominators.items() if child != parent)
    return {node: float(len(nx.descendants(tree, node))) for node in tree if node != target}


def path_weight(graph: nx.DiGraph, target: str, *, decay: float = 0.5) -> Scores:
    """Weight of all citation paths from the target, discounted per hop.

    Raw path counts explode on a dense graph; discounting each hop keeps the
    sum bounded while still rewarding a work the lineage reaches many ways.
    """
    subgraph = acyclic(ancestry(graph, target))
    weights = dict.fromkeys(subgraph, 0.0)
    weights[target] = 1.0
    for node in nx.topological_sort(subgraph):
        for cited in subgraph.successors(node):
            weights[cited] += weights[node] * decay
    return {node: weight for node, weight in weights.items() if node != target}


def _personalized_pagerank(
    graph: nx.DiGraph, target: str, *, alpha: float, iterations: int = 100, tolerance: float = 1e-09
) -> Scores:
    """Power iteration on a walk that always teleports home to `target`.

    Hand rolled because networkx routes `pagerank` through scipy, and an
    ancestry subgraph is small enough that one more dependency buys nothing.
    """
    ranks = dict.fromkeys(graph, 0.0)
    ranks[target] = 1.0
    for _ in range(iterations):
        updated = dict.fromkeys(graph, 0.0)
        stranded = 0.0
        for node, rank in ranks.items():
            citations = graph.out_degree(node)
            if not citations:
                stranded += rank
                continue
            share = alpha * rank / citations
            for cited in graph.successors(node):
                updated[cited] += share
        updated[target] += 1 - alpha + alpha * stranded
        drift = sum(abs(updated[node] - ranks[node]) for node in ranks)
        ranks = updated
        if drift < tolerance:
            break
    return ranks


def time_decayed_pagerank(
    graph: nx.DiGraph, target: str, *, half_life: float = 10.0, alpha: float = 0.85
) -> Scores:
    """Personalised PageRank from the target, halved every `half_life` years back.

    Age alone earns citations, and an undamped walk backwards through a citation
    graph keeps going until it reaches the founding of the discipline. The decay
    is what stops a reading list at the work that made this paper possible
    rather than at Euclid.
    """
    subgraph = ancestry(graph, target)
    ranks = _personalized_pagerank(subgraph, target, alpha=alpha)
    return {
        node: rank * 0.5 ** (year_gap(subgraph, target, node) / half_life)
        for node, rank in ranks.items()
        if node != target
    }


SCORERS: dict[str, Scorer] = {
    "in_degree": in_degree,
    "gateway": gateway,
    "path_weight": path_weight,
    "time_decayed_pagerank": time_decayed_pagerank,
}


def rank(scores: Scores, top: int = 10) -> list[tuple[str, float]]:
    """Highest scoring works first, ties broken by id so runs are reproducible."""
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top]


def overlap(left: Scores, right: Scores, top: int = 10) -> float:
    """Share of one ranking's top-k that also appears in the other's.

    The number the whole PR exists to produce: a scorer that overlaps in_degree
    completely has no reason to exist.
    """
    if not left or not right:
        return 0.0
    chosen = {node for node, _ in rank(left, top)}
    return len(chosen & {node for node, _ in rank(right, top)}) / len(chosen)


def compare(graph: nx.DiGraph, target: str) -> dict[str, Scores]:
    """Every baseline, scored on the same ancestry."""
    return {name: scorer(graph, target) for name, scorer in SCORERS.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", default="gnn", help="field config name under fields/")
    parser.add_argument("--target", required=True, help="OpenAlex id of the paper to trace")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    graph = load_snapshot(latest_snapshot(args.data_dir, args.field))
    scored = compare(graph, args.target)

    for name, scores in scored.items():
        share = 100 * overlap(scores, scored["in_degree"], args.top)
        logger.info("\n%s  (overlap with in_degree: %.0f%%)", name, share)
        for node, score in rank(scores, args.top):
            title = graph.nodes[node].get("title")
            logger.info(
                "  %-14s %8.3f  %s", node, score, "" if is_missing(title) else str(title)[:70]
            )


if __name__ == "__main__":
    main()
