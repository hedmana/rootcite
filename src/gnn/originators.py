"""Rank a paper's ancestors by the model rather than by how often they were cited.

The baselines in `graph.score` weight every citation the same, so a lineage
that runs through a much-cited survey outranks one that runs through the paper
the survey summarised. This scorer replaces that flat weighting with the trained
decoder: at each step backwards the walk prefers the citations the model finds
most plausible, which is the one thing structure alone cannot tell it.

The walk is then decayed by age. Backwards through a citation graph there is
always another ancestor, and without decay the ranking ends at the founding of
the discipline rather than at the work that made this paper possible.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import networkx as nx
import torch
from torch import Tensor
from torch_geometric.utils import to_undirected

from gnn.dataset import observed_view
from gnn.model import LinkPredictor
from gnn.train import load_model
from graph.build import is_missing, latest_snapshot, load_snapshot
from graph.crawler import DATA_DIR
from graph.score import SCORERS, Scores, acyclic, ancestry, overlap, rank, year_gap

logger = logging.getLogger(__name__)


def embed(graph: nx.DiGraph, model: LinkPredictor) -> tuple[dict[str, int], Tensor]:
    """One encode pass over everything on record. Where each node sits, and as what."""
    node_ids, view = observed_view(graph)
    model.eval()
    with torch.no_grad():
        z = model.encode(view.x, to_undirected(view.edge_index, num_nodes=len(node_ids)))
    return {node: order for order, node in enumerate(node_ids)}, z


def citation_probability(
    model: LinkPredictor, z: Tensor, position: dict[str, int], edges: list[tuple[str, str]]
) -> Tensor:
    """How plausible the model finds each citation, scored in one batch."""
    if not edges:
        return torch.empty(0)
    index = torch.tensor(
        [[position[source] for source, _ in edges], [position[target] for _, target in edges]],
        dtype=torch.long,
    )
    with torch.no_grad():
        return torch.sigmoid(model.decode(z, index))


def learned_flow(
    graph: nx.DiGraph,
    target: str,
    model: LinkPredictor,
    *,
    damping: float = 0.85,
    half_life: float = 10.0,
    embedding: tuple[dict[str, int], Tensor] | None = None,
) -> Scores:
    """Weight of a damped walk backwards from `target`, steered by the model.

    Exact rather than iterated: the lineage is acyclic, so one pass in
    topological order settles every node's share.
    """
    lineage = acyclic(ancestry(graph, target))
    position, z = embedding or embed(graph, model)

    edges = list(lineage.edges())
    plausibility = dict(
        zip(edges, citation_probability(model, z, position, edges).tolist(), strict=True)
    )

    flow = dict.fromkeys(lineage, 0.0)
    flow[target] = 1.0
    for node in nx.topological_sort(lineage):
        cited = list(lineage.successors(node))
        shares = [plausibility[(node, work)] for work in cited]
        total = sum(shares)
        if not total:
            continue
        for work, share in zip(cited, shares, strict=True):
            flow[work] += flow[node] * damping * share / total

    return {
        node: weight * 0.5 ** (year_gap(lineage, target, node) / half_life)
        for node, weight in flow.items()
        if node != target
    }


def score_target(
    field_name: str,
    target: str,
    *,
    data_dir: Path = DATA_DIR,
    half_life: float = 10.0,
) -> tuple[Scores, dict[str, Scores]]:
    """The learned ranking for one paper, alongside every baseline it has to beat."""
    snapshot = latest_snapshot(data_dir, field_name)
    graph = load_snapshot(snapshot)
    model = load_model(snapshot)
    return (
        learned_flow(graph, target, model, half_life=half_life),
        {name: scorer(graph, target) for name, scorer in SCORERS.items()},
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", default="gnn", help="field config name under fields/")
    parser.add_argument("--target", required=True, help="OpenAlex id of the paper to trace")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--half-life", type=float, default=10.0)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    learned, baselines = score_target(
        args.field, args.target, data_dir=args.data_dir, half_life=args.half_life
    )
    graph = load_snapshot(latest_snapshot(args.data_dir, args.field))

    logger.info("learned originators of %s", args.target)
    for node, score in rank(learned, args.top):
        title = graph.nodes[node].get("title")
        logger.info("  %-14s %8.4f  %s", node, score, "" if is_missing(title) else str(title)[:70])

    logger.info("\noverlap with the baselines it has to beat")
    for name, scores in baselines.items():
        logger.info("  %-22s %3.0f%%", name, 100 * overlap(learned, scores, args.top))


if __name__ == "__main__":
    main()
