"""Snowball crawl of the citation graph outward from a field's seed papers.

The crawl journals every fetched work and every edge it sees to append-only
JSONL as it goes, so an interrupted run resumes from its last checkpoint rather
than re-spending the API budget. Parquet is written once at the end.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from graph.config import FieldConfig, load_field
from graph.openalex import OpenAlexClient, Work

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

logger = logging.getLogger(__name__)


@dataclass
class CrawlState:
    """Where the crawl has got to. Persisted after every checkpoint."""

    depth: int = 0
    frontier: list[str] = field(default_factory=list)
    next_frontier: list[str] = field(default_factory=list)


class CrawlJournal:
    """Append-only record of a crawl, and the resume point derived from it."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.nodes_path = root / "nodes.jsonl"
        self.edges_path = root / "edges.jsonl"
        self.state_path = root / "state.json"

    def visited(self) -> set[str]:
        """Ids already fetched, recovered from the node journal."""
        if not self.nodes_path.exists():
            return set()
        with self.nodes_path.open() as handle:
            return {json.loads(line)["id"] for line in handle if line.strip()}

    def load_state(self) -> CrawlState | None:
        if not self.state_path.exists():
            return None
        return CrawlState(**json.loads(self.state_path.read_text()))

    def save_state(self, state: CrawlState) -> None:
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state.__dict__))
        tmp.replace(self.state_path)

    def append(self, works: Iterable[Work]) -> None:
        with self.nodes_path.open("a") as nodes, self.edges_path.open("a") as edges:
            for work in works:
                nodes.write(json.dumps(work.model_dump()) + "\n")
                for reference in work.referenced_works:
                    edges.write(json.dumps({"source": work.id, "target": reference}) + "\n")

    def _read(self, path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_json(path, lines=True)

    def to_parquet(self) -> tuple[Path, Path]:
        """Materialise the journal as the raw node and edge tables."""
        nodes = self._read(self.nodes_path).drop_duplicates(subset="id")
        edges = self._read(self.edges_path).drop_duplicates()

        nodes_parquet = self.root / "nodes.parquet"
        edges_parquet = self.root / "edges.parquet"
        nodes.to_parquet(nodes_parquet, index=False)
        edges.to_parquet(edges_parquet, index=False)
        return nodes_parquet, edges_parquet


def _dedup(ids: Iterable[str], exclude: set[str]) -> list[str]:
    seen: dict[str, None] = {}
    for value in ids:
        if value not in exclude:
            seen[value] = None
    return list(seen)


class SnowballCrawler:
    """Breadth-first expansion along citation edges, bounded by hop depth.

    The crawl walks *backwards*, from a paper to the works it references, because
    lineage runs that way. Works outside the field's date range are still
    recorded, but are not expanded through: they bound the crawl without
    silently dropping edges the cleaning stage may want to see.
    """

    def __init__(
        self,
        config: FieldConfig,
        client: OpenAlexClient,
        journal: CrawlJournal,
        *,
        max_nodes: int | None = None,
        checkpoint_every: int = 500,
    ) -> None:
        self.config = config
        self.client = client
        self.journal = journal
        self.max_nodes = max_nodes
        self.checkpoint_every = checkpoint_every

    def run(self) -> CrawlState:
        visited = self.journal.visited()
        state = self.journal.load_state() or CrawlState(
            frontier=_dedup(self.config.seed_papers, set())
        )

        while state.depth <= self.config.crawl.hop_depth:
            pending = _dedup(state.frontier, visited)
            logger.info("depth %d: %d works to fetch", state.depth, len(pending))
            expand = state.depth < self.config.crawl.hop_depth

            for batch in self._fetch(pending, visited, expand, state):
                self.journal.append(batch)
                self.journal.save_state(state)
                if self.max_nodes and len(visited) >= self.max_nodes:
                    logger.info("stopping at max_nodes=%d", self.max_nodes)
                    return state

            state.depth += 1
            state.frontier = _dedup(state.next_frontier, visited)
            state.next_frontier = []
            self.journal.save_state(state)

            # A resumed crawl can start a depth with nothing pending but a
            # frontier waiting for the next one, so emptiness is only decisive
            # once the depth has been rolled over.
            if not state.frontier:
                break

        return state

    def _fetch(
        self,
        pending: list[str],
        visited: set[str],
        expand: bool,
        state: CrawlState,
    ) -> Iterator[list[Work]]:
        batch: list[Work] = []
        for work in self.client.works(pending):
            if work.id in visited:
                continue
            visited.add(work.id)
            batch.append(work)

            if expand and self.config.crawl.date_range.contains(work.publication_year):
                state.next_frontier.extend(work.referenced_works)

            if self.max_nodes and len(visited) >= self.max_nodes:
                break

            if len(batch) >= self.checkpoint_every:
                state.frontier = [work_id for work_id in pending if work_id not in visited]
                yield batch
                batch = []

        if batch:
            state.frontier = [work_id for work_id in pending if work_id not in visited]
            yield batch


def crawl_field(
    field_name: str,
    *,
    data_dir: Path = DATA_DIR,
    max_nodes: int | None = None,
    mailto: str | None = None,
) -> tuple[Path, Path]:
    """Crawl one field end to end and return the raw node and edge tables."""
    config = load_field(field_name)
    journal = CrawlJournal(data_dir / config.name / "raw")

    with OpenAlexClient(mailto) as client:
        SnowballCrawler(config, client, journal, max_nodes=max_nodes).run()

    return journal.to_parquet()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", default="gnn", help="field config name under fields/")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--max-nodes", type=int, default=None, help="stop after this many works")
    parser.add_argument("--mailto", default=None, help="contact address for the polite pool")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    nodes, edges = crawl_field(
        args.field, data_dir=args.data_dir, max_nodes=args.max_nodes, mailto=args.mailto
    )
    logger.info("wrote %s and %s", nodes, edges)


if __name__ == "__main__":
    main()
