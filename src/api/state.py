"""What a request needs and must not rebuild.

Ranking one paper is two costs in a trench coat. The encode pass over the whole
graph is the expensive one and it does not depend on which paper was asked
about; the walk backwards from the target is cheap and does. So the snapshot,
the trained model and the encode pass are loaded once per field and held, and
only the walk runs per request.

Two requests arriving together on a cold field would otherwise both pay for the
load, so it happens under a lock and the second one waits for the first.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
from torch import Tensor

from gnn.model import LinkPredictor
from gnn.originators import embed
from gnn.train import load_model
from graph.build import latest_snapshot, load_snapshot
from graph.config import FIELDS_DIR, FieldConfig, load_field
from graph.crawler import DATA_DIR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Field:
    """One field's snapshot, ready to answer about any paper in it."""

    config: FieldConfig
    graph: nx.DiGraph
    model: LinkPredictor
    embedding: tuple[dict[str, int], Tensor]


class Library:
    """The fields this service can answer about."""

    def __init__(self, data_dir: Path = DATA_DIR, fields_dir: Path = FIELDS_DIR) -> None:
        self.data_dir = data_dir
        self.fields_dir = fields_dir
        self._loaded: dict[str, Field] = {}
        self._lock = threading.Lock()

    def names(self) -> list[str]:
        """Every field with a config, whether or not it has been crawled yet.

        A request naming a field is only ever matched against this list. Nothing
        from a request is joined into a path.
        """
        return sorted(path.stem for path in self.fields_dir.glob("*.yaml"))

    def snapshot(self, name: str) -> Path | None:
        try:
            return latest_snapshot(self.data_dir, name)
        except FileNotFoundError:
            return None

    def load(self, name: str) -> Field:
        if ready := self._loaded.get(name):
            return ready
        with self._lock:
            if ready := self._loaded.get(name):
                return ready
            self._loaded[name] = ready = self._build(name)
            return ready

    def _build(self, name: str) -> Field:
        snapshot = latest_snapshot(self.data_dir, name)
        logger.info("loading field %s from %s", name, snapshot)
        graph = load_snapshot(snapshot)
        model = load_model(snapshot)
        return Field(load_field(name, self.fields_dir), graph, model, embed(graph, model))
