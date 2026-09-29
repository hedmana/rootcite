"""How much two works share in what they are about, from their titles and abstracts.

A paper is built on work about the same things. An optimiser, a dataset or a
regularisation trick is cited by nearly every paper and is about none of them,
and in the citation graph alone it looks just like a real precursor: both are
hubs. TF-IDF over titles and abstracts tells them apart, with no model to train
and no dependency to add.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import networkx as nx

from graph.build import is_missing

# Common English only. A word common to a whole field is discounted by its own
# document frequency, so no field's vocabulary has to be written down here.
# fmt: off
STOPWORDS = frozenset((
    "the", "and", "for", "with", "that", "this", "from", "are", "our", "can", "which", "have",
    "has", "been", "using", "used", "use", "new", "two", "its", "their", "these", "such",
    "into", "than", "also", "based", "via", "over", "more", "not", "all", "one", "only",
    "between", "both", "each", "other",
))
# fmt: on

_WORD = re.compile(r"[a-z][a-z0-9-]{2,}")


class Content:
    """A unit TF-IDF vector for every work in a graph."""

    def __init__(self, graph: nx.DiGraph) -> None:
        counts = {node: Counter(_words(graph.nodes[node])) for node in graph}
        frequency = Counter(word for words in counts.values() for word in words)
        self._vectors = {
            node: _unit(
                {
                    word: (1 + math.log(count)) * math.log(len(counts) / frequency[word])
                    for word, count in words.items()
                }
            )
            for node, words in counts.items()
        }

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity: 1 for the same words, 0 for none shared or none on record."""
        left, right = self._vectors.get(a, {}), self._vectors.get(b, {})
        if len(left) > len(right):
            left, right = right, left
        return sum(weight * right.get(word, 0.0) for word, weight in left.items())


def _words(attributes: dict) -> list[str]:
    text = " ".join(
        str(value) for key in ("title", "abstract") if not is_missing(value := attributes.get(key))
    )
    return [word for word in _WORD.findall(text.lower()) if word not in STOPWORDS]


def _unit(vector: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
    return {word: value / norm for word, value in vector.items()}
