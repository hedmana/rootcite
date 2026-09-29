"""Recover the references OpenAlex lost, from Semantic Scholar.

About one reference in eight that OpenAlex lists points at a record it has since
deleted: the id answers 404, and nothing about the work it named survives. GCN
loses its links to Bruna's spectral networks, to Henaff and to Planetoid that
way. Semantic Scholar keeps its own reference lists, and knows most works by the
Microsoft Academic Graph ids that OpenAlex's older ids are made from.

So every crawled work with such an id has its Semantic Scholar references
matched back to crawled works: by MAG id where OpenAlex kept it, or by title and
year where OpenAlex re-filed the work under a new id. Only citations between
works the crawl holds are added. A work it never reached stays out, as it would
had OpenAlex kept the reference.

An API key raises the rate limit. It is read from `S2_API_KEY`, never from code.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from pydantic import BaseModel, Field

from graph.build import PREPRINT_SLACK, RECOVERED_EDGES, is_missing
from graph.config import load_field
from graph.crawler import DATA_DIR

logger = logging.getLogger(__name__)

S2_API = "https://api.semanticscholar.org/graph/v1"
MAX_IDS_PER_BATCH = 500
REFERENCE_FIELDS = "references.title,references.year,references.externalIds"

# OpenAlex numbers the works it added itself from 4.2 billion up, past every MAG id.
FIRST_OPENALEX_ID = 4_000_000_000

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class SemanticScholarError(RuntimeError):
    """The API could not satisfy a request."""


class Reference(BaseModel):
    title: str | None = None
    year: int | None = None
    external_ids: dict[str, Any] | None = Field(default=None, alias="externalIds")


def mag_id(work_id: str) -> str | None:
    """The MAG id an OpenAlex work id was made from, if it was made from one."""
    digits = work_id.removeprefix("W")
    return digits if digits.isdigit() and int(digits) < FIRST_OPENALEX_ID else None


class SemanticScholarClient:
    """Reference lists for batches of works, with backoff on the shared rate limit."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = S2_API,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 8,
        backoff: float = 2.0,
        timeout: float = 120.0,
    ) -> None:
        self.max_retries = max_retries
        self.backoff = backoff
        key = api_key or os.environ.get("S2_API_KEY")
        self._client = httpx.Client(
            base_url=base_url,
            transport=transport,
            timeout=timeout,
            headers={"x-api-key": key} if key else None,
        )

    def __enter__(self) -> SemanticScholarClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self._client.close()

    def references(self, work_ids: list[str]) -> Iterator[tuple[str, list[Reference]]]:
        """Each work Semantic Scholar knows by its MAG id, with the works it cites."""
        for start in range(0, len(work_ids), MAX_IDS_PER_BATCH):
            chunk = work_ids[start : start + MAX_IDS_PER_BATCH]
            papers = self._post({"ids": [f"MAG:{mag_id(work_id)}" for work_id in chunk]})
            for work_id, paper in zip(chunk, papers, strict=True):
                if paper:
                    cited = paper.get("references") or []
                    yield work_id, [Reference.model_validate(reference) for reference in cited]

    def _post(self, body: dict[str, list[str]]) -> list[dict[str, Any] | None]:
        for attempt in range(self.max_retries):
            response = self._client.post(
                "/paper/batch", params={"fields": REFERENCE_FIELDS}, json=body
            )
            if response.status_code in _RETRY_STATUS:
                retry_after = response.headers.get("Retry-After", "")
                time.sleep(
                    float(retry_after) if retry_after.isdigit() else self.backoff * 2**attempt
                )
                continue
            if response.is_error:
                raise SemanticScholarError(f"{response.status_code} from {response.request.url}")
            return response.json()
        raise SemanticScholarError(f"gave up after {self.max_retries} retries")


def _normal(title: object) -> str:
    return "" if is_missing(title) else re.sub(r"[^a-z0-9]", "", str(title).lower())


def recover_edges(
    nodes: pd.DataFrame, references: Iterable[tuple[str, list[Reference]]]
) -> pd.DataFrame:
    """Citations Semantic Scholar records between crawled works, in OpenAlex ids.

    A title shared by works more than a year apart, or by two records of one
    year, names no single work, so it matches nothing.
    """
    known = set(nodes["id"])
    by_title: dict[str, list[tuple[str, object]]] = defaultdict(list)
    for work_id, title, year in nodes[["id", "title", "publication_year"]].itertuples(index=False):
        if key := _normal(title):
            by_title[key].append((work_id, year))

    def match(reference: Reference) -> str | None:
        mag = (reference.external_ids or {}).get("MAG")
        if mag and f"W{mag}" in known:
            return f"W{mag}"
        candidates = [
            work_id
            for work_id, year in by_title.get(_normal(reference.title), [])
            if reference.year is None
            or is_missing(year)
            or abs(year - reference.year) <= PREPRINT_SLACK
        ]
        return candidates[0] if len(candidates) == 1 else None

    edges = {
        (source, target)
        for source, cited in references
        for reference in cited
        if (target := match(reference)) and target != source
    }
    return pd.DataFrame(sorted(edges), columns=["source", "target"])


def recover_field(
    field_name: str,
    *,
    data_dir: Path = DATA_DIR,
    client: SemanticScholarClient | None = None,
) -> tuple[Path, int, int]:
    """Recover a field's lost citations beside its raw crawl.

    Returns where they went, how many were matched, and how many of those
    OpenAlex had not listed.
    """
    raw = data_dir / load_field(field_name).name / "raw"
    nodes = pd.read_parquet(raw / "nodes.parquet")
    listed = pd.read_parquet(raw / "edges.parquet")
    queried = [work_id for work_id in nodes["id"] if mag_id(work_id)]
    logger.info("asking Semantic Scholar for %d works' references", len(queried))

    with client or SemanticScholarClient() as s2:
        recovered = recover_edges(nodes, s2.references(queried))

    already = set(zip(listed["source"], listed["target"], strict=True))
    new = sum(edge not in already for edge in recovered.itertuples(index=False, name=None))
    path = raw / RECOVERED_EDGES
    recovered.to_parquet(path, index=False)
    return path, len(recovered), new


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", default="gnn", help="field config name under fields/")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    path, matched, new = recover_field(args.field, data_dir=args.data_dir)
    logger.info("%d citations matched, %d missing from OpenAlex", matched, new)
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
