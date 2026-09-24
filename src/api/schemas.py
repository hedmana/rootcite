"""What the service says, as distinct from what it computes.

The internal types carry what the pipeline needs; these carry what a client can
render. Nothing here exposes a path, a score it cannot interpret, or a field of
a snapshot row that happened to be present.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from graph.build import is_missing
from llm.narrative import Ancestry


class Work(BaseModel):
    work_id: str
    title: str | None = None
    publication_year: int | None = None
    authors: list[str] = Field(default_factory=list)


class Originator(Work):
    score: float


class Link(BaseModel):
    citing: str
    cited: str
    direct: bool = Field(description="cites it outright, rather than through works not shown")


class Lineage(BaseModel):
    field: str
    target: Work
    originators: list[Originator]
    links: list[Link] = Field(
        description="who leads to whom among the target and its originators, transitively reduced"
    )
    baselines: dict[str, float] = Field(
        description="share of this ranking's top-k that each unlearned baseline also picked"
    )


class Cited(BaseModel):
    work_id: str
    title: str | None = None
    contribution: str


class Dropped(BaseModel):
    work_id: str
    reason: str


class Account(BaseModel):
    field: str
    target: Work
    summary: str
    claims: list[Cited]
    dropped: list[Dropped] = Field(
        description="claims verification threw out, and what it told the model about why"
    )
    attempts: int
    partial: bool


def work(graph, node: str) -> Work:
    """A node as a client sees it. Snapshot rows carry pandas absence, not None."""
    attributes = graph.nodes[node]
    title = attributes.get("title")
    year = attributes.get("publication_year")
    return Work(
        work_id=node,
        title=None if is_missing(title) else str(title),
        publication_year=None if is_missing(year) else int(year),
        authors=[str(name) for name in attributes.get("authors") or []],
    )


def account(graph, field_name: str, target: str, ancestry: Ancestry) -> Account:
    return Account(
        field=field_name,
        target=work(graph, target),
        summary=ancestry.summary,
        claims=[
            Cited(
                work_id=claim.work_id,
                title=work(graph, claim.work_id).title if claim.work_id in graph else None,
                contribution=claim.contribution,
            )
            for claim in ancestry.claims
        ],
        dropped=[Dropped(work_id=r.work_id, reason=r.reason) for r in ancestry.rejected],
        attempts=ancestry.attempts,
        partial=ancestry.partial,
    )


class FieldSummary(BaseModel):
    name: str
    display_name: str
    description: str
    ready: bool = Field(description="whether a snapshot exists to answer about")
