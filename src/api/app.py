"""The pipeline over HTTP, for a client that cannot run a snapshot itself.

Two answers at two prices. A ranking is an encode pass the service has already
paid for plus a walk backwards from the target; a narrative is model calls,
seconds of latency and real money. They are separate routes so a client can ask
for the first without buying the second, and the expensive one is a POST
because it is not something a browser should issue by following a link.

Handlers are synchronous on purpose. Torch and the model backend both block,
and a synchronous handler runs in a threadpool, which is where blocking work
belongs. Declaring them `async def` would park that work on the event loop and
stall every other request behind it.

A request never reaches the filesystem with anything it supplied. A field name
is matched against the configs that exist before it is used, a work id has to
look like an OpenAlex id before it reaches a graph lookup or a prompt, and no
response carries a path, a key or a backend's own error text.
"""

from __future__ import annotations

import logging
import os
from typing import Annotated

import networkx as nx
from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import Account, FieldSummary, Lineage, Link, Originator, account, work
from api.state import Field, Library
from gnn.originators import learned_flow
from graph.config import load_field
from graph.score import SCORERS, acyclic, ancestry, overlap, rank, skeleton
from llm import load_provider
from llm.base import Provider, ProviderError
from llm.narrative import tell

logger = logging.getLogger(__name__)

DEFAULT_ORIGINS = "http://localhost:5173"

WorkId = Annotated[str, Path(pattern=r"^W\d+$", description="OpenAlex work id")]
FieldName = Annotated[str, Path(pattern=r"^[a-z0-9_-]{1,40}$")]


def allowed_origins() -> list[str]:
    """An explicit allowlist, defaulting to the Vite dev server and nothing else."""
    configured = os.environ.get("ROOTCITE_API_ORIGINS", DEFAULT_ORIGINS)
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


def _present(graph: nx.DiGraph, work_id: str) -> None:
    if work_id not in graph:
        raise HTTPException(404, f"{work_id} is not in this field's snapshot")


def resolve(request: Request, field: FieldName, work_id: WorkId) -> Field:
    """A field name becomes a loaded snapshot, or it becomes an error. It never
    becomes a path: the name is matched against the configs that exist first.

    The work id is taken here, unused, so that its shape is checked before this
    runs. A dependency's own parameters are validated ahead of its body, and
    loading a cold field is seconds of work; a request whose id could never name
    a paper should not be able to buy that.
    """
    shelf: Library = request.app.state.library
    if field not in shelf.names():
        raise HTTPException(404, f"no field named {field!r}")
    if shelf.snapshot(field) is None:
        raise HTTPException(409, f"field {field!r} has no snapshot yet; crawl and build it")
    return shelf.load(field)


def resolve_backend(request: Request) -> Provider:
    """Whatever the environment configures, built on first use and kept."""
    if request.app.state.provider is None:
        request.app.state.provider = provider = load_provider()
        logger.info("narrating with %s / %s", provider.name, provider.config.model)
    return request.app.state.provider


Loaded = Annotated[Field, Depends(resolve)]
Backend = Annotated[Provider, Depends(resolve_backend)]


def create_app(
    library: Library | None = None,
    provider: Provider | None = None,
    origins: list[str] | None = None,
) -> FastAPI:
    """Wire the service. The library and the backend are arguments so a test can
    drive every route without a snapshot on disk or a model to call."""
    app = FastAPI(title="rootcite", version="0.1.0", summary=__doc__.splitlines()[0])
    app.state.library = shelf = library or Library()
    app.state.provider = provider

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins if origins is not None else allowed_origins(),
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "fields": {n: shelf.snapshot(n) is not None for n in shelf.names()}}

    @app.get("/fields")
    def fields() -> list[FieldSummary]:
        return [
            FieldSummary(
                **load_field(name, shelf.fields_dir).model_dump(
                    include={"name", "display_name", "description"}
                ),
                ready=shelf.snapshot(name) is not None,
            )
            for name in shelf.names()
        ]

    @app.get("/lineage/{field}/{work_id}")
    def lineage(
        loaded: Loaded,
        work_id: WorkId,
        top: Annotated[int, Query(ge=1, le=50)] = 10,
    ) -> Lineage:
        graph = loaded.graph
        _present(graph, work_id)
        learned = learned_flow(
            graph, work_id, loaded.model, embedding=loaded.embedding, content=loaded.content
        )
        ranked = rank(learned, top)
        shown = [work_id, *(node for node, _ in ranked)]
        return Lineage(
            field=loaded.config.name,
            target=work(graph, work_id),
            originators=[
                Originator(**work(graph, node).model_dump(), score=score) for node, score in ranked
            ],
            links=[
                Link(citing=citing, cited=cited, direct=direct)
                for citing, cited, direct in skeleton(acyclic(ancestry(graph, work_id)), shown)
            ],
            baselines={
                name: overlap(learned, scorer(graph, work_id), top)
                for name, scorer in SCORERS.items()
            },
        )

    @app.post("/narrative/{field}/{work_id}")
    def narrative(
        loaded: Loaded,
        backend: Backend,
        work_id: WorkId,
        top: Annotated[int, Query(ge=1, le=20)] = 8,
        attempts: Annotated[int, Query(ge=1, le=5)] = 2,
    ) -> Account:
        graph = loaded.graph
        _present(graph, work_id)
        scores = learned_flow(
            graph, work_id, loaded.model, embedding=loaded.embedding, content=loaded.content
        )
        if not scores:
            raise HTTPException(404, f"{work_id} has no ancestors in this field's snapshot")
        try:
            told = tell(graph, work_id, scores, backend, top=top, attempts=attempts)
        except ProviderError as failure:
            # The backend's own message can carry an endpoint or a key, so it is
            # logged and not returned.
            logger.warning("backend could not narrate %s: %s", work_id, failure)
            raise HTTPException(502, "the model backend could not answer") from failure
        return account(graph, loaded.config.name, work_id, told)

    return app


app = create_app()
