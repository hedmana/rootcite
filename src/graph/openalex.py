"""Client for the OpenAlex API.

Only the endpoints the citation crawler needs: resolving a work, expanding a
work's references and citations, and enumerating the works under a topic.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Iterator
from typing import Any

import httpx
from pydantic import BaseModel, Field, model_validator

OPENALEX_API = "https://api.openalex.org"

# OpenAlex caps an OR-joined filter at 50 values.
MAX_IDS_PER_FILTER = 50
MAX_PER_PAGE = 200

WORK_FIELDS = (
    "id,title,display_name,publication_year,publication_date,cited_by_count,referenced_works,"
    "abstract_inverted_index,primary_topic,authorships"
)

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class OpenAlexError(RuntimeError):
    """The API could not satisfy a request."""


def strip_id_prefix(value: str) -> str:
    """Turn `https://openalex.org/W123` into `W123`, leaving bare ids alone."""
    return value.rsplit("/", 1)[-1] if value else value


def abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str | None:
    """Rebuild abstract text from OpenAlex's position-indexed token map."""
    if not index:
        return None
    tokens = sorted((position, token) for token, spots in index.items() for position in spots)
    return " ".join(token for _, token in tokens)


class Work(BaseModel):
    """A single work, flattened to the fields the pipeline actually uses."""

    id: str
    title: str | None = None
    abstract: str | None = None
    publication_year: int | None = None
    publication_date: str | None = None
    cited_by_count: int = 0
    referenced_works: list[str] = Field(default_factory=list)
    topic_id: str | None = None
    authors: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _flatten(cls, payload: Any) -> Any:
        if not isinstance(payload, dict) or "id" not in payload:
            return payload

        topic = payload.get("primary_topic") or {}
        return {
            **payload,
            "id": strip_id_prefix(payload["id"]),
            # OpenAlex leaves `title` null on some records while display_name survives.
            "title": payload.get("title") or payload.get("display_name"),
            "abstract": abstract_from_inverted_index(payload.get("abstract_inverted_index")),
            "referenced_works": [
                strip_id_prefix(ref) for ref in payload.get("referenced_works") or []
            ],
            "topic_id": strip_id_prefix(topic["id"]) if topic.get("id") else None,
            "authors": [
                author["author"]["display_name"]
                for author in payload.get("authorships") or []
                if author.get("author", {}).get("display_name")
            ],
        }


def _chunked(values: Iterable[str], size: int) -> Iterator[list[str]]:
    chunk: list[str] = []
    for value in values:
        chunk.append(value)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


class OpenAlexClient:
    """Synchronous OpenAlex client with cursor pagination and backoff.

    Supplying `mailto` joins the polite pool, which OpenAlex rate-limits far
    more generously. It defaults to the `OPENALEX_MAILTO` environment variable
    so a contact address never has to live in the repository.
    """

    def __init__(
        self,
        mailto: str | None = None,
        *,
        base_url: str = OPENALEX_API,
        transport: httpx.BaseTransport | None = None,
        per_page: int = MAX_PER_PAGE,
        max_retries: int = 5,
        backoff: float = 1.0,
        timeout: float = 30.0,
    ) -> None:
        self.per_page = min(per_page, MAX_PER_PAGE)
        self.max_retries = max_retries
        self.backoff = backoff

        contact = mailto or os.environ.get("OPENALEX_MAILTO")
        self._client = httpx.Client(
            base_url=base_url,
            transport=transport,
            timeout=timeout,
            headers={"User-Agent": f"rootcite (+{contact})" if contact else "rootcite"},
            params={"mailto": contact} if contact else None,
        )

    def __enter__(self) -> OpenAlexClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def work(self, work_id: str) -> Work:
        """Fetch one work by OpenAlex id."""
        payload = self._get(f"/works/{strip_id_prefix(work_id)}", {"select": WORK_FIELDS})
        return Work.model_validate(payload)

    def works(self, work_ids: Iterable[str]) -> Iterator[Work]:
        """Fetch many works, batched into as few requests as the API allows."""
        for chunk in _chunked((strip_id_prefix(i) for i in work_ids), MAX_IDS_PER_FILTER):
            yield from self._paginate({"filter": f"openalex_id:{'|'.join(chunk)}"})

    def works_by_topic(
        self,
        topic_id: str,
        *,
        from_year: int | None = None,
        to_year: int | None = None,
    ) -> Iterator[Work]:
        """Enumerate works under a topic, optionally bounded by publication year."""
        filters = [f"primary_topic.id:{strip_id_prefix(topic_id)}"]
        if from_year is not None:
            filters.append(f"from_publication_date:{from_year}-01-01")
        if to_year is not None:
            filters.append(f"to_publication_date:{to_year}-12-31")
        yield from self._paginate({"filter": ",".join(filters)})

    def citing_works(self, work_id: str) -> Iterator[Work]:
        """Works that cite the given work, i.e. its descendants."""
        yield from self._paginate({"filter": f"cites:{strip_id_prefix(work_id)}"})

    def referenced_works(self, work_id: str) -> Iterator[Work]:
        """Works the given work cites, i.e. its immediate ancestors."""
        yield from self.works(self.work(work_id).referenced_works)

    def _paginate(self, params: dict[str, str]) -> Iterator[Work]:
        cursor = "*"
        while cursor:
            page = self._get(
                "/works",
                {**params, "select": WORK_FIELDS, "per-page": self.per_page, "cursor": cursor},
            )
            results = page.get("results") or []
            if not results:
                return
            for payload in results:
                yield Work.model_validate(payload)
            cursor = (page.get("meta") or {}).get("next_cursor")

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.max_retries):
            response = self._client.get(path, params=params)
            if response.status_code in _RETRY_STATUS:
                time.sleep(self._retry_delay(response, attempt))
                continue
            if response.is_error:
                raise OpenAlexError(f"{response.status_code} from {response.request.url}")
            return response.json()

        raise OpenAlexError(f"gave up after {self.max_retries} retries for {path}")

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            return float(retry_after)
        return self.backoff * 2**attempt
