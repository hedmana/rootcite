"""Field configuration: the only place the pipeline learns what it is about."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

FIELDS_DIR = Path(__file__).resolve().parents[2] / "fields"


class DateRange(BaseModel):
    start: int | None = None
    end: int | None = None

    def contains(self, year: int | None) -> bool:
        if year is None:
            return False
        return (self.start is None or year >= self.start) and (self.end is None or year <= self.end)


class CrawlConfig(BaseModel):
    hop_depth: int = Field(ge=0)
    date_range: DateRange = Field(default_factory=DateRange)


class FieldConfig(BaseModel):
    name: str
    display_name: str
    description: str
    topic_id: str | None = None
    seed_papers: list[str] = Field(default_factory=list)
    crawl: CrawlConfig

    @field_validator("description")
    @classmethod
    def _collapse_whitespace(cls, value: str) -> str:
        return " ".join(value.split())


def load_field(name: str, fields_dir: Path | None = None) -> FieldConfig:
    """Load `fields/<name>.yaml`, or a path to a config file."""
    path = Path(name)
    if not path.suffix:
        path = (fields_dir or FIELDS_DIR) / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no field config at {path}")

    config = FieldConfig.model_validate(yaml.safe_load(path.read_text()))
    if config.name != path.stem:
        raise ValueError(f"field config {path.name} declares name {config.name!r}")
    return config
