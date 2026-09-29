import pytest
from pydantic import ValidationError

from graph.config import DateRange, load_field

CONFIG = """
name: {name}
display_name: Test Field
description: >
  A description
  spread over lines.
topic_id: T1
seed_papers: [W1]
crawl:
  hop_depth: {hop_depth}
  date_range:
    start: 2000
    end: 2020
"""


def write_config(tmp_path, name="test", hop_depth=2, filename=None):
    path = tmp_path / (filename or f"{name}.yaml")
    path.write_text(CONFIG.format(name=name, hop_depth=hop_depth))
    return path


def test_shipped_gnn_config_loads():
    config = load_field("gnn")

    assert config.name == "gnn"
    assert config.topic_id == "T11273"
    assert len(config.seed_papers) >= 5
    assert all(paper.startswith("W") for paper in config.seed_papers)


def test_description_is_collapsed_to_a_single_line(tmp_path):
    write_config(tmp_path)

    assert load_field("test", tmp_path).description == "A description spread over lines."


def test_config_can_be_loaded_by_path(tmp_path):
    path = write_config(tmp_path)

    assert load_field(str(path)).display_name == "Test Field"


def test_missing_config_is_reported_with_its_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="absent.yaml"):
        load_field("absent", tmp_path)


def test_name_must_match_filename(tmp_path):
    write_config(tmp_path, name="declared", filename="onfile.yaml")

    with pytest.raises(ValueError, match="declares name 'declared'"):
        load_field("onfile", tmp_path)


def test_a_forward_crawl_needs_a_topic_to_bound_it(tmp_path):
    path = write_config(tmp_path)
    path.write_text(
        path.read_text()
        .replace("topic_id: T1", "topic_id: null")
        .replace("hop_depth: 2", "hop_depth: 2\n  forward_depth: 1")
    )

    with pytest.raises(ValidationError, match="forward_depth needs a topic_id"):
        load_field("test", tmp_path)


def test_negative_hop_depth_is_rejected(tmp_path):
    write_config(tmp_path, hop_depth=-1)

    with pytest.raises(ValidationError):
        load_field("test", tmp_path)


@pytest.mark.parametrize(
    ("bounds", "year", "expected"),
    [
        ((2000, 2020), 2010, True),
        ((2000, 2020), 2000, True),
        ((2000, 2020), 2020, True),
        ((2000, 2020), 1999, False),
        ((2000, 2020), 2021, False),
        ((None, None), 1800, True),
        ((None, 2020), 1800, True),
        ((2000, None), 3000, True),
        ((2000, 2020), None, False),
    ],
)
def test_date_range_bounds_are_inclusive(bounds, year, expected):
    assert DateRange(start=bounds[0], end=bounds[1]).contains(year) is expected
