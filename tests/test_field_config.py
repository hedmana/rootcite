from pathlib import Path

import pytest
import yaml

FIELD_CONFIGS = sorted((Path(__file__).parent.parent / "fields").glob("*.yaml"))


@pytest.mark.parametrize("path", FIELD_CONFIGS, ids=lambda p: p.stem)
def test_field_config_satisfies_contract(path):
    config = yaml.safe_load(path.read_text())

    assert config["name"] == path.stem
    assert config["display_name"]
    assert config["description"]
    assert "topic_id" in config
    assert isinstance(config["seed_papers"], list)

    crawl = config["crawl"]
    assert isinstance(crawl["hop_depth"], int)
    assert set(crawl["date_range"]) == {"start", "end"}


def test_at_least_one_field_config_exists():
    assert FIELD_CONFIGS
