import json

import pandas as pd

from graph.config import CrawlConfig, DateRange, FieldConfig
from graph.crawler import CrawlJournal, SnowballCrawler
from graph.openalex import Work

# A -> B, C ; B -> D ; C -> D, E ; D -> F
CITATIONS = {
    "A": ["B", "C"],
    "B": ["D"],
    "C": ["D", "E"],
    "D": ["F"],
    "E": [],
    "F": [],
}


class FakeClient:
    """Stands in for OpenAlexClient, recording which ids each call asked for."""

    def __init__(self, citations=None, years=None):
        self.citations = citations or CITATIONS
        self.years = years or {}
        self.requested = []
        self.cited = []

    def works(self, work_ids):
        work_ids = list(work_ids)
        self.requested.append(work_ids)
        for work_id in work_ids:
            if work_id in self.citations:
                yield Work(
                    id=work_id,
                    title=work_id,
                    publication_year=self.years.get(work_id, 2015),
                    referenced_works=self.citations[work_id],
                )

    def citing_works(self, work_id, **bounds):
        self.cited.append((work_id, bounds))
        citing = [work for work, references in self.citations.items() if work_id in references]
        yield from self.works(citing)


# The works citing A, and one citing those in turn, which only a backward crawl misses.
DESCENDANTS = {**CITATIONS, "X": ["A", "B"], "Y": ["A", "Z"], "W": ["X"]}


def make_config(hop_depth=2, start=None, end=None, seeds=("A",), forward_depth=0, fill=0):
    return FieldConfig(
        name="test",
        display_name="Test",
        description="test field",
        topic_id="T1",
        seed_papers=list(seeds),
        crawl=CrawlConfig(
            hop_depth=hop_depth,
            forward_depth=forward_depth,
            fill_cited_by=fill,
            date_range=DateRange(start=start, end=end),
        ),
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def crawl(tmp_path, config, client=None, **kwargs):
    client = client or FakeClient()
    journal = CrawlJournal(tmp_path)
    SnowballCrawler(config, client, journal, **kwargs).run()
    return journal, client


def node_ids(journal):
    return sorted(node["id"] for node in read_jsonl(journal.nodes_path))


def edge_pairs(journal):
    return sorted((e["source"], e["target"]) for e in read_jsonl(journal.edges_path))


def test_crawl_expands_breadth_first_to_the_hop_depth(tmp_path):
    journal, _ = crawl(tmp_path, make_config(hop_depth=2))

    assert node_ids(journal) == ["A", "B", "C", "D", "E"]


def test_hop_depth_zero_fetches_only_the_seeds(tmp_path):
    journal, _ = crawl(tmp_path, make_config(hop_depth=0))

    assert node_ids(journal) == ["A"]


def test_edges_beyond_the_last_hop_are_recorded_for_the_cleaning_stage(tmp_path):
    journal, _ = crawl(tmp_path, make_config(hop_depth=2))

    assert ("D", "F") in edge_pairs(journal)
    assert "F" not in node_ids(journal)


def test_every_traversed_edge_is_journalled(tmp_path):
    journal, _ = crawl(tmp_path, make_config(hop_depth=2))

    assert edge_pairs(journal) == [
        ("A", "B"),
        ("A", "C"),
        ("B", "D"),
        ("C", "D"),
        ("C", "E"),
        ("D", "F"),
    ]


def test_a_work_reached_twice_is_fetched_once(tmp_path):
    journal, client = crawl(tmp_path, make_config(hop_depth=3))
    requested = [work_id for call in client.requested for work_id in call]

    assert requested.count("D") == 1
    assert node_ids(journal).count("D") == 1


def test_works_outside_the_date_range_are_recorded_but_not_expanded(tmp_path):
    client = FakeClient(years={"A": 2015, "B": 2015, "C": 1990})
    journal, _ = crawl(tmp_path, make_config(hop_depth=2, start=2000, end=2020), client)

    assert "C" in node_ids(journal)
    assert "E" not in node_ids(journal)
    assert ("C", "E") in edge_pairs(journal)


def test_max_nodes_halts_the_crawl(tmp_path):
    journal, _ = crawl(tmp_path, make_config(hop_depth=2), max_nodes=1, checkpoint_every=1)

    assert node_ids(journal) == ["A"]


def test_an_interrupted_crawl_resumes_from_its_checkpoint(tmp_path):
    config = make_config(hop_depth=2)
    journal, _ = crawl(tmp_path, config, max_nodes=1, checkpoint_every=1)
    assert node_ids(journal) == ["A"]

    resumed = FakeClient()
    SnowballCrawler(config, resumed, CrawlJournal(tmp_path), checkpoint_every=1).run()

    assert node_ids(journal) == ["A", "B", "C", "D", "E"]
    assert "A" not in [work_id for call in resumed.requested for work_id in call]


def test_state_is_checkpointed_during_a_depth(tmp_path):
    crawl(tmp_path, make_config(hop_depth=2), max_nodes=1, checkpoint_every=1)
    state = json.loads((tmp_path / "state.json").read_text())

    assert state["depth"] == 0
    assert state["next_frontier"] == ["B", "C"]


def test_parquet_tables_deduplicate_the_journal(tmp_path):
    journal, _ = crawl(tmp_path, make_config(hop_depth=2))
    journal.append(list(FakeClient().works(["A"])))

    nodes_path, edges_path = journal.to_parquet()
    nodes = pd.read_parquet(nodes_path)
    edges = pd.read_parquet(edges_path)

    assert sorted(nodes["id"]) == ["A", "B", "C", "D", "E"]
    assert len(edges) == 6
    assert set(nodes.columns) >= {"id", "title", "publication_year", "cited_by_count"}


def test_empty_journal_produces_empty_tables(tmp_path):
    nodes_path, edges_path = CrawlJournal(tmp_path).to_parquet()

    assert pd.read_parquet(nodes_path).empty
    assert pd.read_parquet(edges_path).empty


def test_a_forward_crawl_takes_in_the_works_citing_the_seeds(tmp_path):
    client = FakeClient(DESCENDANTS)
    journal, _ = crawl(tmp_path, make_config(forward_depth=1), client)

    assert node_ids(journal) == ["A", "B", "C", "D", "E", "X", "Y"]
    assert {("X", "A"), ("X", "B"), ("Y", "A"), ("Y", "Z")} <= set(edge_pairs(journal))


def test_each_forward_hop_reaches_one_generation_further(tmp_path):
    journal, _ = crawl(tmp_path, make_config(forward_depth=2), FakeClient(DESCENDANTS))

    assert "W" in node_ids(journal)


def test_the_forward_crawl_stays_inside_the_topic_and_years(tmp_path):
    client = FakeClient(DESCENDANTS)
    crawl(tmp_path, make_config(forward_depth=1, start=2000, end=2020), client)

    assert client.cited == [("A", {"topic_id": "T1", "from_year": 2000, "to_year": 2020})]


def test_without_a_forward_depth_no_citing_works_are_asked_for(tmp_path):
    client = FakeClient(DESCENDANTS)
    journal, _ = crawl(tmp_path, make_config(), client)

    assert client.cited == []
    assert "X" not in node_ids(journal)


def test_an_interrupted_forward_crawl_resumes_without_losing_citers(tmp_path):
    config = make_config(forward_depth=1)
    crawl(tmp_path, config, FakeClient(DESCENDANTS), max_nodes=6, checkpoint_every=1)

    resumed = FakeClient(DESCENDANTS)
    SnowballCrawler(config, resumed, CrawlJournal(tmp_path), checkpoint_every=1).run()
    journal = CrawlJournal(tmp_path)

    assert node_ids(journal) == ["A", "B", "C", "D", "E", "X", "Y"]
    assert resumed.requested == [["X", "Y"]]


def test_works_cited_often_enough_are_fetched_but_not_followed(tmp_path):
    journal, _ = crawl(tmp_path, make_config(hop_depth=1, fill=2))

    # D is cited by B and C, E only by C, and D's own reference F is not followed.
    assert node_ids(journal) == ["A", "B", "C", "D"]
    assert ("D", "F") in edge_pairs(journal)


def test_the_fill_runs_once(tmp_path):
    config = make_config(hop_depth=1, fill=1)
    crawl(tmp_path, config)

    again = FakeClient()
    SnowballCrawler(config, again, CrawlJournal(tmp_path)).run()

    assert "F" not in node_ids(CrawlJournal(tmp_path))
    assert again.requested == []


def test_without_a_fill_threshold_nothing_is_filled(tmp_path):
    journal, _ = crawl(tmp_path, make_config(hop_depth=1))

    assert node_ids(journal) == ["A", "B", "C"]


def test_max_nodes_is_exact_regardless_of_checkpoint_size(tmp_path):
    journal, _ = crawl(tmp_path, make_config(hop_depth=2), max_nodes=2, checkpoint_every=500)

    assert len(node_ids(journal)) == 2
