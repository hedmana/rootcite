# rootcite

**Trace a paper back to its structural originators in a citation graph.**

A reference list tells you who an author chose to thank. Citation counts tell you
who got popular. Neither tells you which earlier work actually made a paper
possible. `rootcite` answers that third question: given a paper, it walks the
citation graph backwards and ranks the ancestors that are structurally
load-bearing — the ones whose removal would leave the paper unreachable.

A graph neural network learns the structure. A language model turns the resulting
ancestry subgraph into a grounded narrative, cited back to the abstracts it read.

The MVP is pointed at the graph neural network literature. Nothing in `src/`
knows that: the field is a config file.

## Architecture

```
                    fields/<field>.yaml
                            |
                            v
  OpenAlex API  ->  snowball crawler  ->  citation graph snapshot
                                                  |
                            +---------------------+
                            |                     |
                            v                     v
                  GNN link-prediction     time-decayed centrality
                            |                     |
                            +----------+----------+
                                       v
                             originator score per node
                                       |
                                       v
                     LangGraph: retrieve abstracts -> narrative
                                       |
                                       v
                            FastAPI  ->  React + graph viz
```

<!-- TODO: replace with a rendered architecture diagram once the stack is wired end to end (PR 15). -->

## Field-agnostic by construction

The pipeline takes its subject matter from one YAML file:

```yaml
name: gnn
display_name: Graph Neural Networks
description: >
  Neural network architectures that operate directly on graph-structured data...
topic_id: null          # OpenAlex topic delimiting the field
seed_papers: []         # OpenAlex work IDs the crawl starts from
crawl:
  hop_depth: 2
  date_range: { start: null, end: null }
```

Pointing the tool at a different literature means writing a second file in
`fields/`, not editing the crawler, the model or the prompts.

## Running it

Each stage writes into `data/` and the next reads from there.

```sh
uv run python -m graph.crawler --field gnn      # snowball crawl from the seed papers
uv run python -m graph.build --field gnn        # clean it into a dated snapshot
uv run python -m gnn.dataset --field gnn        # cut the temporal link-prediction split
uv run python -m gnn.train --field gnn          # fit, and score on citations it never saw
uv run python -m llm.narrative --field gnn --target W2519887557
```

The last command is what the rest is for: the ancestors that carried the paper,
why, and cited to the abstracts it read. `python -m gnn.originators` prints the
same ranking without the prose, next to the unlearned baselines it has to beat.

## Serving it

The same two answers over HTTP, for a client that cannot run a snapshot itself.

```sh
uv run python -m api                            # 127.0.0.1:8000, docs at /docs
curl localhost:8000/lineage/gnn/W2519887557     # the ranking, no model involved
curl -X POST localhost:8000/narrative/gnn/W2519887557
```

Ranking is an encode pass the service pays for once per field and then reuses.
Narrating is model calls, so it is a separate route and a POST: a client can ask
what the lineage is without buying an account of why.

Nothing here authenticates. It binds loopback unless `--host` says otherwise,
and browser access is an explicit allowlist, `ROOTCITE_API_ORIGINS`, defaulting
to the Vite dev server alone.

## Using it in a browser

```sh
uv run python -m api                            # in one shell
cd frontend && npm ci && npm run dev            # in another; localhost:5173
```

Pick a field with a snapshot, give it an OpenAlex work id, and it ranks the
originators, drawn as a graph with older works to the left: hover or select one
to see the paths from the paper to it. Narrating is a separate button because
it spends model calls. The client needs Node 20.19 or later and finds the API
at `VITE_API_URL`, defaulting to `http://127.0.0.1:8000`.

## Model backends

The narrative layer talks to one interface with three backends behind it. Which
one answers is set in the environment, never in code.

| Variable                | Meaning                                       |
| ----------------------- | --------------------------------------------- |
| `ROOTCITE_LLM_PROVIDER` | `claude`, `openai` or `local`                 |
| `ROOTCITE_LLM_MODEL`    | Overrides the backend's default model         |
| `ROOTCITE_LLM_BASE_URL` | Where an OpenAI-compatible server listens     |
| `ANTHROPIC_API_KEY`     | Read by the Anthropic SDK                     |
| `OPENAI_API_KEY`        | Read by the OpenAI SDK                        |
| `ROOTCITE_LOCAL_API_KEY`| Only if your local runtime was started with one |

With nothing set the choice falls to whichever key is present, and finally to a
model on this machine, which is the one option that needs no account. Ollama,
LM Studio, vLLM and llama.cpp all serve the OpenAI chat endpoint, so any of them
works as `local`. A local backend never reads `OPENAI_API_KEY`, and plain
`http` is accepted only to a loopback address:

```sh
ollama pull llama3.1
uv run python -m llm "Reply with the single word: reachable."
```

## Layout

| Path         | Contents                                              |
| ------------ | ----------------------------------------------------- |
| `fields/`    | Field configs. The only field-specific code in the repo. |
| `src/graph/` | OpenAlex client, crawler, graph construction, baseline scorers |
| `src/gnn/`   | PyG dataset prep, training, originator scoring        |
| `src/llm/`   | Provider interface, LangGraph narrative orchestration |
| `src/api/`   | FastAPI service                                       |
| `frontend/`  | React + Vite client                                   |
| `notebooks/` | Sanity checks and exploration. Not shipped code.      |
| `data/`      | Crawl output and graph snapshots. Gitignored.         |

## Roadmap

**Phase 0 — scaffolding**
- [x] PR 1 — Repo init: skeleton, license, field config contract
- [x] PR 2 — Dev environment: packaging, lint, CI, compose stub

**Phase 1 — data pipeline**
- [x] PR 3 — OpenAlex API client
- [x] PR 4 — Snowball citation graph crawler
- [x] PR 5 — Graph construction and cleaning

**Phase 2 — GNN model**
- [x] PR 6 — PyG dataset prep with temporal link-prediction split
- [x] PR 7 — GNN training pipeline
- [x] PR 8 — Originator scoring

**Phase 3 — narrative layer**
- [x] PR 9 — Swappable LLM provider interface
- [x] PR 10 — LangGraph orchestration
- [x] PR 11 — Narrative quality pass

**Phase 4 — API and frontend**
- [x] PR 12 — FastAPI backend
- [x] PR 13 — Minimal frontend
- [x] PR 14 — Interactive graph viz

**Phase 5 — polish**
- [ ] PR 15 — Docker Compose full stack
- [ ] PR 16 — Seed dataset ship
- [ ] PR 17 — Public demo deploy

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Solo project, deliberately run with a
real workflow: an issue per change, a PR per issue, atomic commits inside it.

## License

MIT. See [LICENSE](LICENSE).
