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
- [ ] PR 1 — Repo init: skeleton, license, field config contract
- [ ] PR 2 — Dev environment: packaging, lint, CI, compose stub

**Phase 1 — data pipeline**
- [ ] PR 3 — OpenAlex API client
- [ ] PR 4 — Snowball citation graph crawler
- [ ] PR 5 — Graph construction and cleaning

**Phase 2 — GNN model**
- [ ] PR 6 — PyG dataset prep with temporal link-prediction split
- [ ] PR 7 — GNN training pipeline
- [ ] PR 8 — Originator scoring

**Phase 3 — narrative layer**
- [ ] PR 9 — Swappable LLM provider interface
- [ ] PR 10 — LangGraph orchestration
- [ ] PR 11 — Narrative quality pass

**Phase 4 — API and frontend**
- [ ] PR 12 — FastAPI backend
- [ ] PR 13 — Minimal frontend
- [ ] PR 14 — Interactive graph viz

**Phase 5 — polish**
- [ ] PR 15 — Docker Compose full stack
- [ ] PR 16 — Seed dataset ship
- [ ] PR 17 — Public demo deploy

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Solo project, deliberately run with a
real workflow: an issue per change, a PR per issue, atomic commits inside it.

## License

MIT. See [LICENSE](LICENSE).
