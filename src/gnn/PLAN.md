# GNN improvement plan

Written 2026-09-29 against snapshot `gnn-20260924`.

## Baseline (held-out AUC)

| Scorer | val | test |
|---|---|---|
| GNN | 0.661 | 0.546 |
| Recency (-year gap) | 0.714 | 0.776 |
| Target in-degree | 0.509 | 0.364 |

Why the GNN loses:
- Val/test sources have no out-edges in their message graph (cold start); every train positive is a message edge. Best epoch 3 of 23.
- Negatives are sampled once; degree features include supervision edges.
- Uniform negatives make recency sufficient.
- `learned_flow` weights a paper's existing references; nothing measures that.
- Crawl is backward-only, capped at 3000 works, 81% of edges dangle. Train cut is 2010, so it mostly learns pre-GNN citation habits.

## Gates

1. Proxy: GNN beats heuristics on link prediction.
2. Product: `learned_flow` ranks true originators above the `graph.score` baselines.

## Steps

One PR each, named for what it does.

- [ ] **Link evaluation against heuristics.** Heuristic scorers (recency, in-degree, logistic regression on gap/degree/age). Year-matched negatives, uniform kept as secondary. Per-source MRR and Hits@10 over ~100 candidates. 5 seeds, 3 rolling cut years.
- [ ] **Cold-start training.** Per epoch, hide out-edges of ~20% of train sources and supervise on them. Resample negatives per epoch. Degree features from message edges only. Year gap as decoder pair feature. `topic_id` embedding. Then a small sweep on val MRR. Exit: beat the best heuristic beyond seed spread, else go to crawl expansion.
- [ ] **Originator ranking evaluation.** Gold lineages for 20-30 targets in `fields/gnn.gold.yaml`, from survey history sections. nDCG@10 and Recall@10 vs all baselines. Ablation: `learned_flow` with uniform edge weights. Stability: top-10 overlap across seeds and 10% edge dropout.
- [ ] **Citation influence labels.** Semantic Scholar `isInfluential` per citation; per-paper AUC of the decoder on influential vs incidental references.
- [ ] **Influence-aligned objective.** Fine-tune the decoder listwise on influential references, link prediction as pretraining. Validate on the gold set.
- [ ] **Crawl expansion** (parallel, after link evaluation). Raise `max_nodes`, add forward expansion via `citing_works` filtered by `topic_id`. Rerun link evaluation.

Optional: blinded LLM pairwise judge, trusted only after it agrees with the gold set.

## Open decisions

- Semantic Scholar as a second data source (citation influence labels).
- Who verifies the gold set (originator ranking evaluation).
- Abstract embeddings need a new dependency; deferred until originator ranking evaluation shows text is the bottleneck.
