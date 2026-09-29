# GNN improvement plan

Written 2026-09-29 against snapshot `gnn-20260924`.

## Baseline

`python -m gnn.evaluate`: 5 seeds x 3 cuts (tested after 2008, 2010, 2013), 100 negatives per citation, mean ± std.

| Sampler | Scorer | AUC | MRR | Hits@10 |
|---|---|---|---|---|
| uniform | model | 0.594 ± 0.060 | 0.122 ± 0.033 | 0.251 ± 0.052 |
| uniform | recency | 0.714 ± 0.050 | 0.153 ± 0.070 | 0.375 ± 0.121 |
| uniform | logistic | 0.748 ± 0.043 | 0.189 ± 0.036 | 0.403 ± 0.097 |
| year_matched | model | 0.603 ± 0.032 | 0.136 ± 0.026 | 0.251 ± 0.035 |
| year_matched | logistic | 0.611 ± 0.017 | 0.158 ± 0.016 | 0.279 ± 0.026 |

The model is below a logistic regression on year gap and target in-degree on every metric.

Why the GNN loses:
- Val/test sources have no out-edges in their message graph (cold start); every train positive is a message edge. Best epoch 3 of 23.
- Negatives are sampled once; degree features include supervision edges.
- Uniform negatives make recency sufficient.
- `learned_flow` weights a paper's existing references; nothing measures that.
- Crawl is backward-only, capped at 3000 works, 81% of edges dangle. Train cut is 2010, so it mostly learns pre-GNN citation habits.

## Gates

1. Proxy: the model beats the logistic baseline on MRR under both samplers, beyond seed spread.
2. Product: `learned_flow` ranks true originators above the `graph.score` baselines.

## Steps

One PR each, named for what it does.

- [x] **Link evaluation against heuristics.** `gnn.evaluate`: recency, in-degree and a logistic regression on both. Uniform and year-matched negatives. AUC, and MRR and Hits@10 over 100 negatives per citation. 5 seeds, 3 rolling cuts.
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
