# GNN improvement plan

Written 2026-09-29 against snapshot `gnn-20260924`.

## Results

`python -m gnn.evaluate`: 5 seeds x 3 cuts (tested after 2008, 2010, 2013), 100 negatives per citation, mean ± std.

| Sampler | Scorer | AUC | MRR | Hits@10 |
|---|---|---|---|---|
| uniform | model, before | 0.594 ± 0.060 | 0.122 ± 0.033 | 0.251 ± 0.052 |
| uniform | model, cold-start | 0.810 ± 0.030 | 0.199 ± 0.027 | 0.443 ± 0.057 |
| uniform | recency | 0.714 ± 0.050 | 0.153 ± 0.070 | 0.375 ± 0.121 |
| uniform | logistic | 0.748 ± 0.043 | 0.189 ± 0.036 | 0.403 ± 0.097 |
| year_matched | model, before | 0.603 ± 0.032 | 0.136 ± 0.026 | 0.251 ± 0.035 |
| year_matched | model, cold-start | 0.684 ± 0.027 | 0.190 ± 0.009 | 0.341 ± 0.028 |
| year_matched | logistic | 0.611 ± 0.017 | 0.158 ± 0.016 | 0.279 ± 0.026 |

`logistic` is a logistic regression on year gap and target in-degree. Paired MRR over it, per run: uniform +0.010 ± 0.013 (11/15 wins), year-matched +0.031 ± 0.013 (15/15). The proxy gate holds under year-matched negatives; under uniform only AUC clears it (15/15), MRR is within spread.

On the expanded snapshot (`gnn-20260929`, 9249 works, tested after 2022, 2023, 2024):

| Sampler | Scorer | AUC | MRR | Hits@10 |
|---|---|---|---|---|
| uniform | model | 0.926 ± 0.014 | 0.523 ± 0.012 | 0.783 ± 0.018 |
| uniform | recency | 0.527 ± 0.003 | 0.029 ± 0.002 | 0.029 ± 0.012 |
| uniform | logistic | 0.918 ± 0.004 | 0.506 ± 0.012 | 0.757 ± 0.016 |
| year_matched | model | 0.879 ± 0.016 | 0.447 ± 0.014 | 0.729 ± 0.009 |
| year_matched | logistic | 0.857 ± 0.005 | 0.444 ± 0.009 | 0.725 ± 0.005 |

Paired MRR over logistic: uniform +0.017 ± 0.013 (14/15), year-matched +0.002 ± 0.006 (10/15). In the GNN era citations pile onto a few hubs, so popularity alone nearly matches the model once the year is matched, and recency is worthless.

Why the model lost before:
- Val/test sources have no out-edges in their message graph (cold start); every train positive was a message edge. Best epoch 3 of 23. Fixed by cold-start training.
- Negatives were sampled once; degree features included supervision edges. Fixed by cold-start training.
- Uniform negatives make recency sufficient.
- `learned_flow` weights a paper's existing references; nothing measures that.
- Crawl was backward-only and capped at 3000 works, so training cut at 2010 and learned pre-GNN citation habits. Fixed by crawl expansion, except references OpenAlex holds under dead ids.

## Gates

1. Proxy: the model beats the logistic baseline on MRR under both samplers, beyond seed spread.
2. Product: `learned_flow` ranks true originators above the `graph.score` baselines.

## Steps

One PR each, named for what it does.

- [x] **Link evaluation against heuristics.** `gnn.evaluate`: recency, in-degree and a logistic regression on both. Uniform and year-matched negatives. AUC, and MRR and Hits@10 over 100 negatives per citation. 5 seeds, 3 rolling cuts.
- [x] **Cold-start training.** Each epoch cuts 20% of citing papers out of the graph and supervises only their citations, against fresh uniform negatives. Degree features are counted over the graph the model is shown. Tried and dropped, MRR against plain cold-start:
  - Mixed uniform and year-matched negatives: -0.018 uniform.
  - Year-gap and popularity decoder terms: +0.004 to +0.006, within noise, and they change the checkpoint format.
  - Hashed title words: -0.02, overfits.
  - `topic_id`: +0.06, but it leaks. OpenAlex assigns topics from citations as of the crawl, the reason `cited_by_count` is excluded.
  - Held-out share, learning rate, depth, width, dropout, weight decay, and MRR instead of AP for checkpoint selection: nothing beyond noise.
- [ ] **Originator ranking evaluation.** Gold lineages for 20-30 targets in `fields/gold/gnn.yaml` (not `fields/*.yaml`, which the API lists as fields), from survey history sections. nDCG@10 and Recall@10 vs all baselines. Ablation: `learned_flow` with uniform edge weights. Stability: top-10 overlap across seeds and 10% edge dropout.
- [ ] **Citation influence labels.** Semantic Scholar `isInfluential` per citation; per-paper AUC of the decoder on influential vs incidental references.
- [ ] **Influence-aligned objective.** Fine-tune the decoder listwise on influential references, link prediction as pretraining. Validate on the gold set.
- [x] **Crawl expansion.** Uncapped backward crawl, plus `forward_depth: 1`: the in-topic works citing the seeds. 2789 works and 15k citations became 9249 and 104k. GAT is seeded by its primary record. Cleaning re-dates 262 works OpenAlex dates years after their citers, and a year heavier than its split's share no longer empties the split after it. Gold originators reachable from their targets: 88/94 to 90/94. The other four are references OpenAlex holds under dead ids (about 12% of all references), among them GCN's to Bruna, Henaff and Planetoid.

Optional: blinded LLM pairwise judge, trusted only after it agrees with the gold set.

## Open decisions

- Serve the expanded snapshot: move `data/expanded/gnn/snapshots/gnn-20260929` into `data/gnn/snapshots`, then `gnn.dataset` and `gnn.train`.
- On the expanded snapshot the model only ties popularity under year-matched negatives: revisit training there, or let originator ranking evaluation decide.
- Semantic Scholar as a second data source: citation influence labels, and the references OpenAlex holds under dead ids.
- Who verifies the gold set (originator ranking evaluation).
- Abstract embeddings need a new dependency; deferred until originator ranking evaluation shows text is the bottleneck.
