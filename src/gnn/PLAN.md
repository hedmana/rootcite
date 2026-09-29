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

On the served expanded snapshot (`gnn-20260929`, 9248 works, tested after 2022, 2023, 2024), trained for up to 600 epochs with patience 100:

| Sampler | Scorer | AUC | MRR | Hits@10 |
|---|---|---|---|---|
| uniform | model | 0.930 ± 0.003 | 0.515 ± 0.009 | 0.784 ± 0.009 |
| uniform | recency | 0.530 ± 0.003 | 0.029 ± 0.002 | 0.030 ± 0.012 |
| uniform | logistic | 0.915 ± 0.004 | 0.491 ± 0.012 | 0.749 ± 0.016 |
| year_matched | model | 0.887 ± 0.010 | 0.446 ± 0.010 | 0.723 ± 0.008 |
| year_matched | logistic | 0.857 ± 0.005 | 0.433 ± 0.013 | 0.718 ± 0.007 |

Paired MRR over logistic: uniform +0.024 ± 0.008, year-matched +0.013 ± 0.003, 15/15 wins each; AUC wins 15/15 too. The proxy gate holds on the expanded snapshot. On a graph this dense, one step from a random start ranks by popularity alone, and learning more dips below that before it rises past it; with patience 40, training stopped at that first step, and the model only tied popularity.

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
- [x] **Originator ranking evaluation.** `gnn.lineages` against `fields/gold/gnn.yaml`: 23 papers and 63 originators, each checked against how the paper cites the work, split into `dev` (may steer changes) and `test` (report only) before any tuning. The walk now also follows shared subject (`gnn.content`, TF-IDF over titles and abstracts), chosen on dev only. Served snapshot:

  | Scorer | dev found | dev Recall@10 / nDCG@10 | test found | test Recall@10 / nDCG@10 |
  |---|---|---|---|---|
  | learned (model and content, served) | 23/32 | 0.903 / 0.717 | 17/31 | 0.626 / 0.556 |
  | model only | 22/32 | 0.806 / 0.650 | 16/31 | 0.523 / 0.340 |
  | equal-weight walk | 22/32 | 0.806 / 0.571 | 14/31 | 0.474 / 0.344 |
  | time_decayed_pagerank | 21/32 | 0.778 / 0.567 | 14/31 | 0.474 / 0.303 |
  | gateway | 5/32 | 0.106 / 0.090 | 4/31 | 0.167 / 0.080 |
  | in_degree | 2/32 | 0.097 / 0.092 | 2/31 | 0.076 / 0.038 |
  | path_weight | 0/32 | 0 / 0 | 0/31 | 0 / 0 |

  Learned top-10 overlap across seeds 0.94 dev, 0.97 test; with 10% of citations dropped 0.92 and 0.93. Tried on dev and left out: walk damping, age half-life and sharpening the model's weights (a hit either way, noise at 12 papers), and Semantic Scholar's influence rate per work (Adam's is higher than ChebNet's). `path_weight` ranks 1980s foundations first, the failure time decay exists to prevent.
- [x] **Recovered citations.** `graph.semanticscholar`, an optional stage between crawl and build: 6811 MAG-era works' Semantic Scholar references matched back to crawled works by MAG id, else by title and year. 22582 citations matched, 6808 of them lost from OpenAlex; GCN regains Bruna, Henaff and Weston. Served.
- [x] **Citation influence labels.** 5202 Semantic Scholar `isInfluential` labels from 141 non-gold papers. The decoder's plausibility alone predicts them at AUC 0.70, hand features at 0.71, and the model's embeddings add nothing. The flag mostly counts how often a paper mentions a work, so tools earn it too.
- [x] **Influence-aligned objective.** Not adopted. Walk weights from a model of those labels tie content weighting on dev (24 against 23 of 32 found) and add nothing on top of it, for an extra pipeline stage and a rate-limited API at ranking time.
- [x] **Crawl expansion.** Uncapped backward crawl, plus `forward_depth: 1`: the in-topic works citing the seeds. 2789 works and 15k citations became 9249 and 104k. GAT is seeded by its primary record. Cleaning re-dates 262 works OpenAlex dates years after their citers, and a year heavier than its split's share no longer empties the split after it. Gold originators reachable from their targets: 88/94 to 90/94. The other four are references OpenAlex holds under dead ids (about 12% of all references), among them GCN's to Bruna, Henaff and Planetoid.

Optional: blinded LLM pairwise judge, trusted only after it agrees with the gold set.

## Open decisions

- Gold set verification, then commit `fields/gold/gnn.yaml` and rerun `gnn.lineages`.
- Abstract embeddings need a new dependency; deferred until originator ranking evaluation shows text is the bottleneck.
