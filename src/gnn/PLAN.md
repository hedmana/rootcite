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
  | learned (model and content, served) | 23/32 | 0.837 / 0.695 | 17/31 | 0.614 / 0.542 |
  | model only | 24/32 | 0.806 / 0.594 | 17/31 | 0.553 / 0.344 |
  | equal-weight walk | 21/32 | 0.754 / 0.514 | 12/31 | 0.402 / 0.308 |
  | time_decayed_pagerank | 22/32 | 0.766 / 0.536 | 13/31 | 0.447 / 0.289 |
  | gateway | 4/32 | 0.079 / 0.070 | 3/31 | 0.121 / 0.096 |
  | in_degree | 3/32 | 0.181 / 0.175 | 3/31 | 0.121 / 0.109 |
  | path_weight | 0/32 | 0 / 0 | 0/31 | 0 / 0 |

  On the filled-in graph, with ChebNet's title corrected (its OpenAlex record carried a lecture's title and abstract, which content weighting read as off-topic). Before the fill the learned ranking found 24/32 dev and 17/31 test, nDCG 0.756 and 0.561. Learned top-10 overlap across seeds 0.98 on both; with 10% of citations dropped 0.92 dev, 0.86 test. Tried on dev and left out: walk damping, age half-life and sharpening the model's weights (a hit either way, noise at 12 papers), and Semantic Scholar's influence rate per work (Adam's is higher than ChebNet's). `path_weight` ranks 1980s foundations first, the failure time decay exists to prevent.
- [x] **Recovered citations.** `graph.semanticscholar`, an optional stage between crawl and build: 6811 MAG-era works' Semantic Scholar references matched back to crawled works by MAG id, else by title and year. 22582 citations matched, 6808 of them lost from OpenAlex; GCN regains Bruna, Henaff and Weston. Served.
- [x] **Citation influence labels.** 5202 Semantic Scholar `isInfluential` labels from 141 non-gold papers. The decoder's plausibility alone predicts them at AUC 0.70, hand features at 0.71, and the model's embeddings add nothing. The flag mostly counts how often a paper mentions a work, so tools earn it too.
- [x] **Influence-aligned objective.** Not adopted. Walk weights from a model of those labels tie content weighting on dev (24 against 23 of 32 found) and add nothing on top of it, for an extra pipeline stage and a rate-limited API at ranking time.
- [x] **Fill-in of cross-field works.** `crawl.fill_cited_by: 20` fetches, without following, the uncrawled works 20 or more crawled works cite: 1282 candidates, 827 alive, 10332 works in the raw crawl. Recovered citations rise from 6808 to 10344. Gold originators reachable: dev 28/32 to 31/32, test 30/31 to 31/31: Bahdanau, Interaction Networks, Kearnes and DistMult. Found stays 17/31 on test and drops one on dev. The fill also brings in heavily cited datasets and baselines (Planetoid, QM9, TransE, RESCAL), which compete for the same places; telling "built on" from "compared against" needs how a paper cites a work, not whether it does.
- [x] **Crawl expansion.** Uncapped backward crawl, plus `forward_depth: 1`: the in-topic works citing the seeds. 2789 works and 15k citations became 9249 and 104k. GAT is seeded by its primary record. Cleaning re-dates 262 works OpenAlex dates years after their citers, and a year heavier than its split's share no longer empties the split after it. Gold originators reachable from their targets: 88/94 to 90/94. The other four are references OpenAlex holds under dead ids (about 12% of all references), among them GCN's to Bruna, Henaff and Planetoid.

Optional: blinded LLM pairwise judge, trusted only after it agrees with the gold set.

## Open decisions

- Gold set verification, then commit `fields/gold/gnn.yaml` and rerun `gnn.lineages`.
- Abstract embeddings need a new dependency; deferred until originator ranking evaluation shows text is the bottleneck.
