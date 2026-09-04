# Six-set CPU benchmark

Run on 2026-09-04 using an Apple M4 Pro (14 CPU cores), with PyTorch restricted to four CPU
threads. MPS was not enabled. The model was `laion/larger_clap_music` at revision
`a0b4534a14f58e20944452dff00a22a06ce629d1`.

## Workload and performance

- Six MP3 DJ sets: four tided/drum-and-bass sets and two Mila Stern sets.
- 10.10 hours of evaluated source audio.
- One 2h 06m Mila Stern recording was capped to its first two hours to respect the API limit.
- Ten-second windows every 30 seconds, batch size four.
- 1,217 windows considered and retained; the conservative quality filter rejected none.

| Set | Duration | Windows | Pipeline time |
|---|---:|---:|---:|
| Dark & Minimal Vol. 1 | 91.4 min | 184 | 13.08 s |
| Deep & Soulful Vol. 1 | 81.8 min | 164 | 10.46 s |
| Deep & Soulful Vol. 2 | 102.2 min | 205 | 13.16 s |
| Mila Stern — Bucht der Träumer | 120.0 min | 241 | 15.26 s |
| Mila Stern — Fusion / Tanzwüste | 119.0 min | 239 | 15.04 s |
| Turning Tides Mix #1 | 91.6 min | 184 | 11.66 s |

Total measured pipeline time was 78.66 seconds, or 64.6 ms per retained window. Peak resident
memory was 0.93 GiB in the persistent batch process. A separate warm probe measured a cached
model load at 1.60 seconds; the first checkpoint fetch is network-dependent and excluded here.
These numbers describe this machine and PyTorch build, not a general latency guarantee.

## The aggregation finding

The normalized global means are almost collinear: within-group cosine averaged 0.998524 and
cross-group cosine averaged 0.996441. Global-only nearest-neighbour matching placed only four of
six sets with the expected creator group. In particular, Deep & Soulful Vol. 2 and Mila Stern's
Bucht der Träumer chose one another.

The original unconstrained two-cluster aggregation also produced clusters representing about
0.6% of the windows for two sets. Those were not useful summaries of a second musical mode.
Aggregation version 3 enforces a 10% minimum support during spherical assignment. PCA remains
diagnostic only and is never exported as a music embedding.

For comparison, the benchmark evaluates this symmetric cluster score:

```text
cluster_similarity = 0.5 × Σ(query_weight × best_candidate_cluster_cosine)
                   + 0.5 × Σ(candidate_weight × best_query_cluster_cosine)
```

| Ranking method | Same-group nearest neighbours | Within/cross mean margin |
|---|---:|---:|
| Global cosine | 4 / 6 (66.7%) | 0.002083 |
| Symmetric weighted clusters | 6 / 6 (100%) | 0.002289 |
| 25% global + 75% cluster | 6 / 6 (100%) | 0.002238 |

The prototype uses global cosine for cheap database candidate retrieval, then the 25/75 profile
blend for map neighbours and reranking. This is a directional six-set result, not a validated
production coefficient. Tune it on a larger held-out, human-labelled collection.

### Outlier and medoid ablation

Two additional full runs tested the embedding-space outlier proposal. Local density was measured
from each clip's nearest cosine neighbours, with a median/MAD threshold and a hard 10% cap. The
trim removed 7–10% of clips in each set.

| Aggregation variant | Same-group nearest neighbours |
|---|---:|
| Untrimmed spherical centroids with minimum support | 6 / 6 |
| Density-trimmed spherical centroids | 4 / 6 |
| Density-trimmed spherical medoids | 3 / 6 |

The medoids retained too much track- or transition-specific variance, and the density trim removed
clips that were useful for creator separation. The application therefore keeps untrimmed centroids
as the default and exposes 5%/10% density trimming only as an experimental control. Every export
records the cap, core size, and actual number removed.

## Musical map

The first two axes fitted on the six global vectors explain 74.1% and 20.9% of their variance.
Most tided sets land on one side and both Mila sets on the other, while Deep & Soulful Vol. 2 is
the important visual overlap. Map position is exploratory and collection-relative; the neighbour
label uses the full 512-dimensional profile blend, never the 2D coordinates.

## Recommended next evaluation

Add multiple creators across overlapping genres and label similarity independently of creator
identity. Compare global cosine, symmetric cluster coverage, and a catalogue-centered score using
creator-balanced Recall@K. Freeze the reranker and map reference catalogue only after evaluating on
held-out sets.
