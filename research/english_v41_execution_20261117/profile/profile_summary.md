# Real graph profile and static cost

Publication snapshot: 2026-11-17, before the bounded 3x1 A/B execution. The original profile derivation is retained in [profile_static_join.json](profile_static_join.json), with repository-relative source links and the original derivation hash. [static_graph_cost.json](static_graph_cost.json) preserves the complete saved static-cost input. Its `generated/export_manifest.json` path names the original generated member; that generated file is not included here.

**Decision: run one bounded A/B for the nine unpadded 3x1 embedding nodes.** Their measured CPU share is 49.20923%. No candidate has been executed yet.

Source: kws-data [run 44000006165](https://github.com/jiying2007/kws-data/actions/runs/44000006165), [artifact 14292027025](https://github.com/jiying2007/kws-data/actions/runs/44000006165/artifacts/14292027025). Exact original [graph profile](../resource_results/diagnostic/graph-profile.json), [baseline result](../resource_results/baseline/RESULT.json) and [independent review](../resource_results/independent-result-review.json) are included. The diagnostic replays the existing 16-second exposed LibriSpeech speaker-1272 splice once. All 81 graph/node keys, op IDs and 188 calls per row match the fixed graph.

## Measurement scope

The fractions below divide by the same instrumented profile total: 25.860251016 wall seconds and 25.860211495 process CPU seconds. Each node includes its finite-output check and clock overhead. The graph validation rows are separate. Ring copies, normalization, the head and callbacks are outside the node totals. This is the whole 16-second replay including warm-up, not a steady-only node profile.

The separate uninstrumented baseline remains the headline performance evidence. The diagnostic happened to finish slightly faster; that difference is run variability, not an instrumentation speedup or isolated head-time estimate.

## Graph groups

| Group | Wall s | Process CPU s | CPU share |
|---|---:|---:|---:|
| mel_nodes | 2.557464 | 2.557502 | 9.8897% |
| embedding_nodes | 23.302036 | 23.301961 | 90.1074% |
| graph_metadata_validation | 0.000751 | 0.000748 | 0.0029% |

## Operation groups

| Group | Nodes | Wall s | Process CPU s | CPU share |
|---|---:|---:|---:|---:|
| embedding_1x3_padded_width | 9 | 9.108580 | 9.108572 | 35.2223% |
| embedding_3x1_valid | 10 | 12.742373 | 12.742049 | 49.2728% |
| embedding_3x3_padded_width | 1 | 0.240304 | 0.240302 | 0.9292% |
| graph_metadata_validation | 2 | 0.000751 | 0.000748 | 0.0029% |
| mel_512tap_conv1d | 2 | 2.177320 | 2.177297 | 8.4195% |
| mel_projection_matmul | 1 | 0.069774 | 0.069781 | 0.2698% |
| non_MAC_operator | 56 | 1.521151 | 1.521463 | 5.8834% |

## Top 10 by process CPU

| Graph / node | Kind | Wall s | Process CPU s | CPU share | Effective MAC per call |
|---|---|---:|---:|---:|---:|
| embedding / 6 | embedding_3x1_valid | 2.292496 | 2.292401 | 8.8646% | 3,981,312 |
| embedding / 26 | embedding_3x1_valid | 2.251938 | 2.251699 | 8.7072% | 3,732,480 |
| embedding / 13 | embedding_3x1_valid | 2.202858 | 2.202854 | 8.5183% | 3,760,128 |
| embedding / 19 | embedding_3x1_valid | 2.071976 | 2.071952 | 8.0121% | 3,538,944 |
| embedding / 32 | embedding_3x1_valid | 2.040697 | 2.040683 | 7.8912% | 3,483,648 |
| embedding / 3 | embedding_1x3_padded_width | 1.932891 | 1.932883 | 7.4743% | 4,006,656 |
| embedding / 16 | embedding_1x3_padded_width | 1.740213 | 1.740191 | 6.7292% | 3,603,456 |
| embedding / 29 | embedding_1x3_padded_width | 1.705183 | 1.705163 | 6.5938% | 3,421,440 |
| embedding / 23 | embedding_1x3_padded_width | 1.256216 | 1.256210 | 4.8577% | 2,433,024 |
| mel / 0 | mel_512tap_conv1d | 1.093951 | 1.093936 | 4.2302% | 12,763,648 |

## Candidate and baseline

Candidate nodes: 6, 13, 19, 26, 32, 39, 45, 52, 58. Total: 12.725940281 wall seconds / 12.725610648 process CPU seconds. Node 62 has a single output position and stays on the generic fallback.

| Existing baseline input | Whole-stream wall RTF | Steady wall RTF | Steady CPU / audio-second |
|---|---:|---:|---:|
| libri-speaker-1272 | 1.636129244 | 1.735694311 | 1.735570146 |
| libri-speaker-1462 | 1.636807847 | 1.736862769 | 1.736731437 |

Amdahl scenarios using the diagnostic share alone: if the whole candidate group became 2x faster, the graph profile would become 1.326x faster; a 4x group improvement gives 1.585x. The infinite-group-speed bound is 1.969x. These are conditional calculations, not measured gains; the group includes finite checks that the candidate keeps unchanged.

Model constants plus the fixed head blob are 2,596,020 bytes (2.4757576 MiB); this excludes executable code and runtime allocations and is not process PSS. This exposed-input host replay provides no independent FAR/FRR, word-end latency, microphone or SSC305 acceptance evidence.
