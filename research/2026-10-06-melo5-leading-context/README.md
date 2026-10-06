# One fixed leading-context observation

All five inputs use exactly 24,000 digital-zero leading samples, the unchanged
entire original 16 kHz PCM, and 4,800 digital-zero appended samples. The original
A20 model, runtime, decoder and thresholds are unchanged. M6 is excluded.

| Saved condition | Positive clips with expected event | Nonwake clips with an event |
|---|---:|---:|
| Original short EOF | 0/2 | 0/3 |
| Prior fixed 300 ms tail | 1/2 | 1/3 |
| Fixed 1,500 ms lead + 300 ms tail | 2/2 | 1/3 |

The new events are M1 K1 (score 0.9020349616759333), M2 K2
(0.219211423605427), and false M3 K1 (0.8362519881482239). M4/M5 remain quiet.
All three events become available at sample 38,400, or original-source-relative
14,400 samples (0.900 s). The corresponding original-EOF differences are
145.3125 ms, 87.25 ms and 122.125 ms. These are input availability quantities,
not acoustic word-end latency. The M3 false activation remains.

The complete new trace contains 46 callbacks, 1,290 fbank rows, 428 computed model
rows and 427 decoder-searched rows. One row after M1 activation is not searched;
all model logits remain in the log. There are no events in the 245 pure-leading
model rows. There was one acquisition, without warmup, repetition or a sweep.

M2 gains eligible 小 support at source-relative center 72: posterior
0.03240063413977623 becomes 0.4122069180011749. Ordered row-local support for
小→窝→小→窝 appears at centers 51→66→72→84 alongside the saved K2 activation.
This does not reconstruct beam history or identify which spoken occurrence a
token represents. Leading silence changes frontend, model and decoder state
together. A common post-filter on the already emitted event scores cannot retain
M2 while removing the higher-scored false M3 event; no threshold was selected.

`original_A20.raw.jsonl` retains all scientific fields and numeric spellings,
with only the process ID removed. `observations.json` is unchanged historical
output. `report.json` preserves all three unchanged PR485 summaries, paired
observations, audit counts and scientific limitations. `manifest.json` and
`geometry.json` bind the exact input recipe. `resources.json` preserves measured
resource evidence while removing absolute host clocks. `M2-token-support.json`
contains reproducible compact row-local evidence, without duplicated full rows.

Resources are local and unpaced: collector CPU 0.161421 s and launch-to-reap wall
0.168724903 s. Native/child lifetime high-water RSS is 12,424 KiB; sampled maximum
RSS is 2,880 KiB. High-water values may include fork/pre-exec history; the cause
of the difference is unestablished. Sampling does not prove continuous bounds,
single-thread behavior for the whole lifetime, or child-process absence.

These five exposed single-voice clips do not establish real-speech generalization,
FAR/FRR, training admission, voice rights, independent-speaker validation,
embedded performance or deployment readiness. The original machine six-clip
gate remains FAIL (2/6 both-ASR plan matches); human adjudication is a separate
layer. The native 44.1 kHz clips were heard; derivatives were not independently
heard. Earlier original and tail controls remain immutable.

Saved-only verifier and source bindings are in pipeline PR #484 at
`research/melo5-leading-context-v1`. The actual scorer is separately pinned to
PR #485 commit `6f2461ff11cddf0a1264f5e2a04282c40b5dc4ce` and remains unchanged.
