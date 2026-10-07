# Qwen20 A20 descriptive observations: complete trace, failed supervision

**Overall run: FAILED_NO_RETRY.** The single collector invocation exited 0 and
saved all 20 clips. Its separate supervisor encountered PermissionError reading
`/proc/7/io` near exit. Saved-trace identity, completeness and descriptive counts
passed independent audit. That scoped PASS does not clear the execution failure:
final process I/O and child-process observations remain UNKNOWN. No retry occurred.

This additive record does not modify or enlarge the older frozen 341-asset
archive. It contains no later experiment. It changes no model, decoder, threshold,
shipping default, ABI or qualification flag.

## Descriptive results

- K1 你好小窝: 4/5 matching clips; Serena missed
- K2 小窝小窝: 4/5 matching clips; Eric missed
- Ten near-foil clips: one activation, Eric 你好你好 → K1
- No positive wrong-keyword activation or repeated activation
- Original positive split counts: train 6/6, development_a 1/2, development_b 1/2

All nine original events, both misses, every no-event clip, all callback results
and every saved logit remain available. The original 12 train / 4 development_a /
4 development_b asset assignment is unchanged. These are historically exposed,
historically human-reviewed stock-voice synthetic clips. Stock voice names do
not identify actual recorded human speakers. No new independent holdout exists.

These are clip observations, not product FRR, continuous FA/hour, confidence
intervals, word-tail latency, calibrated probabilities, board performance or
new numerical qualification. No training, warmup, alternate candidate, threshold
sweep or repeated acoustic pass occurred.

## Full surviving evidence

`evidence/raw.jsonl.gz` is deterministic gzip: empty filename, mtime 0, level 9.
It decompresses to the original 197,933 bytes and 317 JSONL records, SHA256
`0ae1cf74a15ba2d14f8d8994ac4cf972f76153973cddb9666043757f18e11c86`.
No raw field was removed, rewritten or selectively published.

- 559,360 samples / 34.96 seconds; 127 feeds/callbacks, 108 full + 19 short
- 20 explicit finishes; 3,454 fbank rows; 1,144 model/decoder-input rows
- 1,111 actual decoder search rows; 33 skipped after first callback activation,
  while the decoder clock still advances all 1,144 input rows
- All 6,864 logit scalars, 1,144 × 6; all original callback-final decoder results
- Explicit empty event arrays and one zero-row callback: repeat-nihao Serena,
  retained 320 + tail 320 samples, below the 800-sample whole-waveform gate
- Probability vectors, beam paths and candidate rejection reasons: NOT_CAPTURED

Event availability is consumed input samples / 16,000. Decoder frame numbers
use a 10 ms coordinate grid. Neither is an annotated word endpoint or word-tail
latency. Callback service timings include JSON writes and flushes.

Native main duration was 0.418725089 s wall / 0.419632249 s process CPU, excluding
dynamic-loader startup. Native getrusage maxRSS was 9,884 KiB and may include
pre-exec inherited high-water state. The last successful sample at 0.403481857 s
reported one thread and 2,864 KiB RSS; sampled peak was 2,932,736 bytes. Partial
I/O samples are retained without being promoted to final totals. This short x86
observation is not a benchmark or physical-board measurement.

## Genuine supervisor failure

The executed launcher read status/I/O before polling for exit. Its read_proc
caught FileNotFoundError only; optional-I/O PermissionError propagated into the
supervisor failure handler. Timing and collector return code 0 are consistent
with an exit-time race, but the exact cause is not proven. No denied process
path was retried or bypassed.

All five stored successful samples lack a children field. The original broad
FileNotFoundError catch and empty-default check did not establish child visibility.
UNKNOWN is not an observed empty list. Hard CPU/address-space/alarm/file-size
limits were configured and no crossing was recorded; failed observation still
precludes an end-to-end run-guard PASS.

`resource-record.json` is byte-exact. `execution-summary.json` is a truthful new
technical summary, retaining the original ledger hash and failure reason without
copying execution-approval records. Context removes machine nodename and replaces
workflow-specific prose with technical contention context. The public audit is
explicitly a projection, not the original audit byte stream.

## Saved-logit diagnostic and limits

`diagnostic.public.json` retains all 20 greedy diagnostics, callback argmax
positions, nonblank runs, three focus-clip scalar comparisons and every event-score
rank. Only JSON parsing, argmax, collapse, subtraction and sorting were used.
No softmax or native beam search was replayed.

- K1 Serena: blank-separated 小 repetitions and strong terminal 窝; all 58 rows
  searched. Exact native miss mechanism is unobserved
- K2 Eric: weak terminal 窝 relative to blank; all 47 rows searched. Candidate
  existence and rejection reason are unobserved
- Eric 你好你好: directly recorded K1 false event. At frames 60/63, 小 and 窝
  compete with reversed local rank order; actual accepted beam path is NOT_CAPTURED
- None of those three clips encountered the 800-sample gate, which applies to
  retained waveform plus incoming samples, not tail length alone
- False-event score 0.3225957193632025 exceeds true-hit scores 0.2544486336329864
  and 0.30555281076. This is ranking only, with no changed-threshold simulation,
  selection, calibration or hypothetical hit/false-alarm claim

Greedy equality is not a wake criterion. Repeated greedy tokens also occur in
true hits. Missing path/probability/rejection traces prevent unique attribution.
No claim is made about extra context, padding, future rows or speech beyond EOF.

## Reproduction and hash boundaries

The corresponding source-only research directory is
`jiying2007/kws-pipeline:research/native_a20_qwen20`. It contains the exact original
collector/scorer, archival non-executable launcher, explicit offline projection
adapter, scalar diagnostic derivation, tests and disabled future supervisor.
Use the tools commit that pins this record's immutable commit and manifest hash.
No new code commit is implied by this data record alone.

With both repositories checked out offline, run from kws-pipeline:

```sh
python3 -B research/native_a20_qwen20/validate.py \
  --data-root ../kws-data/research/2026-10-04-qwen20-descriptive
```

This verifies complete file inventories, exact raw hash, input identities,
geometry, state gates, every descriptive result and all numerical diagnostic
arrays. It performs no model/audio/native-library/process execution. The frozen
scorer is unchanged. The input projection replaces reviewer receipt objects with
immutable public annotation references, changing its hash; the adapter changes
only that hash label in a separate in-memory raw view. All numerical records,
labels, splits, order and gates remain unchanged. Original hash labels are
references, not recomputed hashes of the omitted metadata. The archived raw is
never rewritten. The public protocol projection was never acoustically executed.

`EVIDENCE-MANIFEST.json` inventories all data-package files except itself.
`PROVENANCE.json` maps original/public sizes, SHA256 and exact transformations,
including unchanged source files housed in kws-pipeline. The containing reviewed
Git commit is the inventory trust anchor; self-consistency alone is insufficient.

Native source is pinned at kws-pipeline commit
`804553286fd9caa294769cc4d44cc0c43dc66e88`; library SHA256 is
`68457311f93540405dfa0bf5fdc1cd74421fea22b44d5920fec465be5de4d912`.
The 1,565,280-byte A20 float payload SHA256 is
`a80b233d3c958d25f49085f487ef5cde839bd69f803ed057b59bbd20adc6e43d`.
See the [immutable full A20 archive](https://github.com/jiying2007/kws-data/tree/7af8f8597b0b7fbe8761c9a2400f2e4028395551/research/2026-10-03-native-a20-packages)
and [immutable Qwen20 inputs](https://github.com/jiying2007/kws-data/tree/d9a1f3171bceae45908bc8ac84045d3bca23f43f/datasets/qwen3-reviewed-v1).
No model weights, native runtime files, library/collector binaries or WAV/PCM
payloads are duplicated here. EXTERNAL-REFERENCES.json gives source-level pins.

## Rights and scope

Numeric outputs are newly recorded research evidence. Existing asset-specific
rights and third-party source notices remain in the immutable referenced sources.
Public synthetic audio availability is not blanket commercial clearance or
real-person endorsement. The upstream Qwen model license does not automatically
license all generated audio. No new output relicense is asserted.

The separately reviewed future supervisor remains unconditionally disabled. Its
26 author and seven independent pure-mock tests do not establish live host guard
readiness. Exceptional cleanup and general filesystem/write failures still need
integration review; a kill/exit race there may interrupt final evidence writing.
Future integration cannot retroactively repair this failed historical attempt.

### Optional exact original input-metadata reconstruction

Supply the already-public `datasets/qwen3-reviewed-v1/audio-review.jsonl` from
input commit `d9a1f3171bceae45908bc8ac84045d3bca23f43f` with the validator's
`--annotations` option. The tool verifies its exact file and per-line hashes,
restores the omitted receipt objects in memory, and verifies the reconstructed
original manifest SHA256 `75b322f33857152bc59d4700cba9b85dd818a36408d0e10a4fe837926b78ed58`.
It writes nothing and still performs no acoustic execution. The annotation file
is deliberately referenced rather than duplicated here.
