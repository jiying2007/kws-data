# A20 source-domain observations and decoder diagnostics

One additive scientific record, with four closed phases. This is research evidence,
not a product qualification, new dataset admission, model release or decoder change.
All four raw JSONL files are exact saved bytes. Reports labeled PUBLIC_PROJECTION
are selected metadata/report fields with separate original and public hashes;
they are not original execution artifacts. Historical PREPARED/PENDING statuses
are preserved alongside later audit outcomes rather than rewritten as PASS.

## Results and limits

| Closed phase | Saved observation | What it does not establish |
| --- | --- | --- |
| DEMAND DLIVING ch01 | One original-amplitude 300.004 s stream; 1,001 callbacks, 9,999 model/search rows, 0 activations | Source label UNKNOWN, zero verified-negative hours; no FAR/Poisson bound or wall-time soak |
| FLEURS20 admission, PCM16 derivation, once A20 | 20 human read-speech clips, 247.56 s, 19 sentence groups; 835 callbacks, 24,716 fbank row counts, 8,232 model rows / 8,226 searched | 20 per-clip resets, not one continuous session; no speaker independence, unseen holdout or pretraining-disjointness claim |
| Exact Qwen3case reconstruction | 3 exposed clips; 128 saved FP32 logit rows, 127 searched rows; all 14 callback scores bit-identical | Historical probabilities/beam state were NOT_CAPTURED; the new trace is conditional reconstruction. Original Qwen20 remains FAILED_NO_RETRY |
| Six invented path-witness fixtures | 91 retained pb/pnb states exact; 7 match/reject checks; 5 events; score-formula error 0 ULP | Establishes a gap in a stronger same-path representative-tuple contract; does not establish an original port bug or FAR/FRR improvement |

FLEURS events remain UNADJUDICATED: row16 K1 at 15.3 s, score
0.12134267955397253; row17 K2 at 6.3 s, score 0.14145915542700957.
Times are per-clip input availability, not word-end latency. Sentence 1519 is
shared by rows 7 and 17; sentence groups are not speaker groups. Speaker IDs and
pretrained overlap remain unknown. Target-text absence is only transcript evidence.

Qwen's two misses never formed the full keyword candidate; no complete keyword
candidate was rejected by the activation guard. The repeated-nihao event was reproduced at rank 7/8 with representative
frames 45/54/63/63 and score 0.32259571936320253. A valid merged CTC prefix can lack
a joint same-path explanation for all representative node frames/probabilities.
The rank-2 positive and legitimate late-peak controls remain valid. Persistent
rejected-match score history is a separate factor from CTC prefix mass. The
existing decoder is unchanged. Immutable alignment backpointers are only a future
candidate idea, not implemented or validated by this record. The six-fixture
native diagnostic is closed; there is no seventh core fixture.

Original Qwen20 and strict N0 v1 failures are retained by pinned references.
Historical monitoring still has lifetime child absence NOT_PROVEN and terminal
lifetime I/O UNKNOWN where recorded. Host timing includes the documented collector
and output scopes; it does not qualify a board, ARM deployment, AFE or real-time
capture. No new performance measurement is made here.

## Reproduce only the saved-record checks

The companion source is `research/native_a20_domain_diagnostics` in kws-pipeline.
With the two repository checkouts available locally:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 kws-pipeline/research/native_a20_domain_diagnostics/src/verify_saved.py --data-root kws-data/research/2026-10-04-domain-decoder-diagnostics
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s kws-pipeline/research/native_a20_domain_diagnostics/tests -v
```

These commands check hashes/schema/counts/events, the decimal reconstruction's
rational cells, saved callback scores and an exact Cartesian CTC oracle for the
six already frozen invented tables. They do not load audio/features/weights,
compute softmax, launch a native program, replay a decoder, retrain or probe a host.
The 28 adversarial tests (27 original plus one publication-gate test) are tiny invented validator tests, not new native fixtures.
The public package cannot reproduce the waveform-to-logit forward pass, samplewise
conversion audit, original resource monitoring, or audio absence adjudication.
See [reproducibility map](REPRODUCIBILITY.json), [rights and privacy](RIGHTS-SOURCES-PRIVACY.md),
[full raw field review](RAW-SCHEMA-REVIEW.json), and [original/public identity map](PROVENANCE.json).
The byte manifest proves internal integrity; trust requires externally pinning
its hash or the published commit. Neither a manifest nor a hash proves rights or
scientific truth.
