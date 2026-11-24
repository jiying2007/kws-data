# B17 authenticated paired TRAIN evidence — 2026-11-24

B17 completed its single registered public Actions training run. Both arms start
from the deployed B0 checkpoint, receive the same 192-step schedule and positive
retention objective, and differ only in ordinary TRAIN margin-ceiling weight
(control 0; challenger 0.08). This evidence authorizes no model selection or
deployment. The saved result is `TRAINED_PAIR_REQUIRES_FRESH_INDEPENDENT_DEV`.

## Original execution and artifact

| Item | Authenticated value |
| --- | --- |
| Public run | [45319580193](https://github.com/jiying2007/kws-data/actions/runs/45319580193), attempt 1, success |
| Run head | `9654cfe1c28ed84fe7d1d2a91dab450d8efb1abf` |
| Actual single predecessor | `e46a54fb3e39b3a7a7abd8e8b1afdc00d1ec8cde` |
| Original artifact | [14887048023](https://github.com/jiying2007/kws-data/actions/runs/45319580193/artifacts/14887048023), 2,506,252 bytes, 18 members |
| Original ZIP SHA256 | `1e31bd9c801657c7850fa6fb05c5468cbe8b4ab5bb459b0146c07c4c905ff86d` |
| Frozen experiment SHA256 | `f9ba0ce59b0deed805b313063e648470e61af6c9122d08f6fe9aa5ea9e324831` |
| Frozen prospective assessment SHA256 | `7fdfe9cd47b18f64429fece944468af8adfde6c6ba012a57236a8a33aae1a948` |

`RECEIPT.json` records the original server run, commit and artifact metadata,
every original member's byte count and SHA256, and the six exported model hashes.
`AUDIT.json` records the saved-data verification result. Model, audio and logit
blobs remain in the authenticated original Actions artifact.

## Observed TRAIN result

The prerequisite diagnostic was evaluated before optimization on the 79 existing
synthetic TRAIN originals and nine exact owner-reviewed natural windows from
eight readers. The verifier recomputes all 528 saved arrays' hashes, greedy
CTC tokens and complete `[1,2,3]` events, and B9-minus-B0 margin measurements.
It confirms the registered mechanism gate: 43 ordinary source groups show
positive margin elevation, and 11 positive source groups include a B9 complete
event on an approved original/gain where B0 misses. The registered minima were
two ordinary groups and one positive group. These observations concern TRAIN
mechanism evidence.

| Evidence | Observed and verified |
| --- | --- |
| New prerequisite diagnostic forwards | 396 |
| Reused saved B9 positive arrays | 132 |
| Saved diagnostic arrays | 528 |
| Positive retention | 120 eligible original/gain conditions; 43 originals; 34 groups |
| Unique ordinary B0 teacher arrays | 124, matching the sampled ordinary source/gain/prefix combinations |
| Actual updates | 192 per arm; 384 total, with complete source-matched traces |
| Student forwards | 1,832 optimization + 2 checkpoint reload parity + 1 ONNX export trace per arm; 3,670 total |
| Trained parameters | 43,876 finite parameters per arm |
| Arm parameter differences | 43,870 float values differ |
| PT and NPZ numerical identity | Exact for all 20 state tensors in each arm |
| Required runtime tests | 12 passed after the pinned Torch installation; zero numerical skips |
| DEV/CAL/LOCKED forwards and new TTS/ASR calls in training | 0 |

The saved model identities are:

| Model | PT SHA256 |
| --- | --- |
| B0 initialization and ordinary teacher | `1ae8a2b20c8b6991f925dd4c626af2f36b308adee8733a3a7215818716f565d7` |
| B9 positive teacher | `7dba86a3fcda6e5f05cb80b414dc372c70a5a03b3aee783f168441a4871e7655` |
| B17 control | `144370e96fcf58da50114280d728867781934e3c6538d2b2d769babb7934c81e` |
| B17 challenger | `61c79221d2cc2299929b18812cd4bce2ef0383e3505ad017ed90bf625401bc07` |

## Reproduce the saved-data audit

From a checkout containing this directory, with NumPy available, provide the
unmodified original artifact ZIP:

```sh
python -IB research/english_v6_b17_training_20261124/verify_saved.py /path/to/b17-train-original.zip
```

The verifier prints a recomputed JSON result. Use `--output /path/to/audit.json`
to save that result explicitly. An optional `--repository /path/to/checkout`
supports running a staged copy against its checkout; the default repository is
inferred from this file's normal repository location. No machine-specific path
or current-working-directory import is required.

Verification checks the frozen receipt and entire original ZIP before reading
its members, validates all 18 member hashes and lengths, compares the 15 input
ZIP hash receipts against frozen source pins, authenticates the existing owner
review, and checks all scheduled source identities and roles. It independently
recomputes the saved diagnostic decisions, every B9 positive retention span and
ordinary teacher grid, both complete update traces, the forward-count arithmetic,
the numerical PT/NPZ tensor identities and actual differences between arms.
PT decoding is restricted to known CPU float storage and state-dictionary tensor
constructors; it imports no Torch module and constructs no model.

The audit runs zero model forwards and zero optimizer updates. It checks saved
original input hash receipts and owner-reviewed identities without downloading
the historical audio again. It does not repeat original PCM feature extraction
or the historical matched-teacher forwards. Independent development and
replication quality gates, C11/ARM parity, SSC305 hardware execution and field
false-alarm measurement require their own evidence. `promotion_allowed` remains
`false` throughout this training record.
