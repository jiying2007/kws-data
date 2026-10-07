# Latest human correction: 019 / clip-0004

On 2026-10-02 at 01:08:58 UTC, the user corrected the listening judgment:
“019发音不标准，听起来像小屋小窝，不是小窝小窝”.
This changes only 019's complete 小窝小窝 presence from positive to negative.
Its other target stays negative; every other human bit is unchanged.
The resulting labels are **8 positive, 18 negative, 6 unknown** (26 known).

Replaying the same two text scorers on the same 32 model outputs gives:

- Frozen v1: 0/26 covered for both models, all known bits unknown
- Post-hoc v2 Qwen: 14/26 covered; 13 true negatives, 1 false negative,
  0 true positives/false positives; 12 unknown (7 positive, 5 negative)
- Post-hoc v2 SenseVoice: 13/26 covered; 13 true negatives, 0 false negatives,
  0 true positives/false positives; 13 unknown (8 positive, 5 negative)
- Paired: 12 both covered, **0 jointly wrong**, 11 both unknown, 3 one-unknown
  disagreements. The remaining false negative is Qwen on 008-Serena / clip-0001

The earlier shared-false-negative conclusion about 019 is withdrawn. Both
models' full-keyword absence decisions agree with the corrected human judgment.
Their raw spellings still differ from that listening description. The user's
pronunciation observation does not establish a complete acoustic transcript,
measured tones, timing, or an ASR-wide quality claim.

The original labels, v1/v2 results and raw ASR outputs remain preserved as a
historical layer. `human-overlay.json` records the one-bit correction and its
basis. `human-labels.corrected.json` and `scientific-manifest.corrected.json` are
separate derived evaluation inputs. `recompute.py` runs the unchanged scorers;
it never imports an ASR runtime, changes raw output, or runs model inference.

Only the corrected evaluation manifest hash changes in each model envelope.
All 32 `model_evidence` rows compare equal to their previous evaluation, and every
readout record other than 019 compares equal. The original execution manifest
continues to identify the historical pre-forward declaration. The generic
`predeclared_human` field inside scorer readouts refers to the supplied evaluation
manifest; this correction was received after model execution and must not be
misread as an annotation frozen before that execution.

Start with `compact-summary.json`, `human-overlay.json` and
`evaluation-provenance.json`. The two full readouts are under `frozen-v1/` and
`posthoc-v2/`. These are selected development probes, and v2 remains post-hoc.
Absence of covered errors for SenseVoice does not establish automatic labeling:
all eight known positives still receive unknown under this conservative policy.
