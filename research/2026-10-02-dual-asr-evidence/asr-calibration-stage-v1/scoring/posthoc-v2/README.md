# Historical first-v2 overview

This overview preserves the earlier label set. The later user correction to
019 changes its 小窝小窝 label to negative and withdraws the shared-false-negative
claim. See the publication root README and human-review-019-v1 for the latest
8-positive / 18-negative / 6-unknown labels and recomputed results.

# Post-hoc target-local diagnostic, v2

This is a separately versioned diagnostic of the already completed dual-ASR run.
It is not an untouched-holdout qualification. No model was rerun, and no raw text,
token, WAV, human label, original rule, or original readout was changed.

## Result on 26 retained known human target bits

- Qwen3-ASR-0.6B: 14/26 covered (53.8%); 12 true negatives, 2 false negatives,
  0 true positives and 0 false positives; 12 unknown (7 positive, 5 negative)
- SenseVoiceSmall: 13/26 covered (50.0%); 12 true negatives, 1 false negative,
  0 true positives and 0 false positives; 13 unknown (8 positive, 5 negative)
- Both cover 12 known bits, including one shared false negative; 11 known bits
  are jointly unknown and 3 have one-unknown disagreements
- All 6 human-unknown bits remain unknown in final human labels

These counts describe ASR-plus-text-rule decisions versus the retained human
labels, not general ASR accuracy. No pooled accuracy, model ranking, CER/WER,
acoustic tone/count/timing, FAR/hour, training admission or release claim follows.

Shared failure: 019-Eric / clip-0004 is human-positive for 小窝小窝. Qwen wrote
“小五，小五，稍后再播放那段提示音。” and SenseVoice wrote
“小舞小五稍后再播放那段提示音”. Both revised text decisions are negative.
The additional Qwen false negative is 008-Serena / clip-0001; its “小屋，小屋”
is negative under the diagnostic while SenseVoice's “小窝小屋” remains unknown.

Zero definite positives has two distinct causes: spelling/phonetic deviations
are common, and the conservative boundary policy still abstains on exact target
characters separated by commas. In particular, Qwen produced “小窝，小窝” on
015-Eric / clip-0014 and “你好，小窝” within 037-Eric / clip-0016. These are
reviewable character candidates, not proof of a contiguous acoustic phrase.
They were not joined, repaired, or promoted to positive/gold.

## General bug fixed

The original Rules.derive propagated any heteronym/OOV anywhere in a transcript
into every unmatched target decision. With full official lexicons, this produced
0/26 coverage for both models. That original outcome is preserved separately.

V2 keeps lexical ambiguity as a descriptive diagnostic and applies uncertainty
to target-plausible spans. It considers all dictionary readings and OOV
possibilities over n−1/n/n+1-token spans at the unchanged one-edit tolerance.
All resulting phonetic/OOV/boundary candidates are unknown-only. Exact literal
contiguous text remains the only positive route; global execution, completeness,
quality and uncertainty-marker blockers remain unchanged. Punctuation is retained.

The policy was developed after seeing v1's zero coverage, using invented generic
fixtures. Source implementation and independent code checks did not inspect the
16 human annotations or optimize against them. Actual diagnostic evaluation was
then run with its own rule hash and explicit provenance. This timing makes it
post-hoc regardless of test coverage.

## Files and validation

- evaluation-v1/summary.json: totals, exact error/disagreement/boundary cases
- evaluation-v1/transcript-comparison.json: all 16 raw transcript pairs and labels
- evaluation-v1/readout.json: full derivation, local candidates and metrics
- evaluation-v1/evaluation-provenance.json: unchanged execution identity and new
  evaluation-rule binding; only rules_sha256 differs in the reused input envelopes
- calibrate.py and scorer-v1-to-v2.diff: reviewed source and narrow change
- POLICY-NOTES.md and implementation-receipt.json: implementation-stage record
- tests/test_target_local_v2.py: 19 generic tests
- review/test_phonetic_distance_oracle.py: 450 seeded cases and 4,320 concrete
  reading/OOV realizations against an independent Levenshtein oracle
- v1-regression-results.txt and v2-contract-regression-results.txt: 38 tests each
- LICENSE: original scorer's Apache 2.0 license

The implementation receipt's no-actual-data statements refer to source/test
creation. The later evaluation is documented separately under evaluation-v1.

Reviewed scorer SHA256:
1bccdd66bca3cd67109df1eff63770b5d7e9ba3a9a10b578a665b7bbcc9575b4
Revised evaluation rules SHA256:
c8f484427f670c2be76ff6389f9e8114cfb72578968b150fbfe5711eb54222de
Diagnostic readout SHA256:
c95d26e8e8edf9e53ef5ea5862b6049d8c3e54a995cdf4240101ef5de2053743
Original frozen readout SHA256:
c332fd6390de1ccf5f3b21f3f6a38724721deb45e28f44d74993c2195b43bac3

The original complete execution/raw/frozen-rule archive is
asr-dual-calibration-frozen-v1-20261002.zip, SHA256
b629479217934db38792a7d6ddcb349f91a98f9a8924c8f073ec4905f03bc194.
The public research directory contains the expanded frozen dictionaries,
manifest, and raw receipts alongside this separately versioned v2 subdirectory.
The top-level publication manifest identifies omitted and privacy-redacted files.

Recommendation: keep ASR as a review aid. The observed shared miss means dual-ASR
agreement is not an automatic training-label authority. Any prospective quality
claim requires a separately frozen protocol and fresh independent annotation.
No further threshold or boundary change was made after seeing v2 results.
