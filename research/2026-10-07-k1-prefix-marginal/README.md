# K1 first-prefix marginal: negative result

**FAIL; STOP_CANDIDATE_NO_TUNING.** The fixed whole-observation gate required
M3 rejection and N1 acceptance. N1 remained accepted (A=0.872928816141001),
but M3 (`你好小屋`) falsely accepted K1 (A=0.7411511568874126,
B=0.19759056685088203, R=0.06125827626170545). M1 accepted; M2/M4/M5 rejected
K1. M2's K2 behavior is outside this candidate, so this records no K2 gain.

This append-only numerical archive retains the existing generated-audio and
whole-clip diagnostic history. It neither duplicates nor alters original raw
inputs, audio or models. The complete saved result is 11,283 bytes, SHA256
`76e35a4f16a004f6e3724b282c00775c350e3d02a7d833c4e8e5856bf68fc54a`.

## Evidence

- `attempt-v1/attempt.jsonl`: unchanged first numerical failure; no EOF
- `attempt-v1/underflow-diagnosis.json`: unchanged conversion-only cause
  diagnosis and explicit reconstructed partial DP accounting
- `recovery-v2/attempt.jsonl`: unchanged attempt 2 start, six EOF records and
  completed failed gate
- `recovery-v2/results.json`: unchanged all-six masses, logs, exact rational
  certificates, fixed gates, underflow counters and resource receipts
- `review/saved-audit.json` and `.md`: independent receipt/source-identity audit
  without reopening the raw inputs or replaying a DP

Frozen source, disabled release requests, the original and revised numerical
adapters, independent oracle/edge checks, and offline synthetic CI are in the
companion `jiying2007/kws-pipeline` research draft branch
`research/k1-prefix-marginal-v1`, directory `research/k1-prefix-marginal-v1`.
This archive does not release another saved-row execution.

The input identities are fixed by the unchanged source manifest:

- [M1–M5 original A20 rows](https://github.com/jiying2007/kws-data/blob/dc90c3aa325700fdf17da449452437c90373818e/research/2026-10-06-melo5-leading-context/original_A20.raw.jsonl),
  69,596 bytes, SHA256 `53fe05237bdc3544462243b36566cf784a380f373aa7da1572f00ed69da8b043`
- [N1 original A20 rows](https://github.com/jiying2007/kws-data/blob/8c301d7a4709b066633ebfd9316159281ee320ea/research/2026-10-07-n1-wholeclip-leading/original_A20.raw.jsonl),
  17,794 bytes, SHA256 `013360aae469da5671994b1de4495687cb2324880108236dc5bb0d83d3b2e3ad`

## Honest attempt and resource accounting

Attempt 1 failed because its adapter prohibited a computed exponential zero.
From the failure coordinate and frozen callback order, it attempted 69 softmax
rows, computed 68 DP transitions, committed 59 rows after callback rollback,
and produced zero EOFs. Those partial counts were reconstructed rather than
logged. Its measured CPU, wall time and peak memory are unknown. Both complete
sources, all 537 rows, had already been loaded. The separate cause diagnosis
examined 69 rows and made zero DP transitions.

Recovery retained the complete observation geometry and fixed decision. It
permitted only actually computed zeros with exact recovered-FP32 logit delta
<= -512; every positive term was retained. It completed 537 rows in 57
callbacks with six DP instances and six EOFs. It recorded 81 exponential
zeros across 19 rows, no additional division-only zeros, 0.038824465 CPU
seconds and 0.042778524999448564 wall seconds. Peak memory was not measured;
64 MiB was the configured address-space cap, not an observed peak.

Across both attempts: 606 attempted softmax rows, 605 computed DP transitions,
596 committed DP rows and six EOFs. The conversion-only diagnosis is separate.
Exactly zero new model, frontend, decoder, audio, TTS, ASR or training calls
were made in this saved-row evaluation stage or this publication preparation.

## Interpretation limits

The certificate proves the fixed binary comparison A > B+R for exact normalized
ratios of the supplied rounded binary64 weights. The ideal removed-tail bound
6*T*2^-512 in total variation excludes ordinary exp/subtraction/fsum/division
rounding. Neither statement proves acoustic calibration or an ideal-real
softmax result.

First-prefix A/B states absorb all suffixes. Later 屋 evidence cannot retract
an earlier erroneous 窝 branch; same-occurrence versus separate-word evidence
remains unresolved. The observed acoustic confuser remains the bottleneck.
EOF input availability is not word latency; this is not keyword-anywhere or
continuous detection. The examples were already exposed and use one stock
voice, with no held-out FAR/FRR, generalization, board or product qualification.
No fit, threshold search, crop, reset choice, alternate class grouping or
post-failure rescue evaluation is claimed.

Actual execution-release/approval records are excluded. Their historical
hashes remain in unchanged receipts, without exposing their contents. No local
workspace paths, Library identifiers or private conversation identifiers are
included. The companion's manifest projections disclose path-only metadata
transformations separately from the unchanged scientific evidence.
