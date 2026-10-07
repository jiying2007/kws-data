# Melo6 human transcript supplement

This append-only supplement records one human review of the six previously
saved native WAVs. It changes no original audio, machine result, triage,
source freeze, or historical manifest. The original preregistered all-six
machine gate remains FAIL: 2/6 dual exact-intent matches. No new synthesis,
ASR, playback, or KWS evaluation was performed to build this supplement.

Current full labels are M1 `你好小窝`, M2 `小窝小窝`, M3 `你好小屋`,
M4 `小屋小屋`, and M5 `你好你好`. All five match their planned strings.
M6 is partial: `[首字听不清]天天气很好`. Its current full text is null,
first character is UNKNOWN, and prompt match is UNKNOWN. The earlier M6
`明天天气很好` label is retained solely as superseded history. Neither
that old label nor the two ASR readings may complete the first character.
There are five human prompt matches, zero confirmed current mismatches,
and one unknown, separate from the unchanged machine result.

`human-labels.json` stores technical transcript revisions without conversation
identifiers or private transport references. `audio-bindings.json` links
M1–M6 to the original native WAVs, source IDs, ASR IDs and derivative hashes,
verified against the immutable original data manifest. Human listening was
on 44.1-kHz mono float32 native WAVs; ASR used 16-kHz PCM16 derivatives.
Those derivatives were not independently heard. The reviewer had already
seen ASR hypotheses, so the review is not fully blinded. Confidence,
pronunciation analysis and acoustic tail completeness are not established.

`vocabulary-binding.json` pins the retained six-class research alphabet to
an immutable source commit and file hash. M1–M5 have full text representable
under that alphabet. M6 lacks full text and its observed suffix contains
out-of-vocabulary characters. These are clip-level transcript/representability
findings. M1/M2 now provide word-level K1/K2 positive coverage. Split allocation
for this increment is UNASSIGNED; frozen split-integrity qualification and a
qualified positive development set are not established. One identity group alone
cannot fill all isolated train/development/held-source partitions. Identity-held-out
evaluation, cleared voice rights, and KWS improvement remain unestablished. All six CTC targets
remain null and all six training admissions remain false. Partial or OOV
speech must never be forced to blank CTC.

`adjudication-result.json` is reproducible with the source supplement's
`research/melo6_human_adjudication/adjudicate_saved.py`. Its four inputs are
this directory's labels, audio bindings and vocabulary binding, plus the
unchanged `../2026-10-06-melo6-blind-asr-results/comparison-result.json`
(24,485 bytes; SHA256
`7af81332103cf8174d8cbda9deda00e988027493e84f4643518993f20cdf280e`).
The source README gives the full invocation and nine saved-JSON regression
tests. The checker imports no models, reads no audio and makes no network calls.
`content-manifest.json` lists each sibling file's exact size and SHA256,
except itself. Earlier machine UNKNOWN fields remain an accurate historical
snapshot and are not rewritten by this later human annotation.
