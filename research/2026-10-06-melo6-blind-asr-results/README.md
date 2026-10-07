# Melo6 blind ASR saved results

The preregistered all-six exact-text gate **FAILS**. Of six fixed phrases,
2 have both recognizers match intent, 3 have recognizer disagreement, and
1 has both recognizers agree on off-plan text. All six remain quarantined.
Human transcript, pronunciation and acoustic completeness remain UNKNOWN;
gold labels and CTC targets are null, training admission is zero, and there
is no automatic retry, source advancement or KWS improvement claim.

The original run completed technically: each recognizer retains six started
and six successful completion receipts. Technical completion is separate
from the failed scientific screen.

This sibling archive adds the original, unchanged ASR ZIP and the original,
unchanged comparison JSON. It does not modify the 49-file generation archive
at `../2026-10-05-melo6-source-screen/`, its content manifest, or its exact
directory verifier. `provenance.json` binds the original source, input data,
run and artifact. `review.json` records the independent 1,020-check saved-byte
audit. `publication-allowlist.json` lists all 75 original ZIP members; none
is excluded. The original ZIP contains JSON research evidence, not audio,
model weights or raw logs.

`content-manifest.json` gives the size, SHA256 and Git blob SHA-1 of every file
in this directory except itself. Compare its own hash to the publication
receipt. The original ZIP is 274,118 bytes, SHA256
`373bae98f5ca858fc3f931403981c336e9f8fa009f5a5fdff762a3197ff1e299`;
its 75 members total 1,303,996 bytes. The comparison is 24,485 bytes, SHA256
`7af81332103cf8174d8cbda9deda00e988027493e84f4643518993f20cdf280e`.
Reproduce it with the source supplement's `research/melo6_saved_results`
adapter and the explicitly pinned public inputs. Reproduction makes no
model calls and changes no original evidence.

`ANNEX.txt`, `triage.json` and `triage-verification.json` provide separate
dictionary-based text diagnostics. They preserve the original failure and
do not establish waveform content or substitute a relaxed acceptance rule.
For any later listening review, collect a transcription before showing
further hypotheses where possible and record prior hypothesis exposure.
Only a reviewer unexposed to those hypotheses can provide a blinded pass.
No playback or human adjudication is included here.

The earlier TTS output-signature guard failure remains a failure: one session
was constructed, zero synthesis calls occurred, and all six prior rows were
not run. Recovery constructed one session and generated the six fixed phrases
once each. A separate source-tree publication retry repaired an interrupted
technical publication step without changing scientific inputs or repeating
TTS/ASR. Those histories are not scientific evidence of success.

Saved resource records report setup and both model-stage wall/CPU measurements,
sampled RSS/disk and controlled downloads. Sampled maxima are not continuous
peaks or hard limits; CPU limits are per process; actual sample gaps are
UNKNOWN. Model-stage and adapter-call times are not pure decoding or KWS/A20
benchmarks. Saved records and hash bindings support consistency, not an
independent system-wide execution census or timestamp service.
