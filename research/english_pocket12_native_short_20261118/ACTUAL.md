# Pocket12 actual result: complete execution, quality STOP

The [single public run 44208480336](https://github.com/jiying2007/kws-data/actions/runs/44208480336)
used commit `99c9f6f4b94c47a7ad533f3241bc7b2db348a1b7` and completed all **60/60**
planned high-level model requests: 12 TTS generations and 12 observations from
each of Qwen, Whisper, Parakeet and XLSR. All 60 returned and saved their outputs.
There were no retries, warmups, failed requests or missing observations.

The frozen decision is **`STOP_POCKET12_SOURCE_FEASIBILITY_FAILED`**. Every row
remains permanently excluded from training. No KWS training, new KWS weights,
model promotion or authoritative natural-US-pronunciation conclusion resulted.
The original [RESULT.json](RESULT.json) and [EVALUATION.json](EVALUATION.json) are
preserved byte-for-byte from the public artifact.

## What the actual audio supports

| Group | Rows | Gate passed | Positive candidates | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Hey Mero | 2 | 0 | 0 | Neither complete phone sequence matched a frozen Mero hypothesis. |
| Hey Vosen | 2 | 2 | 2 | Both voices supported the predeclared `/s/` hypothesis with all three ASR votes. |
| Marrow / Bosen contrasts | 4 | 0 | 0 | All were rejected by the fixed whole-phone control patterns; none became a positive candidate. |
| Ordinary sentences | 4 | 4 | 0 | All three ASRs matched the requested ordinary text and no target phone pattern was observed. |

These are diagnostic, machine-corroborated results on synthetic audio. They are
not independent real-speech recall/FAR measurements or a demonstration that the
KWS model itself improved.

### Mero remains the main source-quality blocker

George's Mero and Marrow were both observed by XLSR as normalized `heɪmɛrɔ`.
Mary's Mero was `heɪmɛrʌ` and Marrow was `heɪmɛrə`. Mero's final vowel therefore
did not match the frozen whole-phone hypotheses, and the intended Marrow `/æ/`
contrast was not corroborated. Whisper and Parakeet mainly transcribed these as
Merrill, Merrow or Meryl. Qwen alone transcribed George's positive request as
Mero; that does not override the full evidence.

The matching George phone strings do not prove its two waveforms are identical:
their PCM hashes differ. They establish insufficient separation in this observer
evidence. A synthesis defect, accent variation and shared observer limitations
cannot be cleanly distinguished by this twelve-row probe alone.

### Bosen failed a narrower control contract

Both Bosen outputs were observed as `heɪbozən`, preserving the `/b/` onset. The
predeclared control pattern required `/s/`, so the observed `/z/` did not pass.
This is **not evidence of `/b/` becoming `/v/`**, nor a measured false wake.
The original `/s/` contract and STOP result remain unchanged; no post-observation
alias or phone-pattern relaxation was applied.

### Geometry and native generation were verified

All twelve whole clips had 6–10 full-support windows under the unchanged integer
geometry. The four positive rows had 6, 8, 8 and 7 windows. The quality stop is
therefore not explained by carrier cropping or too few support windows.

Actual native initialization succeeded once with a strict load of 214 stored
BF16 tensors and no missing or unexpected keys. Both precomputed voice states
had six F32 layers with consistent active prefixes. All twelve native outputs
were complete, finite, distinct and unsaturated, totaling 16.40 seconds; each
was 1.20–1.52 seconds. The complete native audio was resampled exactly once,
and each WAV payload matched the PCM consumed by every observer. The generation
stage, including native initialization and saving, took 7.970481 seconds; this
does not include installation and is not a deployment latency benchmark.

## Workflow status and archive correction

The generation and all four observer jobs succeeded. The combine job wrote a
complete evaluation with `combination_error=null`, then intentionally returned
exit code 1 because its frozen exit policy accepts only a quality PASS. The
workflow's failure status reflects that STOP, not an additional inference error.

An independent archive audit found a separate packaging issue: GitHub's upload
step used `include-hidden-files: false`, so the original combined ZIP omitted
`frozen_source/.github/workflows/english-pocket12-once.yml`. Its manifest declared
413 files plus itself, but the ZIP contained only 413 total files. Source audio
and all four observer packets were complete; the missing item was one workflow
source file.

The original eleven artifact ZIPs are preserved unchanged. The missing 12,076-byte
workflow was fetched from the exact execution commit and checked against the
already-frozen plan/manifest SHA256
`0b89b6576fec61d5574861c41878f56b6175344cfa81b12e2870000d9ece5df6`.
The exact file and fetch provenance are retained in [archive_sidecar](archive_sidecar).
[RESTORATION.json](RESTORATION.json) records a separate derived archive containing
all 414 files with every declared byte count and hash verified. This was a saved-byte
restoration with zero additional model calls; it changed no audio, observation,
decision or original artifact.

| Archive | SHA256 |
| --- | --- |
| Original combined ZIP, artifact 14384827730 | `dfb912679ad1fd11bd0bf1871fc8756604ece0c90c42615ba50e39bfcec3cecb` |
| Derived ZIP with the exact source sidecar | `62cd480eab7cb4feba9684470fd1e12b604e4a22916dc67fe095494a2a5ecc9a` |

## Consequence for the English model work

Pocket's native CPU generation route is now demonstrated, and both Vosen requests
received same-input support. The combined Mero/contrast admission gate did not
pass, so this evidence does not authorize corpus expansion or training. Future
source work needs a prospectively frozen Mero pronunciation/contrast design and
independent evidence of its acoustic separation. It must use fresh samples and
retain this batch's exclusion and STOP. The current research model and its
`0.8 / 3 / 13` runtime operating point remain unchanged.
