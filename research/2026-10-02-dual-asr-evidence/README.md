# Dual-ASR calibration evidence: preserved v1 and post-hoc v2

This research archive publishes the completed Qwen3-ASR-0.6B and SenseVoiceSmall
CPU FP32 runs on the same 16 synthetic WAVs: **32/32 primary decodes completed**,
with raw tokens/transcripts, exact checkpoint-load receipts, waveform bindings,
per-clip results, resource measurements, source/tests, dependency hashes, public
model configuration/tokenizer files, and review evidence.

## Latest findings after the 019 human correction

The user corrected 019-Eric / clip-0004 on 2026-10-02 at 01:08:58 UTC:
“019发音不标准，听起来像小屋小窝，不是小窝小窝”. Only its 小窝小窝
presence label changes from positive to negative; its other bit is unchanged.
The current human labels are **8 positive, 18 negative and 6 unknown**.

- Frozen v1 remains **0/26 covered for both models** because its whole-transcript
  lexical uncertainty gate abstains on every known target bit
- Post-hoc v2 Qwen: **14/26 covered**, 13 true negatives and 1 false negative
  (008-Serena / clip-0001); 0 definite true/false positives, 12 unknown
- Post-hoc v2 SenseVoice: **13/26 covered**, 13 true negatives and 0 false
  negatives; 0 definite true/false positives, 13 unknown
- Both cover 12 known bits with **zero jointly wrong covered bits**. The earlier
  shared-false-negative conclusion about 019 is withdrawn. Both full-keyword
  absence decisions agree with the corrected human judgment
- Qwen has comma-separated exact-character candidates on clip-0014 (“小窝，小窝”)
  and clip-0016 (“你好，小窝”). The unchanged boundary policy leaves these unknown
- All six human-unknown labels remain unknown. SenseVoice's eight known positives
  all remain unknown under this conservative policy; this is insufficient
  evidence for automatic training-label admission or an ASR accuracy ranking

The original labels/results remain a historical layer. The separate
`asr-calibration-stage-v1/human-review-019-v1/` overlay replays both unchanged
scorers with one corrected human bit. All 32 model-evidence rows are identical.
No ASR model, raw output, waveform, scorer policy, or decoder setting changed.

**V2 is post-hoc diagnostic evidence using the same outputs.** It was designed
following the v1 coverage problem, independently checked with generic synthetic
fixtures, and then evaluated. This later human correction is also explicitly
timestamped. Neither evaluation is untouched-holdout qualification. No model
was rerun for v2, the correction, or publication.

These selected probes do not establish general ASR accuracy, CER/WER, acoustic
tone/count/timing, FAR/hour, automatic training admission, or release readiness.
No CI success is claimed by the publication; retained validation records identify
local checks and their exact scope.

## Start here

All paths below are relative to this directory:

- `asr-calibration-stage-v1/human-review-019-v1/`: current human overlay, corrected
  compact summary, audit chronology, inputs and both full rescored readouts

- `asr-calibration-stage-v1/calibration-result-v1/`: original frozen baseline,
  complete transcript comparison, original envelopes and bundle-validation receipts
- `asr-calibration-stage-v1/scoring/posthoc-v2/`: reviewed diagnostic source/diff,
  synthetic tests, independent distance oracle, provenance and v2 readout
- `asr-calibration-stage-v1/primary-runs-v2/{qwen06,sensevoice}/`: all 32 raw
  decodes and receipts, immutable run contracts/plans and explicit outcomes
- `asr-calibration-stage-v1/execution-plans-v2/`: exact pre-forward model recipes,
  frozen sample order, generation settings and code inventory
- `asr-calibration-stage-v1/asr_stage/`, `pcm/`, `vendor/`, `tests/`: implementation
  and contract tests (the latter three under `asr-calibration-stage-v1/`)
- `runtime-qualification/`: 140-package hash-pinned dependency closure, input and
  source-build manifests, offline installation/probe receipts and recipes
- `public-model-assets/`: original Qwen public configuration/tokenizer/vocabulary
  files, plus model-license notices. SenseVoice config/tokenizer/CMVN bodies are
  referenced by immutable official URL/hash rather than copied, because the model
  license differs from the package license. Neural weight files are excluded
- `input-references.json`: the unchanged 16 WAV paths, byte lengths and SHA256s
- `publication-projection.json`: exact correspondence to earlier frozen archives,
  omissions and privacy-only edits
- `SHA256SUMS` and `verify_publication.py`: complete publication-byte verification

The original audio is preserved at
[the input commit](https://github.com/jiying2007/kws-data/tree/c8c1a92f428bf515737bcaa5c4933de0fba50f7b/research/2026-10-01-asr-calibration).
This branch starts from that commit and adds only this new research directory.
Audio is referenced rather than duplicated.

## Exact identities and execution limitations

- Qwen model revision: `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`
- SenseVoice model revision: `3847d57b6bdf2dd8875cb1508d2af43d80a16bf7`
- Corrected-human v1 readout SHA256:
  `a103c097d7b00b77c5f103989a57567b2ca7ac32ae1eba084cfa1ce1af58fe22`
- Corrected-human v2 readout SHA256:
  `ddf860a3b6a0fc06963425e962ae4ad1976d455b8b6c8d1fb50533b894f6ac43`
- Historical original v1 readout SHA256:
  `c332fd6390de1ccf5f3b21f3f6a38724721deb45e28f44d74993c2195b43bac3`
- Historical original v2 readout SHA256:
  `c95d26e8e8edf9e53ef5ea5862b6049d8c3e54a995cdf4240101ef5de2053743`
- Reviewed v2 scorer SHA256:
  `1bccdd66bca3cd67109df1eff63770b5d7e9ba3a9a10b578a665b7bbcc9575b4`

Both models ran serially in fresh CPU-only processes. All inputs were unchanged
mono 16-kHz PCM16; normalization was int16/32768 with no resampling, trimming or
mixing. Qwen used FP32/eager attention, at most 256 generated tokens, and no forced
language, hotwords or text context. SenseVoice used the source-default native
fbank with dither 1.0, LFR 7/6, 80 mel bins, Hamming windows and 25-ms/10-ms
frame settings. Native-fbank dither bitwise reproducibility is not established.

Offline flags and local verified assets were used. Networking was not kernel
isolated. Sampled process-group RSS/headroom guards are not cgroup enforcement.
Observed peaks were 2.70 GiB for SenseVoice and 5.52 GiB for Qwen; these are measured
runs, not general hardware minima. Exact historical filesystem paths in hashed
run inputs and receipts are retained as evidence; they are not portable paths.

## Preservation, disclosure and local replay

The main scorer sources, frozen rules, human labels, all raw results/receipts and
both readouts are unchanged. Seven support/provenance files remove private Library identifiers, generalize
internal workflow wording, replace archive-only README directions, or add historical-label notices. The publication manifest
records their old/new hashes. The lexicon exporter changes only a descriptive
provenance string; the historical exporter hash still identifies its original
source. The original child `SHA256SUMS` and inventories
record the earlier source archives, including their omitted audio/status files;
use this directory's top-level `SHA256SUMS` for the public projection.
The original frozen archives remain unchanged outside this public projection.

Run `python3 -B -I -S verify_publication.py` from this directory to check every
published byte and all preserved input WAVs. This performs no model import or
network request. Local test replay receipts list the exact commands and results.
Historical absolute paths mean the original inference entry point is not a
one-command portable rerun; a new run must prepare and freeze its own paths and
run identity rather than reuse the original primary output directory.

The full official pypinyin dictionaries and their license notices are retained
under `asr-calibration-stage-v1/scoring/`; source-model package license notices are
under `asr-calibration-stage-v1/source-audit/licenses/`. Existing upstream notices
apply to copied code/data. Publication does not change an upstream license.
