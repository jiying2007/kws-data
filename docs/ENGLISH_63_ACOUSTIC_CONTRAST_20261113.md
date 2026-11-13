# English R0 follow-up: 63 original WAVs, same-voice acoustic contrast

**Scope:** reproducible *acoustic distance only*, not phoneme ground truth,
not a KWS classifier result, not real-human FRR, not continuous FAR, and
not SSC305 runtime performance. The existing owner-only FSMN-R0 remains
unpromoted because 5/6 exposed development nearphone challenges had a wake
proposal. Neither its weights nor its training scripts are touched.

## Why this experiment

All 63 original 16-kHz Qwen3-TTS English WAVs and all three independent
ASR results already exist. Their source-linked read-only 63-row ledger
succeeded in [kws-data run 43267739514](https://github.com/jiying2007/kws-data/actions/runs/43267739514),
after earlier source-import failure was retained and repaired by #101.

The 63 cells are exactly 7 preset Qwen voices x 9 requested prompts.
Each voice generated two intended Hey Vosen variants, two intended
Hey Mero variants, Hey Wosen, Hey Rosen, Hey Miro and two ordinary
conversation prompts. Punctuation variants are **not new speakers**.

English name spellings are unreliable phonetic labels; in particular,
the earlier three-ASR 0/14 exact-target spelling for Vosen is **not**
0/14 correctly spoken and is not a model FRR. This experiment compares
actual waveform acoustics without using requested strings as true audio
transcripts or using acoustic similarity to auto-promote labels.

## Frozen method (no fitting or sweep)

For each existing WAV (already reauthenticated using original SHA256 and
PCM16 16-kHz header receipts), use 25ms Hann frames, 10ms hop,
40 triangular mel-energy bands at 70-7600Hz and log energy.
Trim simple low-energy leading/trailing frames with a deterministic
0.002 / 6%-of-peak RMS rule, retain 3 context frames, then perform
within-clip CMVN (variance floor 0.25). This local normalization means
absolute signal quality and accent validity require separate measurement.

For **each of seven synthetic voice families**, compute full-path
time-normalized, frame-Euclidean DTW. There are three frozen contrasts:

| Contrast | Same-phrase reference | Nearphone probe |
| --- | --- | --- |
| Vosen/Wosen | Hey Vosen vs Hey, Vosen! | Hey Wosen |
| Vosen/Rosen | Hey Vosen vs Hey, Vosen! | Hey Rosen |
| Mero/Miro | Hey Mero vs Hey, Mero! | Hey Miro |

The nearphone distance is the **minimum** of the probe vs either
previously captured positive variant. Ratio = nearphone distance /
same-phrase-variant distance; a ratio <=1 flags that a nearphone is
at least as acoustically similar as the variant. The value 1 is merely
a descriptive *within-source geometric comparison*, not an optimized
KWS threshold; there is no statistical/physical false alarm inference.

Seven family-level outcomes are retained individually; the two development
families remain held out from training. The summary does **not** multiply
two punctuation variants into more independent speakers or use the
development group to tune a model. No threshold, model or source sweep.

## Exact reused artifacts and reproducibility

| Source | Run | Original ZIP SHA256 |
| --- | --- | --- |
| 63 real new Qwen WAVs | 43242278962 | f180004a27ea14c0575f4dd0e2a8476ca048442c76c2dcc186ede6d731cdffcc |
| 63-row three-ASR review | 43267739514 | 3786baa9fd3575437a05e97956fafdf3298eac23de2538b485b8342ef4d25133 |

The exact review JSON SHA256 is
60054a3532491631a47b094ea7ff2bc86ef16530ff510f968bc9edcdd253a9ad.
The one-off trigger pins both ZIPs, original source policy, all 63 PCM
identity checks, saved 189 native ASR observations and the frozen
comparison roster. Re-executing this read-only analysis never invokes
TTS, ASR, another neural model or the FSMN optimizer; **zero** new
model calls. Source-only regressions use *invented* PCM, not the real
dataset; the protected-main one-off run is required before calling the
actual saved-data result PASS. Output is a source-bound JSON research
artifact, not duplicated original WAVs/weights.

Run the existing captured artifacts locally:
python -I -B automation/english_acoustic_contrast.py --tts-zip
PATH_TO_ORIGINAL_TTS_ZIP --review-zip PATH_TO_ORIGINAL_REVIEW_ZIP
--output ABSOLUTE_NEW_PRIVATE_JSON_PATH --execute

## Next scientific gate

Use the acoustic risk ranking to prioritize **independent phone/accent
validation**, especially Vosen/Wosen and Mero/Miro, without treating
these synthetic prompts as gold. A separate frozen phoneme recognizer
can supply *independent acoustic evidence*, but not ground truth.
Only correctly reviewed acoustic labels from source-verified speech
can be considered for formal training admission. Near-miss intended
labels remain quarantined if they could actually realize a full wake.

A subsequent fixed pretrained FSMN experiment must hold the architecture
and model source constant, use speaker/source-family-disjoint train and
development, and report complete wake-event matching and nearphone
false-trigger impact. Independent real-human FAR/FRR, ambient
continuous negative hours, SSC305 board latency and memory are NOT_RUN.
