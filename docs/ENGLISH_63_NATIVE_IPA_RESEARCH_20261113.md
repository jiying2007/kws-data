# English 63-WAV independent phoneme observations (2026-11-13)

## Decision and scientific boundary

Continue **noncommercial KWS research** with a bounded fourth, independent
acoustic recognizer. Do not change either Hey Mero or Hey Vosen based on
spelling votes. The previously saved 189 three-ASR transcriptions and
Log-Mel/DTW measurements are independent diagnostic inputs, not human
acoustic truth or model recall/FAR.

- Original TTS run: https://github.com/jiying2007/kws-data/actions/runs/43242278962 — original 63 WAVs, never regenerated.
- Previous ASR ledger: https://github.com/jiying2007/kws-data/actions/runs/43267739514 — 189 verified ASR observations.
- Waveform DTW run: https://github.com/jiying2007/kws-data/actions/runs/43334622072 — Vosen/Wosen 6/7 synthetic preset voices have nearphone-vs-variant ratio <=1 (a descriptive geometry comparison, not FAR).
- Saved owner-only strong-pretrained FSMN-R0 remains NOT PROMOTED after 5/6 development nearphone wake proposals.

## Frozen native IPA model

Independent CTC phoneme recognizer:
https://huggingface.co/bookbot/wav2vec2-ljspeech-gruut

- Model revision: 7be1e6f27c8163ce790688d62fd9ce1731177c2e.
- model.safetensors SHA256:
  9bc8cd20e1348fc384ba4b9cd60a5c761dbb3f0b7071d1bdc33331ed1b358f29.
- Approximately 378 MB / 94.4 million parameters; model card declares Apache-2.0.
- Trained on LJSpeech phoneme labels from Gruut: it is biased toward a
  particular narrator, dataset and phonemization policy. Invented English
  names and international accents may fail; this is NOT phonetic gold.
- Download only seven named asset files at exact git revision; verify the
  known safetensors SHA256 before loading; retain manifest hashes and CPU
  runtime packages. Offline inference, no remote code and no pickle weight.
- Model sees ONLY each original verified mono 16-kHz PCM, never text prompt,
  target keyword, source split, prior ASR text or nearphone label.

## Pre-registered experiment, not a label classifier

One-time protected-main trigger permits at most 63 new independent phoneme
model forwards, exactly once per original saved WAV; zero new TTS, original
three ASR, KWS forwards or optimizer steps. No cron or repeated schedule.

For each of the seven Qwen synthetic voice families, preserve the actual
raw IPA tokens and unstressed sequence; separately compare:
- Hey Vosen vs Hey, Vosen! against Hey Wosen.
- Hey Vosen vs Hey, Vosen! against Hey Rosen.
- Hey Mero vs Hey, Mero! against Hey Miro.

Comparison uses full-utterance phone-token Levenshtein. Any nearphone at
equal/lower distance than positive wake variants is only a candidate for
pronunciation investigation, not a false wake event or true negative.
No threshold, source, model or pronunciation sweep is authorized.
Serena/Uncle Fu remain development-only. Seven Qwen preset voices are NOT
seven independent human speakers.

All 63 rows stay human_gold=false, phonetic_label_authenticated=false,
trusted_negative=false, training_admitted=false. Independent genuine human
FRR, long natural-negative FAR/hour, physical SSC305 and streaming end
latency remain NOT_RUN.

## Reproduction, failure retention and future admission

Source-only PR checks use invented phone tokens and metadata, no actual
phoneme model or new source audio. A protected main trigger creates a
new private result directory with started.json; any failure creates
failed.json with attempted forward count and remains consumed, no implicit
retry. Only successful complete source-bound native results permit a machine
observation report. Model and audio never uploaded to GitHub; per-cell
phone observations and SHA-bound source receipts can be retained.

Next gate requires independent human/phonetic pronunciation adjudication
for contested nonce names and accents. Only separately verified and
leakage-checked speech may be considered for future FSMN training.
