# English confirmed-IPA TTS probe — 2026-11-15

## Purpose

Generate only four new synthetic pronunciation probes to test whether the pinned Qwen 1.7B CustomVoice model can follow the owner-confirmed target readings. This is not a new training corpus, acoustic gold, or model qualification. No existing WAV is regenerated.

## Exact targets

- Hey Mero: /heɪ ˈmɪəroʊ/
- Hey Vosen: /heɪ ˈvoʊsən/

Two existing English preset identities are used for a small source-diverse probe: Aiden and Ryan. Each phrase is generated once per voice. The generated audio is retained only for pronunciation review. The text prompt and instruction are not accepted as evidence that the actual output matches the IPA.

## Why a separate 1.7B probe is required

The previously used Qwen3-TTS 0.6B CustomVoice revision is `85e237c12c027371202489a0ec509ded67b5e4b5`. In the official Qwen3-TTS implementation, the 0.6B CustomVoice branch explicitly clears `instruct`; therefore, the old seven-voice TTS campaign's plain orthographic text prompts were not conditioned by the newly confirmed IPA instructions. Those 63 existing WAVs and their ASR observations remain immutable and exposed. They are not silently re-labelled as matching the confirmed pronunciation.

The new probe uses the official Qwen3-TTS 1.7B CustomVoice revision `0c0e3051f131929182e2c023b9537f8b1c68adfe`, whose CustomVoice API supports per-utterance `instruct`. The official model card declares Apache-2.0. The workflow pins this exact revision, downloads no alternative model, and hashes every file. The Qwen API's instruction control is still not a pronunciation guarantee; human acoustic review remains mandatory.

## Fixed budget and isolation

- Exactly four calls: two phrases × two voice presets.
- One model download, one model load, no retries, no ASR, no training.
- CPU-only pinned runtime; BF16 model load; SDPA attention.
- Model download ceiling 6 GiB and at least 1 GiB free disk reserve.
- Every call has a deterministic cell identity and seed.
- Each output WAV/PCM is stored with exact hashes and a terminal receipt.
- Failed/interrupted cells are consumed and never retried by this workflow.
- Output is marked `pronunciation_verified=false`, `human_gold=false`, `training_admitted=false`, and `qualification_allowed=false`.

The one-shot workflow executes only on a protected push to main and only on attempt 1. PR checks execute source-only tests and do not download the model or synthesize audio.

## Required review after generation

For each of the four WAVs, an independent reviewer must assess:
- whether the actual spoken phrase is complete;
- Mero's stressed vowel and following rhotic/diphthong sequence against /mɪəroʊ/;
- Vosen's initial /v/, stressed /oʊ/, and unvoiced medial /s/ against /voʊsən/;
- whether the instruction altered prosody or introduced artifacts;
- whether the output is clear enough for a positive research sample.

Until that review is recorded, every generated sample remains unadmitted. A failed pronunciation is not a negative sample. The four probes do not create independent human speakers or independent evaluation data.

## References

- Qwen3-TTS official source: https://github.com/QwenLM/Qwen3-TTS
- Exact model card/revision: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/tree/0c0e3051f131929182e2c023b9537f8b1c68adfe
- Qwen3-TTS inference API: https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py
