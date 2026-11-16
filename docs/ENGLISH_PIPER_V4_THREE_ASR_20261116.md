# Piper v4 16-cell blind ASR — 2026-11-16

## Exact immutable source

- Generation run: 43898728142.
- Artifact ID: 14240252131.
- Artifact name: `english-ipa-piper-probe-v4-43898728142`.
- Artifact ZIP SHA-256: `9a611676829195c8ac07c07e50705cd8e98822703c60ab66e21ee5be03567193`.
- Exactly 16/16 per-cell `completed.json` receipts and WAV/PCM hashes were read back.

The original generator's aggregate `result.json` incorrectly says `INCOMPLETE_NO_RETRY` because its terminal status condition still compared `completed == 4` after the roster expanded to 16. The original run is retained as failed at the workflow level; **no retry or TTS regeneration is performed**. The downstream worker only accepts this source if all 16 cells, source plan, model lock, speaker roster, WAV hashes, PCM hashes and receipt identities independently verify. It does not rewrite the original artifact.

## Blind ASR plan

- 16 source WAVs × Qwen3-ASR, Whisper-small and Parakeet TDT v3.
- Maximum 48 recognizer calls, 16 per provider, one attempt per cell, zero retries.
- Each recognizer receives audio only; no target phrase, IPA, speaker identity or other transcript is supplied.
- Source IDs are eight new LibriTTS speakers; v3 speaker IDs p3922/p8699 are not repeated.

## Evidence boundaries

Three-way agreement and spelling normalization are machine review cues, not acoustic gold. ASR mismatch is not automatically a trusted negative, especially for accent/allophone variation. No audio is relabelled, no model is trained, and no sample is admitted by this workflow. Human pronunciation review and the independent phone observer remain separate.
