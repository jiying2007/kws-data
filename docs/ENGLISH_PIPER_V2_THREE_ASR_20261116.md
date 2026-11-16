# Final-IPA Piper v2 blind ASR — 2026-11-16

## Controlling pronunciation standard
The final user-supplied pronunciation standard, integrated in kws-data PR #140, is:
- American Hey Mero: /heɪ ˈmiːroʊ/ (MEE-roh)
- American Hey Vosen: /heɪ ˈvoʊzən/ (VOH-zən; voiced /z/)

Earlier /mɪəroʊ/ and /voʊsən/ samples are immutable historical evidence, not target positives for this final standard.

## Exact source
- Piper v2 generation run: 43789385672.
- Artifact: 14191228436, `english-ipa-piper-probe-v2-43789385672`.
- SHA-256: `8ab7ff273ce5ee98e4c4bd83f43fd80b82c7ae03ac89449ded93a4cd61bedc61`.
- Four cells: two final phrases × the same two pinned LibriTTS speakers.
- Piper 1.8.0 and the same verified model/config SHA.

## Blind screen
Exactly four WAVs × three independent recognizers (Qwen3-ASR, Whisper-small, Parakeet TDT v3): maximum 12 calls, one attempt per cell, no retries. The recognizer receives only the WAV, never the target phrase, IPA, speaker ID or another transcript. The source audio is verified by WAV/PCM hashes before any ASR call.

Three-way spelling agreement is a machine review cue, not acoustic gold. No output is automatically admitted as a positive, negative or training row. No TTS, training, threshold tuning or model promotion occurs in this workflow.
