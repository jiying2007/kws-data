# Stress-aligned Piper v3 blind ASR — 2026-11-16

## Controlling pronunciation

- Hey Mero: /heɪ ˈmiːroʊ/ — light Hey, primary stress on Mero's first syllable; raw eSpeak `heɪ mˈiːɹoʊ`.
- Hey Vosen: /heɪ ˈvoʊzən/ — light Hey, primary stress on Vosen's first syllable; voiced /z/ and unstressed schwa; raw eSpeak `heɪ vˈoʊzən`.

The earlier Piper v2 probe marked Hey with primary stress and is superseded for this prosodic target. Its WAVs remain immutable and are not relabelled.

## Exact source

- Piper v3 generation run: 43886265084.
- Artifact: 14235001433, `english-ipa-piper-probe-v3-43886265084`.
- ZIP SHA-256: `1085fb90a94372bb06c9ab358e3855e3da6666868db4242805aaa496be134884`.
- Four WAVs: two phrases × two pinned LibriTTS speakers; exact source WAV/PCM hashes and model lock are verified before decoding.
- Piper 1.8.0; pinned LibriTTS-high model revision and CC-BY-4.0 dataset attribution are preserved.

## Source-pin correction

The first main ASR workflow attempt stopped at the pre-download identity check because the workflow YAML still contained the v2 source run/artifact/digest. All provider jobs made zero ASR calls. The workflow pins were corrected to the exact v3 artifact; binding revision 2 records the deliberate fresh first-main attempt. No audio or TTS was regenerated.

## Blind screen

Exactly four WAVs × three independent recognizers (Qwen3-ASR, Whisper-small, Parakeet TDT v3): maximum 12 calls, one attempt per provider/cell, no retries. Each recognizer receives only the WAV, never the target phrase, IPA, speaker ID or another transcript.

Three-way spelling agreement is a machine review cue, not acoustic gold. The results do not automatically become positive, negative or training rows. No TTS, optimizer, training, threshold tuning or model promotion occurs in this workflow.

## Next gate

Review transcript/phoneme observations against the requested prosody, then obtain independent listening before any sample can be admitted. Keep pronunciation_verified, human_gold and training_admitted false until those gates are actually satisfied.
