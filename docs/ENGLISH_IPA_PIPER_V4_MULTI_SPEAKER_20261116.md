# English Piper v4 eight-speaker pronunciation screen — 2026-11-16

## Why v4 exists

The previous v3 source set contained only two LibriTTS speakers and four WAVs. Blind three-ASR observations did not provide an accepted positive for either target; the independent phone observer showed substantial Mero mismatch and a Vosen schwa mismatch in one speaker. v4 broadens speaker coverage instead of repeatedly regenerating either consumed speaker.

## Frozen targets

- Hey Mero: /heɪ ˈmiːroʊ/ — unstressed/light Hey, primary stress on Mero's first syllable, long /iː/.
- Hey Vosen: /heɪ ˈvoʊzən/ — unstressed/light Hey, primary stress on Vosen's first syllable, voiced /z/, unstressed schwa.

Raw eSpeak sequences remain exactly `heɪ mˈiːɹoʊ` and `heɪ vˈoʊzən`.

## Fixed new voice roster

Eight previously unused LibriTTS train-clean-360 speaker identities from the pinned Piper high multi-speaker config:

| Piper speaker ID | Config speaker index |
|---|---:|
| p4535 | 2 |
| p6701 | 3 |
| p3638 | 4 |
| p922 | 5 |
| p2531 | 6 |
| p1638 | 7 |
| p8848 | 8 |
| p6544 | 9 |

The two v3 speakers p3922 and p8699 are excluded from v4 generation. The exact config bytes and full speaker map are rechecked by the worker before synthesis. The upstream Piper voices repository is MIT; the LibriTTS train-clean-360 voice dataset is attributed under CC-BY-4.0. Dataset attribution is retained.

## Bounded generation

- 8 new speaker families × 2 phrases = exactly 16 new TTS calls.
- One pinned Piper 1.8.0 model load, one attempt per cell, no retries.
- Exact model revision, ONNX SHA-256, config MD5, source speaker IDs, WAV and PCM hashes retained.
- No ASR, phoneme observer, training, labels, optimizer, threshold tuning or qualification in this stage.
- No prior WAV is overwritten, relabelled or regenerated.

## Review gates

The next stage must run Qwen3-ASR, Whisper-small and Parakeet TDT v3 blind on all 16 exact WAVs (48 maximum calls), plus the independent phone observer (16 forwards). ASR spelling agreement is not acoustic gold. No sample enters formal training until pronunciation is independently reviewed and rights/source-family policy is satisfied.

The eight-speaker expansion is a coverage experiment, not a model-training dataset and not proof that either wake phrase has passed acoustic acceptance.
