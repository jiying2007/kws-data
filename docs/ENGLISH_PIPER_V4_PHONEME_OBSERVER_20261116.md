# Piper v4 independent phoneme observer — 2026-11-16

## Exact source

- Piper v4 source run: 43898728142.
- Artifact ID: 14240252131.
- Artifact name: `english-ipa-piper-probe-v4-43898728142`.
- ZIP SHA-256: `9a611676829195c8ac07c07e50705cd8e98822703c60ab66e21ee5be03567193`.
- Sixteen WAV/PCM source cells are verified by per-cell receipts and exact hashes.

The source run's aggregate status is `INCOMPLETE_NO_RETRY` because of a stale four-cell terminal count in the v4 generation script. The actual artifact contains 16 attempts and 16 completed receipts; this observer independently checks source-plan/model-lock hashes and all 16 cell/WAV/PCM identities. It does not regenerate or modify any audio.

## Targets and roster

- Hey Mero: /heɪ ˈmiːroʊ/ — light Hey; first-syllable stress on Mero.
- Hey Vosen: /heɪ ˈvoʊzən/ — light Hey; first-syllable stress on Vosen; voiced /z/ and unstressed schwa.
- Eight new LibriTTS speakers: p4535, p6701, p3638, p922, p2531, p1638, p8848, p6544.

The phoneme model receives PCM only, not phrase, IPA, speaker identity, or ASR transcripts. Exactly 16 forwards, no retries.

## Limits

The phone model's inventory and alignment are a machine observation, not acoustic gold. The outputs must be reviewed alongside the 48 blind ASR transcripts and independent listening. No sample is automatically admitted as a positive or negative. No TTS, ASR, KWS inference, optimizer update, training admission or model promotion occurs here.
