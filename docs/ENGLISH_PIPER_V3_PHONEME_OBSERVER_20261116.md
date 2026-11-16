# Piper v3 independent phoneme observation — 2026-11-16

## Exact source

- Piper v3 source run: 43886265084.
- Artifact ID: 14235001433.
- Artifact ZIP SHA-256: `1085fb90a94372bb06c9ab358e3855e3da6666868db4242805aaa496be134884`.
- Four exact 16-kHz mono PCM16 WAVs, verified by source receipts and PCM/WAV hashes.

## Targets

- Hey Mero: /heɪ ˈmiːroʊ/ — light Hey; stress on the first syllable of Mero.
- Hey Vosen: /heɪ ˈvoʊzən/ — light Hey; stress on the first syllable of Vosen, voiced /z/, unstressed schwa.

The observer receives PCM only. It does not receive phrase, IPA, speaker identity, ASR transcripts or target labels. It runs exactly four forwards using the pinned `bookbot/wav2vec2-ljspeech-gruut` model revision and records raw/stressless phone token sequences.

## Limits

This is an independent machine phoneme observation, not acoustic gold. Its model's phone inventory, alignment and stress representation may not exactly match the owner target. Combine it with the blind three-ASR results and independent listening; do not automatically accept or reject samples.

No new TTS, ASR, KWS inference, optimizer update, training admission, human gold or model promotion is performed in this workflow.
