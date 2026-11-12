# English KWS: exact 63-WAV / three-ASR machine review (2026-11-13)

This is a **research-only, machine-observation** report. Source material was
the already-completed seven-preset Qwen TTS run
[43242278962](https://github.com/jiying2007/kws-data/actions/runs/43242278962)
and three independent ASR run
[43246104418](https://github.com/jiying2007/kws-data/actions/runs/43246104418).
No original audio was regenerated; the earlier ten-WAV cohort remains separate.
The saved R0 owner-only checkpoint was neither loaded nor modified for this audit.

## Immutable evidence

| Source | Artifact ID | Artifact ZIP SHA256 |
| --- | ---: | --- |
| New TTS 63 WAVs | 13954744587 | f180004a27ea14c0575f4dd0e2a8476ca048442c76c2dcc186ede6d731cdffcc |
| Qwen3-ASR 63 | 13956812495 | aa483b6c7f360356c4851f1f7d65e8b0db20414129ded0de95b4e2b7575ac601 |
| Whisper-small 63 | 13957321094 | ea7041d3eaa6bbc9e44bc48ba0a90ce521c532a0e0471071fcab52fa38f7f489 |
| Parakeet TDT v3 63 | 13958425005 | cff9be11d89b4ae9f74fa7150b0ab957bc5ccb9dd50be5c1f55bdec390607113 |

TTS source head: bec1a1421d1db2de17afb9c6e3155343becdb980.
Three-ASR source head: 25a328f41aa7b8a2473a1a30024d3e7fcb8dd2c0.
Every ZIP identity above has been independently rehashed against the saved
GitHub Actions ZIP. The source-linked no-model audit verifies every 16 kHz
WAV and PCM hash, generation receipts, exact 63-cell roster, per-provider
model/revision, 189 terminal native-file hashes and completed ASR spans.

## Observed machine-only agreement

Normalization ignores case and punctuation; it does not map different
spelling, accents, languages, missing words or phonemes into one label.

| Prompt intent | Samples | Three-ASR unanimous text | All three match requested text |
| --- | ---: | ---: | ---: |
| Hey Vosen (intended wake) | 14 | 1 | 0 |
| Hey Mero (intended wake) | 14 | 1 | 1 |
| Near-miss (Wosen/Miro/Rosen) | 21 | 7 | 5 |
| Ordinary conversations | 14 | 12 | 12 |
| **Total** | **63** | **21** | **18** |

Unanimous **other** spelling is not gold. In particular, new nonce-name
spelling discrepancies do not prove an utterance is acoustically invalid.
Some non-English orthographies are kept as distinct ASR observations.
A high exact rate on simple ordinary conversation does **not** qualify the
challenging negative set or measure wake false accepts per hour.

## Current data and model admission

- **No** TTS request, ASR transcript or classifier prediction has been
  elevated to verified acoustic truth. All 63 rows remain
  training_admitted=false and human_gold=false.
- Intended wake words require separately source-bound acoustic/phoneme review,
  with a declared pronunciation reference for **each** candidate. Hey Vosen
  and Hey Mero remain parallel; use actual independent speech and
  accent-coverage evidence before changing labels.
- Confusable near-miss prompts are quarantined. Their audio might contain
  a real target pronunciation; do not auto-label them as negatives or use
  them to optimize against development/challenge leakage.
- Five preset families are train-candidate only; Serena/Uncle Fu are
  development-only. These are seven synthetic preset identities, not
  independent human speakers.
- Existing weak-intent 400-update FSMN-R0 remains a non-promoted research
  result; 5/6 held-out synthetic near-misses triggered a wake proposal.
  Re-running the same 63 WAVs or 189 ASR calls is not an improvement.
- No model evaluation here is physical SSC305, continuous FAR/hour, independent
  human FRR or acoustic pronunciation approval.

## Reproducible audit

automation/english_seven_voice_review.py accepts the four **original
Actions ZIPs** using --tts-zip, --qwen-zip, --whisper-zip,
--parakeet-zip, and a new private --output path. It hard-pins all
four GitHub artifact hashes and checks every WAV/PCM/source/native record
before generating a 63-row private observation ledger. It performs **zero**
new TTS/ASR/model-training calls and fails on mismatch, duplicate, source
split leakage or missing results. Its synthetic-only tests include
tampered receipt, mutated original WAV, bad ZIP path/digest, non-ASCII
ASR observation, and no-gold admission counterexamples.

A single protected-main trigger starts the actual read-only audit once;
no schedule and no polling. The resulting machine ledger is retained as an
Actions artifact. Only a subsequent independently authorized pronunciation
adjudication can advance samples toward stricter corpus admission.
