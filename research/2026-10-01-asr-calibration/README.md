# Fixed 16-clip ASR calibration input

This research archive contains exactly 16 previously human-reviewed synthetic WAVs (2,010,304 bytes; 1,004,800 frames; 62.8 seconds), unchanged from their original 16 kHz mono PCM16 derivations. The original generation receipts, recovered source hashes, listening copies, and published bytes agree. No new TTS generation, waveform resampling, trimming, mixing, or model download occurred during this publication.

## Consumption and evidence boundaries

- Pin the full Git commit and `manifest.json` SHA-256 before fetching any member. Fetch each `audio/clip-NNNN.wav` and verify its full WAV and raw PCM hashes. The opaque ordering is fixed by `decoder-inputs.json`.
- Give ASR decoders only the opaque WAVs and `decoder-inputs.json`. Do not mount `labels.json`, `source-crosswalk.json`, intended text, speaker names, or prior outputs until decoding is complete. Public archival availability itself does not establish a hermetic decoder isolation boundary.
- `labels.json` preserves 26 human-supported target-presence bits (9 positive, 17 negative) and 6 unknown bits. Ten clips have two reviewed bits; six earlier clips have one confirmed positive bit each. Unknown opposite bits must not become negatives.
- Human review establishes complete target presence/absence within its original scope. It does not establish full transcripts, exact occurrence counts, phonetic/tone truth, or acoustic timestamps. Generation text is intent, not an audio transcript.
- These are selected, already observed development clips, including a targeted error-review subset. They do not estimate a global label-error rate, generalization, continuous FAR, or product qualification. Publication does not admit them to training or change historical failed model conclusions.

## Source and rights

`source-crosswalk.json` binds opaque IDs to original recording identities and exact source-generation receipt hashes. The original receipts identify Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice at revision `85e237c12c027371202489a0ec509ded67b5e4b5`, using official Eric and Serena preset voices. These are synthetic voices, not recordings of human participants. Only the approved 16 derived WAVs are published; no raw 24 kHz parent audio, third-party weights, binaries, credentials, or unrelated recovered files are included.

`rights.json` distinguishes the pinned upstream model-card Apache-2.0 declaration, maintainer authorization to publish this fixed research batch, and the unestablished blanket commercial-output license. The model license is not automatically a license for every output or downstream purpose.

The repository's existing 42-clip canonical training/development catalog is unchanged. This directory is an auditable research input archive, not a new admitted training dataset. `manifest.json` is an integrity inventory for this directory only.
