# Twelve synthetic reference voices — original-audio reuse

**Purpose:** Assemble 12 synthetic neutral speech references for voice cloning, without generating a single new voice or changing the rejected historical KWS training labels. Research only.

## Frozen inputs

- Original successful Kokoro native corpus run: [44218734553](https://github.com/jiying2007/kws-data/actions/runs/44218734553), original source commit `7e9797789cba7d1476d02f14e21af91dc3cd0712`.
- Exact public artifact: `14389757284`, archive bytes `74,476,997`, SHA256 `712e083b1bc7e1451c8eb43c2f99c94549d34f8d71db81d8cf2ef965e1df72da`.
- Original complete 144-row manifest and 4,144 file members plus FILES.json are authenticated against original per-file digests. This corpus's original admission remains STOP; no sample is re-labeled.
- For each preset listed in the frozen role map, select exact **Hello there.** and **Good morning.** neutral WAVs by their original IDs and source-group ID. Retain the same original train 6, development 2, calibration 2, locked-test 2 role mapping.

## Materialize actual references

Each reference WAV contains the **complete unmodified original PCM16** from those two short neutral utterances, concatenated with exactly 1,600 silent PCM samples (100 ms at 16 kHz). The derived audio is explicitly a **voice-cloning reference only**, never a natural utterance or KWS positive/negative. Text passed with the reference is `Hello there. Good morning.`.

This uses 24 already generated and authenticated neutral WAVs to create 12 fixed ~2.65–3.83-second references, with reproducible PCM SHA256, output WAV SHA256, all source IDs, role, source-group ID and original model lineage. No new TTS, no ASR, no phone model, no training.

**Independence constraint:** The twelve Kokoro voice presets are distinct research style identities, not verified distinct human speakers. Every future voice clone inherits its donor preset source-group. Do not mix training/calibration/locked-test source groups or claim human-source independence. Unknown ancestral identity remains unknown.

## Boundary

This reference-only corpus can seed a **separate new** Qwen3 Base cloning campaign after the quality of a phonetic render has been verified independently. It has **zero automatically admitted training samples**. Source-only campaign completion is not a model-training or FAR/FRR result.

No additional hour/scheduled jobs, Work/Codex, private repository or old audio generation. Only one exactly pinned public-main source reuse job.
