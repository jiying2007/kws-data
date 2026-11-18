# English canonical KWS — new Piper LibriTTS source identity pilot

**Pre-audio source protocol**: no new Piper inference, ASR, CTC or KWS training happened at the moment of registration. The complete upstream source configuration and role roster are pre-frozen in `automation/english_piper_fresh24_protocol.json`.

## Verified predecessor and exact remaining gate

- [Fresh24 five-model review #44306842000](https://github.com/jiying2007/kws-data/actions/runs/44306842000) performed 840 genuine original-PCM model forwards, with 58 Silver / 110 UNKNOWN and 4/1/3/0 source groups (train/dev/cal/locked). Its original screen remains unchanged.
- [Fresh18 direct-phone review #44312719394](https://github.com/jiying2007/kws-data/actions/runs/44312719394) performed 630 genuine original-PCM model forwards, with 26 Silver / 1 Weak / 99 UNKNOWN and one new locked-test source group. Frozen cumulative total: **4 train, 1 dev, 3 calibration, 1 locked**, against **6/2/2/2**. Actual preserved screening artifact #14429884666, 15,068 bytes, ZIP SHA256 `4a685afa1d8e8c35767900290a7fd590f2153fb69f3b2e5ae68d5dada58b11e1`.
- Original result remains `STOP_CUMULATIVE_6_2_2_2_SOURCE_COVERAGE`. No new optimizer update, weight, calibration or genuine board performance was produced by these campaigns.

## New TTS source-family experiment, not another Kokoro style replay

All 54 previously available official Kokoro voice presets were already used across prior campaigns. The new source uses **Piper LibriTTS English high, 904-speaker model**, specifically fresh source indices **400–423**, rather than generating any old 25/60/168/126 WAV again. Model revision and original ONNX SHA256, config MD5 and sizes are pinned from the earlier validated public Piper implementation. Each numerical source speaker is recovered from the official fixed `speaker_id_map`; any reused LibriTTS numeric speaker ID from all old Piper/Libri cohorts fails before a model forward.

Preassigned group roles are **12 train, 6 development, 6 locked-test** by the fixed voice index sequence; there is no new calibration role because three old calibration sources already passed. These are independently identified source **speaker embeddings**, not independent verified human test recordings; the Piper model shares training ancestry and the evaluation claim is synthetic-only.

Every source will synthesize exactly seven predeclared direct raw eSpeak IPA inputs, with the same canonical brand reading: `heɪ mˈɛɹoʊ` (Hey Mero, /heɪ ˈmɛroʊ/) and `heɪ vˈoʊsən` (Hey Vosen, /heɪ ˈvoʊsən/) each at lengths 0.92 and 1.08; one ordinary `What time is it?`; and diagnostic-only old-vowel Meero `/iː/` and voiced-/z/ Vozen. The two controls are **not** ordinary negatives. Requested text is never ground truth.

One public first-main successor of original main `6fc03ff706b8949b9a6c066f4478890db8de28aa`, attempt one, can execute **168 total native requests**. Each original 22.05 kHz PCM is resampled whole to 16 kHz without crop, with WAV+PCM SHA256, source pNNN identity, native IPA and duration gate preserved. An error keeps complete partial receipts without retry or replacement.

## Non-leaking mandatory continuation

If authentic 168 WAVs exist, an independent native Qwen3-ASR/Whisper-small/Parakeet/XLSR/LV60 run on the complete **same** audio must use the **previously frozen** canonical-vs-contrast full-CTC likelihood rule Git blob `0a688325d0d36e83ef6623ac678511df24a92374`, no target prompting, no threshold change and no silent UNKNOWN-to-negative conversion. Source groups remain fixed across train/dev/locked. New qualified source groups add only to immutable old 4/1/3/1, requiring at least **6/2/2/2**.

Even a source-level PASS will require a separate exact PCM/ancestry/source review data-binding for the disabled **fresh-seed** English KWS trainer, proper development-selection/calibration/locked-once event evaluation and reproducible SDK. Neither synthetic source success nor CI alone proves independent real 3–5m FAR/FRR or SSC305 resource conformance.

No Work/Codex usage, private repository, paid API, timer or hidden old-audio regeneration. Research scope and source licenses are retained: Piper engine GPL-3.0, source data LibriTTS CC-BY-4.0, model/source identity traceable.
