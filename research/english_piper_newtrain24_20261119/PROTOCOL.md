# English KWS canonical: new train/development source campaign

**Pre-audio source plan**, target only high-quality independent canonical synthetic voices. This cannot by itself admit a dataset or promote the English wakeword model.

## Grounded reason

- All four canonical data stages previously achieved real machine-reviewed 6/2/2/2 coverage across Qwen, direct Kokoro and Piper sources (8/4/3/3), and exact 63 Silver/Weak original PCM were bound to a new 49,283-parameter English causal head.
- First real model [44332105007](https://github.com/jiying2007/kws-data/actions/runs/44332105007) completed training but failed original once-opened locked test (Hey Vosen 6/8 missed, Hey Mero 4/8 missed). Those locked results are permanently excluded from next model selection.
- More steps alone [44426347276](https://github.com/jiying2007/kws-data/actions/runs/44426347276) and bounded dropout/AdamW challenger [44427448763](https://github.com/jiying2007/kws-data/actions/runs/44427448763) both genuinely trained new heads and found **no meaningful development gain**: Vosen 2/8 misses, Mero 2/10 misses, 4 false/extra development events at best. Small original train corpus is a likely limiting factor, not a factual proof about any unmeasured real microphone FAR/FRR.

## This new source fixes the training/voice coverage, not the threshold

Original official Piper 904-speaker LibriTTS model high, fixed revision `de3dcfdf1912bb49726dc4aa11c26017ce2ac62a`, ONNX SHA256 `9127a559e11603f10b366d1a20ac7426826081dbc521de4c2420c57728d73f0f`, unchanged source/license. **24 previously unused source indices 424..447**, all pNNN speaker mapping cross-checked against original 0..57 and 400..423 and previous Libri numeric IDs, with fail-closed collision detection. The new indices are reserved for **18 train + 6 new development** original speaker-embedding groups. They are distinct speaker embeddings, not asserted independent human-acoustic reference tests or genuine far-field recordings.

Seven frozen generation requests per group: two direct-IPA Hey Mero `/heɪ ˈmɛroʊ/`, two direct-IPA Hey Vosen `/heɪ ˈvoʊsən/` with /s/ not /z/, one ordinary background phrase, and old /iː/ Meero and voiced /z/ Vozen diagnostic controls which **cannot become supervised negatives**. Whole original 22.05kHz PCM is resampled exactly once to 16k, no crop, full WAV+PCM SHA256, no original source audio regenerated, one public first-main run and zero automatic retries. Budget **168 original Piper TTS forwards**.

After real source artifact success, separate original five-native unprompted Qwen3-ASR, Whisper-small, Parakeet, XLSR and LV60 phoneme observations must validate each original full PCM with the *previously frozen canonical-vs-contrast CTC margin policy*, Git blob `0a688325d0d36e83ef6623ac678511df24a92374`. Only newly qualified train/dev source groups can augment original training, and a new **training-snapshot-bound seed** may be optimized on them. All previous calibration/locked sources and the already opened first locked KWS test are excluded from hyperparameter/model selection. A future genuinely unexposed calibration/locked set is required before performance claims.

No Work/Codex, paid ASR/TTS, private repo, scheduled patrol, old model retroactive admission, or physical SSC305/human far-field assertions.
