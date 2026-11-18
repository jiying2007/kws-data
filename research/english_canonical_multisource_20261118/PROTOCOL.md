# Canonical English KWS — exact six-artifact training snapshot and first native head

Status at authorship: **source gate PASS; new training not yet run**. All results here are research-only with synthetic source audio; there is no claim of independent human 3–5m FAR/FRR, commercial use or actual SSC305 CPU/RAM.

## Verified independent machine evidence

- Source 1: new Qwen3 voice clones 168/168 whole WAVs: [source run 44289263966](https://github.com/jiying2007/kws-data/actions/runs/44289263966), original ZIP 14422271973 SHA256 26396073f5be2f91e35575bedb586737405162388e32bfe85633ed3d796a9583; [review run 44306842000](https://github.com/jiying2007/kws-data/actions/runs/44306842000), 840/840 blind native forward calls, screen ZIP 14429385634 SHA256 5a9b226b8fe475331d9f03d3ca4f91d5bb94c5270113f6d0cf3e8b4c36fe8316. Qualified train/dev/cal/locked groups **4/1/3/0**; original STOP unchanged.
- Source 2: new direct Kokoro original phonemes 126/126 WAVs: source run 44310115097, original ZIP 14429578824 SHA256 34cf7879f974f62c9f6046a72a964002f30bf59d2186a423ac382a76ff69d0e9; review run 44312719394, 630/630 blind model forwards, screen ZIP 14429884666 SHA256 4a685afa1d8e8c35767900290a7fd590f2153fb69f3b2e5ae68d5dada58b11e1. New qualified groups **0/0/0/1**; cumulative STOP unchanged.
- Source 3: genuine new Piper LibriTTS original source identities 168/168 WAVs: source run 44321329022, ZIP 14434542792 SHA256 d1f75716fefd954d59c3289c37e7e6670965ee98917436676896312e4b781aa1; review run 44323694022, five separate blind native models SUCCESS, screen ZIP 14436377961 SHA256 c6b8573bae44ffa110b776482da74919d16e6b1437c6c3e1f462fd1b7a28a519. New qualified groups **4/3/0/2**, and original combined report PASS.
- Total original source group qualification = **8 train / 4 development / 3 calibration / 3 locked-test**, minimum **6/2/2/2**. This is synthetic per-source *acoustic machine* admission, not a human-gold or product-grade far-field claim.

## Actual identity-preserving research snapshot

The source binding script first validates the public protected-main run identity, the exact **six source/review artifact IDs, full archive bytes, ZIP hashes and all 462 original WAV/whole-PCM hashes**, then verifies each source request against its independently preserved Silver/Weak/UNKNOWN screen row. Every source-group classification is recomputed from 0/Mero/Vosen evidence. No old Meero, voiced-z control, UNKNOWN, failed-length positive or unqualified group can enter the new optimizer.

The exact 63 selected independently machine-reviewed rows are sealed by original source corpus, group and role: **28 train / 13 development / 11 calibration / 11 locked-test**, with all roles and source ancestry immutable. Labels remain machine Silver/Weak; positive targets are canonical Mero /heɪ ˈmɛroʊ/ and voiceless Vosen /heɪ ˈvoʊsən/. Nothing is extracted from unrelated previous 60/25 probes or legacy Meero training weights.

## One fixed new model execution, no exploration sweep

Only the first public-main successor of 4992666d4356d76301ad8ccbd71fad12db3d7caa, attempt 1, may run numerical training. Frozen frontend uses byte-pinned openWakeWord v0.5.1 melspectrogram and 96-dimensional speech embedding ONNX, licensed research-only pending external model-asset rights. Fixed lightweight causal prediction head = 16×96 input → 32 ReLU → 3 outputs, **49,283 new parameters**. Newly initialized deterministic seed 20261118, exactly **96 AdamW optimizer updates**, source-balanced 3-class training batches; train-only normalization and whole-PCM-generated synthetic non-speech context are explicit.

Development may only diagnose the one fixed final checkpoint. After that model is saved and immutable, the calibration set alone selects from the **predeclared** threshold grid; threshold and weight SHA must be saved *before the locked-test PCM is even read*. The locked-test prediction/evaluation then occurs once, and uses the saved frozen model/threshold without further selection. No quality figures are claimed before the actual saved results are inspected.

The source-only PR steps run normal/-O/-OO byte and policy tests; PR-specific archive audit downloads and rehashes every real source but performs zero numeric inferences. Only the main event contains the optimizer. Result artifact includes new NPZ weights, exact 197,908-byte fixed head C binary, train/dev/calibration/locked receipts and finite replay status. Later C/ARM/SSC305 parity and independent human 3–5m FAR/FRR remain separate gates, not automatically satisfied by this run.

No ChatGPT Work, Codex, private repositories, timers, paid TTS or ASR calls, automatic posthoc label reassignment or legacy compatibility paths are introduced.
