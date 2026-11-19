# English KWS — developer-only retraining of fresh canonical causal head V2

**Objective:** actual 49,283-parameter new model training and measured development improvement, without reusing the already exposed old 22-bag synthetic locked test for model selection. This is research, not a release-ready product.

## Evidence of v1 failure and immutable provenance

- Exact original successful numeric fit: [run 44332105007](https://github.com/jiying2007/kws-data/actions/runs/44332105007), [artifact 14438929272](https://github.com/jiying2007/kws-data/actions/runs/44332105007/artifacts/14438929272), immutable zip SHA256 b94c34a5cd95eb0bde6f1dd5f8ca4b49636073ce34f1a46c7ecbb9dbfabcb84e; 96 actual optimizer steps, exact full model bin SHA256 49a73ad2c9856b8c8d44fe07846de74a83ace80d15f8fe22ed58aff4d404b040, calibration threshold 0.90.
- Original frozen test was opened once (not to tune): Hey Vosen missed 6/8 (75%), Hey Mero missed 4/8 (50%), 10 false-or-extra events. Synthetic research model **quality FAIL**. It is forbidden to use this old locked set for v2 model/checkpoint/threshold selection. The original numerical evidence and weights remain immutable, not edited.
- Original source admission remains 8 train, 4 development, 3 calibration, 3 locked source groups, with 63 audio clips selected from original actual 462 WAVs and five independent ASR/phone reviews; only machine weak/silver, no human-acoustic gold.

## Prospective V2 optimization: exact train/dev only

- One public-first-main protected event, new torch seed **20261119**, **384 real AdamW train-only updates** with exactly three predeclared candidates at steps **96, 192, 384**; no old Meero head initialization, all source roles and model assets remain SHA-pinned.
- Exactly the original 49,283-param 16×96→32 ReLU→3 streaming causal head, original same v0.5.1 frozen openWakeWord feature ONNX (research license only), same 16k full PCM, train-only normalizer, class/source-balanced weak multi-instance loss. Source audio is loaded only for train and development roles; no new TTS or old source inference replay.
- At **one fixed 0.8 development working threshold**, choose from three predeclared step checkpoints by deterministic tuple `(worst_per_word_development_miss_rate, total_dev_false_or_extra_events, mean_per_word_development_miss_rate, update_step)`. Fixed before new V2 training and **does not consult any calibration or locked audio or results**.
- Save all 384 optimizer draws/loss receipts, full development metrics for every candidate, exact NPZ and 197,908-byte C-binary head weight checksum. Fit and output must be verified by real numeric GitHub Actions. Zero prod promotion implied.
- No calibration threshold or new locked test in this PR. Once dev-only weights are genuinely frozen, generate **new source-disjoint** calibration and locked-test audio (new Piper speaker IDs not included in the prior 63/train/dev/cal/old locked), blind review and evaluate *once*. The previous synthetic locked 22 bags remain exposed and never re-qualify as independent for V2.
- Even a better dev model is not commercial qualification: need independent real-world human FAR/FRR and SSC305 3–5m physical checks. Current openWakeWord feature model license must be verified for commercial rights before any shipment.

No Work/Codex quota, private repository, paid API, per-hour timer or repeated voice-clone generation. Original 6 reviewed ZIPs and all previous STOP/PASS decisions remain immutable.
## Exact first-run zero-numerics recovery

The first main V2 run [44425553398](https://github.com/jiying2007/kws-data/actions/runs/44425553398), head `edf154f3b7d2e977fe78b567d023a289e7054ba3`, successfully passed source-contract and byte-audited all six original public source/screen archives. It **failed at the initial command-line parsing of `--guard`**, before Python 3.12 setup, input download for the numeric stage, frozen model loading or any optimizer call. Original numerical job `135229803469` has all model/optimizer/weight steps `skipped`; no training artifact exists. This failure is a software execution entry issue, not a new trained candidate.

The narrow recovery must attest exact original public-main run ID, failure HEAD, initial failed step, all downstream steps skipped and zero artifact count by read-only GitHub Actions API. Only the first protected-main successor of `edf154...`, attempt 1, may re-enter the numeric stage. Old target audio, 6 source ZIPs and previous locked synthetic evaluation remain frozen and are never replayed as new data or used for model selection.
