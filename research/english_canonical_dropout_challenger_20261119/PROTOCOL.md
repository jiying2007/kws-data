# Canonical English KWS — regularized dropout challenger, train/dev only

This is a second **actual** dev-only research model, not another replay of old locked KWS results. Source inputs are the already independently verified original six GitHub archives and all 63 machine Silver/Weak research clips. Models and weights are source-hash bound; old Meero heads and data are forbidden.

## Root cause evidence

- Original V1 training [44332105007](https://github.com/jiying2007/kws-data/actions/runs/44332105007): genuine 96 train-only AdamW updates, new 49,283-parameter streaming head, 0.90 calibration threshold. Its original previously opened locked synthetic evaluation has Vosen 75% miss and Mero 50% miss, 10 false-or-extra events, and is **not release-quality**. It cannot be used for any further model/threshold selection.
- Dev-only V2 [44426347276](https://github.com/jiying2007/kws-data/actions/runs/44426347276), [original artifact 14484377578](https://github.com/jiying2007/kws-data/actions/runs/44426347276/artifacts/14484377578) genuinely completed 384 new optimizer updates with three preregistered candidate checks. Best dev checkpoint was step 96 (Vosen misses 2/8, Mero misses 2/10, total 4 false-or-extra); step 192 worsened Vosen to 3/8 and step 384 to 5/8, showing increased steps alone do not improve generalization. It produced new BIN SHA256 `4808bc3aae00a3adefdeb5474fa4a1dd96f0045024f5a6946e38b0c725c999a6`, but no new calibration or locked test was opened and no release approval was granted.

## Distinct predeclared algorithmic challenger

- Exact same source- and role-isolated 8 train / 4 development / 3 calibration / 3 historical locked source-group binding, original 63 Silver/Weak research clips, original full-PCM 16k and two pinned v0.5.1 openWakeWord ONNX research feature assets.
- New training seed **20261121**, AdamW **learning rate 0.0003**, train-only **Dropout(0.25)** between ReLU and the last 3-class layer. The inference model is still exactly 16×96→32 ReLU→3, **49,283 parameters**, same 197,908-byte native C-binary, with no runtime dropout or model-size increase.
- Exactly 384 train-only updates, candidate checkpoints only **96/192/384**, one unchanged fixed 0.8 development threshold. Select by `(worst per-word dev miss rate, total false-or-extra events, macro per-word dev miss rate, update)`; no threshold or model choice consults old calibration, old locked or previous locked performance.
- Source-level English Mero `/heɪ ˈmɛroʊ/` and Vosen `/heɪ ˈvoʊsən/` training data remain exclusively machine Silver/Weak, with old Meero and voiced z excluded. Nothing uses previous heldout audio as a new train/dev label.
- After real original GitHub Actions CPU training, save native BIN/NPZ, 384 optimizer draws/losses, candidate development event metrics and all hashes. If dev regression worsens, do not promote the challenger. Real quality requires a fresh, genuinely source-disjoint new calibration and locked test generated **after** this checkpoint is frozen.

Human 3–5m FAR/FRR and physical SSC305 parity are NOT_RUN; public research ONNX feature assets are not commercially approved. No Work/Codex/private repository/paid API/cron or repeated TTS/ASR data generation.