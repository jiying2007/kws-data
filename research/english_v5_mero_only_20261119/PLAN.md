# English KWS V5.0 — Hey Mero only

**Active English target:** `Hey Mero` /heɪ ˈmɛroʊ/ (American English; light *Hey*, first-syllable stress on *Mero*). **No other English wake keyword is active in V5.0.**

## Real research execution

The source of truth for new execution is `automation/english_v5_mero_only.json`. `automation/english_v5_mero_fsmn_ctc_once.py` authenticates the eight already-saved original source/screen GitHub Actions artifacts, using the existing source-only historical verifiers for byte/role/SHA proof. It then creates **a new target-specific snapshot**:

- Old source class 2 (independently admitted canonical Mero) becomes CTC `[HEY, MEH, ROH]`.
- Old explicit ordinary class 0 becomes CTC empty target (background/blank).
- Historical second-keyword class 1, old long-i pronunciation, any diagnostic contrast, failed/missing/unknown row are **excluded**, never silently transformed into ordinary or Mero.
- Only the original, source-disjoint **train/development** rows may be opened by the model. The exposed old locked-test PCM and the existing calibration PCM are never used for gradients, threshold selection or new model comparison.
- A source contributes supervised samples only when it has *both* original machine-admitted canonical Mero and independent ordinary examples. At least six distinct train source groups and two development source groups are mandatory; a genuinely new train and development source group must survive.

No old audio is generated again. Training uses original untrimmed 16 kHz PCM, 40-bin causal (no-centered FFT) log-mel features, four residual left-memory FSMN blocks, and four CTC labels (blank plus exactly three Mero pronunciation sub-tokens). CTC is optimized for an original whole-utterance positive sequence or an explicitly ordinary blank-only sequence. New model weights are initialized from scratch, not warmstarted from any two-word model.

This single first attempt has a fixed CPU training recipe, deterministic data augmentation **only on train**, a train-only normalization, and one final 240-step checkpoint. The fresh, source-disjoint development examples are scored **after** weight freeze; no hyperparameter/threshold choice is made from them. The experiment saves actual model state, source/step receipts, NPZ and ONNX when the exporter succeeds.

## Three completion levels

**L1 — research:** actual finite optimizer updates, weights, SHA-256, source-authenticated train/dev and development token sequence results. Success here is not acoustic product qualification.

**L2 — software:** the exact frozen V5 FSMN-CTC weights must be lowered to an allocation-bounded C/ARM32 streaming implementation. Verify PCM→log-mel→FSMN→CTC token state, event timing, reset/discontinuity and the installed SDK against the Python reference, including true continuous negative exposure. Neither an ONNX file nor generic ARM/QEMU timing alone proves this level.

**L3 — physical SSC305 / acoustic:** requires a new calibration and never-opened locked test, independent wake and long natural negative recordings, true 3–5 m/AEC/robot noise exposure, and real SSC305 BSP/toolchain/board measurements. Formal target remains FRR ≤5% for Hey Mero, FAR ≤0.1/h and P95 word-end-to-callback ≤500 ms with the already-declared confidence requirements.

## Decision rules

A failed model or early one-source development result is not a new baseline or a reason to relax labels. If source qualification, training geometry, actual weights, model export or independent measurements fail, preserve the exact STOP and reason without claiming release. Next high-value work uses the original five-provider phoneme/ASR failure categories to improve *Mero* acoustic input quality and add genuinely independent natural negatives — not repeat already verified TTS. Strong English pretrained KWS/CTC baseline and a low-cost DS-TCN are bounded comparisons, not permission to reopen exposed tests.

No private repository, Work/Codex quota, paid TTS/ASR service, recurring patrol, automatic repeated audio generation or synthetic-only product FAR/FRR claim.
