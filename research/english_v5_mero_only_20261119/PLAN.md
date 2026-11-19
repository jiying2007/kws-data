# English KWS V5.0 — Hey Mero only


## Current exact model disposition (2026-11-19)

**Hey Mero remains the only active English keyword. The first V5.0 scratch causal FSMN-CTC is REJECTED as a quality candidate, although its real weights exist.**

- True 43,876-parameter, 240-AdamW-update model: [run 44453447586](https://github.com/jiying2007/kws-data/actions/runs/44453447586), model artifact 14496732088 and ONNX SHA256 \`0059dfa28da027d736a87a37b5c03ca0ed2c99b402190eb433016cd7601d16f4\`.
- Original source-disjoint machine-reviewed development: 6/6 Mero missed and 0/5 ordinary false accepts. This is neither a measured real-human FRR nor sufficient FAR exposure.
- Immutable [saved-weight token diagnosis run 44454433293](https://github.com/jiying2007/kws-data/actions/runs/44454433293) found ~97.1% blank argmax on the 11 exposed development audio clips. First-20 and last-20 training step mean loss was ~15.13 and ~1.14, but this did not produce a reliable completed greedy Mero sequence.
- Distinct [exact full-word CTC-vs-blank readback run 44456534389](https://github.com/jiying2007/kws-data/actions/runs/44456534389) found only 0.80 exploratory pairwise dev rank fraction across 6 positives and 5 ordinary negatives, **not strictly separable**; train-only sampled 3+3 ranked 1.0. This rules out using a decoder-only fixed threshold on the exposed development audio as a proven solution.
- Never change the historic V5 weight bytes, current readback artifacts, split, old calibration/locked, or original outcomes. No conversion of dropped historical other-keyword audio into negative supervision. No release, SSC305 or actual far-field claim.

Next genuinely useful execution is **new independently source-qualified Hey Mero acoustic quality and English-pretrained teacher/encoder comparison**, plus diverse natural continuous ordinary/near-miss negatives. It should be one bounded model/data test with honest development outcome; if none outperforms, report failure rather than relabeling the old test, training repeatedly on the same 24 clips, or relaxing false-trigger criteria.

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
