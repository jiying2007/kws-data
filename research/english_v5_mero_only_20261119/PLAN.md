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


## Bounded 512-update sequence-margin FSMN — actual result (2026-11-19)

**Decision: REJECTED as a quality promotion**. The experiment was actually executed on [run 44464214887](https://github.com/jiying2007/kws-data/actions/runs/44464214887), using real immutable source/screen ZIPs and the hash-authenticated [Zh-En teacher event artifact 14500132512](https://github.com/jiying2007/kws-data/actions/runs/44461057432) only as **train-role binary whole-phrase confirmation**.

- Previous random-start 240-update FSMN: source-disjoint synthetic **6/6 Mero misses**, 0/5 ordinary triggers; blank argmax ≈97% (frozen predecessor, no new inference).
- New **43,876-param causal 4-layer FSMN-CTC** actually completed **512 AdamW updates** with frame-length-normalized, class-balanced whole-sequence-vs-all-blank contrast, no old trained weights and no teacher pseudo-phoneme time tags.
- Saved new `models/mero-fsmn-ctc-sequence-margin.pt`, `.npz`, `.onnx`, source/teacher hashes, actual gradient loss trajectory, per-source development receipts; artifacts retained under run above.
- Original train+development: **24 original machine-admitted train** (14 Mero, 10 ordinary; 10 sources), **11 source-disjoint, previously exposed development** (6 Mero, 5 ordinary; 5 sources). Only original train is augmented (clean/SNR10). No historical other word, old calibration/locked, or human-gold claim.
- Actual frozen development: **2/6 Hey Mero full-CTC detected, 4/6 missed; ordinary 0/5 triggered**. The unexposed locked/test and real continuous FAR are still NOT_RUN.
- Optimizer first sampled loss ≈6.42385, last ≈0.00027755, yet new-source development misses remain: pronounced **overfitting** rather than a resolved phonetic transfer. Neither adding blind epochs to the same 14 positive clips nor post-hoc threshold tuning is justified.
- Teacher had a genuine **14/14 TRAIN and 6/6 exposed DEV** event result but returned zero token/timestamp arrays in all real positives; it cannot supply trusted phone alignment. No teacher parameters were transferred to student. Strong teacher performance is *not* student quality certification.

**Next non-repetitive research decision:** do not repeat this failed 512-step same-source fit. Increase independently screened canonical Hey Mero audio from **genuinely distinct generator/parent-speaker groups** and real ordinary/near-miss continuous negatives; independently qualify source/phonetics, keep train/dev/calibration/new locked disjoint, then train with acoustic pretraining or verified phone-level teacher information. A pretrained model may remain a source-qualified accuracy reference, but no L1 student quality, L2 ARM32 or L3 SSC305 qualification is awarded from these exposed short clips. Do not auto-run a new model on plan-only or reused data.


## V5 Hey Mero mono-only full-source recovery and actual 70-clip FSMN (2026-11-19)

**First evidence: genuine source admission improvement without regenerating audio.** On [audit run 44470829024](https://github.com/jiying2007/kws-data/actions/runs/44470829024), eight exact saved source/screen archives and original five independent native screen results were re-evaluated **without changing a single Silver/Weak/UNKNOWN tier, source role or PCM SHA**. Historical V5 had unnecessarily applied the old tri-class complete-group test before a mono Hey Mero-only stage. Existing **24 train/11 development clips** were preserved exactly; **35 additional original independently screened clips** qualified under same-group ordinary(0)+Mero(2) requirements. New strict V5 source coverage: **train 30 Mero + 24 ordinary across 24 original synthetic source groups; development 9 Mero + 7 ordinary across 7 source groups**. This is not 70 new recordings, a resplit, 70 independent human voices, or independent locked evaluation. Historic calibration/locked PCM was not opened.

**Second evidence: actual NEW model, not another run of the failed 35-clip recipe.** [Run 44471889167](https://github.com/jiying2007/kws-data/actions/runs/44471889167), archived artifact **14504422674**, ZIP SHA256 `1c6c96129fe0223a92e09b6141d70c90ff72f7a3bd8144e29827f0cddaa476d0`, completed **360 real optimizer steps** with train-only group/class balancing, ±5% speed, limited mel feature masks, additive SNR10 and train-only normalization. A **43,876-parameter** causal FSMN-CTC state, NumPy weight package and ONNX were saved and reloaded; frozen development achieved **7/9 full Hey Mero CTC detections, 2/9 missed, 0/7 ordinary triggers**. This is a clear advance over the previous 2/6 on a smaller development population, but the cohorts differ and previously observed source identities prevent a statistically independent FRR comparison or production qualification. Model promotion stays **REJECTED**. ONNX SHA256: `a1446d6b67d9e4e78cf97537b95b7b912493a39e9fde042b2cbb50889602aaba`; saved PT SHA256: `483c7ede52f2308a461202da7823af38d1353e98287aba3f926705b1e0ea9fd3`.

**Third evidence: precise remaining failure identity and acoustic hypothesis.** The **two remaining** complete-phrase misses are both the independently screened *Silver* `newpiper-415-mero-fast` and `newpiper-415-mero-slow` clips from the **same synthetic Piper speaker group `piper-libritts-p1112`**, not two independent speakers. Both had also missed in the historical 512-step model, whereas p8008 and p7294 improved in the 70-source model. A separate frozen-weight diagnostic locally replayed the exact saved PT/NPZ and unchanged p1112 WAV; it reproduced the published negative full-sequence margins `-1.705454` and `-0.322062`. With **diagnostic-only 1.5× waveform gain**, both produced the complete ordered greedy CTC sequence and margins `+1.946263` and `+1.938336`. Original p1112 RMS of about **0.0282/0.0234** was lower than comparison Piper dev positives around **0.0382/0.0398**. This is **an amplitude sensitivity hypothesis from exposed development**, not permission to use after-the-fact gain, raise a threshold, relabel these clips, or claim an independent model-quality pass.

**Non-repetitive next objective:** a single preregistered **TRAIN-ONLY waveform/feature gain augmentation** experiment (including low-level speech and ordinary negatives) while preserving the p1112 source as DEVELOPMENT; no dev gradients, no model selection from this exposed set, no revisiting old calibration/locked tests. The real success gate requires a newly source-independent (not merely re-ranked previously observed) acoustic challenge and continuous ordinary/nearphone negatives. Only afterward may a genuinely eligible lightweight C/ARM32 package advance to SSC305. The executed 70-source numeric job is consumed and disabled. No Work/Codex, private repository, paid TTS/ASR or timed patrol.

