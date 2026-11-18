# English canonical fresh24 — prospective source experiment

**Status before first numerical run: FROZEN / no fresh WAVs / no admission.** Research-only synthetic audio. The data and model are not commercially qualified, and source presets are not real-human identities.

## Pre-existing evidence (never re-admit)

- MeloTTS CMU run [44271169127](https://github.com/jiying2007/kws-data/actions/runs/44271169127): 25/25 new audio; first [44274204197](https://github.com/jiying2007/kws-data/actions/runs/44274204197) and recovered [44276152391](https://github.com/jiying2007/kws-data/actions/runs/44276152391) produced all 125 independent ASR/phone observations. Original source gate remains **5/25 ordinary Silver, Mero 0/5 and Vosen 0/5**, with 20 UNKNOWN.
- Older Qwen 12-donor / 60-clone cohort [44257296925](https://github.com/jiying2007/kws-data/actions/runs/44257296925), [44259758593](https://github.com/jiying2007/kws-data/actions/runs/44259758593): original independent evaluation remains **0/6,0/2,0/2,1/2** complete groups. Any posterior diagnosis of 10/12 is **posthoc exposed diagnostic, not admission**.
- Original 12 Kokoro donor source assets remain permanently excluded from this experiment; no old Meero training bag or historical speaker aliases are input to the new optimizer.

## Frozen BEFORE creating a new waveform

The sole source roster, official LFS SHA256, speaker ancestry, 24 immutable group roles and all seven fixed text/phonetic control requests reside in [english_fresh24_protocol.json](../../automation/english_fresh24_protocol.json), enforced by [source worker](../../automation/english_fresh24_generate_once.py). All 24 official Kokoro 1.0 assets are distinct and not in the prior 12-style cohort; mixed-language presets are **synthetic voice identity diversity, not proof of independent humans or a guarantee of English pronunciation quality**.

Source roles are prospectively **12 train / 4 development / 4 calibration / 4 locked-test**. Minimum *qualified* source groups remain **6/2/2/2**. For each preset, two newly rendered official neutral utterances become a new immutable synthetic reference (no use of original old WAVs); only then one pinned Qwen3-TTS-0.6B Base model generates exactly seven fixed new requests per group: two Mero canonical renderings, two Vosen, one ordinary, old-vowel Meero **contrast only**, voiced Vozen **contrast only**. Planned numerical ceilings are **48 neutral Kokoro forwards + 168 new Qwen clone forwards**. No retry, no rank-based choice, no role reassignment, and no silent source substitution.

Both words are frozen: Hey Mero **/heɪ ˈmɛroʊ/**, Hey Vosen **/heɪ ˈvoʊsən/**. Requested text and direct model phonemes are not gold. Two old/voiced contrasts are never normal negatives for supervised loss.

The pre-registered [full CTC likelihood / contrast score](../../automation/english_fresh24_ctc_admission.py) computes exact log-probability across each complete saved XLSR and LV60 CTC tensor, canonical versus old /iː/ and voiced /z/ or /b,w/ alternatives. A target requires both independent model margins strictly positive, at least 2/3 complete unprompted ASRs, no contradictory names, full positive clip duration limit, exact bytes/role lineage, and no old-reading transcript. Unscorable / missing stays UNKNOWN; ordinary negatives require actual positive ordinary ASR votes and no wake indication.

## Stage transitions (no premature promotion)

1. The one-shot public-main workflow validates immutable policy and source before model installation; it generates and fully authenticates 24 new donor WAVs.
2. If 24 donors are authentic, the same original run generates up to 168 **new** Qwen clone WAVs with exact source-group pinning. No ASR or KWS training happens at this source stage; preserve partial failure receipts.
3. Only a **new exact-artifact-bound** five-observer workflow may invoke Qwen3-ASR, Whisper-small, Parakeet, XLSR, and LV60 on the same saved full PCM. The policy file is already frozen; never adjust after seeing the outputs. Preserve logits, exact SHA, original transcripts, unknowns and contrast diagnostic outcomes.
4. Data admission uses train-only samples for optimization, development for model selection, calibration once to select thresholds, and locked-test once **after weights and thresholds are frozen**. Do not use locked audio, votes or scores in training/threshold search, even if saved as a public research artifact.
5. Only if source roles meet real **6/2/2/2 qualified groups** and all hashes/review gates succeed may a fresh-seed English head train using the existing KWS source/SDK, with old Meero warm-start disabled. Keep independent human far-field FAR/FRR and physical SSC305 evidence separately **NOT_RUN** until measured.

The official source assets are Apache-2.0 research inputs; downstream model/recording rights must remain traceable. This experiment creates no private GitHub repository, paid API job, Work/Codex usage, scheduled monitor, commercial-release gate, or false product qualification.

The new workflow is guarded against every branch, PR numerical inference, non-public repository, second run attempt, or unexpected main predecessor. CI test fixtures are **not** acoustic approval; only actual saved WAV/model receipts qualify.
