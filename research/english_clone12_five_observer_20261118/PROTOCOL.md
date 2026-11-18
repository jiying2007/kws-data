# 60-source English synthetic corpus — first prospective five-observer admission

This research stage is predeclared **before any real ASR or phoneme inference on the new 60-clone corpus**, not a re-reading of old STOP evidence. No old trained head, legacy Meero or commercial corpus is imported.

## Immutable actual audio

- Qwen3 Base 0.6B original source: [run 44257296925](https://github.com/jiying2007/kws-data/actions/runs/44257296925), commit `337ae4f4f318483696a28f0ebc635877b623606a`, source artifact **14406304889**, 1,564,938 bytes, SHA256 `e31cd2dbe920764932000799a7b4bdcf2fd5af12be191a69c2925e5bb891bc4e`.
- Source references: [run 44256271861](https://github.com/jiying2007/kws-data/actions/runs/44256271861), artifact **14406282094**, SHA256 `8b86b46759f6f2bd084cd37e1c804c72058fe52ec57f6c221c912eeca96092e5`.
- Readback: **60/60** 16 kHz mono PCM16 WAV hashes and original 12 donor group IDs match; one group has 5 fixed rows. Twelve synthetic style identities use original 6 train, 2 development, 2 calibration, 2 locked-test groups; this does not assert 12 independent human voices.
- Two positive-request clips exceed the frozen **1.792-second** limit: development af_aoede / Vohsen (1.84s), locked-test af_nova / merrow (1.84s). Both remain in the 60-row denominator as unknown; same-group prospective alternate variants exist. No crop, rate change, selection after reviewing locked-test phonetics, or group reassignment.

## Exact five independent observers and bounded call budget

Original frozen providers run in five separate public GitHub Actions jobs on **the exact same authentic full WAV files**:

- Qwen/Qwen3-ASR-0.6B, revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`.
- openai/whisper-small, revision `31243bad24cc746f07d4c8bfdd2d974872cb1803`.
- nvidia/parakeet-tdt-0.6b-v3, revision `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`.
- facebook/wav2vec2-xlsr-53-espeak-cv-ft, revision `2c733782da5604684829819a5eb744c193fe9398`.
- facebook/wav2vec2-lv-60-espeak-cv-ft, revision `ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4`.

Each provider makes **at most 60** unique, no-retry model calls (maximum 300 total). The providers never see the requested text, brand class, donor role, other observers' predictions or KWS scores. Preserve unprompted three-ASR full strings. Preserve both phone-model entire float32 logit arrays, native token IDs' CTC reproducibility through saved vocabulary, native input sample counts and SHA256; recompute CTC from saved logits in a separate model-free combination job. Missing or contradictory observations become UNKNOWN, never background.

## Prospective source-intake rule

`automation/english_clone12_admission_policy.py` freezes the new **dual independent phone + three ASR** rule from previous separate seven/ten-clip phonetic experiments (runs 44253051549 and 44254394313). It does not retroactively approve the prior Kokoro/Piper corpora.

- Mero can only be supported by *both* `h?eɪmɛr` phone cores. **XLSR's prior demonstrated /ɔ/ for GOAT** is permitted only if the independent LV60 confirms the full /oʊ|o/ tail. No /iː/ Mero (Meero) is admitted.
- Vosen requires both independent phone models to preserve the medial unvoiced **/s/**, a front GOAT-family vowel and schwa-family unstressed vowel. Voiced /z/, old /ɪ/ or unresolved nasal endings are not made positive.
- At least two of three actual independent ASR phrases must match the preregistered exact whole-phrase Mero (Mero/Marrow/Merrow) or Vosen (Vosen/Vohsen/Vosin) spelling family; third ASR cannot contradict with a competing wake name. Three votes = silver, two without contradiction = weak; target text never supplies a vote.
- Ordinary negatives need at least two ASRs observing the fixed complete ordinary phrase, no possible wake phrase in any of three ASRs, and no one-sided canonical phone evidence.
- Never assign a class from the generator request. Request mismatch is a **veto only**. A 1.792-second overlength positive stays UNKNOWN.
- Exactly 60 source rows, five per original group, remain accounted for. Usable groups need independently admitted classes `{0,1,2}` in the same source group. Training snapshot requires all original role minima **6/2/2/2** and all five providers' complete authentic evidence. No threshold sweep, resplitting or locked-test selection.

## Deliverables and explicit STOP behavior

Outputs are original unchanged five observer artifacts and one `SCREEN.json` with every row's SHA256, tier, class or UNKNOWN reason, per-role group coverage, complete denominator, and PASS/STOP. A workflow's source-contract success is **not** data-quality PASS.

Even if the data-screen passes, candidate `training_allowed` fields remain false until a separate pinned trainer binding validates the source/manifest/artifacts and explicitly enables the fresh non-Meero optimizer. This stage does not execute training, measure live FAR/FRR, or claim physical SSC305 qualification.

No Work, Codex, private GitHub repositories, hourly monitoring or cloned real-world persons. Public Actions is event-driven from the single exact protected-main successor.
