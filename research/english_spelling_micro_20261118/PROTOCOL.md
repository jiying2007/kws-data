# English phonetic rendering micro-campaign (2026-11-18)

Research-only. Do not interpret phonetic spelling, requested brand class or transcript text as gold labels.

**Actual parent evidence**: [#210](https://github.com/jiying2007/kws-data/pull/210) produced 8/8 actual Qwen3 clips and clone returns. Its first phone observation failed before inference; [#211](https://github.com/jiying2007/kws-data/pull/211) completed 14/14 XLSR/LV60 unprompted complete-PCM forwards on exact original clips in [run 44253051549](https://github.com/jiying2007/kws-data/actions/runs/44253051549).

Readback:
- Aiden Mero: XLSR \`h eː m eː ɹ ɔ\`; LV60 \`p eɪ m ɛ ɹ o\`.
- Ryan Mero: XLSR \`h eɪ m ɛ ɹ ɔ\`; LV60 \`h eɪ m ɛ ɹ o\`; actual 2.64s exceeds the 1.792s training positive cap.
- Aiden cloned Mero: both observers support /i/ rather than canonical /ɛ/; exclude.
- Aiden/Ryan Vosen: both observers decode medial /z/; exclude as canonical positives.
- Cloned Aiden Vosen: both observe medial /s/; their o-vowels disagree with raw brand /oʊ/ output convention.

This is insufficient for source-group / train/dev/calibration/test admission and exposes a probable *speaker/model-dependent* pronunciation problem plus observer front/back vowel differences.

## Frozen controlled next test — no old TTS replays

Brand display and training target remain **Hey Mero /heɪ ˈmɛroʊ/** and **Hey Vosen /heɪ ˈvoʊsən/**. The newly generated speech *render prompt* differs from brand spelling. Exactly 10 new Qwen3 requests:
- Aiden/Ryan CustomVoice: \`Hey marrow.\`, \`Hey merrow.\`, \`Hey Vohsen.\` and \`Hey Vozen.\` each (8).
- Qwen3-Base voice clone from the exact original **synthetically generated Aiden reference**: \`Hey marrow.\`, \`Hey Vohsen.\` (2).

\`Hey Vozen.\` is a predeclared voiced-consonant *negative control*, never a canonical positive. Cloned Aiden shares the original Aiden source-group ID, not an independent voice. The two render forms for Mero are alternative attempts at a DRESS-vowel brand sound; do not assume either is equivalent without independent confirmation. No old render string \`Hey Mero.\` or \`Hey Vosen.\` is regenerated.

Generate ten full native Qwen3 waveforms (new deterministic seeds, once); resample entire PCM once to mono 16k PCM16; check unique hashes against the 8 original WAVs and each other; preserve per-row output, model revision, original reference and failure receipts. No replacements, no retries, no uncontrolled prompt search, no human identity/voice cloning.

Then run **20** real independent unconditioned XLSR-53/LV60 phone forwards on these ten complete WAVs with original frozen model weights. Preserve raw phone tokens, logits SHA, CTC hashes, complete PCM SHA, original render text and intended target **separately**. The observers never see the render text or target class.

## Gate and scope

This is a quality *diagnostic*, with no positive/negative training admission at this stage. Future data admission requires actual three independent ASR reads, independent phoneme evidence, robust vowel contrasts with negative controls and a predeclared new rubric where justified. Do not retrospectively reinterpret consumed old STOP outcomes or loosen the current candidate gate.

After a robust canonical phonetic route is proved, a *separate preregistered* 12+ independent-voice cohort must satisfy the existing train/dev/calibration/locked-test role minimum 6/2/2/2. Keep each reference voice and derived clone in one source group and never use prompts as label truth. Only then enable the disabled new-canonical trainer; do not promote if the data are still UNKNOWN.

No private GitHub repo, Work or Codex, no hourly patrol; only existing public GitHub Actions on exact first main successor. Artifact model budget: 10 new TTS, 20 new phone, 0 ASR and 0 optimizer steps.
