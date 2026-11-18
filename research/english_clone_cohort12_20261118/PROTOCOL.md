# English KWS — 12 synthetic donor styles × 5 fixed Qwen3 clone renderings

Status: **prospective source generation only**. This campaign cannot grant a KWS training label or model promotion.

## Evidence and pronunciation decision

The owner has selected only **Hey Mero /heɪ ˈmɛroʊ/** and **Hey Vosen /heɪ ˈvoʊsən/** as brand canonical targets.

The two prior real inference campaigns establish why direct labels cannot be taken from requested text:

- [Run 44253051549](https://github.com/jiying2007/kws-data/actions/runs/44253051549): 14/14 genuine XLSR-53 and LV60 phoneme forwards on 7 high-fidelity Qwen3 clips. Native Mero often lacked /ɛ/ or /oʊ/; Vosen often became /z/.
- [Run 44254394313](https://github.com/jiying2007/kws-data/actions/runs/44254394313): 10/10 new controlled text renderings and 20/20 genuine phone forwards. `marrow/merrow` improved independent /ɛ/ evidence; `Vohsen` preserved /s/ in multiple cases. Original raw phone outputs still disagree on some vowels: XLSR /ɔ/ vs LV60 /oʊ/. These are **research findings**, not a waiver or new gold truth.

## Immutable donor sources

Use **only** the [real completed GitHub artifact 14406282094](https://github.com/jiying2007/kws-data/actions/runs/44256271861/artifacts/14406282094) (run `44256271861`, commit `7c352e15466dd62b21a4733e96f08d51d938b647`), ZIP SHA256 `8b86b46759f6f2bd084cd37e1c804c72058fe52ec57f6c221c912eeca96092e5` from [#213](https://github.com/jiying2007/kws-data/pull/213). It contains 12 newly materialized **reference-only** WAV files, each made from the old verified neutral Kokoro audio without any new TTS calls. Every reference has fixed source lineage and fixed role:

| Role | Donor styles |
| --- | --- |
| train (6) | af_bella, af_sarah, am_michael, am_puck, af_nicole, am_fenrir |
| development (2) | af_aoede, am_echo |
| calibration (2) | af_kore, af_alloy |
| locked-test (2) | af_nova, am_onyx |

These are 12 *synthetic preset styles*, not verified human speakers. Derived clones inherit their **exact original source-group and role**, not a new independent identity.

## Fixed, bounded new synthesis

Under **Qwen3-TTS-12Hz-0.6B-Base**, with the original per-donor reference text `Hello there. Good morning.`, create exactly five new unprompted research WAVs per donor, total **60 native forward requests**:

| ID suffix | Generation rendering | Intended canonical class | Truth status |
| --- | --- | --- | --- |
| mero-marrow | Hey marrow. | Mero | UNVERIFIED |
| mero-merrow | Hey merrow. | Mero | UNVERIFIED |
| vosen-vohsen | Hey Vohsen. | Vosen | UNVERIFIED |
| vosen-name | Hey Vosen. | Vosen | UNVERIFIED |
| ordinary-time | What time is it? | background candidate | UNVERIFIED |

Every render has an exact donor group, reference WAV and PCM SHA256, model commit, deterministic distinct seed, original entire waveform converted once to 16 kHz PCM16, WAV/PCM SHA256, sample counts, independent test role and age-free research policy. No old Meero /iː/ data or any historical Mero/Vosen audio is put into the optimizer. The 1.792-second positive duration gate is recorded as an *observed constraint*; long positive clips stay intact and **not admitted**.

Every single generated row keeps `observed_class=null`, `human_gold=false`, `training_allowed=false`; an identical text rendering is not a training truth label. No retries, post-hoc sample replacements, cropping, altered phoneme assumptions or background labels from unknowns.

## Next mandatory phases before any data training

1. Independently run the original **Qwen3-ASR**, **Whisper-small** and **Parakeet TDT v3** recognizers on each full new audio, retaining their raw output without prompts. No requested text or brand label reaches the decoders.
2. Independently observe complete PCM with XLSR-53 and LV60 and retain complete authentic logits + CTC hashes. Do not reassign the locked test set based on responses.
3. Pre-register robust cross-observer phonetic consensus and contrast controls before determining silver/weak/unknown. All 60 rows remain in the denominator, and no 12 group count is guaranteed to meet coverage requirements.
4. Admit only evidence-supported positive and ordinary training rows, with complete train/development/calibration/locked-test **6/2/2/2 independent source groups**. Then activate the disabled fresh-canonical English trainer with old Meero head warm-start disabled.
5. Evaluate FAR/FRR, streamed C/ARM output parity and SSC305 resources separately before model deployment. Synthetic-only evaluation never proves real-acoustic far-field acceptance.

This PR implements **actual fresh corpus generation** and post-generation byte readback, not completed ASR, training labels, weights or downstream quality claims. GitHub public Actions CPU only; no Work/Codex, new scheduled patrols or private repositories.