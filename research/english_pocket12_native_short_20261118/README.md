# English Pocket12 native short-input feasibility probe

This prospective experiment tests a new synthetic voice family for the English
`Hey Mero` / `Hey Vosen` data work. It does not train or promote a KWS model.

The preceding [context64 run](https://github.com/jiying2007/kws-data/actions/runs/44202856867)
completed all 336 planned high-level calls but failed its frozen quality contract.
Its saved [outcome](../english_context64_controls_20261118/README.md) records why:
the outside-target crops included the start of the following word, some requested
contrasts were observed as target-like, and target hypothesis support was uneven.
Those samples and rules remain unchanged.

Pocket12 asks a narrower question: can native, complete short outputs from another
model family support both wakewords and preserve the selected contrasts? All
observers receive the same complete waveform. There are no carrier sentences,
child crops, padding sweeps, retries, or automatic corpus expansion.

## Frozen design

| Item | Contract |
| --- | --- |
| TTS | `pocket-tts==4.0.1`, CPU, native Python API |
| Model | `kyutai/pocket-tts-without-voice-cloning` at `a7056f505a4a8ca72d8b5fb1440071d6f1a35739` |
| Voices | `george` / `vctk-speaker:315`, `mary` / `vctk-speaker:333` |
| Inputs per voice | `Hey Mero.`, `Hey Vosen.`, `Hey Marrow.`, `Hey Bosen.`, `Hello there.`, `Good morning.` |
| Generation budget | 12 high-level TTS requests, once each |
| Observation budget | Qwen, Whisper, Parakeet, and XLSR: 12 requests each |
| Total budget | At most 60 high-level model requests; zero retries or warmup |
| Audio | Preserve native 24 kHz float32; resample the whole output once to 16 kHz PCM |
| Use of every probe row | Diagnostic only; permanently excluded from training |

The source IDs are documented VCTK speaker identities. They do not establish an
American accent or person-level independence from every other corpus. No new
human recordings or voice-cloning input are required. Voice states, model,
tokenizer and wheel are pinned by byte length and SHA256 in the recipe.

Generation observes the native strict checkpoint load, checks its keys, shapes,
storage dtypes, finite values and loaded values, and verifies the six-layer
precomputed voice-state geometry. Native formatted text and token IDs are saved
as input provenance, not as evidence that the requested words were spoken.

## Evaluation fixed before generation

Candidate classification receives only the three ASR outputs, the XLSR phones,
and the original integer whole-support geometry. It does not receive request
labels. The frozen context64 reference policy supplies aliases, contradictory
wakeword vetoes, phone normalization and Mero `/ɛ/` or `/i/` and Vosen `/s/` or `/z/`
hypotheses. These hypotheses are reported separately; synthesis cannot establish
the authoritative natural US pronunciation.

The feasibility gate requires:

1. All 12 source outputs and all 48 observer captures, with complete call ledgers.
2. Both target words supported for both voices: all four positive rows, each with
   three frozen ASR alias votes, one whole-phone hypothesis and at least three
   whole-support windows under the unchanged KWS geometry.
3. All four intended contrasts corroborated by their frozen whole-phone patterns,
   with zero positive candidates. A requested contrast is not a gold negative.
4. All four ordinary rows reproduced exactly by all three ASRs, with no target
   phone pattern and zero positive candidates.

Failure is retained as `STOP_POCKET12_SOURCE_FEASIBILITY_FAILED`; missing evidence
is `INCOMPLETE`. A successful workflow means execution finished, not that the
quality gate passed. No FAR, recall, production readiness or new KWS weights can
be inferred from these twelve synthetic rows.

## Execution and evidence

The checked-in plan permits execution only on its exact first public `main` push
successor and run attempt 1. Pull-request checks are standard-library source and
mock checks only. Generation and each observer use separate fixed runtimes.

Source, observer and runtime artifacts are uploaded even after failure. A final
combiner retains frozen source, source audio, raw observer evidence, evaluation
and file hashes. Missing terminal evidence is never silently counted as zero
model calls. The generator preserves all twelve denominator rows even when
initialization or an individual request fails.

The run must be read back before any follow-up is proposed. Passing this probe
would justify a separately frozen, source-isolated data experiment; it would not
admit these probe samples to training or automatically start a training run.

## Upstream sources

- [Pocket TTS implementation](https://github.com/kyutai-labs/pocket-tts)
- [Pinned model repository](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/tree/a7056f505a4a8ca72d8b5fb1440071d6f1a35739)
- [Voice origins](https://huggingface.co/kyutai/tts-voices)

The pinned model and cited VCTK voice sources declare CC-BY-4.0. Actual asset
bytes and native runtime compatibility are verified by the execution receipts.
