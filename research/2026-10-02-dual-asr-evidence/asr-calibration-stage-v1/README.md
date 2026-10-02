# Historical frozen-v1 overview

This overview preserves the earlier label set. The later user correction to
019 changes its 小窝小窝 label to negative and withdraws the shared-false-negative
claim. See the publication root README and human-review-019-v1 for the latest
8-positive / 18-negative / 6-unknown labels and recomputed results.

# Completed dual-ASR calibration: frozen v1 result

Both real CPU FP32 model runs completed on the same 16 exact WAVs (62.8 seconds):
32/32 primary decodes succeeded. Raw tokens, transcripts, PCM bindings, exact-load
receipts, per-clip receipts and clean supervisor termination records are retained.

The unchanged, prospectively frozen text scorer returns **0/26 known-bit coverage
for each model**. All 26 known bits are jointly unknown. This is not 0% accuracy,
not evidence of zero errors, and does not rank the two models. Human labels remain
9 positive, 17 negative, 6 unknown. The full dictionaries expose unrelated
whole-transcript heteronyms; every model target bit receives that blocker.
No transcript repair, punctuation removal, label promotion, scoring-rule change
or outcome-driven model retry was used for this result.

- SenseVoice: 16/16 successes, 39.48 seconds including exact loading/verification;
  peak sampled process-group RSS 2.70 GiB, minimum host available memory 6.09 GiB
- Qwen3-ASR-0.6B: 16/16 successes, 223.62 seconds including loading/verification;
  peak sampled process-group RSS 5.52 GiB, minimum host available memory 4.20 GiB
- One prior SenseVoice load stopped before any forward because its standard
  checkpoint metadata needed a reviewed preserving recipe. That failure remains
  in primary-runs-v1. The completed v2 run used a new frozen run ID
- Actual no-weight constructor checks matched all 917 SenseVoice and 612 Qwen
  state keys/shapes. Qwen nested FP32 configuration and contiguous processor
  inputs were verified. Exact weight readback passed for both models
- Source/asset/PCM/receipt tests and full licensed pypinyin exports are retained

Start with calibration-result-v1/summary.json, transcript-comparison.json and
readout.json. The exact readout SHA256 is
c332fd6390de1ccf5f3b21f3f6a38724721deb45e28f44d74993c2195b43bac3.
The original scorer, frozen rules/dictionaries, inputs and raw evidence are all
preserved so a separately versioned diagnostic can be compared without replacing
this result. No revised scoring policy is part of this archive.

These are selected synthetic probes and a text-evidence calibration. They do not
establish general ASR accuracy, CER/WER, acoustic tone/count/timing, FAR/hour,
training admission, or SSC305 release qualification. Full human transcripts are
unavailable. Offline flags/local assets were used; kernel network isolation and
cgroup limits are not claimed. SenseVoice native-fbank dither RNG reproducibility
is not established. No paid service, GPU or subscription was used.

## Historical preparation record

The remainder records the state before actual execution and is superseded by
the completed evidence above where it describes work as not yet run.


This is local implementation preparation, not a model execution result. No model
weights were downloaded or loaded by these tests; no ASR/TTS inference ran.

## Reviewable implementation

- `pcm/pcm_binding.py`: bounded fd-relative nofollow read, exact canonical PCM16
  WAV parser, immutable normalized float32 bytes and waveform/PCM descriptors
- `asr_stage/architecture.py`: independent source-derived candidate expected
  state schemas, not schemas reverse-engineered from checkpoint keys
- `asr_stage/assets.py`: exact official revision/file allowlists, full local body
  length/hash checks, no downloads or model imports
- `asr_stage/runtime_support.py`: deferred CPU tensor descriptor, streamed native
  byte hashing, exact version/source-member and effective-attention checks
- `asr_stage/adapters.py`: deferred CPU FP32 model loaders and single-clip raw
  evidence capture, with fresh-process and no-repeat-primary-attempt rules
- `asr_stage/decoding.py`: minimal metadata-prefix extraction; never repetition,
  homophone, spelling, punctuation or intended-text repair
- `asr_stage/results.py`: explicit-outcome bridge into PR458's existing importer
- `vendor/checkpoint_guard/`: previously independently reviewed 88-test new
  SenseVoice exact-apply guard, copied unchanged
- `vendor/qwen_loading_gate_v2.py`: recovered Qwen single-file gate, with its
  equality-based WeakKeyDictionary seal replaced by an ID+weakref identity seal
- `human-labels.json`: separate immutable scoring projection, 26 known bits and
  six unknowns. Never passed to model loaders or decoders

The modules contain no automatic downloader or runnable inference CLI. The
outside driver owns user authority, source acquisition, exact artifact locks,
process supervision, network policy, deadlines, output publication and cleanup.
An `armed` data field does not itself establish those controls or grant authority.

## Tests and actual signal verification

From this directory:

```
python3 -B -I -S tests/test_stage_contracts.py
python3 -B -I -S pcm/test_pcm_binding.py
```

The stage suite prohibits actual torch/numpy/transformers/qwen_asr/funasr/yaml/KNF
imports. Runtime paths are source proposals pending real package/model integration.
PCM verification additionally inspected all 16 already recovered waveforms,
verified full SHA/size/header/frame identity, and checked conversion of every
sample. There are 1,004,800 frames = 62.8 seconds, 2,010,304 WAV bytes. All are
mono 16-kHz PCM16 with the same canonical 44-byte header. Their float32 conversion
is exact (`int16 / 32768`); no codec binaries, resampling, trimming or mixing.

The decoder manifest is filename/label-free and uses opaque IDs. Its byte SHA256
is `49e7b6b3a1bdbf9e48de698ea3eb6053fc7c13cb2be800b95f8e1e7a86fd8729`.
The separate human projection SHA256 is
`400b11710507a2fd912b8400b21dba61fd1ab374fae317dd07da29b14c9f7b88`.
No missing model outcome is silently fabricated; all 16 rows must be explicit.

## Independent schema candidates

Qwen source: QwenLM/Qwen3-ASR at
`7c6daf77a2421100f5fb066495372c00129d39ff`, package version0.0.6.
SenseVoice source: modelscope/FunASR at
`904cd18681b8083de5e1039bd0ecebc4f49ede60`, released as1.4.16.
`source-audit/sources.json` records exact source hashes and URLs. Installed package
members must equal the inspected bytes, rather than assuming a matching version
implies unchanged internals. The qualified full package closure remains the
outer installer's responsibility.

The constructor-derived candidates contain:

- Qwen0.6B: 612 state keys, 938,008,576 elements, 3,752,034,304 FP32 payload bytes
- SenseVoiceSmall: 917 state keys, 233,999,167 elements, 935,996,668 FP32 payload bytes

These are complete **candidate** state inventories from inspected constructors
and advertised geometry. They have not yet been compared against a real installed
architecture or real checkpoint. Nonpersistent Qwen positional/rotary buffers are
excluded from checkpoint state. The fixed SenseVoice tokenizer vocabulary25055
is independently documented by official native-runtime architecture, but the
actual locked tokenizer must confirm it before loading. No Qwen omitted aliases
are currently permitted; any actual discrepancy fails and requires review.

Schema/content checks prove consistency/application, not model authenticity.
Full-file hashes from the independently acquired immutable model-asset manifest
are required before deserialization. Neither schema nor artifact source locks
may be regenerated from inconvenient actual results to force a pass.

## Real integration decisions

- Qwen FP32/eager, no vLLM/FlashAttention/forced-aligner. Set nested config hints
  and verify all18 audio plus28 text attention modules actually use eager
- Corrected the old duplicate `return_dict_in_generate` keyword: the pinned
  top-level Qwen generate already supplies it to its thinker
- Full generated token IDs, both decoded views, EOS evidence, exact prompt context
  and input processor tensor hashes are retained. Max256 generated tokens; no
  target text/hotword/context and no forced language
- SenseVoice is constructed from locked config with `init_param=None`, avoiding
  FunASR's permissive load helper. The exact-apply/readback guard is mandatory
- Native-fbank backend only; no implicit torchaudio fallback. Normalized float32
  samples are passed as an independent NumPy copy; WavFrontend scales by32768
  internally. Hash before/after inference detects input mutation
- Preserve source default dither1.0, LFR7/6, Hamming80mel, 25ms/10ms. Torch/NumPy
  seed0 does **not** establish KNF C++ dither RNG reproducibility. Record this
  limitation; do not label this recipe fully bitwise deterministic
- Direct SenseVoice inference receives a tokenizer proxy that captures the exact
  collapsed nonblank CTC token IDs passed to SentencePiece. It does not patch the
  model or claim framewise logits. All-blank IDs are preserved as an empty output
  and unknown text evidence, never converted to an acoustic negative
- One fresh process per model, with exactly one thread-bootstrap call. No attempt
  to load both models in one process; failed model construction also consumes that
  process. Readback uses bounded finite-check chunks and zero-copy byte views

Actual PyTorch tensor APIs, model load hooks/ties, tokenizer behavior, NumPy/torch
boundaries, effective attention, frontend and a full model forward remain to be
verified. The first real canary must preserve its result in the primary run and
must not lead to outcome-driven retries or text repair.

## Resource assessment, not artificial minima

Current dot-cloud probe (2026-10-01) reported Python3.12.14, nine CPUs in affinity,
~9.73GiB total/~8.48GiB available memory, no swap and ~29.45GiB free disk. CPU quota
and cgroup limits were not exposed. Docker/GPU tools and speech dependencies were
absent initially; another task is qualifying isolated dependency installation.
Do not rely on ambient NumPy2.3.5: this recipe requires1.26.4.

The passed GitHub dependency lane reported ~1.10GiB probe RSS and6.86GiB container
peak across install/import/probes. Its10GiB ceiling/32GiB disk reserve were chosen
configuration bounds, not minimum hardware requirements. The model payloads
above are ~3.49GiB and0.872GiB; loader buffers, mapped checkpoint pages, imports,
activations/KV cache and transient preprocessing increase actual peaks.

Sequential cloud execution may fit if the isolated dependency stage passes and
actual model loading retains measured headroom. Use observed process-tree RSS,
available system memory, deadlines and termination receipts. Do not claim that
sampled process supervision equals cgroup enforcement. If using GitHub, use a
new model/audio scope rather than reusing the previous no-model authorization.
No paid service/GPU/storage or subscription is part of this proposal.

## Scientific output

Human word1 labels:5 positive,9 negative,2 unknown; word2:4 positive,8 negative,
4 unknown. Keep old ASR-derived opposite bits unknown. Raw outputs may be acquired
without pronunciation dictionaries; real conservative scoring must use licensed
frozen dictionaries, never PR458's invented fixtures. If scoring rules are frozen
after raw output collection, disclose that timing rather than claim prospective
preregistration. Raw evidence is preserved independently in either case.

Report per-target/model confusion counts, abstention/coverage, jointly wrong
answers and all disagreements. No CER/WER without full human transcripts; no
general accuracy, acoustic tone/count/timing, FAR/hour, training admission or
SSC305 release claim follows from this selected 16-clip set. Qwen1.7B remains a
later quality comparison candidate; this first0.6B run is not an optimality claim.
