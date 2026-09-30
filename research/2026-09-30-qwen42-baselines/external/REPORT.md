# Mandarin external KWS baseline: observed-development readback

## Result

42 previously observed Qwen3 synthetic clips were consumed unchanged from native data commit `2f9658ffa9568076ef547615c76861abed84f56e`. With the parameters frozen before inference, the external model emitted 19 events: 19/20 positive clips contained the requested keyword, 0/22 confusable clips emitted events, and no positive had a wrong-keyword or duplicate-target event.

- Native train role: 14/14 positive hits; 0/16 confusable clips with events
- Native development A: 4/4 positive hits; 0/3 confusable clips with events
- Native development B: 1/2 positive hits; 0/3 confusable clips with events
- Human-reviewed audio: 9/10 positive hits; 0/10 confusable clips with events
- ASR-reviewed audio: 10/10 positive hits; 0/12 confusable clips with events
- 你好小窝: 9/9 target-positive clips
- 小窝小窝: 10/11 target-positive clips
- Sole miss: `qwen3-kw2-eric`

This is an observed-development diagnostic. No new blind test, real-human evaluation, 3–5 m assessment, continuous false-activation rate, endpoint latency, or SSC305 resource/quality comparison was performed. Synthetic voice labels are not independent real participants. The native roles are retained; they do not describe external-model training. The external model uses a large prior speech corpus and substantially different resources.

## Frozen execution

`sherpa-onnx==1.13.8`, `sherpa-onnx-core==1.13.8`, NumPy 2.2.6, CPython 3.12.14, Linux x86_64. Only SHA256-verified official PyPI wheels were installed in this directory's venv; no GPU, paid service, global installation or credential was used. The preliminary v1.12.20 proposal was abandoned before installation because that exact PyPI release was unavailable.

CPU provider, two threads, 16 kHz/80 feature bins, max_active_paths=4, keywords_score=1.0, keywords_threshold=0.25, num_trailing_blanks=1; these match the pinned 1.13.8 Python API defaults. Audio was delivered in 320-sample (20 ms) blocks, a fresh stream per original clip, resetting after each result. `input_finished()` was called after original audio; no synthetic silence was appended. This differs from the upstream offline example's 0.66-second tail flush and may affect end-of-clip detections. No padding variant, threshold sweep, pronunciation alteration or tuning was attempted after observing these results.

Keyword lines (all tokens verified in the downloaded vocabulary):

    n ǐ h ǎo x iǎo w ō @你好小窝
    x iǎo w ō x iǎo w ō @小窝小窝

No artificial sandhi alternative was added. Integer IDs 1/2 used in reports come from the native data label mapping, not sherpa internal vocabulary IDs. Token timestamps and audio-available positions are recorded as raw debug observations; neither is represented as ground-truth-aligned endpoint latency. Confidence was not fabricated when the high-level result API did not supply it.

## Model identity, licenses and provenance

Model: `pkufool/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01`, FP32 encoder/decoder/joiner, ModelScope revision `3787015f084cb241dfa0e4ba237703a2d4322d50`.

Original publisher card: https://www.modelscope.cn/models/pkufool/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/summary

The model-specific README at the pinned revision explicitly declares `license: Apache License 2.0`; it is retained as `models/README.md`. This is evidence separate from the runtime code license. It states WenetSpeech L (10,000 hours), approximately 3.3M parameters, pinyin initials/finals and configurable keywords. The inference files total 13,104,605 bytes, including tokens, excluding the small README. Check actual runtime memory separately; file size is not RAM use.

The WenetSpeech project page, https://wenet-e2e.github.io/WenetSpeech/, states noncommercial dataset download purposes and original-owner audio copyright. We downloaded no WenetSpeech training corpus. This model's explicit published license supports the scoped research inference; its training provenance still merits separate review before a commercial release. No broader commercial-rights assurance is made.

Official inference documentation: https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html

Runtime source tag v1.13.8 resolves to `11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf`. Matching API source, upstream offline example and Apache-2.0 LICENSE are retained under `evidence/runtime-*`. C API availability does not mean pure C implementation or prove SSC305 compatibility. Runtime uses native code and ONNX Runtime; core wheel bundles native dependencies.

## Receipts and reproduction

- `evidence/downloads.json`: exact immutable model revision URLs, wheel URLs and expected SHA256s
- `evidence/*-pypi.json`: official package metadata and wheel digests
- `evidence/pip-freeze.txt`: installed package versions
- `config.json`, `keywords.txt`: pre-inference-frozen configuration
- `prepare.py`: download verification (fails on any SHA mismatch)
- `run.py`: input identity verification, stream execution and scoring
- `results/raw-events.jsonl`: 19 raw events
- `results/readback.json`: complete per-clip rows and separate native-role/review-method groups
- `results/run-provenance.json`: hashes binding script, configuration, keyword file, native export receipt and result
- `results/stderr.log`: empty on completed run

Reproduction from these already verified artifacts:

    venv/bin/python run.py

Do not use this command to overwrite the original evidence during review; run a copied workspace/results destination instead. Future reruns should preserve a unique run ID. Fresh performance testing requires genuinely withheld real audio, long negative streams and an explicit matched resource budget.

## Matched boundary controls

After preserving the raw result, two separately authorized, pre-specified boundary conditions were run using the exact derived audio provided by the canonical C-runner investigation:

1. 500 ms (8000 samples) of zero PCM appended
2. 500 ms zero PCM prepended and 500 ms appended

Both conditions returned the same 19/20 positive hits, zero events on 22 confusables, zero wrong-keyword/duplicate events, and the same sole miss (`qwen3-kw2-eric`). Native-role and source-review splits also remain identical. All 19 events in each condition, as in raw, were detected before the explicit EOF flush. Source review applies to the original audio only; the derived silence variants were not relabeled as reviewed or as new examples.

`run_boundary.py` verifies every derived PCM byte against exact zero prefix + original PCM + zero tail, original source hashes, and unchanged labels/roles/review metadata. Model, decoder, keywords and CPU parameters stayed frozen; no additional padding or tuning occurred. The generated condition receipts bind the upstream derived receipt and input-list hashes. Outputs are preserved separately under `results/tail500ms/` and `results/head500ms_tail500ms/`.

Canonical C-runner investigation independently reports 0/20 hits in raw and both matching boundary conditions. Therefore these specific 500 ms start/end-silence controls do not close the observed gap. This does not isolate acoustic-model quality from frontend/decoder/integration causes. It remains an observed-development comparison with unequal model resources and prior training data.
