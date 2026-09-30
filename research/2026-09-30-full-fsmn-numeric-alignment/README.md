# Full FSMN numeric diagnostics and one native cost profile

This archive preserves a failed numerical candidate and its subsequent bounded diagnoses. It is research evidence, not an admitted dataset, model release, fidelity acceptance, product qualification, or SSC305 benchmark. No training or new arithmetic variant is performed by the archive verifier.

## Observed conclusions

- Original whole-network B gate (absolute `1e-4` + relative `1e-5`) failed: sequential FP32 **33/264**, four-way FP32 **35/264**, diagnostic double accumulation **29/264** checks. No gate was widened. Torch whole/split comparison itself failed **3/63** checks; this does not waive B.
- Frozen CMVN **12 calls** were exact; feature-level splice/indexing **29 cases / 67 calls / 8 tests** passed. A fixed local stage-5 affine check had max error **3.814697265625e-5** and zero old-B failures. These are partial checks.
- Full-layer oracle diagnosis: **132 layer + 12 cache** records passed the stated local rounding/propagation checks, while **27** aggregate records still failed old B. The largest forward envelope is **2.3080694676873595e15**, too loose to establish end-to-end fidelity. Its mathematical-consistency pass is not parity or qualification.
- Actual PCM diagnosis: **42 observed development clips × 3 fixed silence conditions = 126** recordings. Target hits were **11/20 raw, 11/20 tail500ms, 12/20 head500ms_tail500ms**; confusable-trigger clips were **0/22** in each condition. All non-score event fields were exact, but **34** triggered-recording scores differed (max **2.694541942516171e-6**, within the unchanged `1e-5` score tolerance). **Full-field exact equality is false.**
- Original retained scalar probability report passes: **25,529,977** elements, maximum absolute difference **8.046627044677734e-6**. Strict final logits fail: **6,126** elements / **63** calls / **42** recording-condition pairs, maximum **0.005767822265625**. fbank `1e-3` absolute gate fails **6/2,368,160** elements, max **0.0012373924255371094**; splice has **11** failing elements. CMVN's fixed `2e-4` gate passes (max **0.0001863241195678711**). Cache/intermediate old-B failures remain.
- A later, separately authorized fixed native cost profile ran **3 × 42 raw clips**, **235.68 audio seconds**, reproducing saved native logits/events. Whole PCM→events CPU was **5.758814368 s**, RTF **0.024434887847929396** (24.435 ms/audio second, 2.4435% of one x86 host core). This measures the numerically failed candidate, not an accepted product. No SSC305 measurement or extrapolation is made.

## What is actually verified

Run from a repository checkout, using Python standard library only:

```sh
python3 research/2026-09-30-full-fsmn-numeric-alignment/verify.py
python3 research/2026-09-30-full-fsmn-numeric-alignment/test_verify.py
```

`verify.py` checks a fixed required logical and physical inventory, original logical lengths/SHA-256, compressed-stream and per-piece hashes, strict UTF-8/JSON, safe paths, symlinked members and ancestors, bounded single-stream gzip expansion, receipt/source bindings, and original failure gates. Four approved major report identities are pinned independently of the editable manifests. It recomputes event equality/score differences/decision counts from retained reference and native events, aggregates per-call scalar error reports, and recomputes profile time/sample denominators and disjoint timing summaries. `summary.json` must match those recomputations.

**It does not reconstruct or independently verify per-element tensor differences from the excluded NPZ arrays.** Probability, fbank, CMVN, logits, layer and cache elementwise errors remain original scalar-report evidence. It does not rerun inference, prove that reported C/Python executions occurred, or turn checksums into external authenticity signatures. Mutation tests cover both all-hash rebound attacks and semantic validators independently of frozen report hashes.

## Timing scope and double counting

The resource-profile original report gives the detailed timing scope. The measured process is Python/ctypes + C frontend/full2599 model + Torch softmax + original Python decoder on an AMD EPYC 9V74 host. Model calls include CMVN. Frontend/splice timing already contains callback-copy timing. Standalone CMVN microcalls and per-clip initialization are outside the whole pipeline interval, so they must not be added to its components or subtracted from model time. The verifier sums the four disjoint boundary groups and separately records nested/outside intervals.

Warmup used the first three raw recordings; the RTF denominator is only the three measured 42-clip passes. Startup imports/audits/preloads are separate. The whole interval retains feature/logit copies and timing-record allocation. Pass I/O includes validation, microcalls and observers; zero warm-host physical reads is not a claim about target flash/DRAM. One thread was sampled at startup/pass boundaries, not proven at every instant. Peak RSS **273,309,696 B** belongs to the Python/Torch research process. Model+CMVN payload size **3,027,732 B**, C PCM state **62,264 B**, C model mirror **22,544 B**, and declared scratch arrays **4,624 B** are scope-limited figures, not a complete C application's RAM budget.

## Preservation, storage, and chronology

All **37 historical logical files** retain exact bytes, lengths and SHA-256 in `logical-files.json`. JSON records larger than 48 KiB use deterministic lossless gzip (`mtime=0`, no filename), divided into physical `.bin` pieces of at most **48 KiB**. The original three previously-gzipped report streams are preserved exactly, including their gzip header bytes, before splitting. Their historical compression manifests remain unchanged. New compressed records preserve their exact original uncompressed bytes. All other leaves are bounded UTF-8 text. `archive-manifest.json` binds every physical leaf except itself; the verifier rejects unlisted files and directories.

`RESULTS.md` and `resource-profile/RESULTS.md` are **unaltered historical reports**. The former's earlier “no resource profiling followed” statement was true at that checkpoint, before the separately authorized resource profile preserved here. Its earlier “no speech run” paragraph likewise predates its appended PCM section. Read these statements in that chronology, not as claims about the final archive.

The synthetic PCM oracle's first generator shared a mutable namespace. The later corrected oracle was generated after first synthetic C output, using independent namespaces and unchanged C arithmetic/gates. That chronology remains in the original report. The actual 126-recording Python reference used one sequential spotter and reported exact historical logits/events. This archive does not fabricate the excluded first synthetic oracle or large reference arrays.

Historical failed C variants and their executed comparators/header are evidence, not reusable current code. `prior-stages/fsmn-before-full-oracle.c` is the guarded historical kernel bound by `local-affine-prereg.json`, before the later memory-helper refactor; its original bytes are preserved. `compare-double-executed.py` and `fsmn-initial-executed.h` were **hash-verified historical reconstructions** matching identities already recorded in the original reports; they were not continuously retained original files. The verifier binds all three historical comparator/source/header identities to those reports.

Current reusable C, loaders, diagnostic scripts and tests are fixed to the [kws-pipeline source snapshot](https://github.com/jiying2007/kws-pipeline/tree/fe501f51243b40015fa4c17c267e66e9274c20d8/research/donor_fsmn). Commit `fe501f51243b40015fa4c17c267e66e9274c20d8` is the actual source-snapshot commit, not a merge SHA; its presence does not imply that draft PR #455 has merged. The original source SHA-256 values in the preserved preflights remain authoritative for their historical executions. Original donor payload, complete tensor manifest, executable libraries, large NPZ trajectories, audio and dependencies are excluded. Existing data catalogs and earlier research archives remain unchanged.

## Historical implementation license

The historical C variants/header derive from pinned Apache-2.0 WeKws sources (FSMN authors Yueyue Nyy and Jing Du). See [NOTICE.md](NOTICE.md) for the exact upstream commit, modifications, attribution and exclusions, and [LICENSE.wekws](LICENSE.wekws) for the verbatim license. These added notices do not alter any historical C or report bytes. They do not grant additional rights to datasets or unpublished donor weights.
