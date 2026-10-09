# 2026-10-08 retained host research / 完整研究成果公开归档

This publication preserves completed research, including negative and inconclusive results. It does not rerun or revive any one-shot experiment, select a winning model, change product code or approve shipping. Old reports saying “private” describe their original delivery state; publication of this separately hashed copy is now authorized. Their original evidence and verdicts remain historical and unchanged.

## Coverage and conclusions

| Package | Members | Retained scope and result |
|---|---:|---|
| `saved36-diagnostics` | 68 | Saved A20 posterior/decoder diagnostics, original failures and source boundaries; no new acoustic qualification |
| `fixed1200-training` | 242 | Fixed D20/D90 1200-update endpoints, donor-derived initialization, training logs, 90-source audio/weak-label identities; `NOT_EVALUATED` remains the original training-stage result |
| `host-runtime-ab` | 167 | A/B export/readback, common host C source/build evidence and 462 synthetic state rows: `PASS_HOST_ONLY`; no strict-FP32 or device qualification |
| `fixed50-inconclusive` | 930 | Fixed synthetic audio, per-item ASR, technical checks and historical versions: `EVAL_INCONCLUSIVE_LABEL_SUPPORT`; ASR is not human ground truth |
| `official-reference-v1-failed` | 118 | `FAILED_NO_RETRY`: D20 internal stage07 failure for features2-whole; 483 reserved rows (479 reference + 4 C), D90 never started. Missing C fulltrace was not reconstructed |
| `local-certificate-v2-failed` | 345 | Local certificates and probability comparisons passed only their narrower gates; D20 PCM9600 second callback (callback2) final logits exceeded the frozen tolerance. `FAILED_NO_RETRY`, 939 reserved rows, D90 not run |
| `reference-admission-hardening` | 8 | Standalone saved-only checker and 19 synthetic safety tests. Original reference's 8 final-logits gates pass but internal partitions fail; no reference admission |
| `source-admission-ledger` | 4 | Source, exposure and weak-research admission records; no commercial-output rights or product training claim |
| `fixed50-blind-listening` | 4 | Offline single-rater observation tool, not gold labels. Actual browser playback/storage validation was not completed. Publicly browsing other evidence can expose answers and invalidate a later claim of blind independent observation |

All 1,886 restored members are covered by `CATALOG.json`. Source/report copies are directly browsable in the pipeline repository; complete data and evidence are stored once per distinct public byte sequence in the data repository. Multiple historical paths with identical bytes share an object. The original 11 delivery archives are not redundantly copied or published as-is.

## Publication transformations and integrity

`CATALOG.json` records every original SHA-256/size, publication SHA-256/size, logical path, object name and explicit transformation list. Private file/conversation IDs, coordinator/one-shot control fields and host-specific path prefixes are redacted. Numerical arrays, audio, weights, binaries and unmodified text remain byte-identical. Original embedded manifests still describe original historical bytes, not the transformed copy; use the outer catalog to verify this publication. Redaction does not retroactively alter numerical provenance or grant execution authorization. Identifiers marked `REDACTED_*`, `WORKSPACE/` and `HOME/` cannot be used as old runtime paths.

`SOURCE-PROVENANCE.json` maps byte-identical historical code to exact existing public repository paths and commit identities where available. Other study-specific code preserves its original headers and provenance; see `THIRD-PARTY-NOTICES.md` and `LICENSES/`.

The archive is an evidence recovery bundle, not a hermetic execution environment. Dependencies, original external donor body and previous unrelated assets are not newly redistributed. Dependency manifests are identity metadata, not included package bodies. Do not unpickle weights from an untrusted source. Recovery and verification never deserialize model objects or execute historical code.

## Rebuild the public evidence layout

In the companion data repository's `research/2026-10-08-d20-d90-host-research` directory:

```sh
python3 -B verify_archive.py
python3 -B verify_archive.py --restore /path/to/new-empty-destination
```

The verifier checks ordered part sizes/hashes, reconstructed ZIP SHA-256, exact object inventory, unique member paths, every public object/member hash and declared transformations. Restoration refuses an existing destination, rejects traversal and writes regular files only. Restoring does not execute code or model forward. The publication copy has new package aliases; historical relative references may require deliberate mapping and dependencies before any independently authorized future experiment.

Pipeline source-copy verification:

```sh
python3 -B research/host-research-2026-10-08/verify_sources.py
```

Publication CI checks bytes and pure synthetic safety helpers only. It is neither a new numerical run nor acoustic/product acceptance. Existing unrelated main/nightly model results remain separate.
