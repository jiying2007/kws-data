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

All 1,886 restored members are covered by `CATALOG.json`. The complete public archive in this data repository is available for verified recovery, including source/report members. The pipeline repository currently provides an index; the expanded source/report tree and its verify_sources.py helper have not been published there. Catalog planned_pipeline_path entries describe proposed destinations, not live links. Complete data and evidence are stored once per distinct public byte sequence in the data repository. Multiple historical paths with identical bytes share an object. The original 11 delivery archives are not redundantly copied or published as-is.

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

The verifier checks ordered part sizes/hashes, reconstructed ZIP SHA-256, exact object inventory, canonical portable member paths, every public object/member identity and declared transformations. It rejects path aliases, case-folded directory collisions, file/directory conflicts, Windows device names and drive/stream syntax, symlink/special ZIP objects and duplicate JSON keys. Verification reconstructs the ZIP in a temporary disk file and streams each unique object once. Limits are 256 MiB compressed archive, 128 MiB per object, 1 GiB restored bytes, 10,000 members, 128 parts and 16 MiB per metadata file. Restoration requires POSIX directory descriptors and an existing trusted parent directory; it creates a new private destination, refuses existing destinations, uses no-follow descriptor-relative traversal and exclusively creates regular files. On an I/O failure a partial destination may remain for inspection; retries must use a new destination. Verification alone does not require POSIX. Restoring does not execute code or model forward. The publication copy has new package aliases; historical relative references may require deliberate mapping and dependencies before any independently authorized future experiment.

There is no runnable pipeline source-copy verification command in the current publication. Use verify_archive.py above to validate and recover the public source/report bytes. Any future expanded pipeline publication must supply its own verified immutable commit and helper before being described as browsable.

Publication CI checks bytes and pure synthetic safety helpers only. It is neither a new numerical run nor acoustic/product acceptance. Existing unrelated main/nightly model results remain separate.
