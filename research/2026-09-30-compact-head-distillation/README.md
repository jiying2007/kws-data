# Failed compact-head distillation archive

This is a failed research candidate, not a released KWS model or product qualification. See RESULTS.md and student-results/failure-readback.json for complete evidence: retention passed but every target-class MAE failed the fixed0.05 gate. The late KL reporting assertion is preserved; ideal KL and unpersisted training telemetry remain unavailable. No extra training or inference replay repaired the outcome.

## Nonredundant model retention

Only the846 newly trained head parameters are included as finite, lossless FP32 JSON: head-float32.json. The frozen pretrained encoder/CMVN belong to the cited donor and are not described as self-produced weights. No full donor or full experimental checkpoint is distributed here.

The original complete checkpoint remains retained locally; its SHA256 is b61fed9ef590a9c922ee2a6ddc6584b3f29330c45bd4c07d386db459a5514bba. Historical scripts and reports may refer to student-results/compact-final.pt; this public archive deliberately replaces that binary with head JSON plus reconstruction proof. Copied historical files are unchanged and distinguished by copied-file-origins.json.

Donor-reference.json binds the exact official source, model-specific Apache2 declaration, revision, byte count and weight SHA. Obtain the donor separately if authorized. This archive does not fetch it automatically or claim the pretrained encoder is newly authored. Numeric state can be reconstructed without any speech inference:

    python rebuild.py --donor /path/to/the/pinned/base.pt

This optional command needs an existing compatible PyTorch environment and uses weights_only=True. It overlays only the two head tensors and verifies EVERY reconstructed tensor's shape, FP32 dtype and SHA against reconstruction-proof.json. An optional --output writes a new checkpoint without overwriting files. Container serialization bytes can differ; exact per-tensor identity, not a replicated PyTorch ZIP container hash, is the contract. rebuild-verification.json records the local successful no-inference reconstruction.

## Archive-only verification

    python research/2026-09-30-compact-head-distillation/verify.py
    python research/2026-09-30-compact-head-distillation/test_verify.py

Those commands use only the standard library and neither load a model nor train/run speech inference. They check complete file identity, finite lossless846-value head encoding, proof and donor links, unchanged original-result identities and outcome arithmetic. Full donor-backed reconstruction is an additional separately recorded check.

## Data and privacy boundaries

No human source audio, synthetic audio, original third-party weights, runtime environment, credentials or private-device paths are included. Saved recording/dataset IDs and pseudonymous speaker/source identifiers are public research provenance; Qwen voice names refer to synthetic presets. Historical /workspace paths identify this cloud research workspace, not a user's private computer.

The fixed40 HI-MIA-CW source clips remain supervised-admitted=false after the failed independentASR screen. Their audio was used only for unlabeled teacher distribution fitting; no transcripts or ASR outputs were used in the loss. This does not create human-verified background supervision. The original12-clip supervised CTC plan was never trained. Source metadata and source licenses are referenced, not original recordings redistributed.

Original experiment code is retained for audit, including the exact reporting bug. repair_statistics.py contains a corrected stable KL function and a never-executed optional replay path; synthetic tests passed, but no statistics-repair.json exists and no rerun is implied. A future new experiment would need a new preregistration, rather than rerunning these historical scripts over their existing outputs.

## Lossless storage format

The three large student prediction JSON files and teacher-reference.json are stored as deterministic gzip (`mtime=0`), retaining their original logical names. compression-manifest.json binds each stored path, compressed byte count/SHA256, and exact expanded byte count/SHA256. Expanded bytes were compared directly with the original local files; no numeric serialization or result changed. Original reports/scripts still use logical `.json` names for historical fidelity.

The standard-library verifier transparently checks these logical files with a1MiB per-file expansion cap,512KiB compressed-file cap and3MiB total expanded-data cap. It verifies CRC/length/SHA and rejects altered, missing or additional storage members. For manual inspection, decompress a copy outside the immutable archive; do not run old experiment scripts against the archive. Compression reduces upload/storage overhead after a tool transfer stalled, without changing the authorized content or evidence.
