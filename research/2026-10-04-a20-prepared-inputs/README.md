# Saved A20 CPU preparation inputs

This isolated research archive retains the exact outputs of CPU preparation run
[37198298586](https://github.com/jiying2007/kws-pipeline/actions/runs/37198298586),
source commit `4a62fc91f7140c6e875692c88235152da4c68e4d` (PR #470).
It permits later, separately admitted training work to reuse 32 validated native
PRE-CMVN features without repeating 32 native and 20 official frontend passes.
Publication and integrity CI do not start or authorize training.

## Contents and boundaries

- 32 native FP32 features: 2,899,200 bytes, 1,812 rows × 400, 199 callbacks,
  5,470 fbank rows; normalization is PRE_CMVN
- 20 official FP32 reconstructions: 1,627,200 bytes, 1,017 rows × 400,
  112 callbacks, 3,070 fbank rows; all 20 match published historical feature hashes
- One complete original-A20 control: 72,960 bytes, padded shape 32 × 95 × 6
- 14 technical JSON files plus the original artifact manifest
- Original first-attempt failure retained separately: 7 JSON members, no raw
  feature or control payload, `FAILED_NO_RETRY` at `METADATA_REQUIRES_DIST_MISMATCH`

The successful run was GitHub run attempt 1 and cumulative preparation attempt 2.
It performed one original-A20 initialization control, zero backward passes,
zero optimizer constructions/updates, and zero decoder/development/F-arm calls.
The original state SHA256 is
`c05623683b4616badda5535eefb0bfaecde63efbf3bfbceec6e7440a5fc19eb2`.
The run produced preparation evidence, not a new trained checkpoint.

The official tensors were reconstructed; historical raw tensors were not
recovered. The largest saved native/official difference is 0.0030474066734313965,
under the preregistered descriptive compatibility diagnostic. It creates no new
numerical acceptance tolerance or numerical-equivalence qualification. All old
FAILs stay FAIL. The original failure's package name and request counts remain
NOT_CAPTURED; later diagnosis does not fill in missing historical observations.

D20 are weak historical rehearsal inputs and Qwen12 are exposed reviewed sources.
These are not fresh holdout evidence. There is no FAR/FAh/FRR, word-tail,
board/capture/soak, real-room generalization or product qualification claim.

## Exact identity and transport

The original successful gzip is 2,472,892 bytes, SHA256
`4a35cde90a5b268a19082a63c26a6ad798c537dda092e486686fef01744b9708`.
It is split at 1,048,576-byte boundaries into three ordinary content-addressed
chunks. `TRANSPORT.json` gives their exact order, offsets, sizes and hashes and
all 68 inner member identities. The gzip is not rewritten or recompressed.
The failed gzip is 5,441 bytes, SHA256
`a2180f5fabd728819f6df09905eb0eb6cdeabfab0937688fcbec784764636bef`.
Both original checksum files are retained byte-for-byte.

`PROVENANCE.json` records original Actions outer-ZIP identities, their two member
identities, and the identical original/public gzip identities. Outer ZIPs are not
duplicated here; their exact payload pair is represented by this archive. The
successful outer ZIP was 2,473,344 bytes, SHA256
`3af3372927dfb69d62a4d79b6f739c36347b4bb6327d95051c47dca2f8cbd487`.
There is no claim that concatenating transport chunks reproduces the outer ZIP.

Pinned source documents at source commit `4a62fc91f7140c6e875692c88235152da4c68e4d`:

- [SOURCE-FREEZE.json](https://github.com/jiying2007/kws-pipeline/blob/4a62fc91f7140c6e875692c88235152da4c68e4d/research/token_preparation/SOURCE-FREEZE.json), SHA256 `7ac5c192a1a28938a80ce792b93def06bf3378b10adb624035e3fafad3828ed2`
- [PROTOCOL.json](https://github.com/jiying2007/kws-pipeline/blob/4a62fc91f7140c6e875692c88235152da4c68e4d/research/token_preparation/PROTOCOL.json), SHA256 `ab1655d93cec7a839bdf1f95f9a09c9940900bbddc9a7a5763b02d44f6e28b6e`
- [TRAIN32.json](https://github.com/jiying2007/kws-pipeline/blob/4a62fc91f7140c6e875692c88235152da4c68e4d/research/token_preparation/metadata/TRAIN32.json), SHA256 `16f8cc4e237e770f4c2bd69ade993e12c33af3f19ffd61900d6fd4307d020389`

## Offline verification and reuse

Use Python 3.8+ and its standard library. No NumPy, Torch, network, package
installation or scientific workload is needed. From the repository root:

```sh
python3 -B -m unittest discover -s research/2026-10-04-a20-prepared-inputs/tools -v
python3 -B research/2026-10-04-a20-prepared-inputs/tools/restore.py verify research/2026-10-04-a20-prepared-inputs
```

For a separately approved consumer, first check out the exact immutable
40-character data commit containing this archive, then supply that same commit.
The restore operation refuses a different or dirty checkout and an existing
output directory. Keep output outside the checkout:

```sh
python3 -B research/2026-10-04-a20-prepared-inputs/tools/restore.py restore \
  research/2026-10-04-a20-prepared-inputs \
  --expected-commit "$PINNED_KWS_DATA_COMMIT" --output "$NEW_OUTPUT_DIRECTORY"
```

Features are under `success/members/native/`; shape, source order, per-source
hashes and callback geometry are in `success/members/native-features.json` and
`initial-control.json`. Future training must pin this data commit, the manifest
and exact source freeze; preserve source labels/roles; apply the original fixed
CMVN exactly once downstream. A branch name, `main`, one-day Actions artifact or
private storage URL is not a training-input identity. Archive validity alone is
not training admission.

The verifier pins `TRANSPORT.json` SHA256
`21687bd7d918c6aab02a513234b78f816d11dbeec4111b28052440f281bfc229`,
checks original checksum bytes, bounded gzip expansion, exact regular members,
JSON duplicate/nonfinite rejection, FP32 finiteness, and inner manifest equality.
The invented tests also reject missing, reordered, oversized and tampered
chunks, unsafe paths, symlinks/hardlinks, duplicate members, concatenated and
truncated gzip, and invalid commit/dirty-checkout consumption.

See [rights, sources and privacy](RIGHTS-SOURCES-PRIVACY.md), the unchanged
[rights receipt](rights-and-provenance.json), and [saved summary](SAVED-SUMMARY.json).
