# Coverage candidate research SDK packaging

This is a new static integration path. It leaves the original snapshot160
packager, inference implementation and runtime sources unchanged.

## Required actual evidence

The packager accepts the authenticated original training archive only as a
source of immutable runtime/frontend/license bytes. A new coverage artifact
must contain its retained candidate, native and ARM/QEMU cached-embedding head
parity, raw replay output, saved Python reference frames and source identities.
It also requires a separately saved independent audit with explicit external
run/head binding, `status=VERIFIED`, retained quality and both parity checks.
An evidence audit passing while the quality gate fails cannot produce a package.

The post-run `PACKAGE_INPUTS.json` has this schema. Fill identities only from
actual authenticated output; this example is intentionally non-executable:

```json
{
  "schema": "english-coverage-package-inputs-v1",
  "runtime_base": {"bytes": 61290521, "sha256": "2aa398cd11b787342d91c056fed188d8e3607929da9fdaf873451b5cf05b07f1"},
  "runtime_base_manifest_sha256": "1dda5e4504cbff66b0b373e6583501e7d61aff3f69ec0167ce188d97b779eaa2",
  "run_id": "ACTUAL_RUN_REQUIRED",
  "head_sha": "ACTUAL_40_HEX_COMMIT_REQUIRED",
  "quality_artifact": {"bytes": 0, "sha256": "ACTUAL_64_HEX_REQUIRED"},
  "independent_audit": {"bytes": 0, "sha256": "ACTUAL_64_HEX_REQUIRED"},
  "auditor_sha256": "FROZEN_AUDITOR_64_HEX_REQUIRED"
}
```

```sh
python3 -I -S -B english_coverage_package.py --check-inputs \
  --training /absolute/training-43867764366.zip \
  --quality /absolute/coverage-quality.zip \
  --audit /absolute/independent-coverage-audit.json \
  --pins /absolute/PACKAGE_INPUTS.json

python3 -I -S -B english_coverage_package.py --build \
  --training /absolute/training-43867764366.zip \
  --quality /absolute/coverage-quality.zip \
  --audit /absolute/independent-coverage-audit.json \
  --pins /absolute/PACKAGE_INPUTS.json \
  --output /absolute/new-english-coverage-sdk
```

All packaging and both SDK verify modes use the standard library only. Head
NPZ payloads are parsed as six finite, C-contiguous LE float32 arrays and copied
directly into the existing portable C binary ABI, then compared byte for byte
with the actual exported candidate. Saved C output is compared with saved
Python scores and clocks; no probabilities are recomputed.

## What the package establishes

The new candidate has passed the predeclared research quality gate and its
native/ARM head replay checks. The frozen frontend source and model files match
the original byte pins. Full PCM frontend execution for this new head and
physical SSC305 acceptance remain `NOT_RUN`; historical proof cannot fill
either field. Class order, fixed 0.8 / three-confirmation / 13-hop-cooldown
events and zero EOF padding are unchanged. Every formal admission/promotion
flag remains false. Same-source noise variants add conditions, not independent
source recordings, and exposed development counts are not natural FAR/FRR.

The new `english_coverage_sdk.py` verifies the complete package, then invokes
the unchanged generic inference CLI only with explicit `--execute`.
There is no unchecked external-head path override.

## Source-only checks

```sh
python3 -I -S -B test_english_coverage_package.py -v
```

Tests use synthetic byte fixtures, never a model, ASR, TTS, frontend or C runner.
