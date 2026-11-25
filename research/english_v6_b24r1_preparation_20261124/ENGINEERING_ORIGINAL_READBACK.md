# Original engineering readback and final delivery binding

No primary, replica or engineering result is asserted by this preparation. The
new reader is a local, offline saved-evidence operation. It never downloads,
compiles, imports an acoustic runtime, executes a model or selects a candidate.
The original numerical tolerances and the fixed 140-call matrix are unchanged.

## Complete original input layout

After the actual first engineering run succeeds, retain the following original
files. Reuse already authenticated quality/input archives; do not regenerate
sources or repeat model calls to populate this directory.

| Location beneath `BUNDLE` | Contents |
| --- | --- |
| `originals/result.zip` | Original `english-v6-b24r1-software-sdk-evidence-RUN` artifact |
| `originals/producer.zip` | Original `english-v6-b24r1-engineering-producer-RUN` artifact |
| `originals/consumed.zip` | Original `english-v6-b24r1-engineering-consumed` artifact |
| `originals/input_proof.zip` | Original `english-v6-b24r1-engineering-input-proof-RUN` artifact, with exactly `FETCH.json`, `CONSUMPTION.json`, `api/runs.json`, and `api/markers.json` |
| `inputs/B0.zip`, `inputs/pair.zip` | Exact original model archives in the engineering registration |
| `inputs/engineering_source.zip`, `inputs/engineering_screen.zip` | The registered fourteen existing exposed engineering originals and saved screen |
| `quality/primary/`, `quality/replication/` | Complete original directories accepted by `english_v6_b24r1_quality_readback.py`, including `api/`, `originals/` and all twelve `inputs/` ZIPs |
| `api/*.json` | Six independently fetched public API responses, plus exact copies of the two original execution responses in `input_proof.zip` |

The `api` response roles and public endpoints are listed below. Fetch the six
ordinary roles through the supported connector endpoints. Copy `runs` and
`markers` byte-for-byte from the original engineering `input_proof.zip`; do
not request their endpoints through a connector that does not support them,
reconstruct an API response from other metadata, or create a new collection
execution. The existing authorized engineering job retains these two raw
responses with its normal token after execution, using the same endpoints as
the unchanged `one_execution` guard. The existing four-artifact layout stays
unchanged.

| Role | Read-only endpoint |
| --- | --- |
| `run` | `repos/jiying2007/kws-data/actions/runs/RUN` |
| `commit` | `repos/jiying2007/kws-data/commits/ENGINEERING_HEAD` |
| `tree` | `repos/jiying2007/kws-data/git/trees/ENGINEERING_TREE?recursive=1` |
| `artifacts` | `repos/jiying2007/kws-data/actions/runs/RUN/artifacts?per_page=100` |
| `runs` | Original `input_proof.zip` member `api/runs.json`, captured by the engineering job from `repos/jiying2007/kws-data/actions/workflows/english-v6-b24r1-engineering-once.yml/runs?branch=research/en-v6-b24r1-engineering-once-20261124&event=push&per_page=100` |
| `markers` | Original `input_proof.zip` member `api/markers.json`, captured by the engineering job from `repos/jiying2007/kws-data/actions/artifacts?name=english-v6-b24r1-engineering-consumed&per_page=100` |
| `pipeline_commit` | `repos/jiying2007/kws-pipeline/commits/88a043c6c00bb3a1a62f45098a023c7601042ded` |
| `pipeline_tree` | `repos/jiying2007/kws-pipeline/git/trees/FROZEN_PIPELINE_TREE?recursive=1` |

Use complete raw API JSON responses. The engineering run must be completed and
successful, public, a first-attempt push on the designated branch, and the
only engineering execution. Its activation commit changes only the original
engineering workflow from one reviewed parent. All four original artifacts
must belong to that exact run/head and match GitHub's digest and byte count.
The saved API bodies themselves are pinned in the reviewed lock. Before any
producer code is imported, the reader also verifies that the local `runs` and
`markers` bytes equal the authenticated original ZIP members exactly. These
two responses are captured after the engineering command while its overall
Actions run may still be in progress; the independently fetched `run` response
must establish the final successful completion. Recursive
Git trees must be complete; the reader checks each saved producer file against
the original tree, without using the current checkout as producer authority.

## Explicit reviewed lock

The lock schema is `english-v6-b24r1-original-engineering-lock-v1`. Its exact
fields are:

| Field | Required original identity |
| --- | --- |
| `schema` | The schema above |
| `repository` | `jiying2007/kws-data` |
| `artifacts` | Exactly `result`, `producer`, `consumed`, `input_proof`; each has `id`, `run_id`, `run_head`, `bytes`, `sha256` |
| `api` | The eight roles above; each has exact `bytes`, `sha256`. The two retained execution responses are independently hashed after extraction and must equal their original `input_proof.zip` members. |
| `evaluator_sha256` | `6e6c552804504aa4a027a71312b59bde6fa97237eec7cb138463803fbe9bc1e1` |
| `quality_reader_sha256` | `17d323a32bd487265d2907a99abce8381b5ee8ba46c1acf9190cb0ddd202afb5` |
| `inputs_sha256` | Actual original producer member `repository/automation/english_v6_b24r1_engineering_inputs.json` |
| `producer_manifest_sha256` | Actual original producer member `PRODUCER.json` |
| `original_result_sha256` | Actual original result member `RESULT.json` |
| `result_manifest_sha256` | Actual original result member `MANIFEST.json` |

The lock is constructed from independently retained original API and archive
bytes. It is not authorized by a summary PASS or a workflow's contract job.
Missing fields, null pins, another evaluator, another quality reader, an
expired artifact, any changed byte, an incomplete inventory or a rerun reject
before a completion report can be written.

```sh
python3 -I -S -B automation/english_v6_b24r1_engineering_readback.py check
python3 -I -S -B automation/english_v6_b24r1_engineering_readback.py verify \
  /absolute/BUNDLE /absolute/engineering-lock.json /absolute/NEW_REPORT_DIRECTORY
```

`verify` starts an isolated stdlib worker, authenticates all producer bytes,
and reuses the exact original engineering helpers under the existing offline
quality-reader guard. Both complete original acoustic readers must reproduce
PASS before engineering PCM is read. The worker then reconstructs the original
fourteen-source admission, pre-execution permit, all 28 conditions, 140 saved
arrays/journals, 84 per-frame event CSVs, numerical differences, full-frame
tokens and event clocks. It also compares every candidate pack tensor byte
with the original NPZ, verifies its B0 frontend, preserves the exact original
PT/NPZ/ONNX exports, and authenticates the installed SDK/header/package bytes
and original logs against the complete result inventory and 13 frozen source
files. There is no pack write, compiler call or inference during readback.

Success creates only `completion.json` and the exact standalone `readback.py`
source. No report directory is created when verification fails. The report
schema is `english-v6-b24r1-verified-original-engineering-v1`; successful status
is `VERIFIED_ORIGINAL_DUAL_QUALITY_SOFTWARE_SDK_CANDIDATE`. It retains the complete
original engineering/API lock, both full quality locks, exact original run
identity, all payload byte pins, readback source pin and the explicit zero-call
and research-only scope.

## Final reviewed selection

Only after this actual readback succeeds may the separate pipeline selection
be prepared. Copy the report fields `candidate_id`, `candidate_pack`,
`pipeline_commit`, `files`, `engineering`, `quality_locks`, and `scope` into the
existing `heym43-software-delivery-v1` manifest. Its `readback.report` pins the
actual completion file; `readback.verifier` pins the actual emitted script.
The existing delivery utility remains the manifest contract authority.

Retain the exact report at
`research/english_v6_delivery/readbacks/REPORT_SHA/completion.json` and the exact
verifier at `research/english_v6_delivery/readbacks/VERIFIER_SHA.py` in the
reviewed pipeline Git change. The immutable delivery manifest is stored as
`manifests/MANIFEST_SHA.json`; the reviewed selection pins its exact path,
bytes and SHA. These are future actions, not effects of the reader.

`completion.files` includes the prescribed fallback `models/B0.pack` pin, but
that file is not in the engineering result. The report explicitly states
`baseline_pack_requires_separate_byte_authentication: true`. Delivery must
authenticate the actual fallback bytes separately: exactly 216,960 bytes and
SHA-256 `ff5d0868c035bf5b1c696115eb62f42c36ddbd3a62781207b1340f5e7d1f9d9b`.
The candidate pack must be copied unchanged from the engineering result.

Stage the four exact original ZIPs together with the emitted `completion.json`
and `readback.py` for `kws-pipeline/research/english_v6_delivery/delivery.py
package`. That tool verifies the reviewed manifest and every staged byte,
then creates the software delivery archive. Installation, installed-byte
verification and the separate zero-forward loader check follow the existing
delivery README. This evidence does not establish physical SSC305 execution,
field FRR/FAR, product decoder settings, CAL access or product promotion.
