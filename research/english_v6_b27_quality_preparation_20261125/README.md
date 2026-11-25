# B27 once-only quality and complete-original readback

These adapters are inert until the real B27 TRAIN and independent new SOURCE
originals have been completely verified and their exact inputs reviewed. No
model, source, observer or optimizer is run by this preparation. The only
scientific change belongs to the separately frozen TRAIN design: natural weight
0.08 to 0.16. Quality retains the original B14 implementation, gains, source
minima, numerical forward loop, raw output capture, and once-only consumption.

## Model and assessment interfaces

`english_v6_b27_primary_eval.py` exports `STEMS`, `MODEL_COMPARISON`, `B0_PT`,
`B9_PT`, `CONTROL_PT`, `INITIAL_PT`, `validate_pair_receipt(receipt, spec)`, and
`validate_training_archive(pair, spec, metadata)`. The replica inherits these
same algorithms under its own explicit panel, inputs, workflow and proof binding.

| Role | Fixed identity |
| --- | --- |
| B0, B9 | Original historical checkpoints and artifact pins |
| Quality control | Original B24 challenger, PT `f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4` |
| TRAIN initialization | B23 PT `784def9df6c2a8bfbecdb7ba2d7b70630337a1cda72561c05392327aa980297f` |
| New challenger | Actual once-trained B27 PT, bound only from its complete original TRAIN proof |
| B27 training and evaluation assessment | `8c7b4571d2ac08546fd00789fc2809df8c4290ad53a38c1f6b33a993331ae2ee` |
| B27 design | `956f0695e6b4e3ac450bc3881c5ef13985d775210cf43cf4da98947a5a1846bf` |

The control and challenger stems are respectively
`control/models/mero43-b24-challenger192` and
`challenger/models/mero43-b27-challenger192`. The historical B24 assessment is
retained separately as `original_B24_training_assessment_sha256`. No old B24
candidate hash or B23 initialization can stand in for the new B27 candidate.

## Offline original-input binding

`bind_b27_quality.py` accepts one bundle of complete original API bodies for B0,
B9, new TRAIN and new SOURCE, and twelve original ZIPs: B0, B9, pair, source,
screen, five native observers, TRAIN producer, and SOURCE producer. Its bundle
schema is `english-v6-b27-{panel}-original-quality-binding-inputs-v1` with `groups`
and `archives` maps. Every reference is an existing original file, never a
follow-latest selector. Binding requires the corresponding complete external
proof directory from the SOURCE preparation.

```sh
python -I -S -B automation/bind_b27_quality.py primary \
  /absolute/original-bundle.json /absolute/proof-root /absolute/fresh-review
```

The binder rechecks the actual trained pair, all source/native bytes and screen,
all historical and reserved population exclusions, complete Git-bound upstream
producer manifests, prospective output capacity, and exact required resources.
It emits new reviewable inputs/producer registrations and an audit into a fresh
directory. It performs no forward or remote mutation. SOURCE resources are
preserved completely; a captured workflow template is a resource, and replica
also preserves the exact already-passed PRIMARY inputs and producer documents.

## Independent readback interface

`english_v6_b27_quality_readback.verify_quality_original(directory, lock,
proof_root)` returns a fully independently recomputed report using an isolated
`-I -S -B` stdlib worker. `fetch_quality_original(directory, lock, proof_root,
reuse_inputs=None)` fetches the original three output artifacts and complete
raw APIs, retains the twelve original inputs, and returns the same fully verified
report. Calling a second identical readback immediately after this fetch is
unnecessary.

The quality directory is exactly eighteen files:

| Directory | Required contents |
| --- | --- |
| `api/` | `run.json`, `commit.json`, `artifacts.json` |
| `originals/` | `result.zip`, `producer.zip`, `consumed.zip` |
| `inputs/` | All twelve original input ZIPs with the registered role names |

The lock schema is `english-v6-b27-original-quality-lock-v1`. It contains exactly
`schema`, `panel`, three complete original output pins (`result`, `producer`,
`consumed`), `evaluator_sha256`, `inputs_sha256`, `producer_manifest_sha256`,
`original_result_sha256`, and `external_proof_binding_sha256`.

The report schema is `english-v6-b27-verified-original-quality-v1`. A report is
returned only for a complete actual `PASS` or complete actual `NO_PROMOTION`.
The worker reopens every source and observer payload, all original producer Git
bytes, every actual raw model return, and every saved array; it recomputes all
frames and the unchanged B14 result. STOP, unknown, partial, missing, or altered
evidence is rejected. The report carries the actual four checkpoint hashes,
the complete original TRAIN report hash, original Q1 failure report hash, full
source inventory and decision, call count, and verified frame count.

Both `all_original_training_proof_verified` and
`complete_original_Q1_failure_verified` must be true. PRIMARY PASS alone sets
`replication_open_allowed`; replica PASS alone sets
`candidate_engineering_validation_allowed`. A complete NO_PROMOTION sets neither
permission. `promotion_allowed`, `product_released`, and
`automatic_next_assessment_allowed` remain false in every quality report.
Engineering must independently require both explicit panel PASS results for the
same actual B27 candidate.

## External proof and process boundary

PRIMARY uses the original 41-file `training-origin` proof. Replica uses that same
complete proof plus the frozen eighteen-file `primary-quality` sibling. The
PRIMARY reader may run inside this narrowly named replica proof tree, but only
the TRAIN proof supplies its authority. Unknown siblings, symlinks, extra files,
or a substitute PRIMARY lock are rejected. A private scope verifies the complete
reader once, then rehashes the entire original closure, reader, lock and canonical
report before any same-scope reuse. Leaving the scope destroys this cache.

The offline worker permits one exact pinned TRAIN child and, for replica only,
one exact pinned PRIMARY child with its original lock and proof paths. It rejects
other subprocesses, repeated children, scientific imports and network access.
This permits full original proof recursion without running a new scientific call.

For engineering reuse of a previously frozen PRIMARY, restore its exact saved
`primary_quality_files` through `english_v6_b27_original_proof.fetch_files` and
`primary_quality_transport`, then run the original reader. Fetching a current API
response may change incidental API bytes and cannot replace the frozen originals.
Explicit reuse directories are accepted only after every byte is rehashed;
regular hardlinks may avoid duplication, while symlinks are rejected.

## Once-only execution and verification

The two scoring workflows remain inert two-token templates until a reviewed
workflow-only first-push activation. Each verifies complete external proof,
original source admission and prospective capacity before model installation or
calls. It uploads complete result, producer and durable consumption originals.
The original four-model forward loop is unchanged; the budget is twelve actual
calls per admitted original, at most 1,440. The whole raw-output envelope is
measured before calls and stays within the prospective 128 MB artifact cap.

Pure stdlib fixtures cover both complete PASS and NO_PROMOTION for both panels,
raw arrays and journals, counters and clocks, B14 decisions, old-assessment or
wrong-model rejection, source/quarantine mutations, independent proof caching,
exact subprocess restrictions, producer resource closure, and inert workflow
guards. The numerical loader, ledger and scoring loop are compared structurally
against their immutable original implementation. Invented fixture results grant
no authority for a real run.

All 23 quality tests passed in normal, `-O`, and `-OO` Python. An independent
review constructed complete quality roots from the actual 439/445-file SOURCE
closures and 407-file TRAIN closure. The resulting PRIMARY/replica roots had
444/448 files. Each passed `producer_contract` and fresh isolated reader/evaluator
import plus `source_check`, with empty stderr, unchanged copied file hashes, and
zero model, source or PCM reads. No original Q1 proof or scientific run was
repeated for these preparation checks.
