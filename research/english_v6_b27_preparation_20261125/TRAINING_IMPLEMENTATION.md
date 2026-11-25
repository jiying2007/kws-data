# B27 TRAIN implementation and original evidence interfaces

This implementation prepares one conditional training phase. The opening is
disabled until the complete original Q1 PRIMARY failure is bound and reviewed.
The prospective assessment is separately frozen; its `optimizer_allowed=false`
is a metadata property and never grants training authority.

## Scientific identity

- Design: `automation/english_v6_b27_natural_weight_design.json`, SHA-256
  `956f0695e6b4e3ac450bc3881c5ef13985d775210cf43cf4da98947a5a1846bf`.
- Prior coefficient definition: the exact original B26 TRAIN-only proposal at
  commit `d5b68dff01abcb66781496369458f7ee006baa87`, SHA-256
  `4f2df013204098f1014e366def6e8b0b6b983fcaba73805db1204a415f116b8c`.
- The original B23 fit code object and B24 export wrapper are reused through
  private globals. Only `NATURAL_WEIGHT` changes from 0.08 to 0.16. Historical
  modules, optimizer settings, source draws, loss definitions and teachers are
  unchanged.
- B23 initializer PT: `784def9df6c2a8bfbecdb7ba2d7b70630337a1cda72561c05392327aa980297f`.
- Immutable B24 control PT: `f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4`.
- One fresh AdamW phase, 192 updates, 1,844 optimization forwards, two checkpoint
  reload forwards and one ONNX export forward: 1,847 total. No old update is
  replayed, and there are zero new teacher or historical-control forwards.
- All 88 original TRAIN sources, 122 protection conditions, 264 saved ordinary
  teacher arrays and the original `COMMON_DRAW.json` remain fixed. The complete
  original B24 TRAIN producer's 338 dependencies are retained.

The new coefficient was defined before the Q1 numerical outcome. Q1 failure is
an eligibility condition only. These files do not change the old B25/B26
opening conditions or any original Q1 budget, reserve or quality threshold.

## Qualification and execution order

The new opener has schema
`english-v6-b27-after-original-b24q1-no-promotion-v1` and condition
`only_after_actual_complete_B24Q1_primary_NO_PROMOTION`. It binds the original
Q1 lock, exact frozen fail-only reader and the complete canonical failure
report SHA. STOP, incomplete returns, unknown consumption and PASS do not
satisfy the opener.

`assessment_binding()` validates every file in the frozen metadata inventory,
the B27 design SHA, original Q1 source exclusion and the deterministic metadata
validator. `source_preflight()` independently verifies the complete Q1 failure,
rechecks actual source metadata exclusions, verifies all four original B24
TRAIN artifacts and repeats the unchanged original TRAIN source/teacher gate
before any acoustic runtime import.

After reviewed preparation, only a workflow-only child of the exact prepared
parent can activate this branch:

`research/en-v6-b27-natural-weight-train-once-20261125`

The workflow path is
`.github/workflows/english-v6-b27-natural-weight-train-once.yml`. Activation
substitutes exactly two `__B27_REVIEWED_NATURAL_WEIGHT_PARENT__` tokens from the
frozen template. Public push identity, first run attempt, actual parent, exact
head and no force/deletion are checked. The producer is uploaded first; then
`fetch`, `consume` and `execute` run from the captured repository. A separately
uploaded `english-v6-b27-natural-weight-train-consumed` marker must exist before
the optimizer starts. Its existence closes retry and replay.

The entry point provides:

```sh
python3 -I -S -B automation/english_v6_b27_natural_weight.py check
python3 -I -S -B automation/english_v6_b27_natural_weight.py fetch --inputs FRESH_INPUTS --prepared-parent PREPARED_SHA
python3 -I -B automation/english_v6_b27_natural_weight.py consume --inputs INPUTS --prepared-parent PREPARED_SHA
python3 -I -B automation/english_v6_b27_natural_weight.py execute --inputs INPUTS --output FRESH_OUTPUT --prepared-parent PREPARED_SHA
```

The last three commands are future workflow operations, not local preparation
or test commands. Fresh paths must be absolute. Original pinned Python package
versions remain NumPy 1.26.4, Torch 2.9.1+cpu and ONNX 1.17.0.

## Original evidence and finite archive capacity

The new producer contains code and metadata only. Models, producer, receipts
and consumed artifacts each retain the unchanged 128,000,000-byte compressed
and expanded limit. The complete producer envelope and receipt envelope are
checked before budget consumption. The receipts reserve 8,000,000 bytes for
new metadata in addition to their complete retained originals.

The receipts retain the four original B24 TRAIN ZIPs and `TRAIN_LIVE.json`,
three original Q1 quality ZIPs and three Q1 API responses, plus `FETCH.json`,
`SOURCE_PREFLIGHT.json`, `CONSUMPTION.json` and `EXTERNAL_INPUTS.json`.
The twelve Q1 inputs and eighteen unchanged TRAIN input ZIPs remain complete
external originals with exact byte counts and whole-file hashes. The large Q1
SOURCE input preserves its original 768,000,000-byte role cap. No historical
role cap is widened, no original is dropped, and no replacement ZIP is treated
as an original artifact.

## Completed TRAIN artifacts and independent readback

| Role | Actual artifact name |
|---|---|
| Models | `english-v6-b27-natural-weight-models-RUN_ID` |
| Producer | `english-v6-b27-train-producer-RUN_ID` |
| Receipts | `english-v6-b27-original-input-receipts-RUN_ID` |
| Consumed | `english-v6-b27-natural-weight-train-consumed` |

The original models contain `control/models/mero43-b24-challenger192.*` and
`challenger/models/mero43-b27-challenger192.*`, each in PT, NPZ and ONNX formats.
The candidate PT SHA is learned only from the complete actual final artifact;
there is no preselected or partial checkpoint. The result schema is
`english-v6-b27-train-natural-weight-v1`; its status is
`TRAINED_NATURAL_WEIGHT_CANDIDATE_REQUIRES_NEW_INDEPENDENT_PRIMARY`.

The independent reader API is:

```python
verify_training_original(directory, lock, inputs)
```

`directory` contains `TRAIN_LIVE.json` and the four `TRAIN_ROLE.zip` originals.
`inputs` contains `no-promotion/{api,originals,inputs}` with exactly 18 original
Q1 files, and `training/` with all eighteen old TRAIN input ZIPs. Lock/report
JSON uses the unchanged compact sorted `failure.canonical()` representation;
ordinary B27 receipt files use sorted, indented JSON plus a newline.

The reader independently validates original API/Git identities, the complete
producer, every saved optimization array and whole frame length, all update
and protection traces, the two already-returned reload arrays, final export
bytes, fresh metadata and complete original Q1 failure. It uses stdlib only
and performs zero new model, source or optimizer calls. Its report schema is
`english-v6-b27-original-training-readback-v1`; `result` retains the actual
full RESULT, and `training_source` retains the exact full SOURCE inventory,
including `base_train` (79), `natural_train` (9) and `train_originals` (88) for
later overlap checks. It grants no promotion or product-release claim.

Reviewable binding commands are available without remote mutation:

```sh
python3 -I -S -B automation/bind_b27_training.py opening --directory ORIGINAL_Q1 --lock Q1_LOCK --output FRESH_OPENING_FILE
python3 -I -S -B automation/bind_b27_training.py training --directory ORIGINAL_TRAIN --inputs EXTERNAL_INPUTS --output FRESH_BINDING_DIRECTORY
python3 -I -S -B automation/english_v6_b27_training_readback.py ORIGINAL_TRAIN TRAINING_LOCK EXTERNAL_INPUTS
```

The training binder derives pins from the actual four-artifact inventory and
then requires complete independent readback before writing the lock, report,
unaltered producer manifest and binding candidate. It does not install the
binding, activate a workflow or open source generation.

When embedded under another offline reader, the parent must allow only the
exact registered B27 TRAIN reader command and canonical lock at the explicit
proof root. Its child is the unchanged frozen Q1 fail-only reader. Historical
offline subprocess policies must not be widened globally.

## Validation

The TRAIN contract suite uses no tensor runtime or real qualification proof.
It checks the single private coefficient, identical original code objects,
one-fit export behavior, exact control copying, first-run guard, disabled
opening, incomplete metadata refusal and prospective archive envelopes. The
readback suite constructs stdlib NPY fixtures and rejects missing, partial,
nonfinite, wrong-draw, cropped, inconsistent-loss and changed-reload evidence.

A read-only compatibility check against the already saved original B24 pair
validated its 1,844 optimization arrays, 340,900 optimization frames, two
reload arrays, 192 update records, 692 protection records and 768 ordinary
margin records. It did not rerun training or any model forward.
