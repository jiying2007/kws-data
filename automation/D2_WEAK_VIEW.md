# D2 native return → selective ASR → research weak view

`active_d2_weak_view.py` is a new source adapter and package schema. The historical
`active_asr_weak_view.py` remains unchanged. This adapter calls its existing
`selective_evidence`, `make_row`, `decide`, and `SourceRoles` implementations.
It makes no TTS, ASR, training, model download, or network calls.

## Inputs

Use either independently frozen D2 increment plan from
[`D2_EXPANSION.md`](../research/auto-tts-voice-clone-20261102/D2_EXPANSION.md):
`existing-eight` (256 new requests) or `new-twenty-four` (1536 new requests).
The complete frozen D1 parent remains embedded and is never generated again.
The first stage uses the original fixed six train/two development identities.
One view contains **1–64 observed native returns**, not all planned cells. Missing
returns, consumed unknown calls and invalid native geometry are accounted for by
the generation/export owner; they do not become fabricated audio or negatives.

Each actual batch supplies:

- A D2 plan and its **canonical JSON** SHA256, separately from the raw file hash.
- The shared `generation_root` containing `plan-lock.json` and
  `cells/<bulk-id>/{started.json,completed.json or failed.json}`. Completed cells
  also have `native-f32.wav` and `audio-16k.wav`.
- One job's `outputs/native` directory, passed as `native_root`. Every observed
  cell retains `call.json`, `returned.json`, `native.json`,
  `returned-array.npy`, and `returned-native.wav`.
- The original native WAV ZIP and `active-asr-inputs-v1` prepared input directory.
  ZIP members use `cells/<bulk-id>/native-f32.wav` for both completed and failed
  native returns. Input preparation may sort these members; the adapter joins
  by member/cell ID, never by request position.
- The selective source map, frozen policy and externally pinned final audit.
  Both primary recognizers are required. Unrequested Whisper is retained as
  `not_requested`, never a negative vote or a third agreement.

For actual jobs, also pass the frozen job recipe path and **raw file** SHA256 plus
its stage file. These are optional only for standalone native evidence intake;
when any is supplied, all three are required. They authenticate the recipe →
stage → plan-file hashes, full stage chunk coverage, selected cell membership,
generation settings, zero retries and one-model concurrency. They do not replace
the upstream queue's durable run/result and export inventory.

## Run

The following paths represent the concrete upstream job/export outputs. Use the
canonical plan SHA and recipe raw SHA recorded by the upstream export.

```bash
python3 -I -S -B automation/active_d2_weak_view.py source-map \
  --inputs /private/d2/asr-inputs/inputs.json \
  --inputs-sha256 INPUTS_FILE_SHA256 \
  --archive /private/d2/export/source-original.zip \
  --plan /private/d2/stage/plan.json --plan-sha256 PLAN_CANONICAL_SHA256 \
  --generation-root /private/d2/stage/generation \
  --native-root /private/d2/stage/queue/jobs/generate-001/outputs/native \
  --job-recipe /private/d2/stage/queue/jobs/generate-001/recipe.json \
  --job-recipe-sha256 JOB_RECIPE_FILE_SHA256 --stage /private/d2/stage/stage.json \
  --output /private/d2/source-map.json
```

Run the existing selective ASR workflow with this source map and the unchanged
policy. Then create a new private package outside the source checkout:

```bash
python3 -I -S -B automation/active_d2_weak_view.py create \
  --inputs /private/d2/asr-inputs/inputs.json \
  --inputs-sha256 INPUTS_FILE_SHA256 \
  --archive /private/d2/export/source-original.zip \
  --plan /private/d2/stage/plan.json --plan-sha256 PLAN_CANONICAL_SHA256 \
  --generation-root /private/d2/stage/generation \
  --native-root /private/d2/stage/queue/jobs/generate-001/outputs/native \
  --job-recipe /private/d2/stage/queue/jobs/generate-001/recipe.json \
  --job-recipe-sha256 JOB_RECIPE_FILE_SHA256 --stage /private/d2/stage/stage.json \
  --asr-root /private/d2/asr-outputs \
  --selective-source-map /private/d2/source-map.json \
  --selective-policy /private/d2/selective-policy.json \
  --selective-audit-sha256 FINAL_AUDIT_FILE_SHA256 \
  --fsmn-tokens /private/runtime/fsmn/tokens.txt \
  --output /private/d2/weak-view

python3 -I -S -B automation/active_d2_weak_view.py verify \
  --package /private/d2/weak-view --manifest-sha256 PACKAGE_MANIFEST_FILE_SHA256
```

The FSMN vocabulary is optional for package creation and mandatory for the
`fsmn-fulltext` loader. Its existing fixed model vocabulary hash is checked.
The package manifest pin is optional for ordinary local verification; publish
or record it alongside the upstream receipts before later consumption.

## API and outputs

`build_source_map(inputs_path, expected_inputs_sha, archive_path, **source_options)`
creates audio/source-only routing metadata. `source_options` are `plan_path`,
`plan_sha256`, `generation_root`, `native_root`, plus the optional owner tuple
`job_recipe_path`, `job_recipe_sha256`, `stage_path`.

`create_view(inputs_path, expected_inputs_sha, archive_path, asr_root, output,
*, selective_source_map, selective_policy, selective_audit_sha256,
fsmn_tokens=None, **source_options)` retains every source/native/ASR evidence byte
needed to replay the batch. It rejects malformed evidence before writing outputs.

`verify_view(package, *, expected_manifest_sha256=None, other_split_sources=())`
replays the evidence and returns the verified report. It verifies the plan and
embedded D1 parent, file/PCM/array joins, reference mode, job ownership when
supplied, ASR inventory and exact routing, all decisions, quantized audio, counts,
and every derived JSONL export. Original source/runtime directories are not needed.

`load_training_rows(...)` and `load_development_rows(...)` accept the same pin and
cross-split identity arguments and the existing `consumer="fsmn-fulltext"` or
`"rnn-four-token"`. They return the old trainer-facing row fields, full observed
text, numeric targets, absolute audio paths, masks and weights after verification.
Callers must apply these weights to loss. Four-token consumers reject unsupported
ordinary text instead of silently discarding its other characters.

`report.json` and `all-observed.jsonl` retain **all** observed rows, including
uncertain ones. `weak-fulltext.jsonl`, `uncertain.jsonl`, `disagreement.jsonl` and
the per-consumer/per-split exports are replay-checked views of those rows.
Uncertain rows have zero loss weight and are excluded from the numeric training
exports. Failed native returns remain uncertain even with agreeing ASR text.
Requested prompts/categories are retained as provenance and conservative intent
constraints; only independently saved ASR transcripts supply weak text targets.
Nothing in this package is human gold, independent test data, or product approval.

When combining packages, use `source_id` (the D2 cell ID), not batch-local
`clip-000`, as the unique recording identity. Reject duplicate cells and pass
the accumulated source identities through `other_split_sources`. The D1 aggregate
loader has a fixed 256-cell contract and must not ingest these D2 packages.

## Regression checks

```bash
python3 -I -S -B automation/test_active_d2_weak_view.py -v
python3 -I -S -B -O automation/test_active_d2_weak_view.py -v
python3 -I -S -B -OO automation/test_active_d2_weak_view.py -v
```

Fixtures use the real D2 planner/bulk runner and real receipt formats with
fictional PCM and decoder outputs. They cover both increments, fixed roles,
failed native zero weights, observed text differing from the request, sorted
archive membership, owner pin/scope failures, deletion of original directories,
rehashed decision/array/ASR/export mutations and cross-package split leakage.
