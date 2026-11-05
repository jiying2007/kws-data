# Independent contextual wake supplement

This is an additional **8 or 16 requested candidates**, outside the frozen
D1 256 and cumulative D2 2048 candidate positions. A plan is not generated audio,
an observed context, a machine label, or an admitted training example.
The existing D1/D2 planners, bulk runner and weak-view modules are unchanged.

The existing campaign planners provide standalone keywords. The multilingual
planner also requires a positive to equal the complete normalized keyword.
They therefore cannot represent these context-positive cases by editing their
saved plans. The independent `kws-context-supplement-plan-v1` schema provides
an explicit, separately pinned extension.

## Exact requested coverage

The parent is the frozen `existing-eight` D2 increment. The same eight references,
their original ancestry, checked/unverified reference modes, source manifest,
registry binding, six train/two development roles and model revision are retained.

| Request type | Keyword 1 example | Keyword 2 example | Default cells |
|---|---|---|---:|
| `prefix` | 你好小窝，请把客厅的灯打开。 | 小窝小窝，请把客厅的灯打开。 | 4 |
| `middle` | 客厅有点暗，你好小窝，请把灯打开。 | 客厅有点暗，小窝小窝，请把灯打开。 | 4 |
| `suffix` | 如果你准备好了就回应我，你好小窝。 | 如果你准备好了就回应我，小窝小窝。 | 4 |
| `repeated-call` | 你好小窝，你好小窝。 | 小窝小窝，小窝小窝。 | 4 |

The default 16-cell plan assigns each original identity two different contexts,
one per keyword. All eight templates appear twice; split counts are 12 train and
4 development candidates. The 8-cell alternative assigns one template per
identity, covering each of the eight templates once (6 train/2 development).
Choose and freeze one budget; do not treat alternative plans as a continuation.

Each cell keeps `intended_kind="positive-intent"`,
`generation_category="full-positive"`, `intended_keyword_id`,
`label_authority="intended_only"`, and `actual_text=null`. `context_class` and
`requested_keyword_occurrences` describe the request only. Cell IDs use a separate
context namespace and are disjoint from the parent's D1/D2 IDs.

## Create a private plan

```bash
python3 -B research/auto-tts-voice-clone-20261102/context_supplement.py \
  --parent-plan /private/d2-existing-eight/plan.json \
  --parent-plan-sha256 D2_CANONICAL_PLAN_SHA256 \
  --sources /private/references/research-source-pool.content-observed.json \
  --audio-root /private/references/normalized \
  --batch-id context-supplement-001 --count 16 \
  --output /private/context-supplement-plan.json
```

Omit `--output` for a read-only preview. The command verifies real reference WAV,
PCM, ancestry and existing source evidence, but performs zero model calls. It
does not alter any D1/D2 artifacts or source registry.

## Native runner contract

The module exposes:

```python
plan(parent, parent_sha, source_manifest, audio_root, *, count=16, batch_id=...)
verify_plan(value)
rebuild(value, source_manifest, audio_root)
runner_recipe(value, selected_cells=None)
run(value, source_manifest, audio_root, output, backend,
    *, max_new_calls, writer=media._write_waves)
```

`runner_recipe` returns `selected_cells`, `max_calls`, and the fixed generation
configuration used by `active_tts_smoke.BoundedBackend`, including
`max_new_tokens=256`. A supplied backend must expose this matching `recipe` and
`synthesize(cell, ref, audio_root)` interface. This module never creates a model.

The independent pipeline wrapper must pin the complete plan/source/model/runtime
files, acquire the shared heavy-model lease, disable network access, enforce job
and total budgets, and instantiate the existing `QwenBulkClone` plus
`BoundedBackend`. The new `run` API submits only the declared pending cells. A
previous `started.json` always consumes the request, even without a terminal;
there are no quality retries or replacements to fill a usable-data quota.

The generation directory has the existing `kws-bulk-clone-execution-lock-v1`,
`cells/<bulk-id>/started.json`, and `completed.json` or `failed.json` records.
Successful outputs include `native-f32.wav` and `audio-16k.wav`. The backend's
separate `native/<bulk-id>/` directory retains `call.json`, `native.json`,
`returned.json`, `returned-array.npy` and `returned-native.wav` **before** signal
validation. A failed signal gate leaves that original return intact.

Result fields match the D2 driver: `attempts_this_call`, `total_started`,
`total_completed`, `model_calls_requested`, and false admission flags. A completed
generation still has `actual_text=null` and awaits independent ASR.

## Independent source and observed-context adapter

The frozen D2 weak view deliberately rejects this schema. Do not mutate its
validator or disguise a context plan as a D2 increment. The independent adapter
must verify the new plan and replay these bindings:

- Context canonical plan SHA and file SHA; embedded D2 canonical parent SHA,
  embedded D1 parent, exact original reference records, source-manifest SHA and
  registry assignment snapshot.
- A separately pinned `kws-context-generation-stage-v1` and
  `kws-context-generation-job-v1`, with the D2 owner's field structure and both
  `d1_cells_generated_again=0` and `d2_cells_generated_again=0` on the stage.
  The selected cells, budget, generation settings and native call receipt must
  agree with the context plan and actual job.
- Every actual original return, native NPY/WAV/PCM hash, generated derivative,
  reference ICL/x-vector mode, start/terminal record, source identity and split.
- The audio-only source map, raw ASR claim/native/result bytes, model identities,
  exact primary/selective call matrix and final audit pin. No intended sentence,
  hotword hint or peer transcript goes to a recognizer.

The existing `weak.decide` already supports a complete agreed sentence containing
the intended keyword, retaining the full observed sentence. Its decision must
remain based on the actual transcripts. Failed native, missing agreement, source
quality uncertainty or an unconfirmed intended keyword retain zero loss weight.
An OOV full sentence must be preserved and rejected for that numeric consumer;
never remove its other characters to manufacture a four-token target.

Context coverage needs an additional **observed text** calculation. Report each
keyword's nonoverlapping occurrences and character offsets in the actual agreed
normalized full transcript. Prefix/suffix/middle require real surrounding text
in the corresponding position; repeated-call requires at least two complete
nonoverlapping occurrences. A standalone decoded wake does not establish the
requested context. Keep requested class and observed coverage separate. These
are text-position observations, not acoustic boundaries or human event counts;
timestamps and acoustic completeness remain unknown without separate evidence.

## Verification

`test_context_supplement.py` exercises the real independent planner and runner
against fictional native-return fixtures, including 16/8 coverage, preserved
parent bytes/roles, non-reused IDs, changed prompts/seeds/authority rejection,
the old validators' schema refusal, two bounded chunks, retained failed native,
consumed unknown skipping and refusal of changed token budgets before a call.
Run it in ordinary, `-O` and `-OO` modes. No test invokes a model or downloads data.
