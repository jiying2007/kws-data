# Additive context supplement v2

This is an explicit new generation version. The frozen context v1 plan, queue,
partial returns, failures and consumed unknowns stay intact. No old request is
retried, relabelled or replaced. No D1/D2 cell is generated again.

The v1 prefix/middle templates include characters absent from the fixed 2601
full-text vocabulary. This independent producer keeps that vocabulary unchanged
and uses representable requested sentences. Preflight is about the **request**:
actual speech can still differ, contain OOV characters, fail signal checks or
remain uncertain. Only later observed ASR evidence can provide weak labels.

## Coverage and ancestry

| Version budget | Prefix | Middle | Suffix | Repeated call | Identity coverage |
|---|---:|---:|---:|---:|---|
| Default 8 | 4 | 4 | 0 | 0 | All original 8, one candidate each |
| Explicit 16 | 4 | 4 | 4 | 4 | All original 8, both keywords per identity |

Both budgets preserve the original six train/two development source roles. The
16 plan has 12 train/4 development candidates and eight requests per keyword.
The complete 16-cell v1 parent and its exact canonical SHA are mandatory. Its D2
and D1 ancestors, references, source modes and registry bindings remain pinned.
Every v2 cell uses a new namespace and is checked against all ancestor cell IDs.
Suffix and repeated-call wording may match a v1 request; each such cell explicitly
records `same_requested_text_as_parent=true` and the new additive revision.
Choose one budget before freezing; alternative plans are not continuations.

Prefix requests are `{keyword}，请把灯打开。`; middle requests are
`现在，{keyword}，请把灯打开。`. Suffix and repeated-call use the original safe
sentence forms. All requested normalized characters must have nonblank IDs in
the fixed vocabulary; no character is discarded or remapped to blank.

## Fixed vocabulary

`fixtures/context-full2601-vocabulary.json` contains the exact existing vocabulary
bytes, not audio or participant data. It derives from the original 2599 outputs
plus the already frozen task-text additions 沃 and 灯. The producer verifies:

- Original file SHA-256: `7cfd3e1f62ef1700be65123257a3073083fafaf50e4615533d82ad002e19203e`.
- Canonical SHA-256: `104575dab3a9f7ed622d8fa29aa8a897e0590cca8aed9214b26ef4cd1c8b4e4b`.
- Original base-tokens SHA-256: `5914b5e048e14dd8e1cce6914333cbbf8b382df7679c26041e35bd1c86db43aa`.

The plan embeds `vocabulary={value,canonical_sha256,file_sha256,base_tokens_sha256}`.
Offline validation reconstructs its exact canonical bytes and replays every
template's preflight. Numeric observed targets are never embedded in the plan.

## Create a new private plan

```bash
python3 -B research/auto-tts-voice-clone-20261102/context_supplement_v2.py \
  --parent-context-plan /private/context-v1-plan.json \
  --parent-context-plan-sha256 FROZEN_V1_CANONICAL_SHA256 \
  --sources /private/references/source-manifest.json \
  --audio-root /private/references/reference-assets \
  --vocabulary /private/fixed2601/vocabulary.json \
  --batch-id context-v2-001 --count 16 \
  --output /private/context-v2-plan.json
```

Omitting `--output` is a read-only preview. Plan creation validates reference
WAV/PCM hashes and source evidence. It performs **zero model calls**.

## Independent runner and downstream contract

```python
plan(parent_context, parent_sha, source_manifest, audio_root, vocabulary,
     *, count=8, batch_id=...)
verify_plan(value)
rebuild(value, source_manifest, audio_root)
runner_recipe(value, selected_cells=None)
run(value, source_manifest, audio_root, output, backend,
    *, max_new_calls, writer=media._write_waves)
```

The new schema is `kws-context-supplement-plan-v2`, stage
`context-supplement-v2`. The old v1 and D2 validators intentionally reject it.
The new native owner uses `kws-context-v2-generation-stage-v1` and
`kws-context-v2-generation-job-v1`; it must assert all three zero-reissue fields:
`d1_cells_generated_again`, `d2_cells_generated_again`, and
`context_v1_cells_generated_again`.

Run/result/native retention contracts match v1. Generation remains fixed at
`max_new_tokens=256`; a supplied bounded backend must have the exact selected
cells, call budget and generation settings. `started.json` consumes a request
even without a terminal. No automatic retry or usable-row quota exists. The
backend retains native WAV/NPY and all receipts before signal checks; failures
keep their native returns. The independent owner enforces time/model/queue
ownership, shared heavy-model leases and authenticated exports. An 8+8 chunk
split uses the two contiguous ranges of the **same** 16-cell plan.

The independent v2 source/weak adapter must replay the v1 parent, all source,
native and ASR evidence, the new owner schemas and the fixed vocabulary binding.
Observed full-text OOV or uncertainty must remain zero loss. Context positions
and repetitions come from the actual agreed full transcript, never from the
requested template or preflight; textual positions are not acoustic timestamps.

## Tests

`test_context_supplement_v2.py` uses actual planners and the exact fixed vocabulary
fixture with fictional PCM only. It covers both budgets, each identity's roles,
keyword/context counts, v1 OOV counterexamples, changed vocabulary and ancestry,
disjoint IDs, false-truth mutations, native 8+8 chunks, retained failed native,
consumed unknowns and altered generation budgets. Run normal, `-O` and `-OO` modes.
