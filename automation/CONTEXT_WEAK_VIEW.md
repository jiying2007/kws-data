# Context native and ASR weak views

`active_context_weak_view.py` consumes only a
`kws-context-supplement-plan-v1` plan verified by `context_supplement.py`.
It does not pass a context plan to the D1 or D2 plan verifier. The nested frozen
D2 parent proves the unchanged eight reference identities and their source
splits; the supplement owns separate cell IDs and generation claims.

The package schema is `kws-active-context-research-weak-view-v1` and its source
adapter is `kws-active-context-native-source-v1`.

## Required source evidence

`source-map` and `create` require the exact plan pin, input manifest pin,
original native ZIP, generation/native roots, job recipe pin and stage bytes.
The owner schemas are `kws-context-generation-stage-v1` and
`kws-context-generation-job-v1`. The selected cells must be the declared chunk,
and both `d1_cells_generated_again` and `d2_cells_generated_again` must be zero.

Every observed clip is joined through its cell and reference to the original
claim, completed or failed terminal, generation call, returned array, FLOAT WAV
and return receipt. Requested text remains `intended_only`. A missing return
creates no observed row; a missing or contradictory native/terminal proof cannot
be repaired by an ASR transcript.

The source map contains audio/source identities and quality routing fields. It
does not inject the requested prompt or another recognizer's text into ASR.
The unchanged selective-ASR evidence verifier checks the requested matrix,
native transcript projections, routing, audit and retained file inventory.

## Complete observed text and context coverage

The existing weak decision requires agreement on the entire normalized
transcript. For a positive-intent source, the intended keyword must actually
occur in that observed transcript. Prefix, middle, suffix and repeated calls
therefore use the same decision rule; no special label override is introduced.

`context_observation` separately records character offsets and non-overlapping
occurrence counts from each authenticated complete ASR transcript and any
agreed full transcript. Confirmed coverage is counted only when the unchanged
weak decision confirms the positive. Requested position/count remains source
metadata and cannot stand in for observed coverage. Character offsets are not
acoustic event times.

The full ASR observations and selected raw transcript remain in
`all-observed.jsonl` and the replayable report. Failed-native, uncertain and
unconfirmed-intent cases retain their original zero-loss decision; no automatic
negative is invented.

## Consumer boundary

The package's `rnn-four-token` and `fsmn-fulltext` exports retain their existing
legacy meanings. `consumer_contracts` explicitly identifies the FSMN map as
2599 outputs. OOV text is rejected as a complete transcript; characters are not
deleted or replaced with blank.

A full2601 model must use a separate context consumer after `verify_view`
replays the entire package, including rows rejected by the legacy exports. That
consumer must verify the already declared vocabulary file and official token
bytes, then project the complete observed transcript. A context package must
never be relabeled as a D1 or D2 input schema to bypass their source checks.

## Entry points

The Python functions `build_source_map`, `create_view`, `verify_view`,
`load_training_rows` and `load_development_rows` follow the analogous D2 API,
with required context owner arguments. CLI actions are `source-map`, `create`
and `verify`; `source-map` and `create` require `--job-recipe`,
`--job-recipe-sha256` and `--stage`.

`test_active_context_weak_view.py` uses fictional PCM and native ASR fixtures.
It exercises the real context plan and bounded runner, whole-package offline
replay, source splits, immutable owner bindings, failed/unknown outcomes,
complete-transcript labels and observed context statistics. It runs without
NumPy, network access or model inference.
