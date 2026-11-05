# Additive context-v2 native and ASR weak views

`active_context_v2_weak_view.py` consumes only a
`kws-context-supplement-plan-v2` plan verified by `context_supplement_v2.py`.
The package schema is `kws-active-context-v2-research-weak-view-v1`; its source
adapter is `kws-active-context-v2-native-source-v1`. These are separate from the
unchanged context-v1, D1 and D2 packages.

The plan embeds its exact context-v1 parent, that parent's D2 ancestor, and the
previously fixed full2601 vocabulary. The verifier authenticates all three
bindings. Every v2 cell has a new ID. A cell may explicitly repeat a parent
request as a new additive revision; that does not authorize retrying or
relabeling an existing generation claim.

## Source and owner proof

`source-map` and `create` require the input-manifest file pin, original native
ZIP, canonical plan pin, generation/native roots, job-recipe file pin and stage
bytes. The owner schemas are `kws-context-v2-generation-stage-v1` and
`kws-context-v2-generation-job-v1`. Selected cells must equal the declared
chunk. The stage must declare no automatic retries, model concurrency one and
all three historical reissue counts zero:

- `d1_cells_generated_again`
- `d2_cells_generated_again`
- `context_v1_cells_generated_again`

The existing native protocol is replayed unchanged: reference identity and
source split, claim, terminal record, generation call, returned array, original
FLOAT WAV and return receipt. Only actual returned clips become observed rows.
A partial observed subset is allowed within its declared chunk; absent returns
are reported in owner evidence and do not become labels or training rows.

The independent `source-map` is derived from authenticated audio and source
identity before ASR. The unchanged selective-ASR verifier authenticates routing,
requested models, native transcripts, final audit and the complete inventory.

## Vocabulary and observed labels

The source adapter records both canonical and file vocabulary pins. A cell's
`requested_text_preflight` has scope `requested_text_only`: it proves the
requested text can be represented by the fixed vocabulary. It is not evidence
of what the generated audio contains, and it does not supply training targets.

The unchanged weak decision requires agreement on the full normalized observed
transcript and the actual presence of the intended keyword. All raw ASR
observations and the complete selected transcript remain in the report and
`all-observed.jsonl`. Failed-native, uncertain or unconfirmed positive rows
retain zero-loss labels and are never automatically labeled negative.

`context_observation` calculates Unicode character offsets and non-overlapping
keyword counts from retained transcripts. Confirmed prefix, middle, suffix or
repeated-call coverage is counted only from an observed positive decision.
These offsets are not acoustic event times. The requested context class and
occurrence count remain intent metadata.

## Entry points and consumers

The Python APIs are `build_source_map`, `create_view`, `verify_view`,
`load_training_rows` and `load_development_rows`. CLI actions are `source-map`,
`create` and `verify`, with the same argument names as the v1 adapter. Both
source-producing actions require `--job-recipe`, `--job-recipe-sha256` and
`--stage`; a v1 owner is rejected.

The `rnn-four-token` and `fsmn-fulltext` exports retain their explicit legacy
consumer contracts, including the 2599-output FSMN mapping. A full2601 model
must instead use the separate v2 pipeline consumer after whole-package replay.
That consumer verifies its existing vocabulary file against the plan's exact
fixed vocabulary and projects the complete observed text. OOV text must remain
whole-text unsupported; neither the request nor a legacy filtered export may
replace an observation.

`test_active_context_v2_weak_view.py` covers authentic fixture native/ASR replay,
offline retention, partial observation coverage, source splits, historical
schema rejection, parent/vocabulary pins, owner/chunk rejection, failed/unknown
native outcomes and observation-derived context statistics. Tests use fictional
PCM with the real producer and evidence formats, without model inference,
NumPy or network access.
