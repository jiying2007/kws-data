# New24 context v3 native and ASR weak views

`active_context_v3_weak_view.py` consumes the separately frozen
`kws-context-supplement-plan-v3` produced by `context_supplement_v3.py`.
The plan contains 72 requests across the existing 24 added source families:
one K1 context, one K2 context and one ordinary-speech request per family.
The original 18 train / 6 development family assignments remain fixed.

The package schema is `kws-active-context-v3-research-weak-view-v1`; its source
adapter is `kws-active-context-v3-native-source-v1`. The embedded new24 D2
parent and context-v2 parent retain the earlier D1, old8 D2 and context-v1
ancestry. The producer rejects historical request hashes and authenticates the
existing full2601 vocabulary. These requested-text checks never certify what
the generated audio actually contains.

## Native source and job ownership

Both `source-map` and `create` require the exact input-manifest file pin,
original native ZIP, canonical plan pin, generation/native roots, job-recipe
file pin and stage bytes. The new owner schemas are
`kws-context-v3-generation-stage-v1` and `kws-context-v3-generation-job-v1`.
The stage covers every planned cell in declared order. Each observed subset
must belong to its owning chunk, with at most eight selected cells.

Generation parameters must agree across the verified plan, stage, recipe and
actual native call. The stage records zero retries, one concurrent model, and
zero reissues of D1, D2, context-v1 and context-v2 cells. A jointly changed
stage/recipe/native parameter set cannot override the verified plan.

The verifier authenticates reference identity, original source split, claim,
terminal record, generation call, returned numeric array, original FLOAT WAV
and return receipt. The cell's normalized prompt hash, complete request hash
and historical-reuse flag remain available in each source row. Missing returns
remain unobserved cells. A failed-native return retains its original bytes and
failure outcome; ASR agreement cannot promote it into a training label.

The independent source map is built from authenticated audio and source identity
before ASR. The unchanged selective-ASR verifier checks routing, requested
models, native transcripts, final audit and the complete evidence inventory.
Neither a request nor another recognizer's transcript is a decoder input.

## Positive and ordinary-speech observations

The shared weak policy requires agreement on the complete normalized observed
transcript. A positive-intent row additionally requires the intended complete
wake phrase. Failure to confirm that phrase leaves an uncertain row; it never
automatically becomes a negative.

An ordinary-speech row has `intended_keyword_id=null`. A confirmed ordinary row
keeps its complete nonempty observed text, positive CTC weight, both keyword
masks enabled and keyword targets `[0, 0]`. Any retained recognizer observation
that contains either complete wake phrase blocks a negative decision, including
an unusable observation. Failed-native, uncertain, unsupported or conflicting
rows retain unknown keyword targets and zero objective weights.

`context_observation` reports observed positive context classes and confirmed
ordinary-speech counts separately. Keyword offsets are Unicode character
positions and non-overlapping occurrence counts in saved transcripts. They are
not acoustic timestamps. Requested positions and requested occurrence counts
remain intent metadata. All raw recognizer observations and complete selected
text stay in `all-observed.jsonl`, including rows excluded from training.

## Entry points and projection

The Python APIs are `build_source_map`, `create_view`, `verify_view`,
`load_training_rows` and `load_development_rows`. CLI actions are `source-map`,
`create` and `verify`; source-producing actions require `--job-recipe`,
`--job-recipe-sha256` and `--stage`. The adapter rejects older owner schemas.

The `rnn-four-token` and `fsmn-fulltext` exports retain their explicit legacy
vocabulary contracts; the latter is still the 2599-output map. The separate
pipeline `training/research_context_v3_inputs.py` authenticates the complete
package before projecting each entire observed transcript into the fixed
2601-output vocabulary. Ordinary negatives never use empty CTC targets. OOV
text remains whole-text unsupported, with complete evidence and zero weights.

`test_active_context_v3_weak_view.py` replays the real native and selective-ASR
formats using fictional PCM, without model inference or network access. It
covers offline self-contained replay, source-family separation, ordinary text
preservation, possible-wake rejection, failed-native zero weights, old schema
rejection, immutable parents and vocabulary, exact generation binding, and
observation-derived statistics. It runs in normal, `-O` and `-OO` modes.
