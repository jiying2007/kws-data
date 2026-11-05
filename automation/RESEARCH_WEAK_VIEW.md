# Saved ASR results → research weak training view

`active_asr_weak_view.py` consumes existing `active-asr-inputs-v1` and
`active-asr-result-v1` evidence. It does not run TTS, ASR, an optimizer or a
deployment command. Output is a new private directory. Historical source,
generation and ASR records retain their original bytes and admission fields.

## Inputs and source identity

~~~sh
python3.12 -I -S -B automation/active_asr_weak_view.py create \
  --inputs /private/asr/inputs/inputs.json \
  --inputs-sha256 ACTUAL_INPUT_MANIFEST_SHA256 \
  --source-archive /private/original-audio.zip \
  --asr-results /private/asr/results \
  --fsmn-tokens /private/fsmn/tokens.txt \
  --output /private/research-view-001
~~~

The default D0 source adapter joins a generated public-preset WAV to its original
`generation.json` outcome and planned row. The original archive, member path,
WAV hash and FLOAT PCM hash must agree. The preset's model/revision comes from
the corresponding runtime receipt. Clip IDs and worker lanes are not speakers.
The same Qwen preset remains one source family across campaigns and revisions.

The preserved D0 source records specify `train_only`. This adapter therefore
produces training rows and **zero independent development rows**. Four presets
do not become 64 independent speakers. `load_training_rows(...,
other_split_sources=...)` checks family, speaker and original/derived PCM/file
identities against supplied development sources; include every calibration or
selection source when forming an experiment. A preset used for development in
another retained experiment cannot also serve as independent development for
an experiment training on that preset.

For a D1 Base/clone batch, use the explicit frozen-plan source adapter by adding
all three options to the same `create` command:

~~~sh
  --d1-plan /private/d1/frozen-plan.json \
  --d1-plan-sha256 CANONICAL_PLAN_SHA256 \
  --d1-generation-root /private/d1/generation
~~~

The expected SHA is the SHA-256 of canonical plan JSON, exactly as recorded in
the bulk runner's `plan-lock.json`. The original plan file is also retained with
its own byte SHA. Build each active-ASR input archive from at most 64 selected
native WAVs named `cells/<cell_id>/native-f32.wav`; this archive contains those
WAVs only. Keep the bulk generation directory with its original `plan-lock.json`,
`started.json`, `completed.json` and `audio-16k.wav` files. A 256-cell plan may be
processed in several bounded ASR batches without changing source assignments.

The adapter verifies the D1 plan layout, reference manifest hash, exact cell
membership, start receipt's plan/reference/seed, completion's source identity,
native WAV and FLOAT PCM hashes, and the original PCM16 derivative's hashes and
geometry. Missing, changed or duplicate source records are rejected. Source
family, speaker, train/development role and `generation_category` come from the
frozen plan. They are never inferred from the ASR text, worker lane or an
ambiguous `hard-negative-intent` field. Original completion/screen schemas stay
unchanged. Retaining a plan and its rights fields does not authenticate the
underlying source permission.

D1 source-group development examples have separate `*-development.jsonl` files
and an explicit `load_development_rows` API. `load_training_rows` continues to
yield only training rows. Both APIs verify all source roles before yielding;
reference WAV/PCM/parent identities and generated derivatives also participate
in cross-split checks. Development labels remain machine weak and do not become
independent human test evidence. The default D0 package format and its zero
development scope remain unchanged.

### Extension outcomes, failed raw audio and pre-ASR source maps

`kws-active-d1-frozen-plan-source-v2` consumes an immutable per-job
`active-tts-stage-outcomes-v1` document. It verifies the caller-supplied outcomes
SHA, the retained stage/job byte hashes, full-plan canonical/file hashes, exact
new/reused roster, per-cell claim owner, reference/mode/text/seed binding, native
numeric NPY values, FLOAT WAV identity and original success or failure receipt.
It does not read the mutable global generation ledger. The package retains the
original absolute paths as evidence while replaying only its own copied files.

Use these options instead of `--d1-generation-root` for a version 2 intake:

~~~sh
  --d1-plan /private/prepared/plan.json \
  --d1-plan-sha256 CANONICAL_PLAN_SHA256 \
  --d1-stage /private/extension/stage.json \
  --d1-job-recipe /private/extension/jobs/chunk-001.json \
  --d1-outcomes /private/extension/queue/jobs/chunk-001/outputs/outcomes.json \
  --d1-outcomes-sha256 ACTUAL_OUTCOMES_FILE_SHA256
~~~

The native-only ZIP is an exact subset of that job's **new** calls. Original
reused cells never become new ASR requests. The extension export lists native
returns with no decodable bounded finite WAV separately; they remain original
outcomes, have no observations, and are not inserted into a ZIP. A finite WAV
that failed the generator's quality gate can be recognized at its original
volume. Its view source is explicitly `generation_outcome="failed-native"`;
`original_generation_row` is the real `FAILED_NO_RETRY` receipt, not a fabricated
completion. Even unanimous ASR text leaves the selected full text null, keyword
targets null, both keyword masks false, CTC mask false, and every loss weight
zero. The raw transcripts remain in the report. Legal short failed audio can
retain its PCM audit representation without becoming a training example.

Before running any sparse ASR, invoke `source-map` with `--inputs`,
`--inputs-sha256`, `--source-archive`, `--output` and the same D1 source options.
The public `build_d1_source_map` function uses the same validation. The resulting
`active-asr-source-map-v2` contains only input/archive/plan hashes and per-clip
global cell identity, original/derived audio hashes and measured source quality
flags. It contains no target or reference text. Quality-failed sources carry
`original_generator_signal_invalid`; all-zero or saturated PCM adds its own
measured flag. The selective controller can therefore freeze routing without
guessing identity from filenames or seeing intended text.

For relocated artifacts, `--d1-artifact-roots-json` maps original absolute
`record.output_root` strings to current readable directories. This changes no
retained source record. Public loader names and the seven existing source
dependencies are unchanged. Tests use fictional numeric/audio/ASR fixtures and
do not establish that any new generation, recognition or training has run.

### Selective ASR observations

The explicit `kws-active-asr-research-weak-view-v2` report consumes a sealed
`active-asr-selective-final-audit-v2` by adding these options to `create`:

~~~sh
  --selective-source-map /private/source-map.json \
  --selective-policy /private/selective-asr-v2.json \
  --selective-audit-sha256 ACTUAL_FINAL_AUDIT_FILE_SHA256
~~~

The policy byte hash is fixed to
`5cba6043b6c7e39aaa39482b31e0e14ae478cd5485b386e7d87f5bec78a3c501`.
Qwen and SenseVoice are requested for every input. Whisper is mandatory for a
primary disagreement, incomplete decoder/text, primary quality flag or source
quality concern. Otherwise the predeclared seed `20261105` and full-plan/global
cell identity apply the declared 20% hash threshold. Local clip numbering
does not affect the draw. The data consumer independently repeats the source
join, normalization, routing and requested-call matrix; a missing requested
result blocks a complete view. The native decoder's original v1 result format
and audio-only conditions remain unchanged.

An unrequested third observation has `status="not_requested"`, `text=null`,
`usable=false`, reason `two_primary_agree_not_sampled` and its routing hash.
There must be no claim, native output or result for that cell/model pair. This
state is neither an error nor a third agreeing vote. Two usable primary full
sentences still receive the existing two-vote weight 0.5. The rule does not
promote uncertainty or a generator failure to a negative example.

Every byte in the upstream sealed input/output inventory is retained, including
actual worker/load receipts, complete native outputs, the source map, policy,
request, routing and final audit. The externally pinned native checkpoint
application audit remains upstream evidence; this data consumer does not load
models or duplicate that model-state verifier. It replays native text,
source/audio identities, call counts, requested/absent states, supervision and
deadline evidence before rebuilding labels. Package replay works after the
original input, source, result and policy directories are removed. Existing v1
packages and their three-route observations keep their original semantics.

The original mono FLOAT 24 kHz WAV and the existing 16 kHz float32 derivative
are retained unchanged. A separate PCM16 WAV is produced for current trainers:
multiply by 32768, round to nearest even integer, then saturate to the PCM16
range. The receipt records source/destination hashes, geometry, peak, RMS and
saturation count. There is no gain normalization, trimming or padding.
Saturation or an all-zero quantized signal prevents automatic training labels.

## Explicit machine-weak decisions

For a v1 batch, all three result records, native outputs and durable call claims are retained.
The consumer checks input/model identity, native-output hashes, empty context,
decoder termination and the source's disclosed decoding conditions. In
particular, Whisper's explicitly supplied Chinese language is preserved; this
is not independent language detection.

Raw transcript strings are never corrected, replaced by generation intent,
truncated to wake words, or merged with another recognizer's text. Comparison
uses the shared orthographic normalizer: punctuation and Han spacing may vary,
while different characters, English word boundaries and contractions remain
different. The complete agreed sentence is the CTC target text.

| Observation | New view decision |
| --- | --- |
| Three usable complete decoders agree on the whole sentence | `machine-weak-three`, weight 1.0 |
| Two usable complete decoders agree on the whole sentence | `machine-weak-two`, weight 0.5; preserve the third observation |
| Intended complete wake is supported by that sentence, even when the third decoder writes a near character | Positive keyword mask/target may be used as machine weak evidence |
| Intended complete wake is not confirmed by the agreed sentence | `uncertain`; never automatically negative |
| Partial/ordinary/near-negative intent has full-sentence agreement and no byte-bound observation contains a complete or ordered wake path | Negative keyword targets, with the full nonblank speech transcript retained |
| Negative intent conflicts with a byte-bound wake observation, including text whose decoder quality prevents a vote | `uncertain` |
| No two whole-sentence votes, empty/incomplete evidence, or signal concerns | `uncertain`; zero CTC/keyword weights and false masks |

Thus an agreed isolated “小窝” can be a negative **wake** example while its CTC
target remains the spoken two-syllable sequence. It is not silence or blank.
The weights are declared experiment choices, not calibrated probabilities.
Every row remains `human_gold=false`; acoustic verbatim completeness remains
unknown. No keyword start/end timestamp is fabricated from the clip boundary.

## Files and existing trainer interfaces

The package retains `report.json`, `weak-fulltext.jsonl`, `uncertain.jsonl`,
`disagreement.jsonl`, both consumer manifests, original evidence and a complete
SHA-256 `MANIFEST.json`. A disagreement can be present in both the usable weak
view and the disagreement view. This does not hide the third transcript.

~~~sh
python3.12 -I -S -B automation/active_asr_weak_view.py inspect \
  /private/research-view-001 --consumer rnn-four-token
python3.12 -I -S -B automation/active_asr_weak_view.py inspect \
  /private/research-view-001 --consumer fsmn-fulltext
~~~

`load_training_rows(package, consumer)` rechecks the retained original source,
all ASR evidence, the decision and the PCM16 derivation before yielding numeric
full-transcript targets and audio paths. `rnn-four-token-train.jsonl` uses the
existing `training.train_ctc.Manifest` JSONL fields (`audio`, `target_ids`,
`speaker_id`, `source_id`). A sentence containing an unsupported character is
excluded from that consumer, with an explicit unsupported reason in the report.
No OOV phoneme becomes blank or “窝”.

The FSMN adapter uses the maintained official 2599-class vocabulary and
raw-ID-minus-one mapping, pinned to SHA-256
`5914b5e048e14dd8e1cce6914333cbbf8b382df7679c26041e35bd1c86db43aa`.
Its yielded rows also expose `recording`, `ctc_ids`, `ctc_text`, `audio_path`,
`audio_sha256` and `samples`, matching the maintained full-text adapter's
trainer-facing information. This new format is read through the explicit
adapter rather than pretending to be the older two-ASR conditioned archive.
The current view performs no 300 ms padding; any later frontend padding must be
declared by the consuming experiment.

Both adapters preserve `ctc_mask`, `ctc_weight`, `keyword_targets`,
`keyword_mask`, `keyword_weight`, full text and label tier. **The existing
`Manifest` feature loader ignores arbitrary weight metadata.** A weighted
experiment must pass these values into its per-example CTC and keyword losses;
loading the JSONL alone does not apply them. The adapter is a data/PCM boundary,
not evidence that any training or accuracy comparison has run.

## Generation mix and 0/5/10 training exposure are separate

The D1 generation layout is documented in
[`BULK_CLONE.md`](../research/auto-tts-voice-clone-20261102/BULK_CLONE.md).
It plans 4 near-negative requests out of 32 per source. This 12.5% generation
share does not prescribe a training sampling share.

The D1 adapter propagates the verified generation category and frozen-plan hash
into both consumer manifests. In particular, an agreed “小窝” from the
`partial-negative` category enters the ordinary/partial replacement pool even
though its preserved old generation intent is `hard-negative-intent`. Unknown
categories are rejected. No category can override uncertain acoustic/text
evidence or supply unsupported full-text CTC targets.

This connection is covered by a small fixture using the actual bulk runner's
file formats and fictional PCM/ASR responses, including source-separated
development rows and all three sampling settings. No real D1 reference set,
D1-generated speech, D1 ASR batch or D1 0/5/10 training experiment has been
executed by these source-contract tests.

`research_weak_sampling.exposure_plan(rows, total_exposures=...,
full_positive_exposures=..., near_percent=0|5|10)` returns indices into verified
training rows. All three comparisons retain the same total draw budget and the
same complete-positive draws and weights. Near-negative draws replace
ordinary/partial-negative draws. Sources and categories rotate within the
budget. Rows and their transcripts are not relabeled or edited.

If a source batch has only near-negative intents and no ordinary/partial
negative pool, the fixed-exposure comparison cannot yet be built. The sampler
reports that missing pool rather than replacing it with extra positives,
silence, or an OOV-to-blank target. D0 alone has that limitation; D1 is planned
to supply the separate ordinary and partial pools.
# Accounted continuation after a consumed call without a return

The explicit source adapter **kws-active-d1-partial-observation-source-v3**
accepts the separately declared **active-tts-accounted-checkpoint-v2**. It keeps
the original timed-out queue terminal and the original returned records. New
returned records require their own completed queue owners. The original
stage is never described as completed, and no old outcomes document is made.

The current bounded protocol accounts for the original prefix of 176, the
21 original returns, one consumed invocation with no return, and only the
58 previously unclaimed cells. The new jobs have 12, 12, 10, 12 and 12 calls.
The first three jobs pause for an authenticated ASR handoff. The additional
queue allocation is 29.9 minutes; its actual receipts are added to the
original 600,186,647,856 ns without rewriting that original allocation.
These are protocol limits, not a statement that future generation or
recognition has run.

Every original and new owner is authenticated through its immutable queue
request, recipe, claim, dependency receipts, terminal, logs and exact output
roster. Original call records bind the reference audio and PCM, source
family, reference text status, actual ICL or x-vector mode, prompt, seed and
generation settings. Returned NPY and native FLOAT WAV values are compared
without loading a numerical package. Valid native returns retain the
existing source-map, selective-ASR decision policy and loss masks.

The one **unobservable-consumed** entry has its original claim, call and start
records. Its missing return and terminal paths are checked explicitly. It
has no input clip, ASR request, observed text, synthetic PCM or training row.
It is never reclassified as FAILED_NO_RETRY. A returned but undecodable
native array is a separate **returned-unrecognizable** exclusion, with the
original return, failure and owner evidence retained.

Source-map and view creation take this mutually exclusive pair of options:

~~~sh
python3 -I -S -B automation/active_asr_weak_view.py source-map \
  --inputs PRIVATE/inputs.json --inputs-sha256 INPUT_SHA \
  --source-archive PRIVATE/native.zip --output PRIVATE/source-map.json \
  --d1-checkpoint PRIVATE/checkpoint.json --d1-checkpoint-sha256 CHECKPOINT_SHA
~~~

The same two checkpoint arguments are accepted by create. The checkpoint
can authenticate several generation owners in a single archive. A final
checkpoint excludes all previously returned cells from new recognition
requests. The archive still contains only the exact eligible input roster;
unknown or reused cells are rejected at this boundary.

The retained package keeps the full original plan at
**evidence/source/d1/plan.json**, its checkpoint, and each referenced file under
a deterministic private evidence name. Original absolute paths inside
receipts remain unchanged. d1_artifact_roots can explicitly relocate input
roots at intake; package replay uses retained files and never consults the
live ledger. New v3 view producers record exactly the seven files in
ACCOUNTED_SOURCE_PATHS; old v1/v2 producer records keep their original shape.

Two public standard-library verification APIs are available:

~~~python
verify_accounted_prior(path, expected_sha256, *, expected_count=256)
verify_accounted_seal(path, expected_sha256, *, expected_count=232)
~~~

verify_accounted_prior validates d1-accounted-prior-evidence-v2, replays
every immutable view, checks source-family and role separation, and returns
kws-d1-accounted-prior-verification-v2. Its normalized fields include the
full plan, plan hash, accounted prefix, observed IDs in plan order,
excluded_returned, unobservable_consumed, pending_ids, view pins and
all replayed evidence_files with absolute path, byte count and SHA-256.
Both unobservable categories include the original proof, family, role,
zero ASR calls, null observed text and training_admitted: false.
accounted_complete means the requested prefix is fully accounted for;
complete_terminal_coverage remains false when an unknown exists.

verify_accounted_seal additionally authenticates the completed ASR queue,
each actual request, source map, native call/result matrix, original and
derived PCM, routing and final audit against the retained views. Exact
recognition and input file rosters reject extra unretained calls. It checks
that the fixed control snapshot, the invoked verifier and every new view
producer have the same seven source hashes. Success returns:

~~~json
{
  "schema": "d1-accounted-selective-seal-verification-v3",
  "status": "verified",
  "verified": true,
  "accounted_prefix_count": 232,
  "full_replay_completed": true,
  "ml_imports": 0,
  "new_model_calls": 0
}
~~~

The result also contains the verified prior, actual queue state, new observed
IDs and evidence pins. Any mismatch raises an exception. The generation
handoff must require these success fields, not merely a dictionary return.
The sealing operation runs all four existing consumer/split loader calls;
subsequent verification can do one complete row replay and verify the four
saved projections from those same fully checked rows.

Tests in test_active_d1_accounted.py use explicitly fictional queue receipts,
numeric audio and saved ASR outputs. They cover mixed owners, the unknown
boundary, actual duration arithmetic, all 231 observed rows in a fictional
accounted prefix of 232, and replay after the original generation roots have
been removed. They make zero model calls and are not evidence of a real
continuation run.
