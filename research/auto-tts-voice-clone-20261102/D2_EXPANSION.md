# D2: cumulative 32-source, 64-request coverage

D2 retains the frozen D1 plan and all 256 historical outcomes, including a
consumed call with unknown output. It never retries D1 to reach an audio quota.
The requested cumulative mix per reference identity is 16 keyword-1 positives,
16 keyword-2 positives, 20 ordinary negatives, 8 partial requests and 4 near
requests. These are generation intents; actual ASR observations determine weak
labels, and usable counts are reported separately.

| Increment | Sources | New requests per source | New calls |
| --- | --- | --- | --- |
| `existing-eight` | Original six train and two development | 8 kw1 + 8 kw2 + 12 ordinary + 4 partial | 256 |
| `new-twenty-four` | 18 additional train and 6 development | 16 kw1 + 16 kw2 + 20 ordinary + 8 partial + 4 near | 1536 |

The inherited 256 D1 cells plus these 1792 new cells form 2048 candidate
positions. An unavailable new reference does not block the original-eight
increment. Each increment has its own plan, output and model-call budget;
completing one never rewrites the other. Keep the same batch ID across the two
increments. Seeds and cell IDs include stage and frozen D1 identity.

## Existing references: prepare immediately, without running a model

Run from the repository root, replacing the private paths and parent digest
with the actual retained inputs. The output directory must already exist;
omitting `--output` performs a read-only source and plan preview.

```sh
python3 -B research/auto-tts-voice-clone-20261102/d2_pool.py \
  --stage existing-eight --batch-id d2-20261105 \
  --parent-plan /private/d1/plan.json \
  --parent-plan-sha256 ACTUAL_CANONICAL_D1_SHA256 \
  --speaker-registry /private/speakers.json \
  --sources /private/research-source-pool.content-observed.json \
  --audio-root /private/references \
  --output /private/d2-existing/plan.json
```

The D1 parent must verify with the unchanged `bulk_pool.py` D1 contract.
The original speaker/family/role, original and normalized audio, source terms,
publisher text and reference IDs are pinned. Byte-bound reference-content
observations may improve after D1 without changing these identities. The
planner checks the existing persistent registry and never assigns a new split.
It preserves the model revision from D1 unless an explicit revision is supplied.

The existing `bulk_runner.py` accepts the D2 increment schema. Its model prompt
cache, native audio writer and consumed-start receipts remain in use:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -B \
  research/auto-tts-voice-clone-20261102/bulk_runner.py \
  --plan /private/d2-existing/plan.json \
  --sources /private/research-source-pool.content-observed.json \
  --audio-root /private/references \
  --model-root /private/pinned-qwen-base --model-lock /private/qwen-base-lock.json \
  --out /private/d2-existing/generated --max-new-calls 8 --execute
```

A supervising queue may call `bulk_runner.run()` with the existing bounded
backend to retain returned arrays even when a technical check fails. A saved
start, failed or unknown result is consumed; it is never repeated to make a
quota. Each invocation allows at most 64 new calls. The 256/1536 limits are
separate from wall-time and total compute budgets enforced by the queue.

## Add the 24 new references using the existing importer

Use the original WAV, transcript, terms and conversion evidence for 24 distinct
new speakers. `prepare_open_reference_pilot.py` and
`research_only_bulk_ingest.py` now support up to 32 references per invocation.
Append to the existing registry with `python3 -m kws_data_contract.source_registry
--registry ... --index ... --execute`; do not initialize a replacement registry.
The current append policy preserves the old six/two assignment and adds
18 train/six development identities.

Produce a source manifest containing **only the 24 new references**. Pass it to
the same D2 CLI with `--stage new-twenty-four`, the same frozen D1 parent and
batch ID, the extended registry, and a separate output directory. The planner
rejects a repeated D1 speaker, family or original/normalized audio disguised as
a new source. Missing sources and reference failures stay explicit.

## Consumption boundary

`plan.references`, `plan.cells`, model and budget fields are shared with the
existing worker. D2 stores the immutable parent under `d1_parent`, a frozen
public projection of the private registry under `speaker_registry`, and the
stage separately. It does not add D1 cells to the generator work queue.

The native ASR dispatcher accepts both plan schemas. Preserve original ASR
receipts and replay the observed weak-label adapter; do not treat requested
text, a model-call count or `GENERATED_PENDING_ASR` as a usable label. Unknown
and technically invalid audio remain represented with masked training loss.
This extension does not claim new independent human speakers or test accuracy.

## Offline checks

```sh
python3 -B research/auto-tts-voice-clone-20261102/test_d2.py -v
python3 -B research/auto-tts-voice-clone-20261102/test_bulk.py -v
python3 -B automation/test_prepare_open_reference_pilot.py -v
python3 -B automation/test_research_only_bulk_ingest.py -v
```

Fixtures use artificial PCM and mock model returns. Tests cover the actual
32-reference intake, cumulative 2048-position mix, frozen D1 identities,
cross-source ancestry aliases, registry continuity, consumed unknown starts,
native ASR receipt replay and absence of requested-label authority.
