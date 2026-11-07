# Sealed natural-window ASR evidence and machine-weak text

`active_natural_asr_weak_view.py` reads prepared natural windows and **existing**
sealed native ASR results. It retains every original transcript and computes a
separate, explicit machine-weak view. It runs no recognizer, queue, model, audit
sealer or trainer. The existing D0/D1/D2 view schemas and decisions retain their
own semantics; this adapter does not infer generation intent for natural audio.

## Frozen decision rule

The source constant `DECISION_RULES` is copied into every report. The policy ID
is `natural-three-clean-fulltext-or-two-fulltext-with-unanimous-keywords-v1`.
The model order is `qwen06`, `sensevoice`, `whisper`. All three native result
records must be complete, quality-usable, terminated, flag-free and have a
nonempty normalized full transcript before either quorum rule applies.

| Whole-window observations | Full-text result | Per-keyword result |
| --- | --- | --- |
| Three normalized full transcripts are identical | Nonempty agreed full text; CTC mask true, weight 1 | Literal presence of each complete target, mask true, weight 1 |
| Exactly two normalized full transcripts are identical, and all three transcripts have identical literal-presence vectors for both complete targets | Nonempty agreed full text; CTC mask true, weight 0.5 | Literal presence of each complete target, mask true, weight 0.5 |
| Any other valid observation set | Full text null; CTC mask false, weight 0 | Targets null, both masks false, both weights 0 |

The complete targets remain `你好小窝` and `小窝小窝`. Normalization uses
`kws_data_contract.reference_text.normalize`: the existing orthographic rule,
including punctuation and Han spacing, without changing different characters.
Raw text from each decoder is retained independently. The selected raw sentence
is also retained beside the normalized CTC candidate; text is never truncated
to a keyword or replaced with an expected sentence.

The existing ordered-character-path check is recorded as a diagnostic. An
ordered but nonliteral path is **not an additional veto**. Near words are not a
general phonetic admission gate. For example, two equal sentences containing
`你好这个小窝` and a third different sentence with the same absent-target vector
can satisfy the two-full-text rule. Conversely, two full transcripts containing
`你好小窝` and a third containing `你好小屋` disagree on the complete target and
cannot receive a quorum. These distinctions are fixed before new observations.

Three absent-target votes with no full-transcript quorum do not become negative
labels. When a full nonempty ordinary sentence does receive a quorum, its
keyword targets can be `[0, 0]`, while CTC keeps that complete nonempty sentence.
There is no empty CTC negative export.

All such labels remain machine weak, never human gold or acoustic ground truth.
`acoustic_verbatim_completeness` and acoustic target presence remain unknown;
event times remain null. A window's integer parent interval is a provenance
boundary, not a keyword timestamp. `accepted_negative_seconds` stays **0** and
`natural_far_available` stays false even when a machine-weak target equals zero.

## Required request and CLI

Create a private JSON request with schema
`kws-natural-asr-evidence-request-v1` and an explicit list of batches:

```json
{
  "schema": "kws-natural-asr-evidence-request-v1",
  "batches": [
    {
      "batch_id": "natural-batch-001",
      "bundle": "/private/natural-inputs-001",
      "declaration_sha256": "ACTUAL_DECLARATION_FILE_SHA256",
      "bindings_sha256": "ACTUAL_BINDINGS_FILE_SHA256",
      "asr_output": "/private/natural-asr-001",
      "asr_request_sha256": "ACTUAL_ASR_REQUEST_FILE_SHA256",
      "final_audit_sha256": "ACTUAL_EXISTING_FINAL_SUMMARY_FILE_SHA256",
      "policy": {
        "path": "/private/selective-asr-v2.json",
        "bytes": 1,
        "sha256": "5cba6043b6c7e39aaa39482b31e0e14ae478cd5485b386e7d87f5bec78a3c501"
      }
    }
  ]
}
```

Replace each placeholder with the actual lowercase SHA-256 of the file bytes;
replace the example policy byte count with the actual count. The bindings file
is `natural-window-bindings.json`. The ASR recipe is `asr_output/request.json`;
the final seal is `asr_output/audit/final-summary.json`. All paths are absolute.
The selective policy bytes have the fixed hash shown above. One request permits
1–64 batches and at most 1,024 total windows; each original bundle permits at
most 8 parents, 64 windows, and 30 seconds per window.

```sh
python3.12 -I -S -B automation/active_natural_asr_weak_view.py create \
  --request /private/natural-view-request.json \
  --request-sha256 ACTUAL_REQUEST_FILE_SHA256 \
  --output /private/natural-view-001

python3.12 -I -S -B automation/active_natural_asr_weak_view.py verify \
  --package /private/natural-view-001 \
  --manifest-sha256 ACTUAL_VIEW_MANIFEST_FILE_SHA256
```

The output must be a new directory outside the source checkout and every input
bundle or ASR evidence directory. A missing seal
or wrong recipe pin fails before creating the output. Later validation failures
may leave a partial private directory as an attempt record; use a fresh output
name for any corrected attempt. Neither command changes an upstream input,
native result, recipe, inventory or seal. No missing requested result is filled
in, retried, labelled negative or counted as an agreeing vote.

## Validation and retained evidence

The natural source adapter checks the exact declaration and bindings, source
IDs derived from parent PCM hash and integer sample interval, parent/session/
family roles, window WAV bytes, archived WAV bytes and the exact PCM16-to-float32
derivative. Derivation is signed int16 divided by 32768; there is no gain,
resampling, padding, clipping or trimming. The unreviewed-natural source-map
flag remains true, which makes all three native recognizers requested under the
existing selective policy.

The shared `replay_selective_evidence` reader validates the existing final seal,
complete inventory, original recipe, routing, native output/result/claim joins,
audio and model identities, native text projection, quality/termination fields,
actual worker receipts, request counts, process supervision, offline guard and
deadline evidence. A complete view therefore contains three real sealed native
observations per window. A valid observation can still be unusable for weak
text. Missing or inconsistent native evidence rejects the view; it is not a
new `not_requested` state for natural audio.

Original model-loading receipts are retained and their sealed bytes are
checked. Their checkpoint/tensor application semantics remain those of the
pinned upstream canonical audit; this data consumer does not repeat that model
state verifier or load model weights.

This data-side replay checks the **retained window bytes and declared ancestry**.
It does not reload complete parent WAVs, source archives or annotations. The
pinned upstream preparation evidence records the original parent check; the
pipeline's natural coverage consumer can additionally rerun the existing
preparation verifier against the actual complete parent WAVs. Retaining a
provenance field does not independently authenticate its upstream source.

Duplicate global source IDs across batches are rejected. Parent PCM identity,
session/channel identity and source family roles must stay consistent. Parent
and window WAV/PCM hashes participate in split checks. Test/heldout partitions
are not accepted, and an eval/development partition cannot be relabelled train.
The eight already observed continuation sessions therefore retain development
roles. The view emits no trainer manifest and sets `training_admitted=false`.

A successful package contains `request.json`, `report.json`, `rows.jsonl`,
`MANIFEST.json`, the exact `PRODUCER_FILES` sources under `evidence/view-code/`,
and every upstream input/native inventory file under each batch. Original
absolute path strings remain evidence. Replay uses retained files and works
after the original directories are removed; it never calls a sealing function.

`verify_view(package, *, expected_manifest_sha256)` is the public read-only API.
It verifies the externally supplied manifest pin, all retained file bytes and
rosters, the producer sources and the original native evidence, then rebuilds
every report row and decision. Changing a report and rehashing its outer
manifest cannot substitute for native evidence. The loaded modules must come
from the same checkout, and producer sources cannot change after import. If the
producer changes, replay the old package with its retained source tree in a
fresh isolated interpreter; do not silently repin the old report.

For coverage consumers, each `report.batches` item provides complete status,
declaration/bindings/input/recipe/final-audit/inventory hashes and
`requested_native_results`. Each row provides source, parent PCM/WAV, dataset,
session, family, split and exact sample interval. Its observations have model,
original text, native identity, status `complete`, requested true and a separate
usable boolean. `decision.full_text_quorum` and `decision.ctc_mask` identify
weak full-text coverage. Complete observation coverage, usable observations,
weak-text quorum coverage and verified negative exposure are separate counts.

## Verification without model calls

```sh
for mode in '' '-O' '-OO'; do
  python3.12 -I -S -B $mode automation/test_active_natural_asr_weak_view.py -v
  python3.12 -I -S -B $mode automation/test_selective_asr_weak_view.py -v
done
```

Fixtures contain fictional PCM and native receipts. They exercise full-text and
per-keyword counterexamples, incomplete/low-quality decodes, missing seals,
wrong recipe and manifest pins, native/report forgery, source and split aliases,
gain changes, retained-only replay, module provenance and source changes after
import. These checks do not represent any new corpus, model or training run.
