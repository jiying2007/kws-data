# Original B24R1 SOURCE STOP readback

This utility authenticates saved evidence from SOURCE run `45534020313`, head
`b65bfe4974432b3d922738d7931d1c2d6d6f367c`, prepared parent
`ea9f9e3b467231efe2746cbe64f867481177d09e`. It grants no execution permission.
The real Parakeet ninth row `b24r1-primary-84-mero-fast` saved a normal native
exit with stdout consisting of one newline, rejected by the unchanged original
parser as `EMPTY_NATIVE_TRANSCRIPT`. This is a SOURCE failure, with no KWS
quality verdict. No unentered call, reserve, continuation, retry or R2 is opened.

The reader was authored before the remaining providers' final evidence was
bound. The completed original readback below now verifies those outcomes using
the independently downloaded final APIs, all original fourteen archives, both
failed job logs, and exact reviewed pins and counts. The code contains no
prefilled final evidence lock or execution opening. Missing final evidence or a
different failure pattern cannot produce a report. It does not reconstruct
missing observations or replace a failed run.

## Completed original readback on 2026-11-25

The corrected full isolated readback completed successfully against the actual
[original SOURCE run](https://github.com/jiying2007/kws-data/actions/runs/45534020313),
whose final GitHub conclusion is `failure`. The report status is
`SOURCE_STOP_EMPTY_NATIVE_TRANSCRIPT`; a successful evidence verification does
not change the failed SOURCE outcome.
The retained [evidence record and terminal state](../english_v6_b24r1_source_stop_20261125/README.md)
provide the complete original lock and canonical report without publishing raw
API, job-log or audio payloads.

| Bound item | Exact bytes / SHA-256 |
| --- | --- |
| [Canonical complete STOP report](../english_v6_b24r1_source_stop_20261125/READBACK.json) | 12,097 bytes / `e3ca7baa272609fc8a4b304fb56b52bd98c8a85a504dd2f4ab0fab298b777ffd` |
| Corrected reader source | `ac3e12810f72f68c3bdd4328dc5933b71c3e05da00524b508f91ce870b83d83d` |
| [Complete original lock](../english_v6_b24r1_source_stop_20261125/ORIGINAL_LOCK.json) | 5,435 bytes / `56c10ef6a669650050fd1d821871489c030f13f224fc85d1c9dc537c69ef37f8` |

| Original provider | Entered / returned / raw saved | Valid captures | Empty failures | Never entered |
| --- | ---: | ---: | ---: | ---: |
| Qwen | 120 | 120 | 0 | 0 |
| Whisper | 120 | 120 | 0 | 0 |
| Parakeet | 9 | 8 | 1 | 111 |
| XLSR | 120 | 120 | 0 | 0 |
| LV60 | 120 | 120 | 0 | 0 |
| **Total** | **489** | **488** | **1** | **111** |

The complete readback verified all fourteen original ZIPs, all five original
API bodies and both failed logs, 1,460 actual producer Git blobs, every native
FILES closure and all 600 planned journal slots. It reparsed all 488 valid
payloads, including 18,344 original phone frames with model/vocabulary binding,
finite-float checks and greedy CTC parity, and reproduced the one original empty
return rejection. The nested original B24 STOP was fully reopened twice and
reproduced its unchanged frozen report.

The first report draft preceded the complete phone semantic checks and is
superseded. Only the corrected report and reader hashes above identify this
completed readback. Independent review found no remaining material blocker in
the corrected saved-evidence reader; its 17 mutation contracts pass in normal,
`-O` and `-OO` modes. These facts provide no source admission or KWS quality
result. `quality_verdict` and `model_quality_claim` are null, all downstream
permissions remain false, the R1 source is permanently stopped, and no retry,
continuation, sealed reserve or automatic R2 is opened.

## Saved evidence interface

Run the stdlib-only reader from any checkout; it imports scientific validators
only from the authenticated actual SOURCE producer archive.

```sh
python3 -I -S -B automation/english_v6_b24r1_source_stop_readback.py check
python3 -I -S -B automation/english_v6_b24r1_source_stop_readback.py verify \
  /absolute/originals /absolute/reviewed-lock.json /absolute/new-STOP-report.json
```

`check` is inert. `verify` starts a fresh isolated worker and creates the report
only after the complete readback succeeds. It does not download anything or
modify the original evidence. The report is canonical ASCII JSON with sorted
keys, compact separators, no trailing newline and no NaN. Its SHA-256 binds the
exact output bytes; it also records the exact reader source SHA-256 and lock.

The originals directory has fourteen root ZIPs:

| Role / filename | Original artifact name |
| --- | --- |
| `source.zip` | `english-v6-b24r1-primary-source-45534020313` |
| `source_producer.zip` | `english-v6-b24r1-primary-source-producer-45534020313` |
| `transport.zip` | `english-v6-b24r1-primary-transport-45534020313` |
| `source_consumed.zip` | `english-v6-b24r1-primary-source-consumed` |
| `native_<provider>.zip` (five) | `b24r1-primary-dev-observer-<provider>` |
| `consumed_<provider>.zip` (five) | `english-v6-b24r1-primary-observer-<provider>-consumed` |

Providers are exactly `qwen`, `whisper`, `parakeet`, `xlsr`, `lv60`. Preserve each
complete original downloaded ZIP, including all raw native payloads. Only
`source.zip` has the already registered 512,000,000-byte compressed and expanded
envelope. All thirteen other original ZIPs retain 128,000,000-byte envelopes.
The nested original B24 proof retains its immutable individual archive limits.

Keep these exact independently fetched API bodies under `metadata/`:

| Filename | Supported original API endpoint in `jiying2007/kws-data` |
| --- | --- |
| `source_run.json` | `/actions/runs/45534020313` after completion |
| `source_artifacts.json` | `/actions/runs/45534020313/artifacts?per_page=100` after completion |
| `source_jobs.json` | `/actions/runs/45534020313/jobs?per_page=100` after completion |
| `source_commit.json` | `/commits/b65bfe4974432b3d922738d7931d1c2d6d6f367c` |
| `source_tree.json` | `/git/trees/c3c804008ae68d7fb1166becdd11469bcf4d5e1d?recursive=1` |

Also retain the complete independently downloaded `PARAKEET_JOB.log` and
`READBACK_JOB.log`. Their pinned job IDs must match the final eight-job API
inventory. Generation snapshots and rewritten API JSON cannot replace these
original final byte bodies. No workflow-specific run list or global artifact
search API is required.

## Exact lock

The lock schema is `english-v6-b24r1-original-source-stop-lock-v1`. Required keys:

- `schema`, `repo`, `run_id`, `run_head`, `prepared_parent`, `branch`, `workflow`:
  the fixed actual identity above and the registered primary SOURCE workflow.
- `artifacts`: exactly the fourteen roles above, each with `id`, `name`, `bytes`,
  `sha256` from its original downloaded archive and final artifact API.
- `api`: exactly `run`, `artifacts`, `commit`, `tree`, `jobs`, each with `bytes`
  and `sha256` of the unchanged API response file.
- `logs`: exactly `parakeet` and `readback`, each with `path`, `job_id`, `bytes`,
  `sha256` from the original job log.
- `source_producer_manifest_sha256`, `source_manifest_sha256`, `history_sha256`,
  `source_assessment_sha256`, `candidate_pt_sha256`, `train_result_sha256`:
  exact original producer, source and TRAIN receipt hashes. The R1 assessment
  remains distinct from the original B24 TRAIN assessment.
- `provider_counts`: the actual final five-provider ledger counts. Each row
  has `status`, `input_started`, `forward_attempted`, `native_returned`,
  `raw_payload_saved`, `payload_valid`, `capture_saved`, `failed`, `not_started`,
  `entered_without_return`, `entered_without_raw`. Derive these from final
  original journals and independently review them before running `verify`.

No lock is usable with null, missing or guessed pins/counts. The known Parakeet
boundary is 9 entered/returned/raw saved, 8 valid/captured, 1 empty-return failure,
111 never entered, no entered-without-return/raw gap. The completed original
readback verified the four other complete providers and totals of 489 original
attempts and returns, 488 valid payloads, and 111 never-entered calls. Unit
fixtures remain invented contract examples; the completed evidence report is
the record of actual execution.

## What is recomputed

Before any captured imports, the reader verifies all original API pins, the
first-attempt failed public push, its workflow-only child commit, final jobs,
exact fourteen artifact identities/digests, every ZIP CRC/member and every
producer file's actual Git blob and mode. It recomputes the actual captured
producer's complete dependency closure and all nineteen dynamic helper pins.

The unchanged captured SOURCE preflight reopens all 120 complete WAV/PCM
originals and synthetic native chunks, natural-source metadata and ancestry,
original B24 TRAIN proof, and the complete nested original B24 STOP. The only
allowed child process is the exact isolated immutable old B24 failure reader
with SHA `2d5564039689d5fb5e31bf5cde51d49dba5bf9d0697c295cd98c96f7706585c6`.
Its full recomputation must reproduce original report SHA
`4063a57f1d20ffe9f534248067e1adcbe9c25ab65cb5bd3682b5db44d3ba39d8`.

Each provider's complete FILES closure, all 120 journal slots and saved counts
are independently reconstructed. All valid ASR payloads and all phone frames
are reparsed with the original captured readers. For phone payloads, the strict
observer reader authenticates original raw-return/validity receipts and frame
geometry; the captured B11 loader's existing pure checks additionally verify
model/revision/weight and vocabulary identity, every finite float in every frame,
and exact argmax/greedy CTC parity with the saved tokens. The all-five loader,
source admission and acoustic quality decisions are never called. The actual failed Parakeet
payload is passed to the unchanged parser, which must reject it with precisely
`EMPTY_NATIVE_TRANSCRIPT`. No successful capture or validity record is invented.
All 111 unentered row directories must remain absent. Original consumed-marker
ZIPs are compared byte-for-byte with their retained budget receipts. The final
failed readback log and artifact inventory must both establish absence of SCREEN.

The report status is `SOURCE_STOP_EMPTY_NATIVE_TRANSCRIPT`. It has no source
admission result, no model-quality verdict, no model claim and no downstream
permission. This describes the SOURCE execution; it is not a claim about model
quality. All calls during readback are zero source/observer/KWS/optimizer calls.
No original B24 gate, R1 opening, parser, threshold or runtime is edited.

## Validation scope

The separate contract workflow runs invented mutation tests in normal, `-O` and
`-OO` modes with no runtime installation or scientific execution. Tests cover
incomplete/final API substitution, wrong Git identity/mode/member, archive
mutation and capacity/path boundaries, lost native returns, reassigned rows,
changed empty stdout/exit, invented validity, nonmatching parser rejection,
wrong phone model/vocabulary, nonfinite floats and changed saved CTC tokens,
scientific imports/network/arbitrary child processes and all-false permissions.
These contracts cannot substitute for the complete original saved-evidence run.
