# B24R1: one separately registered evaluation of frozen B24 weights

Registration timestamp: **2026-11-24 18:41:45 UTC** (2026-11-25 02:41:45 Asia/Shanghai).
The experiment directory retains the `20261124` identifier used by the surrounding trials.
This registration was written after actual B24 training and the original B24 SOURCE failure, before any B24R1 audio, observer call, or independent keyword-model evaluation.

## Why this is a new evaluation registration

The original B24 SOURCE run [45456888767](https://github.com/jiying2007/kws-data/actions/runs/45456888767) ended in failure.
Parakeet returned a real empty transcript on original row `b24-primary-284-mero-fast`: 49 calls entered and returned, 48 valid transcripts, one empty return, and 71 rows never entered.
The frozen whole-panel-stop rule remains effective. The original B24 panel has no passing source gate, no SCREEN artifact, and no independent KWS scoring permission.
The remaining original rows will not be completed, replayed, relabeled as ordinary, or supplemented from another panel.

B24R1 is one bounded new evaluation of exactly the already trained B24 model with a fully independent primary panel and separately sealed replication panel.
It is prompted by source acquisition failure, before any independent B24 KWS output existed.
It changes neither the model nor the source format, native parsers, strict five-observer requirements, source minima, gains, or B14 decision.
No format diagnostic or canary is introduced: there is no unresolved format change to evaluate, and a preliminary observer screen would itself require a new scientific budget and additional exclusions.

This follows the existing B20 contract's distinction between an immutable training assessment and a new assessment for evaluating the same frozen models on new sources.
It does not relabel old training as having used the new metadata.
The original B24 SOURCE STOP and all original artifacts remain part of the record regardless of B24R1's outcome.
A B24R1 SOURCE failure is terminal for this registration. **No automatic R2, repeat sampling loop, or reclaiming an old reserve is authorized.**

## Immutable training identity and a separate evaluation identity

Actual B24 TRAIN run: [45449190562](https://github.com/jiying2007/kws-data/actions/runs/45449190562).
Its head remains `146b9e8022e067855f96e2e6a33f0c8d2e5d23af`; its 192 completed challenger updates are historical facts, not updates authorized by B24R1.

| Identity | Frozen SHA256 |
| --- | --- |
| Original B24 TRAIN assessment | `3afdb17e822bb0b4a49757ac116a28101ff9f1447b22429305299047410fe2ad` |
| Original B24 TRAIN design | `4fa88cf4050aa276323a7306f21d53163d47c27e3c011c2547d5eeebe6980309` |
| B24R1 evaluation design | `79300b64e5870272ebeb54cc4364ee07b7e01518ad1b95752cc3a8ec2938d3a4` |
| B24R1 evaluation assessment | `e783c982f66ed35f903680772842ca8068aad6fb18221e22b9512cd819efa075` |
| B24R1 ancestry | `eca9b1fa081022f0efa4da79f4d398affdce01a6ebdd81e4cad0bcd6405d3c06` |
| B0 weights | `1ae8a2b20c8b6991f925dd4c626af2f36b308adee8733a3a7215818716f565d7` |
| B9 weights | `7dba86a3fcda6e5f05cb80b414dc372c70a5a03b3aee783f168441a4871e7655` |
| Historical control: actual B23 weights | `784def9df6c2a8bfbecdb7ba2d7b70630337a1cda72561c05392327aa980297f` |
| Challenger: actual B24 weights | `f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4` |

The new assessment explicitly records `selection_before_optimizer=false`, `selection_after_training=true`, and selection before new audio and first independent KWS evaluation.
It authorizes zero optimizer updates, zero new training audio, zero restored optimizer state, and no new model selection.
The original B24 warm-start receipt and its original training assessment must still pass their original checks.
The new evaluation binding is separate and cannot replace that receipt.

## Complete exclusions and one deterministic selection

The preserved ancestry contains **2,823 PCM identities**: all 2,583 earlier identities, all 120 B23 original-source identities, and all 120 B24 original-source identities.
It also retains all prior explicit and conservative rosters, canonical exposed people, and never-reclaimed reserves.
All original B24 primary 36 people and sealed replica 36 people are excluded from both source types, including the original rows whose observers never ran.

The historical type-specific exclusion sets remain intact. They are not interchangeable: the natural-corpus exclusion set additionally excludes the whole potentially overlapping Piper population, including unused synthetic voices.
The validator separately verifies that every actually exposed or reserved historical canonical person remains excluded from both source types.

The unchanged SHA256 metadata-ranking algorithm uses the newly frozen `english-v6-b24r1-frozen-b24-evaluation-...-20261124-v1` salts exactly once.
No source outcomes, observed transcripts, quality outputs, audio, or model bytes participate in selection.
The available pool contains 208 synthetic people and 90 eligible natural readers; selection yields 48 synthetic and 24 natural people, all 72 canonical identities mutually distinct across both panels and both source types.

Each panel reserves 96 synthetic originals from 24 people and 24 complete natural originals from 12 people.
The synthetic cells retain positive length scales 1.4 and 1.8, ordinary scales 1.0 and 1.08, the same prompts and native inputs, and zero replacements, crops, padding, or retries.
The natural roster retains exactly two whole eligible originals per reader.
Both the old B24 reserve and the new B24R1 reserve remain unopened at this registration.

## Unchanged scientific gates

Per panel, the native budget remains 600 calls: five fixed native observers for each of 120 whole original sources.
Every required returned payload must be valid under the original parser. A real empty return remains an error and stops that panel.
The source minima remain 12 synthetic positives, 24 synthetic ordinary originals, eight paired synthetic families, 12 natural originals, and eight natural readers.
The maximum primary budget remains 1,440 KWS forwards, with gains `1.0`, `0.75`, and `0.5` and the same four frozen models.

B14 quality assessment `944f5df3c7bc3244977a8a33f7e1c393bffff9f7f5c32d2d8cc79e2628e265d9` and decision function `6aa5f999598e47b41eb0a946acd82fdc5b47d30aea8990871c1c1a01065e4023` remain unchanged.
There is no threshold search, empty-return exception, new acceptance threshold, or training use of any evaluation source.
Workflow completion is not source acceptance or quality acceptance.

## Minimal public provenance and separate full STOP proof

Only two sanitized B24-original metadata exports are added:

- `b24-source-exposure.json`: all 120 original rows, retaining only `id`, `cohort`, `source_group_id`, `pcm_sha256`, and `canonical_person`, plus the exact original primary/reserve people sets.
- `b24-source-origin.json`: fixed source and producer artifact pins, selected run/Git identity fields, original MANIFEST/HISTORY digests, and the captured original assessment's byte count, SHA256 and Git blob identity.

The original source archive is artifact 14950099270, 59,393,379 bytes, SHA256 `b4dd350b773ae925696f692e8f67caf7e83444220eab9cbb1f675ec20176ac4e`.
The original producer is artifact 14951855225, 17,486,742 bytes, SHA256 `bcf47dc4e0d81f9678abf750ea184728a1cc8f7f1684ebbb9e71a36626bc6f7e`.
Both original archive digests and the captured assessment's exact equality to the immutable Git bytes were verified before export.
The source MANIFEST and HISTORY member digests are `5ab44377322029a99b105a998091550d2ef7a3768b01b67501c60d04b6256a12` and `fc5b2e3683222b5fd11136d42cd24949273b57148dd10a144bc6db6f34d8d72d`.

No new raw API record, actor information, log, native transcript, full source manifest, audio, or original ZIP is copied into this public registration.
The metadata validator does not claim to certify the complete native failure predicate: the separate fail-only reader must authenticate all original SOURCE artifacts, original producer, raw returns, ledgers and the real STOP before activation.

## Authority and validation

All metadata execution, source, scoring, replica, engineering, optimizer, and promotion permissions are **false**.
The binding explicitly records that the complete original SOURCE STOP proof is not yet bound.
Execution requires that separate original proof and an independently reviewed activation; this registration alone is never a consumption license.
Only a future authenticated B24R1 primary numerical PASS can satisfy the prerequisite for its own sealed replica, followed by the unchanged engineering and physical/field requirements.
The separate conditional B25 design keeps its original gate and remains sealed. A SOURCE failure or a B24R1 result does not silently substitute for B25's registered opening condition.
Physical and field evidence remain `NOT_RUN`.

The source-only contracts verify all inherited exclusions, old/new assessment separation, exact four-model identity, deterministic rosters, unchanged gates and source cells, the public whitelist boundary, rejection of missing originals or modified permissions, and pure standard-library metadata execution with model libraries, sockets and subprocesses blocked.
Original-byte reexport additionally verifies the actual source ZIP and captured original assessment without decoding audio or invoking an observer.
These checks execute zero scientific calls.
