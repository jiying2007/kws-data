# B24Q1 quality entry and complete evidence capacity

This is an inert implementation registration. It records no Q1 source PASS,
quality PASS, replica opening, engineering result, installation or physical test.
The actual saved outputs and independent original readers must establish each
permission in order. The separate static `evaluation_binding.json` is metadata
registration and is never rewritten by either scoring binder.

## Approved source policy and unchanged quality decision

The new Q1 source orchestration is explicitly bound to [APPROVED_PROPOSAL_SOURCE.json](APPROVED_PROPOSAL_SOURCE.json),
[APPROVAL.json](APPROVAL.json) and
[`english_v6_b24q1_orchestration_contract.json`](../../automation/english_v6_b24q1_orchestration_contract.json).
It does change the whole-panel treatment of an independently proved successful
empty native return. It does not reinterpret the original B24 or R1 STOP under
the new rule. Both stopped assessments and their unopened replicas remain closed.

The primary evaluator, replica evaluator and independent saved-byte quality reader
retain the full source audit alongside the selected quality records:

- The fixed exposure is all 120 original identities, including natural originals
  rejected before observation. The fixed observer population is
  `N = 96 + unchanged pre-observer-eligible natural`; quarantine never reduces N.
- All five providers must complete exactly N attempts and returns. Valid and
  independently proved empty returns are separate counters. The total is exactly
  `5*N <= 600`, with zero fatal or unknown slots and no calls to fill unused space.
- The source reader reopens every raw payload and ledger, recomputes the union of
  quarantined original IDs and replays unchanged five-valid source admission.
  Every quarantine stays unknown, with no admitted class and zero KWS forwards.
- The four disjoint counts are pre-observer rejected, quarantined invalid empty,
  five-valid unqualified and qualified. They sum to 120. Full cohort, requested
  class, cell, person and provider attrition is preserved in the original SCREEN
  and in the quality audit; no whole-population accuracy is imputed.
- Every qualified original is scored, once for each of four fixed models and each
  gain. There is no selected subset of the qualified population.

The four fixed model roles are B0, B9, the historical original B23 challenger as
control, and the original B24 challenger. This is not a newly trained matched
control. The B24 candidate PT remains
`f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4`.
The original TRAIN assessment remains
`3afdb17e822bb0b4a49757ac116a28101ff9f1447b22429305299047410fe2ad`,
distinct from the new Q1 assessment
`681ddb42432bd11dfef0157d2f1c95f7e02824c3a2ed7da64ce9a12bdf9c3139`.

The original B14 decision file SHA remains
`6aa5f999598e47b41eb0a946acd82fdc5b47d30aea8990871c1c1a01065e4023`;
its quality assessment remains
`944f5df3c7bc3244977a8a33f7e1c393bffff9f7f5c32d2d8cc79e2628e265d9`.
Gains remain 1.0, 0.75 and 0.5, with no more than 1,440 forwards per panel.
The source minima remain 12 synthetic positives, 24 synthetic ordinary originals,
8 synthetic paired groups, 12 natural originals and 8 natural speakers.

## Role-specific complete archive limits

All limits use **decimal bytes**, as the original 128,000,000-byte convention does.
Both compressed and expanded complete archive sizes must satisfy the relevant
limit; path, member, CRC and provenance checks remain mandatory.

| Exact archive role | Maximum bytes |
| --- | ---: |
| New Q1 PRIMARY SOURCE | 768,000,000 |
| New Q1 replica SOURCE | 1,280,000,000 |
| New Q1 replica quality producer | 1,024,000,000 |
| PRIMARY quality producer and ordinary originals | 128,000,000 |
| Historical R1 SOURCE, nested unchanged | 512,000,000 |
| Historical ordinary originals, nested unchanged | 128,000,000 |

The limits are prospective transport envelopes. They do not authorize source
generation, observation, scoring or a quality verdict. No nested original ZIP
receives a larger limit because its outer container is larger.

The actual retained R1 proof, pinned in the
[original lock](../english_v6_b24r1_source_stop_20261125/ORIGINAL_LOCK.json), has
14 ZIPs totaling 308,497,677 bytes and five final
API receipts plus two original logs totaling 1,697,604 bytes: 310,195,281 bytes in
all. The unchanged original TRAIN/LIVE evidence totals 50,759,903 bytes. The
unchanged maximum 96-original TTS envelope is 109,588,224 bytes. New metadata has
a 32,000,000-byte reserve. Those terms already total 502,543,408 expanded bytes
before any new natural audio. The conservative ZIP envelope for those terms is
593,072,007 bytes, so the old 512,000,000-byte PRIMARY SOURCE limit is insufficient.
Any separately retained historical metadata transport proof is additionally
counted among the exact known files; it is not removed from this calculation.

The capacity helper reads all 24 fixed natural FLAC STREAMINFO headers after the
complete corpus has been authenticated, retaining duration rejects as whole
originals. It reserves their exact native member and full PCM WAV lengths without
decoding audio. It combines these with every retained original proof byte and
the unchanged TTS upper bound. Before the first generation call, the saved
CAPACITY receipt is recomputed and must match exactly. The generated full source
is checked again before observer entry and on saved-byte readback.

For replica SOURCE, the complete actual original PRIMARY proof is counted only
after its independently reconstructed PASS. The historical R1 proof is nested in
`PRIMARY_QUALITY/inputs/source.zip`; it is not copied a second time at the replica
root. The actual original PRIMARY inventory plus all original TRAIN evidence,
all 24 whole natural originals, the TTS upper bound and metadata reserve must fit
the registered 1,280,000,000-byte envelope before replica consumption. This is an
actual-size gate, not an assumption that the future proof will fit. The replica
quality producer separately counts that complete PRIMARY proof and both original
producer roots against 1,024,000,000 bytes before any replica scoring consumption.

## Original evidence binding and ordered execution

The inert primary binder is:

```sh
python3 -I -S -B automation/bind_b24q1_primary.py \
  /absolute/repository /absolute/original-bundle.json /absolute/new-binding-output
```

The bundle schema is `b24q1-original-historical-control-primary-binding-inputs-v1`.
It retains four complete original API groups (`B0`, `B9`, `train`, `source`) and
twelve exact original ZIP paths: B0, B9, pair, source, screen, all five native
provider artifacts and both upstream producer artifacts. Every original digest,
run, commit, complete Git closure and native screen is reauthenticated before the
binder emits a proposed scoring registration and complete producer manifest.
It uses saved evidence only, performs zero model calls and never changes the
static evaluation binding, workflow or original artifacts.

After original Q1 SOURCE admission and review of those concrete binding outputs,
the PRIMARY workflow can be bound using its two registered predecessor-token
substitutions. Its source and producer pins remain null until that review. The
workflow verifies the whole source/observer/TRAIN proof before loading Torch,
uploads the durable consumed marker first, and retains every actual returned
array, saved logits matrix and per-forward journal.

The independent reader accepts an exact
`english-v6-b24q1-original-quality-lock-v1` for one panel. The lock pins the actual
result, producer and consumed ZIPs, evaluator, input registration, producer
manifest and original RESULT member. Its public APIs are:

```python
fetch_quality_original(absolute_directory, original_lock)
verify_quality_original(absolute_directory, original_lock)
```

The latter reopens all original source evidence, full arrays and producer Git
bytes in a fresh isolated stdlib process, replays complete source admission, and
recomputes the unchanged B14 decision. It returns only a reproduced original PASS.
The narrow process allowance is the immutable R1 STOP worker
`ac3e12810f72f68c3bdd4328dc5933b71c3e05da00524b508f91ce870b83d83d`,
plus exact recursion of this quality reader for a replica. The unchanged R1
worker requires the resolved `ORIGINAL_B24R1_STOP` directory and its exact
canonical frozen lock SHA
`56c10ef6a669650050fd1d821871489c030f13f224fc85d1c9dc537c69ef37f8`.
Recursive Q1 workers accept only an original PRIMARY lock and its complete
`api`, `originals` and `inputs` directory layout. Both subprocess guards allow
only `stdin`, `stdout` and `stderr` options; environment, working-directory,
executable, shell and process hooks cannot be overridden. The unchanged R1
worker manages its own unchanged nested original B24 proof. Network, model
imports and arbitrary subprocesses are denied during saved verification.

Only an original PRIMARY PASS can bind the separate replica opening. The complete
original proof is carried into replica source authorization and reopened before
generation, source admission and scoring. The new replica score binder is
`automation/english_v6_b24q1_replication_bind.py`, with the same three positional
arguments and the corresponding `...replication-binding-inputs-v1` bundle schema.
Only two authentic original quality PASS results can authorize later engineering.
No physical SSC305 test, natural field FAR or 3–5 m FRR is claimed by this route.

### Complete evidence reuse within one audit

PRIMARY preflight, replica preflight, each independent quality worker and each
pure scoring binder now own an explicit, process-local evidence scope. The
first encounter with a fixed proof runs the complete original isolated reader.
Repeated encounters within that same outer audit may reuse only its exact
canonical report bytes. Every encounter still enumerates the complete directory
closure, streams every original file into SHA-256 and checks the complete reader
hash. The first full callback is followed by another complete byte check. The
scope ends with that operation; another operation or isolated process must run
the full original reader again. File times, file lengths and earlier successful
operations cannot stand in for reading every byte.

The replica opening remains sealed with `primary_quality_report_sha256` and
`primary_quality_files` both null. After an actual complete PRIMARY PASS, the
pure helper `english_v6_b24q1_replication_source_ancestry.bind_opening(directory,
lock, fresh_output)` reopens that original proof and freezes its canonical report
SHA and all 18 file pins: three original API bodies, three original quality ZIPs
and twelve input ZIPs. It checks the same 18 whole files before and after the full
reader. The three quality ZIP pins must equal the immutable quality lock; each
input pin must equal the exact original producer SPEC, whose complete bytes and
SHA must also match the captured registration. The output is a new proposed
opening file; this helper never edits the live opening or executes a workflow.

Subsequent replica reads use those already frozen pins. They cannot derive fresh
expected hashes from altered evidence. Even a cache hit repeats the authentic
PRIMARY PASS, exact model, original TRAIN, assessment and full 120-original
inventory checks. A same-length mutation, missing or additional member, changed
lock, changed reader or changed canonical report fails closed. This bounds
recursive verification work without changing admission, numerical decisions,
source limits or scientific call budgets. The score jobs allow 180 minutes for
PRIMARY and 240 minutes for replica to retain complete evidence after scoring.

## Local contracts and limits of this preparation

The new contract tests use stdlib-only invented bytes or previously saved static
registration metadata. They check fixed numerical function bodies, exact model
identities, all120/N accounting, multi-provider empty union, quarantine exclusion,
full raw-array mutation rejection, original result/provenance locks, isolated
worker permissions, inert openings and the measured conservative capacity rules.
They never substitute for original source, PRIMARY, replica or engineering proof.

Run the capacity, primary evaluator, primary binder, quality readback, replica
evaluator and replica ancestry test files under `python3 -I -S -B`, `-O` and `-OO`.
At this preparation, all 92 contracts pass in each of the three modes. The real
pure helper integration also resolves the new source policy and the unchanged
source minima against the frozen metadata, with zero model calls.
The corresponding push workflows remain inert until their exact reviewed parent
and complete original evidence bindings exist. No test or this document opens
them.
