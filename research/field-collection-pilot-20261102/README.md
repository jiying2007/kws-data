# Real-human KWS field collection pilot (private local data only)

## Why

The Qwen12 v2/v3 source evidence has 36/36 native ASR completion receipts,
but only 2/12 samples have strict target three-engine agreement. The corpus
does not yet meet independent KWS training source-admission requirements.
This offline stdlib tool creates a **recording worksheet**, then verifies the
exact audio bytes collected locally; it never records, synthesizes, transcribes,
trains, uploads, authenticates a human or approves a release.

**Do not commit or upload any participant manifest, raw human audio, personal
identifier, reviewer record, consent file or audio audit result to this public
repository.** All generated outputs are meant for local restricted storage.

## Private participants file

Create an access-controlled JSON file such as:

~~~json
{
  "schema": "xiaowo-field-participants-v1",
  "participants": [
    {
      "speaker_id": "anonymous-p01",
      "session_id": "pilot-p01-session01",
      "source_family_id": "field-p01",
      "role": "train-candidate",
      "room_id": "room-a",
      "device_id": "mic-a",
      "afe_profile_sha256": null
    },
    {
      "speaker_id": "anonymous-p02",
      "session_id": "pilot-p02-session01",
      "source_family_id": "field-p02",
      "role": "development",
      "room_id": "room-b",
      "device_id": "mic-a",
      "afe_profile_sha256": null
    },
    {
      "speaker_id": "anonymous-p03",
      "session_id": "pilot-p03-session01",
      "source_family_id": "field-p03",
      "role": "sealed-evaluation",
      "room_id": "room-c",
      "device_id": "mic-a",
      "afe_profile_sha256": null
    }
  ]
}
~~~

Use opaque IDs, not real names or phone numbers. Real identity and consent
must be checked independently. Different ID strings are NOT proof of physical
speaker, microphone, room, or source-family independence.

Generate a new local plan:

~~~sh
python3 -B research/field-collection-pilot-20261102/plan.py plan \
  --participants /restricted/participants.json \
  --output /restricted/field-plan.json
~~~

The planning command prints `plan_sha256`, the canonical complete-plan digest.
Freeze that digest independently in the restricted review record at planning time,
alongside the original participant manifest. Do not recompute the expected digest
from an edited plan during verification: a digest supplied by the same mutable
input is not an independent commitment. This binding does not authenticate human
identity, consent or the truth of participant declarations.

Each participant receives 37 deterministic, hash-shuffled tasks:
24 positive-intent tasks (both wake phrases, 1/3/5 m, front/back, two repeats);
12 confusable tasks (including 屋/沃/吴/王 phonemes, 3/5 m);
one natural-conversation/background task. The planned/declared intent is not
observed actual speech. Blind genuine listeners must later establish spoken
text, completeness, omissions, repetitions and confusing pronunciations.

## Attach recorded WAV bytes

Collect original recordings under restricted local controls. Retain native
capture/AFE metadata and any explicit conversion provenance. For this audit,
each normalized recording must be canonical RIFF WAV, mono PCM16, 16 kHz,
44-byte standard header, without trailing bytes.

For actual recorded slots only, create a local JSON roster:

~~~json
{
  "schema": "xiaowo-field-recordings-v1",
  "recordings": [
    {
      "slot_id": "s-REPLACE_WITH_AN_ACTUAL_24_HEX_SLOT_ID",
      "path": "session01/sample001.wav",
      "wav_sha256": "REPLACE_WITH_64_LOWER_HEX_SHA256",
      "pcm_sha256": "REPLACE_WITH_64_LOWER_HEX_SHA256",
      "frames": 16000,
      "human_verbatim_reviewed": false,
      "rights_approved": false,
      "training_admitted": false
    }
  ]
}
~~~

Placeholders are NOT evidence. The tool hashes WAV and PCM separately in
bounded chunks and checks exact file/frame lengths, paths, no symlinks/hardlinks,
duplicate audio payloads and source-slot IDs.

~~~sh
python3 -B research/field-collection-pilot-20261102/plan.py verify \
  --plan /restricted/field-plan.json \
  --participants /restricted/participants.json \
  --expected-plan-sha256 REPLACE_WITH_INDEPENDENTLY_FROZEN_PLAN_SHA256 \
  --recordings /restricted/recordings.json \
  --audio-root /restricted/field-audio \
  --output /restricted/new-structure-audit.json
~~~

Verification requires both the original participant manifest and the independently
frozen digest. It rebuilds the full plan and rejects any drift in declared role,
intent, AFE identity, room or slot inventory before opening WAV data. Updating
both plan and participants cannot bypass an unchanged frozen digest. Legacy
invocations missing either binding are rejected rather than silently accepted.

This output is create-only and aggregates counts by planned role and intent
without copying raw audio, transcript or personal IDs. Partial recordings
remain partial; all completion and licensing claims remain unauthenticated.
Every output always has `training_admitted=false`, `qualification_allowed=false`,
`human_identity_authenticated=false`, `shipping_approved=false` and
`physical_ssc305_qualification=false`. The `shipping_approved=false` field is
retained solely as an explicit negative assertion in the frozen cross-repository
v1 format; it is **not** an active product approval, and this pilot cannot
create one. Research effects and target measurements are separately reviewed.

## Independent research measurements (outside this pilot)

Independent participant consent and rights; genuinely blind audio listening
with exact labels (not the intended prompt or ASR alone); verified provenance,
source family/speaker/session split isolation and a reviewed train-root admission;
diverse real positives and long continuous negative data; bounded CTC training
and immutable weights; FAR/FRR plus independent real-human/final AFE;
actual SSC305 CPU/latency/noisy 3–5 m performance and soak.

With zero false alarms and a Poisson 95%-upper FAR goal <=0.40/hour, a simple
reference exposure is at least ~7.5 hours of independent negative audio.
That is not a universal research acceptance threshold: dependence, nonzero FAs and
actual final-AFE operating conditions must be accounted for separately.

## Offline source-only tests

~~~sh
python3 -B research/field-collection-pilot-20261102/test_plan.py -v
python3 -B -O research/field-collection-pilot-20261102/test_plan.py -v
python3 -B -OO research/field-collection-pilot-20261102/test_plan.py -v
~~~

All test audio is artificially generated in temporary directories. There are
zero mic/network/model calls and no reviewer/rights or training approval.
