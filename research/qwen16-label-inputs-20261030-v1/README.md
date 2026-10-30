# Source-bound blind label inputs for the completed Qwen16 batch

This consumes the **existing** generation run 41008894901, not a new generation
or a replay of kws-pipeline #665. Its original ZIP is pinned at 4,602,393 bytes,
SHA-256 `78f8b8736db1ab1dafc15a3bdddbbab64799ee0c975b6f3dab647604d633a8d1`,
archived in kws-data commit `1ec9c6b20387620026bdbbcac6c705f3912f004e`.

The immutable original archive and its independent verifier remain unchanged.
All 12 Vivian/Uncle_Fu/Dylan train **candidates** are copied byte-for-byte into
hash-ordered, anonymous `blind/audio/*.wav` names. The only JSON in `blind/` is
the existing `blind-source-screen32-job-v1` audio-only job. Their total duration
is 30.08 seconds. The four Serena development_a clips remain excluded: new seeds
do not make them train or independent sealed evaluation sources.

`control/source-map.json` retains exact source/PCM/frame identity, source family,
original roles and hashes of both original plan and result records. It belongs
on the trusted host, **never inside an ASR worker mount**. Original requests are
not copied into the blind job. Actual transcripts/tokens remain null, all label,
training and qualification flags remain false, and source-rights policy review
remains pending. Existing model license metadata alone is not new usage proof.

`control/capture-request.json` is a prospective three-family capture requirement:
12 inputs per Qwen3-ASR/SenseVoice/Whisper family, at most 36 recognizer calls,
zero retries, zero new TTS and zero training. It is NOT an execution release,
runtime/model/config freeze, ASR receipt or training root. No inference starts
from this code or workflow. A later independently reviewed native capture must
freeze its actual runtime/source/asset/config dependencies, retain the complete
raw observations and recheck the input against the original archive before any
recognizer call. Consensus is not evaluation gold or proof of error independence.

## Reproduce and verify

```sh
python3 -B research/qwen16-label-inputs-20261030-v1/test_prepare.py
python3 -B research/qwen16-label-inputs-20261030-v1/prepare.py --output /tmp/qwen16-label-inputs
python3 -B research/qwen16-label-inputs-20261030-v1/prepare.py --verify /tmp/qwen16-label-inputs
```

A different archive location can be supplied with `--archive`; its immutable
size/hash cannot be overridden. Tests also use `QWEN16_ARCHIVE` as a location
only. A destination must not exist. An interrupted write leaves no trustworthy
completion claim; consumers must run the complete source-bound verification,
not trust `handoff.json` alone. Stable local filesystem ownership is assumed;
this is not a filesystem transaction or protection from later concurrent edits.

Tests use the exact public archive and temporary Git fixtures. They cover blind
membership, source roles, source-row hashes, forbidden labels, malformed JSON,
mutated archives, audio geometry, no overwrite, output mutation/alias rejection,
and exact four-path source registration preserving old retained rows.

The workflow runs all tests in normal/-O/-OO, invokes the unchanged independent
archive verifier, builds/verifies the complete handoff, and uploads the bounded
reproducible output. The source-branch-only registration job appends these four
new source paths to the existing retention manifest, bound to the actual source
commit; no retained record, catalog, verifier or ruleset is weakened. Final PR
and main checks are separate from this initial source registration.

Remaining: native three-ASR capture and reviewed train-root admission, genuine
review for development labels, suitable negatives and isolated evaluation,
bounded KWS training and weights, deployment numerics, final AFE and SSC305.
This input handoff does not complete a production model.
