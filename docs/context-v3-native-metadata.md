# Context-v3 native WAV metadata reconciliation

`automation/active_context_v3_metadata_weak_view.py` is an additive reader for
retained native output. The original `active_context_v3_weak_view.py` and shared
`active_asr_weak_view.py` remain byte-identical. Original packages continue to be
replayed with their original reader; this new reader has separate package and
adapter schemas and an explicit source roster.

The observed failure occurred after eight completed TTS calls in context chunk5,
before any chunk5 ASR call. Seven original native-return and bulk-generation WAVs
were byte-identical. For one row, both files had the same 69,120 original float32
samples and differed only at byte60, inside the RIFF PEAK timestamp: the second
serialization advanced from epoch1793928970 to1793928971. Each original file's
own receipt hash remained correct. The original all-byte equality guard therefore
stopped the pure source-map handoff. Neither original file needs to be rewritten
or regenerated to diagnose or reconcile this serialization difference.

## Strict scope

For every completed source row, the reader first verifies the native receipt's
returned-WAV and NPY hashes, the bulk completion receipt's separate generated-WAV
hash, and exact decoding of both WAVs to the native array's float32 PCM. The
archive remains bound byte for byte to the returned native WAV. The unchanged
completion verifier authenticates the bulk-generated WAV against the hash that
was actually recorded for that container.

When the WAVs differ, the only new acceptance rule requires all of the following:

1. Both complete RIFF containers have the same length, chunk order and offsets.
2. Their chunk sequence is exactly `fmt `, `fact`, `PEAK`, `data`, with the fixed
   mono IEEE float32 24-kHz format and the declared frame count.
3. The original native array is little-endian float32 and its samples exactly
   match both data chunks.
4. Both PEAK chunks have version1 and identical peak value and position.
5. The bulk PEAK timestamp is exactly one second after the native-return PEAK
   timestamp. Every byte before and after the four timestamp bytes is identical.

Backward, two-second or arbitrary timestamp changes fail. Other ancillary
chunks, reordered chunks, changed metadata, changed PCM, changed native arrays
and mismatched receipt hashes fail. The rule is not PCM-only acceptance.
Identical containers retain the previous exact-byte acceptance path.

## Interface and retained evidence

| Interface | Value |
| --- | --- |
| Entry module | `automation/active_context_v3_metadata_weak_view.py` |
| Package schema | `kws-active-context-v3-metadata-research-weak-view-v1` |
| Source adapter | `kws-active-context-v3-metadata-native-source-v1` |
| Container policy | `kws-native-wav-peak-timestamp-plus-one-second-v1` |
| Producer roster | New entry followed by the unchanged original14 files, in their original order |
| Public call signatures | Same `build_source_map`, `create_view`, `verify_view`, `load_training_rows` and `load_development_rows` arguments as the original v3 reader |

Each `report.items[*].source.native_container_relation` records both original WAV
hashes and byte lengths, original NPY identity, native PCM SHA and frame count,
the exact timestamp values and differing byte offsets. Failed-native rows keep
this field null and retain their original failure and zero objective weights.
The adapter records its container policy and the exact reconciled cell IDs.
Legacy exported rows retain the relation and `source_adapter_schema` fields.

The same original native/generation files are retained in new packages, so a
complete verification works after the live source directories have been removed.
No old report, receipt, audio file or source hash is overwritten. A mixed package
consumer must dispatch each package to its own exact, externally pinned reader
revision and must retain the actual adapter schema. Replacing one package's
schema or producer hashes with another reader's values is not permitted.

The whole-transcript ASR policy, source family/speaker/split identity, generation
outcome, signal-quality checks, ordinary-speech negative handling, unknown/native
failure exclusions, and full observed targets are unchanged. Reconciliation
makes no new model call and grants no human-gold or training qualification.

## Tests

The stdlib-only focused suite covers separate authoritative hashes, exact native
array/PCM equality, timestamp carry across multiple bytes, one-second direction
and extent, non-timestamp metadata, additional chunks, PCM and array tampering,
unchanged original-reader rejection, versioned producer/schema dispatch, retained
offline native/ASR replay, ordinary full-text labels, failed-native zero weights,
and rehashed report tampering. CI runs normal, `-O` and `-OO` modes. All audio and
ASR fixtures are explicitly fictional; no model is imported or invoked.
