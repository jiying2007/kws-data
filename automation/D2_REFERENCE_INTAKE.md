# Bounded acquisition of 24 additional D2 reference speakers

`fetch_aishell_subset.py` retrieves a bounded prefix of the official AISHELL-3
archive and keeps only complete, safe, readable WAV members. It neither
downloads models nor generates speech, certifies transcripts, or creates
training labels. Existing source terms and the private research import contract
remain required at the later import boundary.

## Acquisition contract

The declared official source is
`https://openslr.trmal.net/resources/93/data_aishell3.tgz`, observed ETag
`"46fe4e811-5b46158f79040"`, total length 19,057,141,777 bytes. Every range must
return status 206, that exact ETag, the exact requested Content-Range, and the
declared number of bytes. A range gets at most two transport attempts; a
failed download is retained, not called a complete archive. A seed prefix is
read-only and must match an explicit SHA-256. Resuming writes to a new file.

Extraction uses streaming tar reads and selects only complete regular members
matching the declared dataset paths. It rejects symlinks and special members,
limits each member and total expanded bytes, and verifies mono PCM WAV geometry.
An incomplete final member cannot become a reference clip. A truncated gzip/tar
prefix is **not a verified full archive**, even when complete members inside it
are usable. The source subset manifest records this distinction explicitly.

The bounded prefix is not a random or representative sample of AISHELL-3 or of
real users. Its 24 selected identities extend engineering source coverage only;
the resulting synthetic speech is not independent human evaluation evidence.

## Actual 2026-11-05 intake

- Completed bounded prefix: 50,331,648 bytes, SHA-256
  `d265304436bc8a93b5e76cd0d078895d94950639cf2a70cee4bc7bd1b0570b98`.
- 587 complete candidate WAV members were observed before the prefix ended.
- 24 new speaker references were selected, excluding all eight existing D1
  source identities. The persistent registry now contains 32 identities;
  the original six training/two development assignments remain unchanged.
- All 24 selected references were converted by the existing bounded ffmpeg
  preparation path, with original file, normalized file, PCM and conversion
  receipts retained. The unchanged official transcript file SHA-256 is
  `26187c611df08e7274afb1b0019c0688f1b15a80c17b9e85bf83d8dc3b1f937e`.
- The first import is `dataset-provided`, not content-checked. The reference
  ASR workflow must retain its native results and apply the existing full-text
  agreement rule before a separate content-observed manifest can enable ICL.
  Disagreement keeps x-vector-only reference mode.

Raw reference audio, source identities and per-reference text stay in the
private execution bundle. Public source contains the executable contract and
aggregate observations only. Earlier partial downloads and their transport
failures are retained separately from the successful bounded prefix.

## Pipeline

1. Use `fetch_aishell_subset.py fetch` with an explicit byte bound, expected
   source identity and optional verified seed prefix. Read `--help` for the
   exact argument names; create-only destinations prevent silent overwrite.
2. Use its `extract` command with the original speaker registry and a cap of
   24 new speakers. Review the subset's truncation and completeness flags.
3. Create a selection and extend `source_registry` using canonical dataset and
   source-speaker identity. Preserve all eight historical registry entries.
4. Run `prepare_open_reference_pilot.py --max-speakers 24` and import with
   `research_only_bulk_ingest.py --max-refs 24`, preserving the explicit
   research scope, terms, original transcript and conversion receipts.
5. Run `reference_asr_inputs.py`, the existing bounded audio-only ASR worker,
   and `reference_asr_checks.py`. Import observations as a new manifest.
6. Freeze the `new-twenty-four` D2 plan only after the reference observation
   manifest is final. Changing reference status changes the plan identity.

The additional D2 plan contains 1,536 requests, and the existing-eight increment
contains 256 requests. Together with the original 256 D1 requests, this accounts
for 2,048 planned cells. Request count, native audio returns, technical passes,
machine weak-label rows and human gold are reported separately.

## Offline verification

`automation/test_fetch_aishell_subset.py` uses invented tiny archives and mocked
HTTP replies. It checks range/source identity, bounded extraction, incomplete
members, unsafe members, duplicate identity selection, and create-only outputs.
CI runs it normally and with `-O`/`-OO`; it makes no network or model calls.
