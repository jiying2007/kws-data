# Bounded research archive, version 1

This is an offline, lossless byte archive. It does not evaluate audio, load a
checkpoint, import NumPy, deserialize pickle, retrain a model, or contact a
network. Integrity is not authenticity: pin the reviewed Git commit and the
SHA-256 of `archive-index.json` from a trusted source. License and privacy
review remain separate responsibilities.

## Read or restore

Requires Python 3.9 or newer, standard library only. From the research directory:

```sh
python3 tools/archive.py verify . --expected-index-sha256 REVIEWED_INDEX_SHA256
python3 tools/archive.py materialize . /new/empty/output --expected-index-sha256 REVIEWED_INDEX_SHA256
python3 -m unittest discover -s tools -p 'test_archive.py' -v
```

The output path must be absent or empty. Materialization first verifies all
stored bytes and all derivations. It then writes the full logical tree. Existing
files are never overwritten; symlink ancestors/descendants and path traversal
are rejected. Drive/alternate-stream colons, Windows device names, control
characters and other nonportable path components are rejected on every host.
No payload is executed. A `.pt` checkpoint remains an opaque byte file, even
after materialization. Loading such a file with a pickle-based tool is outside
this archive tool's scope and may execute code.

## Storage and identities

- Each unique original byte sequence is keyed by its full lowercase SHA-256
- Logical aliases retain distinct filenames and provenance but share stored bytes
- A deterministic gzip stream uses level 6, empty original filename and mtime 0
- The gzip stream is split into ordered binary shards of at most 65,536 bytes
- Shards are addressed by their own SHA-256: `archive/shards/<sha256>.part`
- An object descriptor `archive/objects/<original-sha256>.json` contains the
  original and compressed byte counts/digests and ordered shard records
- Asset metadata and object-reference indexes are themselves content-addressed,
  hash-pinned JSON pages of at most 65,536 bytes
- `archive-index.json` pins all metadata pages and records measured counts
- JSON encoding is UTF-8, sorted keys, compact separators, terminal newline
- Gzip, descriptor, shard, and final original SHA-256 values are all verified
- The root index, descriptors, and all pages/shards are bounded to 65,536 bytes

Transport may base64-encode a binary shard for a Git blob API call. That does not
change the stored binary format. One 65,536-byte shard needs 87,384 base64
characters, leaving room under a 95 KB JSON request limit. Do not store base64
text while claiming it is a binary shard. Shard MIME is `application/octet-stream`;
the concatenated stream is `application/gzip`. Logical asset MIME and encoding
are recorded independently, including NumPy containers and PCM endian/layout.

The verifier rejects a declared uncompressed object over 128 MiB, total logical
bytes over 512 MiB, malformed or
oversized lengths, concatenated/trailing gzip members, unreferenced objects,
duplicate logical names, missing bytes, and mismatched digests. The fixed upper
bound is an operational safety limit, not a claim about model or audio validity.
Compression bytes can vary with zlib implementation versions. An archive's
recorded compressed/shard hashes are authoritative; reconstructed original
bytes remain stable. Rebuilding with the same implementation and inputs is
deterministic.

## Canonical audio and derived PCM

Store the exact canonical RIFF WAV. A derived logical asset has:

```json
{"logical_path":"benchmark/release/pcm/example.raw","source_logical_path":"audio/fixed12/example.wav","transformation":"wav-pcm-s16le","sha256":"EXPECTED_PCM_SHA256","size_bytes":32000}
```

The decoder accepts only uncompressed mono 16-bit PCM WAV at 16,000 Hz, reads the
actual frame data, and verifies both declared frame length and the expected raw
PCM SHA-256. The `.raw` bytes are little-endian signed 16-bit samples with no
header. This deterministic container extraction is not resampling, inference,
regeneration, augmentation, or a new audio evaluation. Both original WAV and PCM
identities remain independently checkable. Derived-from-derived assets are not
supported. Raw PCM is reconstructed only during verification/materialization;
it is not redundantly stored in the archive.

## Local-only packaging contract

```sh
python3 tools/archive.py pack /local/approved-plan.json /public/research/root
```

The plan is a JSON array, or an object containing `assets`. Each stored entry
requires `logical_path`, local-only `source_path`, exact `sha256`, and
`size_bytes`; it should also supply `mime_type`, `encoding`, `category`, detailed
`provenance`, license/privacy rationale, and any disclosed transformation with
original/public hashes. Extra public metadata is retained. The top-level
`source_path` is never copied to public output. Arbitrary metadata and the asset
contents must already have been reviewed and sanitized. The packer does not
sanitize private text, infer licensing, or grant permission to publish.

For `wav-pcm-s16le` entries, use `source_logical_path` and omit `source_path`.
The packer verifies all source bytes and PCM derivations. Other transformations
must be performed separately into explicitly approved files with a new public
SHA-256; do not relabel modified bytes as an original raw record.

The output root may already contain reviewed README/license/tools files. The
packer writes only `archive/` and `archive-index.json`. It can reuse identical
existing artifacts, but refuses any nonidentical overwrite. A failed build can
leave unreferenced local output; use a new clean staging directory for a revised
plan and independently check the final publication file allowlist. The packer
never deletes files or uploads anything.
