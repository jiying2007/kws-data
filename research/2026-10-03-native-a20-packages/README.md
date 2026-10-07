# Native A20 public-package transport

This optional outer transport restores the exact six previously verified public
ZIPs and their original archive. It does not change any scientific payload,
license, historical failure, inner manifest, or model qualification.

## Pinned content

- 111 content-addressed binary chunks, each at most 1,048,576 bytes
- Six ZIPs, 113,247,446 bytes including ZIP headers
- Exactly 2,243 original public files, 112,489,844 payload bytes
- Original file manifest SHA-256:
  `d0c265888b8e4799101a226eb568f4592f5e63d824493fae4214b0d577b5ed01`
- Outer transport SHA-256:
  `8819c75bacc04182769955700623e3a370ab96984f11282181ab6546d5baa3a7`
- Original archive index SHA-256:
  `3435110711dc158d20ccd3eb1393227090d4a5939e791292e69030a90d78f7e7`

`public-files.json` and `parts.json` preserve the original manifest bytes. The
outer chunks reconstruct the same six ZIPs, without recompression. The original
archive still uses 65,536-byte internal shards. Its verifier is unmodified.

## Offline use

Python 3.8+ and its standard library are sufficient. No network, dependency
installation, Git, authentication, model execution, or scientific rerun is used.
From this directory:

```sh
python3 tools/restore.py verify .
python3 tools/restore.py restore . --output /path/to/new-a20-output
python3 -m unittest discover -s tools -p test_restore.py -v
```

The output path must not exist, and its parent must already exist. The tool
checks exact chunk inventory, each chunk identity/order, each complete ZIP,
all ZIP member headers/CRCs and every allowlisted file hash. It then invokes
only the restored, hash-verified original archive materializer, confirming 341
logical assets, 301 stored objects and 12 exact PCM derivations. The restored
packages, `public/kws-data` tree, logical assets and report are published into
the new local output directory only after verification. Failed verification
does not overwrite an existing output. Use a trusted local parent directory;
do not concurrently modify the bundle or output directory during restoration.

`verify` performs the same restoration and logical checks in a temporary
directory, then removes that temporary output. Allow about 0.5 GB of free
working storage. The original numerical gates and historical FAIL records are
retained; this transport's PASS means integrity, not model or board acceptance.

## Scope and rights

The ZIPs contain only the previously reviewed public file set. Their preserved
`research/2026-10-03-native-a20/LICENSES/` notices and provenance govern source,
model and data use. The 50 synthetic WAVs retain their distinct historical D20
training, Serena diagnostic and fixed12 regression roles. Twelve PCM assets
are exact recorded derivations. Commercial output clearance remains
not established; this wrapper creates no new broad license or product claim.

The older core checkpoint and any already uploaded Git objects are retained.
The standalone six-ZIP manual importer remains byte-valid. Its original
expected-head check must stop if the target branch has since changed; never
force-push, silently change its pin, or run simultaneous API and manual writers.
See PUBLICATION_PROTOCOL.md for publication and final verification boundaries.
