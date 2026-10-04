# Rights, sources and privacy boundary

These are source-identifiable PRE-CMVN acoustic features and numeric control
outputs derived solely from 32 already-public synthetic source WAVs. The basis
for this bounded research publication is that approved public synthetic-source
and scientific-evidence scope, not the dimension or size of the features.
No anonymity, non-invertibility or inability to reconstruct source information
is guaranteed. Public source IDs, preset labels, waveform/PCM hashes, source
URLs and numeric values are deliberately linkable to their public sources.

The data contain D20 historical stock-preset synthetic sources and 12 exposed
reviewed Qwen synthetic sources. They do not add natural recordings, user voice
references or new private voice inputs. Stock-preset names identify generator
presets, not human participants or an endorsement by real people. D20 weak
pseudo-labels are not human-gold; no event/phoneme timing gold is supplied.

Preserve the unchanged [rights-and-provenance.json](rights-and-provenance.json)
and [D20 source notice](D20-SOURCE-NOTICE.md). The upstream provenance includes
Apache-2.0 source/model notices and the BSD-2-Clause frontend notice. Do not
blanket relicense third-party assets. `commercial_output_license` remains
`not-established`; this archive creates no new license grant, commercial
clearance or real-person endorsement. Existing source metadata is pinned to:

- [Public Qwen research rights](https://github.com/jiying2007/kws-data/blob/7af8f8597b0b7fbe8761c9a2400f2e4028395551/docs/RIGHTS.md)
- [Structured generator and output-rights catalog](https://github.com/jiying2007/kws-data/blob/7af8f8597b0b7fbe8761c9a2400f2e4028395551/catalog.json)
- [A20 core license scope](https://github.com/jiying2007/kws-data/blob/7af8f8597b0b7fbe8761c9a2400f2e4028395551/research/2026-10-03-native-a20-core/LICENSES/SCOPE.md)
- [Apache-2.0 notice](https://github.com/jiying2007/kws-data/blob/7af8f8597b0b7fbe8761c9a2400f2e4028395551/research/2026-10-03-native-a20-core/LICENSES/Apache-2.0.txt)
- [BSD-2-Clause frontend notice](https://github.com/jiying2007/kws-data/blob/7af8f8597b0b7fbe8761c9a2400f2e4028395551/research/2026-10-03-native-a20-core/LICENSES/BSD-2-Clause-torchaudio.txt)

The original gzip payloads are unchanged. The successful payload contains 53
finite FP32 buffers, 14 technical JSON files and an artifact manifest. The
failure payload contains 7 JSON files. All nested JSON fields and tar headers
were inspected before archive preparation. Metadata describes public source
identities, fixed software/package versions and hashes, generic GitHub-hosted
runner image/version, numeric resource samples and execution boundaries.
No credentials, private local paths, private storage IDs/URLs, chat, approval
transcripts, private reports, individual host identities or personal metadata
are added. Tar headers have zero uid/gid/mtime, blank user/group names and no
PAX metadata. The public record does not include copies of source WAVs,
model/checkpoint weights, wheels, virtual environments or software binaries.

The verifier and tests are small archive tooling, using standard-library file
and path checks following this repository's existing package-restoration
conventions. Verification reads saved numeric bytes only. No acoustic inference,
frontend recomputation, training, new dependency installation or model download
is performed by the new workflow.
