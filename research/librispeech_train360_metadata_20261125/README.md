# LibriSpeech TRAIN-clean-360 metadata acquisition preparation

This is an independent public metadata acquisition task. It does not register a
candidate or assessment, choose people or utterances, change any existing source
policy, or authorize TTS, ASR, PCM creation, model execution, or scientific panel
opening. The actual workflow is disabled. A later review must explicitly bind
the prepared public commit before the one-shot template can run.

## Verified public input

OpenSLR's [SLR12 release](https://www.openslr.org/12/) identifies
`train-clean-360.tar.gz` as upstream training data and labels the corpus CC BY
4.0. Its [official checksum list](https://www.openslr.org/resources/12/md5sum.txt)
publishes MD5 `c0e676e450a7ff2f54aeade5171606fa`. The original origin's HEAD
response on 2026-11-25 reports exactly **23,049,477,885 bytes**. The original
response metadata is retained in `OFFICIAL_HEAD.json`; its body was not read.

The SHA-256 is
`146a56496217e96c14334a160df97fffedd6e0a04e66b9c5af0d40be3c792ecf`.
This is independently present in the official PyTorch Audio reader at
[commit 654fee8fd17784271be1637eac1293fd834b4e9a](https://github.com/pytorch/audio/blob/654fee8fd17784271be1637eac1293fd834b4e9a/src/torchaudio/datasets/librispeech.py),
`src/torchaudio/datasets/librispeech.py`, release tag `v2.7.0`. That source was
read as text and never imported. Its exact byte identity, the official mirror
URLs, citation references, and source lookup scope are in `OFFICIAL_SOURCES.json`.

The official release listings inspected do not offer an independent complete
TRAIN-clean-360 utterance transcript-only file. `raw-metadata.tar.gz` is described
only as extra creation metadata; no claim that it contains the complete exact
utterance transcript set was verified. SLR11 language-model book text is not a
substitute for split-specific utterance transcripts. This is a limited finding
about the inspected publisher listings, not a claim of global nonexistence.

The publisher's [paper, Table 1](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf)
and the already authenticated complete `SPEAKERS.TXT` establish **921 census
readers** in this TRAIN split. This is a census, not an eligible pool. No count
of readers meeting the existing 4–14-word / two-originals requirement is claimed
or computed here. The existing requirements and all historical exclusions remain
unchanged. No development, test, or CAL speech is acquired.

## Stream and integrity boundary

The new scanner retains the B21 approach of exact original UTF-8 text plus each
text's SHA-256, canonical speaker/chapter paths, and a complete census. B21's
full-on-disk 6.3 GB transport cannot be reused for this 23 GB input on a standard
small runner. The new standalone implementation uses only the Python standard
library and never imports B21 source/model implementations.

There is one fixed full HTTPS GET to
`https://openslr.trmal.net/resources/12/train-clean-360.tar.gz`. The request uses
identity encoding. Redirects, retry, resume, ranges, alternate mirrors, and
fallback selection are prohibited. The HTTP status, final URL, and exact content
length are checked before reading. SHA-256 and official MD5 include every byte
of the complete compressed response.

The gzip stream is traversed once. A small tar parser uses standard-library
`TarInfo.frombuf` to validate every 512-byte header. It permits only ordinary
files and directories with canonical short paths from this split, plus the five
original root text metadata files. Links, sparse files, PAX extensions, GNU
long-name records, duplicate paths, unknown file types, and other split payloads
stop acquisition. These rules are prospective; compatibility with the full
original archive has not yet been exercised.

Each FLAC member is streamed through SHA-256 and discarded. The complete member
index records its original path, original byte length, original SHA-256, tar
header hash and header offset. No FLAC is written to disk or interpreted as an
audio format. **This reads audio container payload bytes; audio decoding, audio
header interpretation, PCM creation, and model calls are zero.**

Every original `.trans.txt` is retained without word filtering. Text is decoded
strictly as UTF-8 and round-trips to its original bytes. The complete official
speaker census must equal the existing SHA-pinned census. Original utterance
IDs must be globally unique and form a bijection with the original FLAC member
IDs; all 921 readers must occur on both sides. Each member's original bytes and
SHA are bound into the index. Both tar end blocks, all subsequent zero padding,
the gzip CRC/trailer and compressed EOF are consumed and checked.

No valid catalog is published until **all** checks have passed, including the
complete 23,049,477,885-byte count and both compressed hashes. During work,
outputs are `INCOMPLETE`; the catalog, member index and their validation report
are staged together and atomically renamed into `receipts/COMPLETE` only after
their size checks. A partial catalog is never labeled complete.

## Prospective operational bounds

| Resource | Frozen limit |
|---|---:|
| Complete archive GET | 1 |
| Retry / resume / redirect / range / fallback | 0 |
| Full transfer, decompression, parsing and hashing | 300 minutes |
| Socket timeout | 60 seconds |
| Expanded tar bytes, including every header/payload/padding byte | 40 GiB |
| All tar member headers, including directories | 150,000 |
| Raw metadata text total / one text member | 64 MiB / 8 MiB |
| One original FLAC member / one path / one transcript line | 128 MiB / 256 bytes / 16 KiB |
| Serialized complete catalog / full member index | 96 MiB / 64 MiB |
| All receipt files / producer files | 176 MiB / 8 MiB |
| Combined artifact roles, including explicit ZIP reserve | 192 MiB |
| Process address space | 1.5 GiB |
| Minimum free disk before GET | 256 MiB |
| Collection step / complete job | 320 / 330 minutes |

The stream does not materialize the 23 GB archive or any audio file. Only the
bounded text and member index are accumulated and saved. The memory cap covers
parser state, text and the complete member inventory. The resource caps are
reviewable limits, not claims that official pages provide exact expanded sizes
or that a particular throughput has been measured. Exceeding a cap leaves this
metadata acquisition incomplete; the code does not increase caps during a run.

The 300-minute bound needs about 1.22 MiB/s of average compressed throughput,
including processing time. Actual GitHub runner throughput has not been tested.
`PROGRESS.jsonl` and final receipts preserve elapsed time, compressed and
expanded bytes, and average throughput. A timeout or transport failure is a
metadata acquisition failure, never a new algorithm result. It neither consumes
nor resets any scientific panel budget, and does not change the failed B27
SOURCE run's existing zero-retry rule.

## Captured producer and review interfaces

`paths()` is the explicit producer closure: the new implementation, tests,
registration, template, workflow, contract workflow, this README, public input
receipts and the existing original census. `prepare` checks every producer file
against its exact Git blob, writes its full byte inventory, and copies all files
into one self-contained producer. Collection then executes **from that captured
producer**, and rechecks its exact file set and hashes before the GET. Thus the
fixed transport registration JSON is part of the captured runtime closure.

Purely inert checks, with no network, corpus, source or model work:

```sh
python3 -SIB automation/test_english_v6_train360_metadata.py -v
python3 -SIBO automation/test_english_v6_train360_metadata.py -v
python3 -SIBOO automation/test_english_v6_train360_metadata.py -v
python3 -SIB automation/english_v6_train360_metadata.py check
```

Future workflow-only activation uses the exact template and replaces only
`__TRAIN360_METADATA_PREPARED_PARENT__` with the reviewed public preparation
commit. The first public push must be its single-parent child and change only
the once workflow. Both API run count and a separate metadata transport marker
enforce one acquisition attempt, including failed runs. The code does not
consult or spend any candidate/source/quality budget or inherit a scientific
opener. No activation has been prepared or performed by this patch.

Future acquisition interface, only inside that registered workflow:

```sh
python -SIB automation/english_v6_train360_metadata.py prepare "$RUNNER_TEMP/train360-metadata" "$PREPARED_PARENT"
# The producer and separate metadata marker must be uploaded before collect.
python -SIB "$RUNNER_TEMP/train360-metadata/producer/repository/automation/english_v6_train360_metadata.py" collect "$RUNNER_TEMP/train360-metadata" "$PREPARED_PARENT"
python -SIB "$RUNNER_TEMP/train360-metadata/producer/repository/automation/english_v6_train360_metadata.py" verify-catalog "$RUNNER_TEMP/train360-metadata/receipts/COMPLETE"
```

The final verifier reopens all retained original text and member identity rows,
checks their bytes/hashes and the complete bijection, and rejects an incomplete
directory. It does not independently download the original archive again.
Authenticating the three original GitHub artifact ZIPs and API provenance after
an actual run remains a separate original-artifact binding step.

## Remaining work and limits

This preparation has not downloaded the archive or selected any new people.
Before any future assessment, a successful full metadata acquisition must be
independently bound to its original public run, original artifact ZIPs and
captured producer. Only then can a separately registered metadata assessment
consider historical exclusions and unchanged eligibility rules.

The new metadata index gives exact whole-FLAC identities. It does not authorize
later source execution, decide which people to use, or retain the FLAC originals
needed for such execution. Any future streaming source reader and original-audio
retention scheme needs its own review for complete archive authentication,
whole-original-FLAC byte retention, proof closure and budgets. Nothing here
amends old source policies, old openers, B14, model comparisons or numeric gates.
