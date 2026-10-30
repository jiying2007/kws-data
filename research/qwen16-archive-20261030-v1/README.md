# Exact original Qwen16 archive — 2026-10-30

Archive only the completed `jiying2007/kws-data` run **41008894901**, attempt 1,
artifact **12956997278**. Source is `1398f5e3c6a608ab865cd20fcad9d1b9e622c30a`.
Original ZIP: 4,602,393 bytes; SHA-256
`78f8b8736db1ab1dafc15a3bdddbbab64799ee0c975b6f3dab647604d633a8d1`.

The existing generation run consumed all 16 attempts. Do not recreate its
execution branch, rerun generation, or launch the overlapping kws-pipeline #665.
This archival workflow performs **zero TTS/ASR/model-training calls** and uses
only a standard public runner. It does not invoke Codex, Work, paid services,
private candidate models or private audio.

## Verified evidence and review

`evidence/run-41008894901/original.zip` retains the exact downloaded ZIP.
`artifact/` preserves all 59 members byte-for-byte: 16 native float WAVs,
16 PCM16 derivatives, 16 claims, and 11 plan/receipt/metadata files.
The separate `verification.json` is observer evidence, never a replacement for
original receipts. Verification checks exact archive and member identities,
RIFF/format/sample extents, raw float and PCM hashes, sequential attempt claims,
recorded EOS, remeasured native metrics, and the original per-voice split.

All 16 PCM identities are distinct; total duration is **43.2 seconds**. This is
positive-intent synthesized material, not independent continuous negative hours.
Twelve rows are train candidates; **Serena's four remain development_a**.
All actual labels remain null and all training/qualification flags remain false.
The 7.84-second sample is retained unchanged; duration alone is not a transcript
or rejection rule. Exact syllables, repetitions, completeness, and naturalness
still require actual review. No hidden trim/gain change or retry selection occurs.

Open `evidence/run-41008894901/review.html` locally for the ordered blind review
interface. It does not display the request text. `review-map.json` maps review
IDs to source WAV identities and contains only pending placeholders, NOT human
review-v2 records. Complete genuine human review or reviewed native three-ASR
capture/admission is needed before any CTC training use; model/voice rights and
source provenance still need their separate admission checks. A future model
also needs suitable negatives, independent evaluation, numerical/deployment
qualification, final AFE and physical SSC305 evidence. No production approval.

## Execution and registration

`QWEN16_ARCHIVE=/path/original.zip REQUIRE_ARCHIVE=1 python3 -B test_archive.py -v`
executes 26 tests. Normal, -O and -OO were run locally on the actual ZIP: all pass.
The first source-only workflow cannot yet find the archive and explicitly skips
the 17 archive-dependent methods; nine format methods still execute. After the
fixed artifact is fetched, all 26 execute in each mode. PR acceptance requires
the committed original ZIP; missing evidence fails rather than silently skips.

The initial independent RMS check used Python's `sum`, whose reduction differed
from the retained generator's sequential accumulation. The verifier now mirrors
the actual recorded accumulation order; equality remains exact, no tolerance or
acceptance threshold was loosened.

A first creation of the one fixed archival branch runs offline tests, reads the
exact successful run/artifact metadata, downloads one archive, validates all
bytes, commits the original evidence, and registers only these new source/archive
paths using their already-created source commit. Every earlier retained record
must match the current Git tree exactly. Main is never directly written; merge
requires the existing protected checks and fresh-main acceptance. The original
catalog, source execution files, production flags and scientific pins stay intact.
