# One continuous candidate: 你好小窝，屋里有人

One Melo model call produced this candidate. The native mono float32 signal
contains65,536 samples at44.1kHz (1.486077s); its fixed mono PCM16 derivative
contains23,778 samples at16kHz (1.486125s). The native signal is unchanged,
with no concatenation, selected pause, added silence, trimming or gain change.

**The audible words 你好小窝，屋里有人 were later human-confirmed** for the
exact native WAV. `current-status.json` points to the appended, hash-bound
`human-adjudication.json`; historical pending records are preserved unchanged.
The reviewer had seen the intended text, so this is not independent or blinded.
Word boundaries, tail integrity and confidence remain unknown; punctuation is
not a time annotation. This remains quarantined research material with null CTC labels,
null terminal-slot selection and no training admission. No ASR, KWS or
training call, retake, alternate voice or seed sweep occurred in this run.

The two WAVs, original generation receipt/freeze and durable one-attempt claim
are retained byte-for-byte. `lineage.json` connects their signal/file hashes,
the fixed conversion and original executed-plan identity. The source is in
[kws-pipeline/research/single_k1_followed_wu](https://github.com/jiying2007/kws-pipeline/tree/research/single-k1-followed-wu-v1/research/single_k1_followed_wu); its public plan explicitly omits
one unused annotation and has a different hash, as documented by
`plan-projection.json`. Generation receipts retain the original executed hash.

`resources.json` preserves the failed installer cap, merged-stream receipt
parser and original mapping-cap stages, followed by their bounded corrections.
No dependency, import or model call is hidden by those corrections. The
remaining asset phase finished in39.150s under its separate180s limit; the
original300s setup window remains failed. Cumulative controlled HTTP bodies
were201,093,479 bytes; the budget additionally charges128KiB conservatively
for two small official frontend connector reads. The overall256MiB ceiling
was unchanged. No model weights or dependency wheels are redistributed.

Generation used one CPU session and one native call. The supervised child took
3.262s including model loading; call/conversion took1.212s. The sampled owned
process-group RSS maximum was356,720,640 bytes across13 observations at0.25s.
Samples are not a continuous peak-memory guarantee, and the timings are not
target-device or detector-performance evidence. The raw safe signature,
runtime and supervision receipts preserve the measured scopes.

Any later audio annotation must record exposure to the intended text, retain
uncertainty ranges, and seal the acoustic interval before KWS score inspection.
No boundary is inferred from clip duration, comma input or TTS intended text.
This archive establishes one generated candidate, not wake quality, a detector
policy, population accuracy, acoustic independence or product qualification.

## Fixed waveform check

`heuristic-preregistered.json` froze one numerical rule before measurement:
10-ms non-overlapping RMS windows below -60 dBFS for at least 20 ms, plus
exact-zero runs. `waveform-evidence.json` records only a leading [0,90) ms
low-RMS candidate in each WAV and no interior candidate meeting that rule.
This does not prove absence of an audible pause and supplies no word boundary.
The portable source `inspect_waveforms.py` reproduced every measured value
without a model call; it takes this archive directory and an output JSON path.
