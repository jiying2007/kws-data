# A20 N0: complete deterministic nonspeech negative-control record

Research-only saved numerical evidence from three fixed 300-second streams. All
three streams completed once; each has an explicit empty event list. This archive
adds a separate record; it does not edit or requalify any prior archive or failure.

## Observed result

- 14,400,000 PCM16 samples / 900 seconds of constructed input; 3,000 complete
  4,800-sample feeds and callbacks, three no-output finish calls
- 89,994 fbank rows, 29,997 model rows, and 29,997 actually searched decoder rows
- 6,012 complete raw records and all 179,982 six-class logit values
- Zero activation events in each domain and in both keywords
- State reset only at complete stream boundaries; no within-stream reset
- Probabilities and native beam paths were not captured; no softmax or beam replay

Domains are low-amplitude triangular pseudorandom noise, 32-tap moving-average
colored noise, and triangular noise with 13 sparse biphasic transients. Seeds,
amplitudes and recipes were fixed before observing model output. All input
identity/time-domain QA fields and 900 one-second block hashes are retained.
No speech, music, TTS, source-clip mixing, looping or outcome-based selection.

## Resource observations and limits

This was offline as-fast-as-possible processing. 900 seconds is input/state
duration, not a 15-minute elapsed soak, capture, I/O, microphone, AFE or board test.
Native main wall was 10.672298922 s and process CPU 10.669888429 s. Supervisor
10.84850817300321 s is the run phase after attempt reservation and before final
record encoding; it excludes admission and pre-reservation hashing. Observer CPU
delta was 0.301312151 s; its reported scope includes admission/final hashing but
excludes Python startup/imports and final record encoding/persistence.

Native getrusage high-water was 39,376 KiB, while sampled /proc peak was
2,912,256 B. These have different scopes; no cause for the difference is inferred,
and neither is model-only RAM. Observer lifetime high-water was 53,669,888 B and
is outside collector limits. Hard collector AS/CPU limits and the sampled RSS
guard are distinct: RSS is not a kernel-hard RSS limit or process-tree bound.

All 108 saved observations are present: 107 LIVE RSS/Threads samples and a final
EXITED_BEFORE_READ observation. Threads=1 in those live samples. The exact owned
children endpoint returned FileNotFoundError/errno2: children and count remain
null with NOT_AVAILABLE, never zero. Lifetime child absence is NOT_PROVEN.
Terminal/lifetime I/O is UNKNOWN; last successful I/O counters are retained
separately. Sampling can miss transient descendants/threads or exit peaks.

## Historical failure and revised observability contract

The original strict harmless sleep probe is still FAILED_NO_RETRY: one probe,
zero candidate runs, critical children monitoring unavailable, cleanup/reap with
return code -9. It never passed and was not retried. A separate one-time read-only
observation of an already-running interpreter showed the same absent interface
while containing /proc/task directories existed. Static source/ELF evidence
supported a single-process design, not an OS sandbox or lifetime proof.

The separately revised v2 contract permits only FileNotFoundError/errno2 on the
exact owned-PID children endpoint as NOT_AVAILABLE/null. PermissionError,
malformed contents, other errors or known nonempty children still stop. VmRSS and
Threads remain mandatory/fail-closed, with at least one LIVE observation required.
The second probe is disabled. One collector completed, return code0, no guard
stop, no warmup or retry. The original raw resource status remains
RAW_COMPLETE_PENDING_AUDIT; a separate final saved-evidence audit reported
PASS_DESCRIPTIVE_WITH_LIMITATIONS. Neither changes the old strict probe or earlier
Qwen20 FAILED_NO_RETRY result.

## Scientific interpretation

This establishes counts only for these exact deterministic constructed domains
and their state/geometry continuity. It does not establish qualifying real-world
negative exposure, independent real sessions, real-world FAR, FRR, product low
FAR, numerical qualification, true-word-end latency, board performance or product
readiness. 0 events / 0.25 constructed hours is a descriptive rate of0/h.
The conditional one-sided95% zero-event Poisson arithmetic gives11.982929/h only
if qualifying exposure and independent stationary Poisson assumptions hold;
these assumptions are not established here. This is not a product FAR bound.
MA32 ideal pre-round FIR zeros/spectrum are not measurements of quantized PCM;
startup and rounding can fill nulls. No FFT spectral QA or strict bandlimit claim.

## Files and identity

`evidence/raw.jsonl.gz` is one lossless gzip member (mtime0), decompressing to all
4,617,273 original bytes, SHA256
`ac881e0407496a92dbcbe5ed54adc2f2367bd36478240ed093f1433c4c6be998`.
No record filtering or rewriting occurred. `metadata/geometry.json.gz` similarly
preserves all1,947,411 original geometry bytes. `PROVENANCE.json` records each
original-to-public hash mapping and every projection rule. Exact input metadata,
generator recipe, decoder configuration and complete saved scorer output remain
byte-identical. The raw header references original frozen identities; modified
public protocol/resource/history projections have separate hashes and were never
executed. Ephemeral run PIDs and exact /proc error endpoint strings remain in
scientific evidence; host nodename and boot identity are omitted.

The corrected r2 interpretation is current: run-phase supervisor time and sampled
RSS wording above supersede earlier imprecise wording. Original evidence remains
unchanged. The public audit is a purpose-written scientific summary, not a copy
of private review, release or transport records.

## Excluded payloads and reproducibility

WAV/PCM payloads (28,800,132 WAV bytes), weights, runtime binaries, collector
binary and previous archive payloads are not redistributed. Their immutable
references and hashes are retained in `EXTERNAL-REFERENCES.json` and input metadata.
Source/generator/scorer tools are in kws-pipeline `research/native_a20_n0`.
`src/generate_inputs.py` is exact original source; `GENERATOR-RECIPE.json` carries
all seeds and rules. No input regeneration was used as publication evidence.
The source record validates this complete saved record without any model, audio,
frontend, decoder, native execution or new probe.

The new data directory is additive to kws-data commit
`617b66bb30148a675805be2f17c61703d31e6e79`. Existing341 logical assets and inherited
archive inventories are not edited. Source is separately additive to
`804553286fd9caa294769cc4d44cc0c43dc66e88`. Publication of source must first pin the
independently reviewed and read-back immutable data commit.
