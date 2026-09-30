# sherpa Mandarin 3.3M: measured x86 CPU / I/O resource profile

This resource-only run is separate from the previously frozen two-thread quality comparison. No approved quality reports, evidence pack, weights or decoder parameters were modified. This is not an SSC305 result, a deadline guarantee, a C working-RAM budget, or a new quality qualification.

## Primary measurement

Host: AMD EPYC 9V74, Linux x86_64; 9 available CPUs/affinity CPUs. CPython 3.12.14, sherpa-onnx/core 1.13.8, NumPy 2.2.6. ORT `num_threads=1` and process-local `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; `/proc/self/status` confirms exactly one resident thread before/after load, warmup, and every measured pass. No CPU affinity change, root operation, cache drop or global setting was used.

One persistent FP32 model load, 42-clip warmup, then three passes over the same 42 original SHA-verified clips. Each pass is 78.56 seconds of audio; total measured exposure is 235.68 audio-seconds. Audio is preloaded float32 (5,027,840 bytes), so input filesystem work is outside steady inference. Each clip gets a fresh stream; model remains loaded. Every feed is up to 320 samples/20 ms, no wall-clock pacing. Decoder defaults remain score 1.0, threshold 0.25, 4 paths, 1 trailing blank. No appended silence.

- Model assets: 13,104,605 bytes total: encoder 12,174,219, decoder 675,349, joiner 253,410, tokens 1,627. This excludes runtime, libraries and loaded execution state
- CPU: 5.454116 seconds, CPU RTF 0.0231420
- Wall: 5.457726 seconds, wall RTF 0.0231574
- CPU RTF across three passes: 0.0236435 / 0.0238493 / 0.0219334
- 11,784 feed+ready/decode timings: p50 0.030515 ms, p95 6.433076 ms, p99 8.965548 ms, maximum 19.499600 ms
- 633 feeds that actually decode: p50 7.422884 ms, p95 10.192676 ms, p99 12.362002 ms, maximum 19.499600 ms

The distinction between all feeds and decode-active feeds matters: most 20 ms feeds only buffer/extract features; this streaming model performs burstier chunk computation. Feed timings include waveform acceptance, readiness checks, all ready decode calls, result checks and any event reset. Stream construction and EOF drain are measured separately in the JSON. A measured maximum below 20 ms on this host is not a target-board real-time guarantee. Detection end latency is not measured.

## Startup, RAM and I/O

Imports: 0.102879 seconds wall, 0.098506 seconds CPU. Constructor: 0.712434 seconds wall, 0.711911 seconds CPU. Asset identity hashing occurs before the timed constructor and warms cache; cold-cache startup is explicitly unknown.

Process RSS (/proc/self/status):

- Before constructor: 46,448 KiB
- After constructor: 86,148 KiB
- After warmup / before measured passes: 91,188 KiB
- After measured passes: 91,516 KiB (89.37 MiB)
- Process-lifetime ru_maxrss after passes: 91,360 KiB; /proc VmHWM: 91,516 KiB (different kernel accounting/sampling surfaces retained rather than reconciled artificially)

These values include Python, NumPy, ORT, loaded weights, preloaded audio and timing/event instrumentation. They are not standalone model workspace; the small growth during measurement also includes accumulating timing arrays. Three short passes do not establish long-run leak freedom.

Constructor /proc/self/io deltas: rchar +25,299,069 bytes, syscr +38, read_bytes 0, write_bytes 0. Constructor faults: +8,012 minor, 0 major. These are process counters, not a per-model-file trace; logical bytes read can exceed asset file sizes.

Across measured passes, including snapshots between passes: rchar +8,687 bytes, syscr +28, read_bytes/write_bytes 0; +82 minor faults, 0 major faults. Counter/status observations themselves create rchar/syscr bookkeeping. The cache-backed run establishes no physical storage I/O in these counters during measurement, not zero RAM traffic or cold-start flash cost. We did not trace mmap calls; minor faults do not by themselves prove a model mapping or disk access.

Each measured pass emits 19 events, matching the existing development observation. This is a protocol sanity check, not three independent quality trials.

## Files and reproduction

- `measurement.json`: complete primary measurements, identities, assets/hashes, configuration and caveats
- `raw-timings.json`: every per-feed, EOF and stream-creation timing plus observed events
- `profile.py`: actual primary measurement script (historical absolute paths retained)
- `diagnostic-unlimited-blas-pool/`: earlier preserved diagnostic that set ORT threads=1 but still had 9 resident process threads; not the primary one-thread result
- `manifest.json`: SHA256/byte inventory

Reproduction command from the original workspace:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/shared/kws-external-baseline/venv/bin/python /workspace/shared/kws-sherpa-resource-profile/profile.py

Use a copied output directory for a new run rather than overwrite this evidence. No inference downloads or training are needed if the pinned existing dependencies/model are retained.
