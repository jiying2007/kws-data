# Frozen C runtime host profile, 2026-09-30

This is a single-thread x86 cloud diagnostic, not an SSC305 measurement or an
architecture budget. The current frozen model still misses all 20 observed Qwen
positive clips. Its speed is not evidence that this quality level is acceptable.

## Identity and sizes

- Source: c84576f4945c16bb7d5ace81f69c85690a122198; manual GCC 14.2.0 `-O2`, strict C11 warnings
- Host: AMD EPYC 9V74 80-Core Processor reported by `/proc/cpuinfo`, 9 visible vCPUs; single-thread harness, no affinity/frequency changes or cache drops
- Model: ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402, 6812 bytes
- Pack: 370ee3eeba27b1d62b38f32f53d8302b47c2b762392101e6ca7eb0ccf8dfb723, 120 bytes
- Runtime arena API on this x86 ABI: 22088 bytes, alignment 8
- Model+pack+arena: 29020 bytes, excluding stack/code/application buffers
- A caller's 20 ms mono PCM16 block is 640 bytes; input rate is 32000 bytes/second
- Model/pack view structures measured 88/1040 bytes in the harness, separate from that core footprint; do not interpret process RSS or timing sample arrays as the engine workspace

`build-receipt.json` binds compiler commands, source/generated inputs, binaries,
and the profiling workload. The original model is feature_dim=32, hidden_dim=64,
vocab_size=5. File sizes are portable; arena/structure sizes need remeasurement in
the target ABI. This profile does not describe the new lightweight prototypes.

## Workload and timers

`qwen42-concat-workload.wav` concatenates original PCM in the pinned native export
order, without added silence. 78.56 seconds per pass; stream state continues across
clip boundaries within the pass and resets between passes. This is a timing
workload, not the per-clip quality experiment. Originals were not modified.

Each of 3 process runs preloads the entire timing WAV outside the measured loop,
runs one full warmup pass, and measures 10 more passes: 785.6 seconds of audio and
39280 320-sample blocks. CLOCK_PROCESS_CPUTIME_ID measures the single-thread loop,
excluding resets and all file I/O. It includes loop/timer/bookkeeping overhead.
CLOCK_MONOTONIC surrounds each accept_pcm16 call; its per-block timing includes
clock overhead and scheduling delay. Reported percentile labels use the existing
benchmark's empirical index `max(1, floor(p * n)) - 1` (not conventional ceil-based
nearest rank). No busy waits or simulated real-time sleep.

The existing unmodified kws_board_bench additionally measures one original
1.84-second Qwen positive clip repeated 100 times. It excludes each fread from its
per-block timer but does not discard an in-process warmup. Its report is retained
as preliminary corroboration, not merged into the warmup-controlled runs.

## Three warmup-controlled runs

Process CPU RTF: 0.000751587, 0.000745830, 0.000727770; median 0.000745830.
That median is approximately 0.746 ms CPU per second of input audio, or 0.075% of
one continuously available host CPU. It is not a predicted SSC305 utilization.

Per-20ms-block wall timings:
- p50: 13.199–13.200 microseconds
- p95: 17.516–18.637 microseconds
- p99: 38.678–56.003 microseconds
- maximum: 1019.650–1957.333 microseconds; scheduler noise is included

## Startup and steady I/O distinction

Warm-filesystem model+pack fopen/read/close: 55.323–80.460 microseconds.
Parsing/validation: 2.113–2.935 microseconds.
Arena malloc+initialization+keyword setup: 22.283–23.976 microseconds.
These are separately measured once per process; no cold-cache or physical flash
latency claim. They exclude dynamic loader/program launch and input WAV preload.
The fopen/read/close timing includes metadata-handling overhead; its logical
payload byte count does not measure filesystem metadata traffic or physical
block-device transfer granularity.

The existing file loader requests exactly 6932 model+pack payload bytes at startup.
The core API/source has no allocator, locks, threads or file I/O on its audio path;
model weights are read-only in-memory views and the caller owns the arena. That
is an API/source inspection, not a syscall instrumentation claim. Harness file
reads, timing arrays and full-workload preload are not deployment requirements.
An integration can feed 640-byte PCM blocks and keep weights resident, rather than
re-reading model files or using per-block disk IPC. AFE/capture/IPC and other board
workloads are excluded and must be measured on the actual target.

For SSC305, measure the same arena API, 20 ms deadline percentiles, cold/warm
startup, target CPU time and I/O under representative camera/AFE/load contention.
Do not extrapolate this EPYC ratio, multiply model bytes into a speed prediction,
or lock a new model budget based on this failing-quality tiny baseline.
