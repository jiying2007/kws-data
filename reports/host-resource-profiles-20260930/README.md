# Host CPU and I/O evidence, 2026-09-30

CPU and I/O are the primary product constraints. Model storage and RAM remain flexible; this pack does not impose fixed limits. These are measured x86 host reference points, **not SSC305 measurements or a ranking at a common operating point**. The user identifies the target as SSC305 dual-core Cortex-A32. No target board was measured. Software work with existing synthetic/open-source data can proceed now; board testing is an acceptance stage, not a prerequisite for the next software deliverable. No models, audio, executables, dependency environments or training are included.

## What was measured

| Reference | CPU RTF on this host | Timed work and denominator |
|---|---:|---|
| Frozen C | 0.000728–0.000752 | Full accept_pcm16 path, 20 ms blocks; concatenated original42 PCM without resets between clips; 3 processes ×10 measured repeats ×78.56 s, after warmup |
| cFSMN full2599 head | 0.011995 aggregate | PCM frontend + model + decoder + WAV read/spotter construction; 300 ms chunks fed offline; 3×42 clips, 235.68 audio seconds |
| sherpa FP32 | 0.023142 aggregate | Preloaded audio, 20 ms feeds with bursty ready/decode; persistent model/fresh stream per clip; 3×42 clips, 235.68 audio seconds |
| sherpa INT8 encoder | 0.020350 aggregate | Same sherpa resource protocol, measured sequentially; decoder/joiner remain FP32 |
| Lightweight A / B | 0.090602 / 0.031742 | Model-only incremental PyTorch calls on cached32dim features; 3000×20 ms feature frames each; no frontend, PCM or decoder |

All are single-thread profiles, with Python/framework overhead included where used. Different batching, reset boundaries, quality, frontends and timed scopes prevent a fair end-to-end speed ranking. Server timings cannot be scaled to SSC305 from clock frequency or parameter count. Small A/B models have poor cross-voice results; the tiny frozen C baseline misses all20 positives. Host speed does not justify promotion.

INT8 and FP32 sherpa retain identical complete recording/event observations on the42 original clips:19/20 positives and0/22 confusable activations. This is a small observed-development sanity check with permissive, unequal decoder operating points across architectures, not long-negative FAR evidence. INT8 assets are5,737,545 bytes versus13,104,605; its roughly12.1% lower measured CPU RTF is a sequential-host observation, not a universal speedup or a C11 deployment result. The runtime here is ORT; no ONNX runtime product dependency is proposed.

## RAM and storage interpretation

See each original report for exact model bytes, logical state, backing storage and process RSS. cFSMN has3,027,732 bytes of FP32 model tensors and22,528 bytes of acoustic cache. Its Python process peak is312.7 MB, which is not a C working-set estimate. `cfsmn/profile-corrected.json` is authoritative: the raw profiler equated a NumPy view's logical size with backing storage; the annotated correction makes backing size unknown. There is no valid summed backing total. Original raw measurements are retained for audit.

Observed cached Python measurement windows have zero storage-attributed read_bytes/write_bytes. Inputs and process observation still cause logical reads; cFSMN rereads WAV files in its timed path. The C core's no-file-I/O audio path is a source/API fact, not a syscall trace of the full system. Zero steady storage I/O is a design goal, not proof about physical flash, cache misses, cold start, capture/IPC or background services on SSC305.

If useful for planning,5% and10% of one target core could be discussed as CPU budget tiers, equivalent to1/2 ms average at50 frames/s. Percentages use one core as the denominator, not combined dual-core utilization. They are suggestions awaiting confirmation, not measured capabilities or hard caps. Average utilization cannot replace p99/max callback deadlines, contention testing and thermal behavior. AFE inclusion must be specified separately; no fixed model/RAM budget is inferred.

## Reproduction and verification

The copied scripts are historical exact measurement scripts, with their original absolute input/environment paths retained. They require separately obtained pinned models, input assets and runtimes to rerun. Do not execute them merely to verify this archive. No inference is performed by the portable verification command:

    python tools/verify_host_resource_profiles.py
    python -m unittest discover -s tests -p test_host_resource_profiles.py

The standard-library verifier checks exact file inventory/size/SHA256, binds measurement scripts, checks CPU-time/audio denominators, cFSMN per-clip totals and annotated correction, sherpa aggregate totals/raw timing counts, and parses the lightweight raw integer NPY arrays inside NPZ without NumPy or pickle. It verifies retained data consistency, not the truth of hardware counters or target performance.

Source folders contain the exact reports, raw retained measurements, scripts and receipts. C retained per-run summaries, not every raw per-block sample; do not claim those samples are archived. cFSMN similarly retains per-clip timing and chunk summary distributions. Manifest excludes only itself; its entries include this interpretation. No existing quality report was changed.
