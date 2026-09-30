# cFSMN x86 resource measurements

Actual measurements, not SSC305 estimates. The baseline is the intact 756,133-parameter donor with full 2,599-class head. Hamming provider route only; threshold 0 and upstream decoder unchanged.

## Timing

CPU: AMD EPYC 9V74 80-Core Processor; x86_64, PyTorch 2.13.0+cpu, intra/inter-op threads 1; /proc reports one thread. CPU affinity allowed 9 cores; no affinity pin, scheduler isolation, or frequency control. Three initial clips warmed the process; then three passes over the original 42 clips, each 78.56 seconds of audio. All 126 measured outputs match retained baseline logit SHA256 and exact event records. These clips are an observed development workload, not a broad worst-case workload.

Contrary to the initial full-clip assumption, the existing audited driver feeds 4,800 samples (300 ms) at a time through the official cached frontend/model/decoder. Files are read into memory one clip at a time and chunks are fed without real-time pacing. This is offline-fed 300 ms chunk timing, not 20 ms callback or hard-real-time certification. No new 20 ms adaptation was made.

| Pass | CPU seconds, whole pipeline | Wall seconds | CPU RTF | Model CPU seconds | Frontend CPU seconds | Decoder search + detection CPU seconds |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.970407 | 0.970421 | 0.012352 | 0.469634 | 0.404993 | 0.047494 |
| 2 | 0.998214 | 0.998309 | 0.012706 | 0.473600 | 0.424526 | 0.050688 |
| 3 | 0.858456 | 0.858468 | 0.010927 | 0.398189 | 0.372836 | 0.044443 |

Aggregate audio 235.68 seconds; pipeline CPU 2.827076 seconds, RTF 0.011995. Model RTF 0.005692; frontend RTF 0.005102; decoder search+detection RTF 0.000605. RTF is CPU seconds divided by audio seconds; these ratios cannot be scaled directly to SSC305 clock rate.

Frontend timing includes PCM16 unpack/conversion, fbank, splice and skip. Model timing includes CMVN, complete head and the retained diagnostic-logit append. Decoder component includes prefix search and detection, but not softmax; softmax and orchestration are in the residual. Whole-pipeline timing includes WAV read, fresh spotter creation, all stages, and timer bookkeeping. Output hash/equality validation is outside the clip timing, and included separately in measurement-loop totals. No model-only replacement or cached-feature replay was substituted.

Each pass has 285 input chunks (including short tails), 281 model calls and 2,529 decoded frames. Chunk CPU p95 ranges 3.616–4.896 ms; maximum observed 10.446 ms. These are instrumented, finite-workload 300 ms chunk measurements, not a bound for every input and not a 20 ms latency claim. The frontend and model retain finite lookahead/delay.

## Bytes and I/O

- Checkpoint container: 3,038,219 bytes, SHA256 d02b09c34f4a8bbb06f0dd1bf5eb58db3395eb7f1fd15c3625fe09d3a2492233
- Learned FP32 tensors: 3,024,532 bytes
- CMVN FP32 buffers: 3,200 bytes; total model tensors: 3,027,732 bytes
- Acoustic cache tensor: 22,528 bytes, shape (1,128,11,4)
- At clip ends, maximum observed feature-history logical bytes 1,280, backed by storage up to 9,600 bytes; residual waveform NumPy logical size up to 5,120 bytes. Its backing allocation was not measured: this array is a view retaining a larger parent allocation
- Those three observed histories total up to 28,928 logical bytes (sum of individual maxima, not proven simultaneous). NumPy backing storage is unknown, so no total backing-byte estimate is supported. This is not full C working memory: transient activations, frontend FFT/mel workspaces, logits/probabilities, decoder state and allocation overhead remain
- Input PCM16 mono 16 kHz traffic is exactly 32,000 bytes/audio-second; an input300ms PCM chunk is 9,600 bytes
- Steady-state /proc/self/io rchar per pass: 2,515,872 / 2,515,874 / 2,515,875 bytes; storage-attributed read_bytes and write_bytes: 0 on all passes. Data was page-cached; this does not mean zero input bandwidth. Proc snapshots themselves contribute small logical-read counts
- Startup after stdlib imports: CPU 1.108 s / wall 1.115 s; rchar 59,369,858 bytes, read_bytes 0, write_bytes 8,192. Includes PyTorch/NumPy imports, audits of all three condition inputs, and donor loading. Not clean cold-start disk I/O; no caches dropped
- Whole research-process peak ru_maxrss: 312,709,120 bytes (~298.22 MiB). VmHWM/VmRSS sampling is also recorded in JSON and can differ slightly due to measurement timing. This includes Python, PyTorch libraries, allocator workspaces and retained per-clip logits; it is not model RAM or a C implementation estimate

## What this supports

The intact donor is inexpensive on this x86 CPU, and frontend work is comparable to the full model. It supplies a measured host reference and observed FP32 reference model/state byte sizes for SSC305 work. Actual SSC305 CPU, end-to-end callback deadlines, cache behavior and full C workspace require a faithful target implementation and board measurement. There is no measured SSC305 tier here, and no legitimate numerical conversion from this desktop/server RTF to one.

Reproduce with the already-existing CPU environment, without installation:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/shared/kws-lightweight-prototypes/venv/bin/python /workspace/shared/kws-cfsmn-resource-profile/profile.py

Files: profile.py, raw profile.json, profile-corrected.json and run.log. The raw profiler mistakenly equated waveform NumPy logical nbytes with backing storage; independent review identified its retained parent-array view. The corrected JSON marks waveform backing bytes null, without changing any timing or rerunning inference. Original baseline bytes and quality controls were left unchanged.
