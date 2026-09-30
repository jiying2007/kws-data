# Official INT8 encoder: observed quality and host resource comparison

The encoder-only official INT8 variant reduced asset size and CPU/tail timings in these sequential x86 measurements. It retained the same observed42-clip outcomes. This is a useful deployment candidate for target-device testing; SSC305 feasibility remains unverified.

## Asset and license provenance

Official model: https://www.modelscope.cn/models/pkufool/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/summary
Snapshot: `3787015f084cb241dfa0e4ba237703a2d4322d50`. Published file metadata is preserved byte-for-byte in `evidence/pinned-files.json`. Immutable download URL and expected SHA are in `evidence/downloads.json`.
Encoder: `encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx`, 4,807,159 bytes; SHA256 `017af32f2c0138f931d05fbc009ee864295e910aff304f77d2f563815fc834fb`.
Existing FP32 decoder/joiner and tokens are retained via local symlinks. No INT8 decoder/joiner was fetched. Four-asset total:5,737,545 bytes. Model-specific Apache2 declaration is retained in `evidence/model-README.md`; runtime Apache2 license in `evidence/runtime-LICENSE`. Upstream training-source rights and real voice/data permissions require separate review; this license evidence is not a blanket warranty.

## Quality

Same42 original synthetic development clips,20 positive/22 confusable, same reviewed labels and two-thread decoding settings.19/20 positive hits,0/22 confusable clips/events,0 wrong-keyword/additional events. The sole miss remains `qwen3-kw2-eric`. Human-reviewed positives9/10; ASR-reviewed10/10. Train14/14; development A4/4; development B1/2. No event depended on EOF flushing. FP32 and INT8 subgroup summaries and per-clip hit/miss identities match, while token timestamps may differ.

These clips were previously observed and provide neither fresh holdout evidence nor continuous FAR or real3–5m performance. Same numeric threshold does not establish matched FAR across quantization.

## CPU1 resource measurements

AMD EPYC9V74 x86; Python3.12.14, sherpa-onnx1.13.8/NumPy2.2.6. Process-local BLAS/OMP caps verified one resident thread. One persistent model;42 preloaded originals (78.56s), one warmup pass, three measured passes (235.68s total). Fresh stream per clip,320sample feeds, zero padding. Scripts are retained. Historical FP32 measurement path/hash are bound in `comparison.json`.

| Metric | FP32 | INT8 encoder | Observed reduction |
|---|---:|---:|---:|
| assets_bytes | 13104605 | 5737545 | 56.22% |
| cpu_rtf | 0.0231420413 | 0.0203499788 | 12.06% |
| wall_rtf | 0.0231573565 | 0.0203506307 | 12.12% |
| all_feed_p99_ms | 8.96554842 | 7.81152461 | 12.87% |
| decode_active_p99_ms | 12.3620019 | 10.9506182 | 11.42% |
| decode_active_max_ms | 19.4996 | 16.104576 | 17.41% |
| final_rss_kib | 91516 | 89152 | 2.58% |
| constructor_wall_s | 0.712434191 | 0.686744895 | 3.61% |
| constructor_rchar_bytes | 25299069 | 10564951 | 58.24% |

INT8 all-feed p50/p95/p99:0.029905/5.052560/7.811525ms; decode-active p50/p95/p99:6.681280/9.003068/10.950618ms.11784 total feeds,633 decode-active feeds. Buffered feeds dominate the all-feed distribution; consult active-feed tails. These wall measurements include waveform acceptance, readiness checks, decoding, result access and event reset; EOF/stream creation are separate. Maxima are observations, not worst-case bounds.

INT8 constructor RSS46,308→83,552KiB; after warmup88,840KiB; end VmRSS/VmHWM89,152KiB. ru_maxrss89,056KiB is retained separately because kernel accounting/sampling differs. RAM includes Python/ORT, loaded assets,5,027,840B preloaded float32 audio and instrumentation; it is not planned embedded C workspace.

Constructor: CPU0.686301s/wall0.686745s; rchar+10,564,951B, syscr+36, minor faults+6896. Steady: CPU4.796083s/wall4.796237s; rchar+8708B/syscr+28 (includes bookkeeping), minor faults+78. Both phases read_bytes0/write_bytes0/major faults0. Asset hashing warmed cache before constructor. No cold-cache benchmark, cache drop, mmap tracing or long soak was performed. Logical read counts are not flash traffic; zero measured storage reads does not guarantee zero future I/O.

Sequential host runs can vary with scheduling/cache/system load. The observed12.1% CPU improvement and lower p99 are not a controlled statistical speedup guarantee, and do not extrapolate to SSC305. Verify ARM32 runtime build/operator support, ABI/libc/NEON, SDK licensing/integration, cross-compiled deployment dependencies, real-time scheduling, target flash reads/page faults, long-run memory and audio/DMA contention. Neither parameter count nor C11 alone is a deployment gate.

## Reproduction and verification

Use existing `/workspace/shared/kws-external-baseline/venv/bin/python`. `run.py` performs the frozen quality pass; `resource-profile/profile.py` requires `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. `validate.py` checks official asset digests/sizes, exact decoding/data parity, subgroup results/provenance, thread counts and repeated event identities. Historical absolute paths/symlinks are documentary and require materializing the same pinned inputs for another workspace. Approved FP32 quality/evidence artifacts were not modified.
