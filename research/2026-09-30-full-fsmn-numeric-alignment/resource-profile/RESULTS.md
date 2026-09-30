# Fixed native-FSMN diagnostic cost on x86

One authorized profile of a model implementation that still fails strict final-logit and fbank numerical gates. No training, optimization variant, threshold change or SSC305 extrapolation. The executable pipeline still uses Python/ctypes and Torch softmax plus the original Python decoder.

AMD EPYC9V74 host. Torch intra/inter-op and OMP/MKL/OPENBLAS fixed1; startup and every pass boundary sampled one OS thread. Samples do not prove an unobserved thread-count maximum. Warmup was the first3 raw recordings in source order, followed by3×42 raw recordings,235.68audio seconds total.

- Timed PCM→events CPU5.758814368s /235.68s =0.02443488785 RTF, or24.435ms CPU per audio second,2.4435% of one host core
- Pipeline wall sum5.734398696s. Wall/CPU difference reflects measurement resolution and scheduling; no real-time deadline guarantee
- C model call including its internal CMVN:5.201670355CPU seconds,22.0709ms/audio second
- C frontend/splice call boundary inclusive of callback copies:0.152644713s,0.647678ms/audio second
- Of that inclusive frontend time, callback copies0.008003267s. Do not add it again. Subtracting gives a rough boundary residual, not isolated C-instruction time
- Torch full2599 softmax0.027976982s,0.118707ms/audio second
- Original Python decode/detection methods0.172264473s,0.730925ms/audio second
- Standalone CMVN microcalls0.060044159s ran outside the pipeline timing. They are not an additional pipeline component and were not subtracted from model time
- Per-clip construction/init CPU0.074619186s total, separately reported; not included in the steady PCM→events timer

The timed whole interval includes Python timing-record allocation, retained feature/logit copies and ctypes/timer overhead. Model trace pointer was NULL, skipping diagnostic trace copies without altering arithmetic. All126 measured clip runs reproduced saved native logit hashes and events exactly. There were no NPZ/log writes in steady passes; JSON was written after measurement.

Each pass `/proc/self/io` delta: read_bytes0, write_bytes0, rchar106, syscr2, no writes. PCM was preloaded; these counters include validation, microcalls and the observers around the pass. Zero warm-host physical-read counters do not prove zero flash reads, page faults or DRAM bandwidth on SSC305.

Startup after stdlib import through imports/audits/model/PCM/reference preloading:CPU1.377562485s,wall1.392770760s. NumPy/Torch imports account for0.992081244CPU s; local model identity validation/load/init0.064109064s. The remainder includes source/PCM and large diagnostic-reference audits, so it is not a minimal production model startup benchmark. Startup logical reads238,203,025bytes, physical reads0; warm-cache scope.

Enumerated sizes:
- Model+CMVN FP32 payload3,027,732bytes
- C PCM state from native sizeof62,264bytes
- C model ctypes layout mirror22,544bytes (includes22,528-byte numerical cache, pointer/flag/alignment)
- Declared df_step scratch float arrays4,624bytes; compiler frames/other locals/caller output/decoder structures are additional
- Python+Torch research-process peakRSS273,309,696bytes, not the future C application's memory requirement. Pass-end samples261,369,856/268,136,448/270,770,176bytes include accumulated measurement records

No strict causal speed comparison with previous differently wrapped/protocol runs is claimed. The dominant measured cost here is the unoptimized scalar full2599 model. The prior numerical failure remains explicit.

Result SHA256 ba58cc18037d56884731e0646f46d013e2dbd08fb5de288d249032ec4bc57629; preflight a1f68618a930b90d0eaab3297785bf56827e1fdc92a0bd1a3762cb685cbd6724.
