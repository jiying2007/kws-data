# Qwen6 saved source screen

The execution completed, but the source panel did not qualify. Six fixed Qwen3-TTS-0.6B preset samples were generated once and each was transcribed once by SenseVoiceSmall and Qwen3-ASR-0.6B. Under the frozen text rule, only Aiden / K2 has weak machine support; five samples remain quarantined. No preset supports both K1 and K2. This is exposed synthetic research evidence, not a human correctness rate, acoustic completeness judgment, quality improvement claim or training dataset.

Source: [draft PR482](https://github.com/jiying2007/kws-pipeline/pull/482), [immutable source commit](https://github.com/jiying2007/kws-pipeline/tree/16ba0941bd7d7d6705d1ee94bb433dbd4fc6aa51), [successful run 37343458180](https://github.com/jiying2007/kws-pipeline/actions/runs/37343458180), attempt 1. The fixed requests are Ryan, Aiden and Ono_Anna × K1 `你好小窝` / K2 `小窝小窝`, seeds 1337–1342. Sohee generation and screening remain zero.

## Result and decision

`result-projection.json` retains all six rows: one agreement with the intended plan, two agreements that differ from the plan, and three ASR disagreements. Normalization removes only whitespace and Unicode punctuation; it does not repair homophones, spelling or repetition. Agreement is correlated machine evidence and the intended plan is not a human label. All six have UNKNOWN human truth and acoustic completeness. No human label was inferred from either model; training admission is zero.

The unchanged prospective panel gate was run with the six actual saved observations and `require_data=True`. It returned `REJECTED` / code 1. The integration wrapper successfully records that rejection; its successful process exit is not a passing gate. All observed rows are exposed screens, with actual text unset and human review pending. Actual positive coverage is zero and the plan remains preparation-only. The six `EXPOSED_OR_UNKNOWN_HELDOUT` conflicts concern future Sohee placeholder rows whose exposure is UNKNOWN. They do not show that Sohee was generated, heard or leaked. The planned 24 coverage rows were not generated, and this result does not open that expansion.

## Preserved evidence

The five original Actions ZIPs are retained exactly: the two earlier failed setup attempts, successful full TTS evidence, blind handoff, and raw ASR/status/resources. The full TTS ZIP is stored as three ordered pieces of at most 1 MiB; concatenating them restores its original 2,519,928 bytes and SHA256. Other ZIPs remain whole. `archive-manifest.json` binds every stored byte, each Git blob identity and every restored ZIP. No model weights, installed environments or raw job logs are included.

[Run37335852099](https://github.com/jiying2007/kws-pipeline/actions/runs/37335852099) and [run 37339583157](https://github.com/jiying2007/kws-pipeline/actions/runs/37339583157) remain `FAILED_NO_RETRY`, with zero TTS and ASR calls. The first specific child cause remains `UNKNOWN_NOT_RETAINED`. The second retained failure was `import_soundfile/OSError`, before model loading. The successful third setup used the official bundled manylinux wheel of the same SoundFile 0.13.1 release and offline native/import checks before model downloads. The Python implementation and dependency metadata stayed unchanged. All five disputed or unsupported outputs are retained.

Full TTS, blind and ASR ZIP digests were checked and all members were privately persisted and read back before the fixed comparison. The ASR job received only the SHA-addressed blind ZIP. `execution-history.json` records the audit identities, original failures, request accounting and resource limitations.

## Offline verification

Use Python 3 standard library only, from this directory:

```sh
python3 -B verify.py
python3 -B test_verify.py
python3 -B verify.py --restore /path/to/new-directory
```

The optional restore directory must be new and outside this archive. Verification checks exact archive membership and hashes, reassembles the original ZIPs, checks 108 extracted members plus seven inner blind members, reruns the unchanged saved comparison, compares all six projected rows, and reproduces the panel observations and gate reports byte-for-byte. It makes zero network or model calls. Without `--restore`, temporary reconstructed files are removed after verification. A `PASS` verifies the saved evidence and its rejected panel result; it is not a dataset qualification pass.

`method/` preserves the exact frozen comparison code and its two source files used solely for AST-selected pure helpers. `panel-integration/` preserves the gate and the observed-screen mapping. Its public provenance projection replaces one internal source locator with an immutable public attestation. The history/plan integrity hashes and adapter pins are mapped explicitly in `PROVENANCE-PROJECTION.json`; these are projected bytes, not the original frozen plan bytes. The gate implementation, scientific roles/IDs/cells/policy, observations and gate report are unchanged. The original full comparison is regenerated locally rather than included as a private report. The original plan, source hashes, normalized ASR values and all negative outcomes remain reproducible.

## Resources and limits

Actual controlled package/model response bodies totaled 6,275,920,806 bytes for the successful pipeline and 12,143,253,734 bytes including the two earlier setups. Other infrastructure transport and unobserved third-party sockets are not globally accounted. The two successful standard public CPU jobs kept the reviewed per-job 50-minute and 4-GiB controlled-download limits. Cumulative scientific calls were 6 TTS and 12 ASR, with zero retries.

The nominal sampler sleep was 0.25 s. Generation, SenseVoice and Qwen-ASR recorded 189, 25 and 82 samples respectively; timestamps and maximum gaps were not retained. RSS and disk peaks are sampled observations, not continuous hard guarantees. Stage timings include setup/model loading and all requested decodes; these are not KWS runtime measurements. CPU limits are per process, not hard aggregate process-tree limits.

The existing `kws-data` CI validates its catalog and explicitly named older research archives. It does not discover this archive's verifier automatically. Its ordinary green result must be reported separately from running this verifier against the exact remote archive bytes. The source repository's ordinary CI likewise remains distinct from the research run.

These stock presets do not establish distinct human actors or novelty relative to model pretraining. Official model licenses do not settle all actor/output reuse rights. This archive records a bounded research pilot without commercial or deployment qualification. It performs no KWS training, threshold tuning, new synthesis, seed/voice search or model replacement.
