# Fixed frozen-encoder two-keyword clip BCE experiment

Status: the one authorized P/R experiment completed successfully under the execution guard. The narrow two-output mean-BCE comparison passed; strict8/8 exact-label keyword gates failed for both arms. See RESULTS.md. No additional training or parameter/threshold sweep was performed.

## Scope

Exactly one P/R pair. P uses the pinned published donor encoder; R uses the identical frozen architecture with upstream default initialization under seed1337. Both use the published400D CMVN and exact pinned Hamming/Kaldi waveform frontend. Only a shared-initialization(seed7331)140→2 affine head with282 trainable parameters is optimized. There are100 full-batch AdamW updates per arm. This is whole-clip classification, not streaming KWS.

Training is precisely12 existing human-reviewed Qwen TRAIN recordings plus the previously fixed32 FLEURS nativeTRAIN recordings. FLEURS labels mean absence of the two complete target phrases in the source-provided transcript. This recipe-specific approval does not modify original source receipts: every FLEURS admitted_for_training=false remains false. The32 sentences are not human-reviewed by us, ASR truth, or proof of cross-source generalization. No CTC character labels are used, so donor dictionary OOVs do not bar this narrower binary task.

The primary readout is8 human-reviewed Qwen A/B development clips. All42 Qwen recordings have supplementary readouts retaining original split and evidence labels;22 ASR-reviewed clips are readout-only. No HI-MIA or later failed Qwen clips are included. The old CTC admission, distillation fidelity, and native full-chain numerical failures remain unchanged.

## Reproducibility and stage boundary

- preflight-lock.json binds dataset, spec, initialization tensors, synthetic checks, unit tests, main driver, launcher and test driver
- All74 WAV and PCM identities are checked against original receipts before and after execution
- Donor source/checkpoint/config/CMVN, dataset commit, Python/NumPy/Torch versions and selected runtime files are hash-bound
- P/R encoder and CMVN tensors, and shared head initialization, have named tensor hashes plus state hashes
- Marked synthetic fbank frames exercise the actual source parser for every real clip's sample count without reading speech into the model
- Four actual FSMNBlock identity paths verify eight-step nominal delay, chunk parity and the exact zero-based index>=8 mask
- Actual donor synthetic silence/ramp checks cover finite feature/cache/pool values, reset replay, and one artificial-data optimizer smoke update
- Unit tests exercise population-std train-only normalization, fixed0.5 inclusive decisions, unweighted development BCE, nonfinite rejection and frozen282-parameter-head contract
- Source receipts and in-progress publishing workspaces are read-only

Preflight preparation initially found an overly strict bitwise equality assertion between two mathematically equivalent FP32 BCE formulations and a Torch module-introspection path error. Both failures and pre-fix code are retained under preflight-revisions. The corrected math test uses1e-7 numerical tolerance; no real experiment condition, training step, seed or decision threshold changed. There has been no real-corpus inference while preparing these fixes.

## Execution gate and safety

The independent reviewer must produce independent-approval.json with approved=true and the exact preflight_lock_sha256. Only then may the authorized single run be launched using the existing Python and launch.py run. The launcher creates an exclusive execution-started.json journal and the driver exclusively creates results/, so the same experiment cannot silently rerun. Do not delete either guard to retry. A real failure stops the experiment and preserves partial logs/evidence.

Independent review also caught a race between the strict single-process watchdog and the original git precheck subprocesses. The locked-v1 preflight is retained in preflight-revisions/locked-v1. The final driver creates no subprocesses; the launcher performs the exact read-only git checks before and after the monitored process, binds repository/commit/clean status and the git binary hash in separate receipts, and keeps unknown child processes prohibited. The final code passed a fresh synthetic test and preflight.

The process uses the existing environment, no installation or download, and no native C kernel. CPU budget is600 seconds for the process group; a sampled watchdog and RLIMIT_CPU protect it. Wall watchdog900 seconds, sampled RSS ceiling2GiB, one thread requested in Torch/BLAS environment and verified in resource receipts. The guard inspects only process/memory counters and explicitly selected environment flags, not credentials. These host safety measurements do not establish SSC305 performance.

The first44 real features are TRAIN only. For each arm, train140D mean and population std(floor1e-5) are fixed before100 updates. Both arms finish optimization before any primary development feature inference. Raw clip logits, uncalibrated sigmoid scores, both decisions, per-output unweighted BCE and per-voice results are retained. Frozen encoder state must be identical after training.

## Prespecified interpretation

A narrow representation advantage requires P's mean unweighted development BCE below R for each of the two outputs. A separate exact-label gate requires both bits correct on all8 clips, including no second-output activation on a positive clip. A BCE win alone is not keyword success. Neither gate establishes calibrated scores, false accepts per hour, a blind holdout result, a streaming detector, wake latency, or a deployable product.

The eight nominal startup outputs are excluded; the final delayed tail is left unobserved by the original no-flush EOF rule. The Qwen-positive/FLEURS-negative corpus mix can create a domain shortcut, and positive weighting does not remove it. This experiment compares frozen representations; R is not fully trained from scratch.
