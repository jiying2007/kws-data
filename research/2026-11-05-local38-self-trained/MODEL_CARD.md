# Local38 self-trained research checkpoint archive (2026-11-05)

This archive preserves exactly two self-trained research checkpoints, including the failed RNN result. It is not a deployable KWS package, a selected model, or an independent generalization result. No pretrained donor weights are included; FSMN is excluded.

## Provenance and training

Architecture/source reference: [kws-pipeline PR #744](https://github.com/jiying2007/kws-pipeline/pull/744), pinned commit [6da11c72eb5d3a9d36398133e91095b9d9caa655](https://github.com/jiying2007/kws-pipeline/tree/6da11c72eb5d3a9d36398133e91095b9d9caa655). Local source snapshot SHA-256: eb4d4432e2e58643a135b7d17b651fdb13d6387494fa62bddb2aefec3bb539e8. The local continuation harness is not distributed in this minimal checkpoint archive; pinned upstream source alone is not a complete reproduction package.

Both models used the same 38 weak-labelled, non-cloned synthetic training recordings, 100 total epochs, 1,000 optimizer steps and 3,800 sample exposures per model. Epochs 21–100 continued the original formal epoch-20 checkpoints. RNN64 has 6,533 parameters; C1-CTC has 15,685. Equal source exposure counts do not imply equal cumulative loss coefficient: batches of four use coefficient 1/4 and the last batch of two uses 1/2. This is not a capacity-only ablation against older tiny models.

Actual runtime: Python 3.12.14, Torch 2.15.0+cpu, NumPy 2.1.3, CPU, one Torch thread, deterministic algorithms, MKLDNN disabled. Python 3.8 compatibility and the upstream Torch 2.13 runtime lock were not verified.

## Fixed results

- Own-training fit: C1-CTC sequence errors 0/38, token edits 0/148, mean CTC 0.009000349174082083. RNN64 sequence errors 35/38, token edits 74/148, mean CTC 0.996399534376044. This measures memorization/fit, not generalization.
- Historical development readback: C1-CTC sequence errors 3/12 and token edits 5/46; RNN64 sequence errors 9/12 and token edits 19/46. There were no execution failures.
- The historical set contains eight previously human-reviewed synthetic recordings and four machine-weak recordings. It shares the Qwen generator family with training. It is previously observed development material, not new human gold, a sealed test, or family-independent evaluation. The readback did not select a model or trigger further training.

No Strict/D0 admission, native-v2 compliance, commercial rights, production export, real-world FAR/FRR, latency, or SSC305 qualification is established. Research/public archival availability does not change these limits.

## Checkpoint inspection and loading

First verify each exact file length and SHA-256 against manifest.json. Files use schema `local38-continuation-checkpoint-v1`. They contain `state_dict` model parameters, `optimizer_state_dict`, Python/Torch RNG state, architecture, recipe, runtime, and input file path/hash provenance. The referenced WAV/ASR files themselves are not embedded. The archive includes neither audio payloads nor raw ASR transcripts.

For controlled research inspection only, use the recorded runtime and `torch.load(path, map_location="cpu", weights_only=True)`. Check schema, kind, architecture and pinned source identity before using `checkpoint["state_dict"]`. Do not use unrestricted pickle loading. Build the exact matching architecture from the pinned source and recorded recipe, and use strict state-dict loading; do not guess missing frontend/decoder behavior. This publication does not rerun loading or inference. Earlier training readback and historical inference verified the retained bytes without changing them. Optimizer continuation from these final checkpoints is not supported or certified (`resume_supported=false`, `future_restore_tested=false`). They are not TorchScript/ONNX/native production exports.

A static ZIP/pickle inventory matched model, Adam-moment, scalar-step and RNG storage sizes; no audio container member, CJK transcript literal or credential pattern was found. This is limited inspection, not proof about arbitrary numerical tensor semantics or a universal safe-deserialization guarantee. Static review receipt SHA-256: a42fc502fa03ed448d35eefc51c10861d8fd03a523be7f5da062305b9cbc38eb.

See RIGHTS.md for the separate source-license and research-use limits.
