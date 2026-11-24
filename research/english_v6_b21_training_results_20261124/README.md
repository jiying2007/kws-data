# B21 actual paired training evidence — 2026-11-24

**Status: `PASS_TRAINING_EVIDENCE_ONLY`.** The original first public training run completed successfully and its untouched weight and producer artifacts passed the independent saved-data audit. Independent primary and replication quality, C11/ARM parity, board execution and promotion remain separate gates.

## Actual execution and source proof

- [Original run 45352494611](https://github.com/jiying2007/kws-data/actions/runs/45352494611), attempt 1, successful public `push`.
- Run commit: `eb386f2b8e489bbd8e69c6a7972708dd667aa685`.
- Reviewed main / single parent: `60f1e6507216ae605d601f3f09ab15177b46aae2`.
- Complete run tree: `1ed2c3a4b3cfba03c3375b1a31eb15a5e9682717` (3,841 entries, not truncated). The only changed file from reviewed main was the runtime training workflow.
- Weight artifact `14900878022`: 1,801,215 bytes; SHA256 `35fa747026589d3db83a9b24508a3d480d88392080b5c07a910c109b5912ba81`.
- Producer artifact `14901261272`: 1,260,473 bytes; SHA256 `377296c5e0ade26c5cd02782f40b033c97ca5007ee267f014f39d136a6f02af8`. All 221 captured scientific code/resource/workflow files match the actual Git tree's exact blob, mode, byte count and SHA256.

The assessment remains `65ad9c7204755f1cd0b639added163387af39604818f9bd26b9da9fe8bd098fc`; the training design remains `76a1fd1353de00f712023a2b20256a8e622557551823cacd8b808d3f0549b0a1`. The run used the reviewed trainer SHA256 `205654f5afa0f8e55e10016354fec5b4c57057600df66daf13e584fea95ad3e3`.

## What the independent audit establishes

Both arms started from frozen B0 and completed exactly 192 updates each, using the same source IDs, prefixes and corrected common schedule. All 79 original synthetic TRAIN examples and all nine owner-reviewed natural TRAIN windows received all three prefix lengths: 237 synthetic and 27 natural original/prefix combinations. The structural schedule repair is common to both arms; this experiment does not claim a numerical replay of historical B17.

The shared ordinary B0 ceiling and positive retention coefficients remained 0.08. The sole arm contrast is the positive competitor set: the control excludes blank and the challenger includes blank. The same 120 B9-admitted original/gain paths cover 43 originals and 34 source groups. The saved preoptimizer mechanism uses 528 authenticated historical TRAIN arrays, with blank-only suppression in 13 qualifying conditions across ten groups (37 frames), and makes no new diagnostic forward call.

The audit reconstructed the common draw independently, checked all 384 saved optimizer records, verified all 264 ordinary teacher cache arrays against their native frame geometry and allowed TRAIN identities, and authenticated all 16 input ZIP hash receipts. Each saved PT checkpoint contains 43,876 finite float32 values and equals its NPZ tensors exactly, with the original acoustic frontend preserved. The two arms differ in 43,869 learned float values. The saved execution counts are 3,670 student forwards and 264 shared teacher forwards, zero DEV/CAL/LOCKED forwards, and zero new TTS/ASR calls.

| Arm | Original PT SHA256 | First / last saved training loss |
| --- | --- | --- |
| Control | `5f8863d4ef3f10fd35edd8278ac9c01bd48138184e18e08bff9d6f44b4699b05` | 6.4743857384 / 1.6540050507 |
| Challenger | `4947d017142f3b01f5dd4cf7706cc3c6978cf1a072ca022c26ea847ec0dc2f04` | 6.4658460617 / 1.7878690958 |

Training loss is recorded for provenance; it is not an independent quality or promotion decision.

## Evidence and reproduction

`AUDIT.json` contains all verified member hashes and model-export pins. `API_RECEIPTS.json`, `GIT_TREE.json`, `PREPARED_MAIN_COMPARISON.json`, `PRODUCER.json` and `PRODUCER_PRECHECK.json` bind the actual artifact bytes to the original run and reviewed main. `original/` contains byte-identical JSON members from the untouched weight artifact, including full source, draw and update records. `JOBS.json` and `JOB.log` preserve the original completed job: three producer checks and all twelve mandatory numeric tests passed.

`verify_saved.py` is a standalone stdlib/NumPy auditor. Its restricted checkpoint decoder reads known float tensor storage without importing Torch or executing a model. It reproduced PT/NPZ identity against authenticated historical B17 before auditing these actual B21 artifacts. The actual audit made zero local model forwards and zero optimizer updates.

```bash
python -IB verify_saved.py B21_WEIGHTS_ORIGINAL.zip \
  --producer-zip B21_PRODUCER_ORIGINAL.zip \
  --receipt API_RECEIPTS.json --git-tree GIT_TREE.json \
  --b17-archive B17_TRAIN_ORIGINAL.zip --output REAUDIT.json
```

The original B17 reference archive must have SHA256 `1e31bd9c801657c7850fa6fb05c5468cbe8b4ab5bb459b0146c07c4c905ff86d` and length 2,506,252 bytes. Binary weight/teacher arrays and original ZIPs are not duplicated in this evidence folder; reproduction requires the original artifacts matching the recorded pins. No source is relabeled, no gate is relaxed, and no independent quality result is inferred from this training evidence.
