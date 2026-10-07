# Cosy49 six-recording follow-up: negative result

Cosy49 step300 was not adopted, and its candidate study was terminated. It lost the original A20 model's correct K1 event on Z5 and added a wrong K1 event on Z6. Original A20 remains the research baseline, but is also insufficient: both models missed Z4 K2 and falsely activated K2 on Z2. The draft pull request remains open for evidence archiving.

This small supplement is separate from the frozen historical Cosy49 archive. It changes neither that archive's 26 chunks nor its 150-file logical restoration contract. The historical exposed-98 gains and three availability delays remain historical observations; they do not establish qualification. The earlier fixed300 candidate also failed, was not adopted, and its candidate study was terminated.

## Six observations

| Alias | Actual human label | Expected wake | Original A20 | Cosy49 step300 |
|---|---|---|---|---|
| Z1 | 你好 | None | None | None |
| Z2 | 小屋小屋 | None | Wrong K2, 1.84 s | Wrong K2, 1.84 s |
| Z3 | 小挖 | None | None | None |
| Z4 | 小窝小窝 | K2 | Miss | Miss |
| Z5 | 你好小窝 | K1 | K1, 1.48 s | Miss |
| Z6 | 你好小屋 | None | None | Wrong K1, 1.50 s |

The original detected 1 of 2 actual wake recordings; the candidate detected 0 of 2. Among 4 actual nonwake recordings, the original falsely activated on 1 and the candidate on 2. Neither arm repeated an event. These are descriptive counts, not FAR, FRR, confidence intervals, or population estimates. Each arm emitted two events overall, which alone hides the regression.

Z3 is literally 小挖, with 挖 outside the fixed vocabulary. Its text edit/exact metrics stay null; only keyword events are interpretable. It is not normalized to 小窝 or assigned a forced CTC/blank target. No human uncertainty was reported. Generation plans differed from actual hearing for Z1 (planned 你好你好) and Z3 (planned 小窝); event truth uses the actual labels.

Human labels were frozen before predictions. Models, native libraries, and decoder settings were frozen before these recordings were opened. The single comparison used each original recording once per arm, in order Z5, Z4, Z1, Z3, Z6, Z2, original arm then candidate arm, without retries, warmup, sweeps, or rescue tuning. All six are now permanently exposed and excluded from future training, development, tuning, and model selection. They cannot be reused as a fresh holdout.

These are six synthetic recordings from one historically exposed stock voice/generator and phrase family. They provide no new-speaker or population qualification. Greedy text in RESULTS.json is an argmax/collapse diagnostic from saved logits, not ASR, a new decoder execution, or proof of the cause of a beam decision. Event time is available input audio at callback, not a measured speech endpoint or detection latency.

## Evidence and verification

The two JSONL files preserve every clip-scoped source line byte for byte, in original order, including starts, feeds, callbacks, finishes, ends, all events, and all logits. Each has 100 records, 41 callbacks, 378 model rows, and 2268 logit scalars. The original searched 376 decoder rows; the candidate searched 378. Early decoder termination after activation does not omit the full input or model computation.

MANIFEST.json binds the complete original raw hashes to the projected public hashes, byte sizes, counts, and the three omitted global record kinds. Global process/load/end records are excluded uniformly. ACTUAL-LABELS.json is an explicit technical-field projection of the frozen labels. No audio or executable/model payload is included. The manifest also identifies the full source-label hash and source model/configuration hashes.

Run `python validate.py` to check the public file hashes, raw geometry, labels, and every event against the saved summary. A custodian holding the complete original raw files can additionally run `python validate.py --original-dir SOURCE_DIRECTORY --labels-source SOURCE_LABELS.json` to verify source hashes, exact line selection, and label projection. These checks only read saved bytes; they perform no model, decoder, audio, or network calls. The manifest's own hash must be checked against its separately reviewed publication pin.

Resource values in RESULTS.json are observations from one ordered pair, with separate native and whole-collector timing scopes. Native maximum RSS is a lifetime counter; sparse current-RSS samples have a different scope. Two live samples per arm do not prove continuous lifetime compliance, descendant absence was not proven, and terminal I/O remains unknown. No speedup, cold-cache, board-resource, or soak claim follows.
