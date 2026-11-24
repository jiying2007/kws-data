# B22 original source evidence and text-only continuation

The original public source execution is [45402179335](https://github.com/jiying2007/kws-data/actions/runs/45402179335), attempt 1, at `54c5f514d0aed0bf276927c8b6631245d833167c`. Its failed outcome and every original consumed marker remain preserved in `initial_failure/`.

## Completed and remaining work

| Operation | Original execution | Continuation authorization |
|---|---:|---:|
| Frozen synthetic source generation | 96 complete | 0 new calls |
| Whole natural source extraction | 24 complete | 0 new decodes |
| XLSR native phoneme observations | 120 complete | 0 new calls; exact saved bytes reused |
| LV60 native phoneme observations | 120 complete | 0 new calls; exact saved bytes reused |
| Qwen text observations | 0 attempted | 120 first calls |
| Whisper text observations | 0 attempted | 120 first calls |
| Parakeet text observations | 0 attempted | 120 first calls |
| KWS scoring / optimizer updates | 0 | 0 in this source workflow |

The three text observers stopped during initialization because the captured execution root omitted the fixed `campaign.py` research helper. Inspection of the complete native loader chain also found that `runner.py` was missing. Both files were already present and immutable in the repository; the continuation captures all 19 original helper pins and runs the actual dynamic import routes from the captured root before any observer budget is consumed. The runtime checks stop at the model-asset boundary and make no acoustic calls.

`automation/english_v6_b22_source_origin.py` independently checks the original public run, commit, jobs, logs, artifact inventory, consumed markers, 1,172 producer files, all 120 original WAV/PCM identities, and the two complete sets of saved native phone outputs. The three text ledgers contain no attempted, returned, failed, or unknown input calls. The continuation is a distinct first execution with distinct durable markers and a ceiling of 360 text observations. It cannot generate audio, decode natural audio, run phone models, train, or score the KWS model.

## Evidence identities and unchanged admission

The continuation retains the entire original source directory byte for byte and adds explicit continuation and origin evidence. It records the original audio-generation identity separately from the actual continuation identity. The two reused phone directories retain their original run identity, while newly observed text records use the actual continuation run. Readback requires those exact identities, complete native returns, the original parsers and unchanged five-observer admission rules.

The source minima remain 12 synthetic positives, 24 synthetic ordinary originals, 8 paired synthetic reader groups, 12 natural ordinary originals, and 8 natural readers. Unknown or rejected originals remain in the preserved source inventory. All 120 originals remain excluded from future independent panels.

No completed source result or promotion is claimed by this preparation. PRIMARY remains closed until the actual complete continuation source passes its frozen admission gate and the original artifacts are bound. Replication remains sealed until original PRIMARY PASS. Engineering validation requires both original quality panels to pass with the same weights. B0 remains the software integration baseline.
