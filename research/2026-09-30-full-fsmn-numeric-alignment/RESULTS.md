# Full donor FSMN numerical alignment: partial components pass, old B fails

Observed local synthetic research, 2026-09-30. No training. The later explicitly authorized PCM behavior diagnosis is reported below; it does not convert the failed original B gate into a pass. Original donor weights and large derived golden/trajectory arrays remain local and are not included here.

## Outcomes

- Feature-level splice/skip/EOF indexing: exact source parity over29 cases/67 calls;8 test groups and ASan/UBSan pass (leak checking unavailable/disabled). This is not yet complete PCM frontend integration
- Strict local Python donor identity loader: fixed canonical payload SHA, full30-tensor schema/hash/layout/finite checks; invalid/rebound fixtures rejected. Standalone C serialized-file hash loader remains unfinished
- CMVN:12 frozen calls exactly equal
- Whole-network old B numerical gate: FAILED for sequential FP32, four-way FP32 and one explicitly diagnostic double accumulation variant; gates unchanged
- One fixed local stage5 check: same-input C versus frozen Torch maximum3.8147e-5, no old B failures; supports upstream propagation as a contributor, without proving the entire network correct
- One approved all-layer oracle/propagation diagnosis:132 layer records and12 call-end cache records pass local theoretical rounding and propagation bounds; exact-value cache/index checks pass. This is numerical-value equality, not a claim to distinguish +0/-0 bit patterns

## Important limit of the new diagnosis

The maximum forward propagation envelope is approximately2.3081e15. Absolute-weight envelope growth makes it uninformative for end-to-end fidelity. A mathematical-consistency PASS must not be described as parity or qualification.

Actual observed final-logit error ranges up to1.10626220703125e-4, with zero final-logit old B element failures across the six routes (absolute+relative criterion). Intermediate stages/cache still fail the original old B criterion:27 failing aggregate records out of132 layer+12 cache records. This aggregation differs from the earlier264 call/stage comparisons, so33/35/29 historical failed-check counts are not directly comparable to27.

Maximum same-input local C/oracle error4.1913444647434524e-5; maximum local C error/bound ratio0.8577414896940483. The fixed current kernel has no accepted end-to-end fidelity status. Whole/split Torch reference itself previously crossed the strict original gate in3/63 layer comparisons; this is reference reduction sensitivity, not permission to widen the gate.

## Reproducibility and incomplete work

Full oracle result SHA256 `ddd7eb6c92ba7f04df4859b1b9ae3627f1f22bcae9acc6231e1d59f12eaee0bc`; preflight SHA256 `9683aa25495828c2c793c8a2541e4a5c36ab4f2a6858a45c53090aed0c4486b8`; retained local arrays SHA256 `b277447239fee5da97c388f7f9257ec85e8a992fa0e6d6f2e34224c76d913e3a`.

Code/components belong in kws-pipeline; these scalar result/provenance records belong in kws-data. Cross-links should be added only after actual commits exist. No new PR has been created for these changes. Do not publish donor payload, full tensor manifest or large derived array bundles. No resource speed claim follows from these correctness diagnostics. Any PCM stage-C test requires separate authorization and unchanged probability/event gates; no speech run was made here.


## Subsequent fixed actual-PCM behavior diagnosis

One reviewed126-clip run, original42 recordings ×raw/tail500ms/head500ms_tail500ms, Hamming, full2599 donor, fixed four-way C kernel, original unchanged Python softmax/keyword decoder. Python reference was frozen first with complete per-call logits/probabilities/stages/indices/availability; every reference recording exactly matched historical baseline logits hashes and events. C PCM composition then ran once. Both reference and native raw arrays remain local (~527MB each), excluded from publication.

- Target hits retained11/20 raw,11/20 tail,12/20 head+tail; confusable-trigger clips0/22 in every condition
- All non-score event fields exactly match, including keyword, times, availability and EOF status
- All34 triggered recording scores differ numerically; maximum score difference2.694541942516171e-6. Scores pass the previously specified1e-5 tolerance, but **full-field exact equality is false**
- Complete probability gate passes:25,529,977 elements, maximum absolute difference8.046627044677734e-6; row sums pass
- Final-logit old formula fails:6126/25,529,977 elements,63 calls across42 recording-condition pairs, maximum absolute difference0.005767822265625
- fbank80 absolute1e-3 gate fails for6/2,368,160 elements (6calls), max0.0012373924255371094; splice has11 failing elements; CMVN fixed2e-4 gate passes (max0.0001863241195678711)
- Cache/intermediate old B failure remains. No arithmetic variants, gate changes or resource profiling followed these results

This supports preservation of these observed decisions and probabilities under the fixed decoder operating point; it is neither strict numerical equivalence nor deployment qualification/FAR evidence. The synthetic frontend bound had limited coverage and did not prove the real-PCM fbank bound universally.

Native result SHA256 `0a4a927610d603b5be3703a0cef880e370dfb61ee367a7457dc3cc9369e9daec`; Python reference receipt SHA256 `1e9c5acd3dc2268ca79189043780ddf1f8674b95bcad6322faa0c03fa6f3ec6d`. Scalar JSONs are stored as deterministic lossless gzip with logical-byte identities in pcm-compression-manifest.json.

The synthetic PCM adapter test's first reference generator shared a mutable upstream namespace between two spotters. Its failed oracle/report remains local. The corrected oracle used independent namespaces and was generated after first synthetic C output, with that chronology explicitly recorded; C arithmetic and gates did not change. Corrected104cases/212calls and7test groups passed normally and with ASan/UBSan (no leak-check claim). This issue did not affect the real126 Python reference: it used one sequential spotter and proved exact agreement to prior historical logits/events.
