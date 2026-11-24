# B21 actual PRIMARY: independently reproduced NO_PROMOTION

The [first B21 PRIMARY execution](https://github.com/jiying2007/kws-data/actions/runs/45387856667) completed successfully at `6b8cc1b559dab3a9622283b36c737a61380ee4b2`, with the sole reviewed preparation parent `1059a5e70098934ac4261c98d6eed8fa213c8338`. Its scientific decision is **NO_PROMOTION**. Workflow success does not change that result. The reserved replication remains unopened, engineering remains disabled, and B0 remains the installed integration baseline.

## What actually ran and was independently read back

The source remains the independently screened B21 panel from [run 45378709327](https://github.com/jiying2007/kws-data/actions/runs/45378709327). All 120 original source identities and all 600 original native observer returns were reauthenticated. The same 90 qualified originals comprise 22 synthetic positives, 48 synthetic ordinary utterances and 20 natural ordinary utterances from 12 readers. The 30 unknown originals remain in the full provenance and exclusion population. No source was regenerated or relabeled during PRIMARY or readback.

The four frozen models each ran 270 times: 90 original utterances at gains `1.0`, `0.75` and `0.5`. The independent readback checked **1,080 complete saved arrays and their 1,080 actual raw tensor returns, covering 171,732 frames**. Every original entry journal, source ID, PCM hash, array shape, finite FP32 byte, argmax class, collapsed token sequence, event frame and sample-availability clock matched. The frozen B14 decision was recomputed from independently decoded arrays, producing the same complete result. No model, TTS or observer call was repeated.

| Observed count | B0 | B9 | Matched control | B21 challenger |
| --- | ---: | ---: | ---: | ---: |
| Positive originals with a wake, gain 1.0 (of 22) | 20 | 20 | 19 | 19 |
| Positive originals with a wake, gain 0.75 (of 22) | 19 | 19 | 19 | 19 |
| Positive originals with a wake, gain 0.5 (of 22) | 7 | 13 | 10 | 10 |
| Natural ordinary events, gain 1.0 (20 originals) | 9 | 20 | 7 | 7 |
| Natural ordinary events, gain 0.75 (20 originals) | 9 | 19 | 5 | 6 |
| Natural ordinary events, gain 0.5 (20 originals) | 7 | 19 | 5 | 6 |
| Synthetic ordinary events, gain 1.0 (48 originals) | 2 | 8 | 1 | 1 |
| Synthetic ordinary events, gain 0.75 (48 originals) | 0 | 4 | 0 | 0 |
| Synthetic ordinary events, gain 0.5 (48 originals) | 0 | 2 | 0 | 0 |

These are original-utterance event counts on this fixed research panel. They are not hours-normalized field FAR or real 3–5 m FRR.

## Why the unchanged primary decision failed

1. **No recovered positive source family.** The registered minimum is one family where B9 wakes, the paired control misses, and the challenger restores the wake. The actual count is zero.
2. **One protected B0 positive was lost.** On `b21-primary-854-mero-slow` at gain 1.0, B0 and B9 each produce one wake; the paired control and challenger produce none, both retaining only token `[1]`.
3. **Three original/gain ordinary conditions regress.** There are four reference comparisons: `b21-primary-natural-8324-286683-0029` at 0.75 adds one event versus the control; `b21-primary-natural-8226-274371-0020` at 0.5 adds one event versus both B9 and the control; `b21-primary-natural-405-130895-0033` at 0.75 adds one event versus B9.

The challenger improves the natural event count against B9 in seven readers at gain 1.0, exceeding the required two. Its natural event totals at each gain do not exceed B0, and it adds no extra positive events. Those passed conditions do not override the three failed conditions above. The challenger and control also have identical positive event totals at every gain.

## Original producer omission and separate retrospective proof

The actual PRIMARY output artifact is complete, but its producer artifact is **not sufficient for isolated source-provenance replay**. The original declared inventories contain 278 primary files and 274 source files. Their bytes and Git identities are valid. The first strict isolated audit nevertheless fails inside the source owner's original ancestry verification because `automation/english_v5_mero43_trainonly_robust_challenger_once.py` is absent.

The source owner's deeper checks compare every file listed in an earlier original producer manifest. Those dynamic manifest dependencies were not all included by the PRIMARY producer's static Python closure. Execution succeeded with the full repository checkout; the restricted uploaded producer did not retain all the bytes needed for an isolated provenance readback. `ORIGINAL_STRICT_READBACK_FAILURE.json` preserves this failure. The production PASS-only quality reader and the original ZIPs were not modified.

A separate retrospective proof authenticates all 314 files in the earlier original source producer artifact `14912996481`, including their manifest sizes, SHA256 values and original source Git blobs. Forty Python dependencies absent from the primary producer are retained under `producer_git_supplement/`. Every one also matches the actual PRIMARY Git tree, so none is a new implementation or a changed scientific rule. The source execution workflow is the additional member in the 41-path difference; it is already present under the original artifact's separate `source_producer` directory and is not copied as a code supplement.

The supplemented audit builds a separate temporary diagnostic directory and repeats the original source admission there. It then independently verifies every saved output and computes **NO_PROMOTION**. It explicitly records `original_producer_archive_complete=false`, identifies the 40 additional files, and grants no permission. This does not repair or relabel the historical producer artifact as complete. A later experiment must close dynamic manifest dependencies prospectively in its own producer registration.

## Immutable evidence

| Original | Artifact ID | Bytes | SHA256 |
| --- | ---: | ---: | --- |
| PRIMARY full outputs | 14917736688 | 5,967,862 | `1be5f7aa27d4b2331c076a98e87e8a50ab51483f71cd08cfdd2b172fea0cefa7` |
| PRIMARY producer | 14917337166 | 18,024,132 | `ff408a5a1874ea4da1d953abd037daee74873a6b59c7a890d67e4b844727d2fa` |
| PRIMARY consumed marker | 14917757566 | 384 | `5c22313474ecc600e629d550e29003e75d50dacc7dfd23d1e0161537b55c0c2b` |
| Earlier original source producer | 14912996481 | 9,028,521 | `4cc1e427c26d771b8e75194b2d5bbd0b5399423f632b54f6ef6e7e583b1bc957` |

- Original `RESULT.json`: `f34493166b2ac21248670a7d58bac408b1a564bfc30891c950666dc7dc5cbd87`.
- Original full logits NPZ: `d2b96d72d4d8795c9bb5139234f67915aefe5285ba2b34c2a6f875d2bfd945f6`.
- Independent supplemented source/output audit: `483be35deb1634bb2d60ffb5c6b539d7b7708cf96e2177b6eec75d5fd275d415`.
- Separate supplement manifest: `c1b7036ff4e96ce83b33db2b9932bb156ce299bfe13907834e2458b7bae72a50`.

The original `FETCH.json` retains the complete actual source and PRIMARY Git trees, original input metadata, one-time run inventory and original consumption state. The bound input registration retains all ten scientific input artifact pins. `FILE_SHA256.json` covers the durable files in this evidence directory. Original output arrays remain identified by their original artifact and NPZ digests; no derived scores replace them.

## Reproduce the readback without inference

Collect the three original PRIMARY ZIPs into `originals/{result,producer,consumed}.zip`; place final run, full commit and artifact metadata in `api/{run,commit,artifacts}.json`; place the ten original scientific ZIPs under `inputs/<role>.zip` using the bound input registration. Keep the earlier original source producer ZIP separately.

The strict script deliberately retains the historical failure. The separate observation script can reproduce the completed NO_PROMOTION record:

```bash
python3 -SIB original_auditors/independent_primary_supplemented_observation.py verify \
  /absolute/path/to/original-evidence ARTIFACT_LOCKS.json \
  --reader audit_helpers/english_v6_b21_fixed_mirror_quality_readback.py \
  --source-producer /absolute/path/to/original-source-producer.zip \
  --source-producer-metadata SOURCE_PRODUCER_ORIGINAL_ARTIFACT.json \
  --supplement-output /absolute/path/to/new-supplement-directory \
  --output /absolute/path/to/new-independent-observation.json
```

Both output paths must be new. The script uses only the standard library, rejects network/acoustic runtime imports and writes only diagnostic evidence. Its seven FP32/CTC fixtures pass; separate peer review verified the original source producer and the two-tree supplement. All model/source/observer call counters remain zero during readback. No replication, CAL/LOCKED, candidate engineering, physical SSC305 or product release claim is made.
