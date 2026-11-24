# B20 frozen B17 primary assessment — no promotion

The first and only registered B20 primary assessment, [run 45337450827](https://github.com/jiying2007/kws-data/actions/runs/45337450827), completed successfully as an execution. Its numerical quality decision is **NO_PROMOTION**. The installed integration baseline remains **B0**, and no B17/B20 challenger has been released.

The actual execution head is `1c03eb9da468c7e267ff32bb4a34f4ee17bd00b9`, with reviewed prepared parent `474f7bf00ccf722cf52b6140847bcef78c5a82a3`. The parent change was merged through [PR #347](https://github.com/jiying2007/kws-data/pull/347), following [PR #346](https://github.com/jiying2007/kws-data/pull/346). The execution commit changes only its registered one-shot workflow. `RUN_FINAL.json`, `COMMIT_API.json`, `JOBS_FINAL.json`, `ARTIFACTS_FINAL.json` and the original `FETCH.json` retain the actual provenance.

## Original source population and actual work

The synthetic source [run 45334413855](https://github.com/jiying2007/kws-data/actions/runs/45334413855) retained 96 original slots and 480 valid native observer returns. Its unchanged source rules admitted 15 positive originals and 48 ordinary originals, with 12 groups containing both classes; 33 synthetic originals remained UNKNOWN. The fixed B17 natural panel contributed 24 original slots, of which 21 ordinary recordings from 11 readers were admitted and three remained UNKNOWN. No natural source was generated or re-observed for B20.

The 120 original source rows therefore produced **84 admitted originals**, scored at three fixed gains by four frozen roles: B0, B9, B17 control and B17 challenger. All **1,008** registered forward attempts returned complete outputs. The original artifact retains 1,008 raw output arrays, 1,008 saved frame arrays and 1,008 write-before-forward attempt journals. Each role has 252 complete forwards, and there are zero unresolved consumed/unknown attempts. The forward count is actual retained evidence, not a workflow estimate.

The B17 checkpoints and the B14 numerical decision rule remained unchanged. No training, optimizer updates, threshold selection, source replacement, inference replay or additional model call was performed by this assessment or its saved-evidence audits.

## Numerical outcome

Each table cell shows the original whole-input event count at gains **1.0 / 0.75 / 0.5**. Counts refer to this short development panel and are not a continuous FAR/h estimate.

| Cohort | Originals | B0 | B9 | B17 control | B17 challenger |
| --- | ---: | --- | --- | --- | --- |
| Synthetic positives | 15 | 12 / 10 / 5 | 9 / 11 / 8 | 11 / 11 / 6 | 12 / 11 / 5 |
| Synthetic ordinary speech | 48 | 2 / 1 / 0 | 6 / 5 / 2 | 3 / 1 / 0 | 3 / 1 / 0 |
| Natural ordinary speech | 21 | 5 / 6 / 6 | 16 / 26 / 20 | 4 / 6 / 6 | 5 / 7 / 6 |

The challenger failed the unchanged numerical gate for these recorded reasons:

- It lost `b20-primary-798-mero-slow` at gain 0.5 relative to the paired control.
- It added ordinary-speech events on `b17-natural-6829-68769-0007` at 0.5, `b17-natural-1221-135766-0013` at 0.75, and `b17-natural-4507-16021-0036` at 1.0 relative to the paired control.
- Natural ordinary events at gain 0.75 exceeded B0: seven versus six.
- The four original/gain cases where B9 had an event and the control had none spanned three source groups; the challenger recovered none. The required positive recovery was absent.

There were no extra positive events. Eight natural reader groups improved relative to the more sensitive B9 reference, but that did not satisfy the other requirements. `RESULT.json` preserves the full original per-source result, and `SAVED_LOGIT_DIAGNOSTIC.json` describes the four failing conditions from saved logits. These descriptions are retrospective diagnostics only and cannot supply training examples, threshold choices or a replacement candidate selection on this already exposed panel.

## Independent source and producer authentication

`SOURCE_AND_PRODUCER_AUDIT.json` and `SOURCE_AUTHORITY.json` reconstruct the admission from the original source/native artifacts before the numerical result is read. All **17** pinned input ZIPs were checked against their artifact ID, run ID, head, exact byte length and SHA256. The audit reopens all original sources, including the UNKNOWN rows, checks the original B19 proof and historical source receipt, checks the frozen model member identities, and reproduces both original synthetic and natural screening decisions. It does not re-download or re-open every historical archive body referenced by the original history receipt.

The reconstructed source population contains exactly 120 originals, the reconstructed admitted population exactly 84, and its complete admission document equals the original `ADMISSION.json`. The separately uploaded consumed marker binds that exact admission. The original fetching receipt records one assessment run and no previous consumed marker.

The **original producer ZIP is incomplete**: it has 310 files and omitted 32 `.github` files. This fact is preserved. The missing files are retained under `producer_git_supplement/`, separate from the original artifact, with exact original commit/tree identity, Git blob SHA, length and SHA256. `PRIMARY_ACTUAL_TREE_API.json` and `SOURCE_ACTUAL_TREE_API.json` bind their contents to the real producer commits. The supplement fills exactly those 32 declared absences in a temporary audit overlay, with no replacement of any original member. The combined static producer closure has 342 files. No artifact was rewritten, and the supplement is never presented as having been included in the original upload.

This authenticates the complete declared static producer closure. It does not claim that a static inventory proves every possible dynamic dependency in arbitrary future code.

## Independent numerical readback

`ARRAY_AUDIT.json` reads every original raw array, saved frame array and attempt journal using Python's standard library. It derives greedy tokens, complete wake events and event sample positions, checks source/model/gain identities, recomputes aggregate counts, and calls the unchanged original B14 quality function. The complete result matches the official **NO_PROMOTION** outcome. The unchanged success-only result reader rejects the same artifact; the audit grants no opening permission.

`ROOT_NUMPY_OUTPUT_READBACK.json` is the earlier independent NumPy result readback. It deliberately made no claim about producer completeness or a complete native-source audit. The later source and standard-library array reports close those separate audit scopes while retaining the original report and its limits.

The official result artifact is `14894229987`, exact ZIP size 5,916,607 bytes, SHA256 `9972b33b74facbf1e9c055eec3820b981b8d0f9c829791e52c17468bb4891eb5`. The saved frame archive SHA256 is `272e69d0f9bfd72aa8a9da606d67db3677991a214dd950df5cd04aa5bcd315cb`. `ARTIFACT_LOCKS.json` records all three original output artifacts. The binary arrays remain in that authenticated Actions artifact; this directory retains the original result and admission, exact locks, runnable audits and descriptive reports.

## Closed branches and release status

**B20 replication is permanently closed for this failed primary result.** The reserved synthetic indices 822–845 and the new train-clean-100 natural reader reserve remain unopened. Earlier sealed reserves and CAL/LOCKED sources remain untouched. Candidate C11/ARM engineering validation is not authorized by this result, and no candidate is installed in place of B0.

A later candidate requires a separately registered TRAIN-only intervention and prospective independent evaluation. This record cannot turn a failed primary into a successful one by changing its threshold, quality rule, denominator or source population.

Physical SSC305 testing, real 3–5 m field FRR and continuous natural FAR/h were not measured here. `OUTCOME.json` is the concise closed-state handoff; its promotion, replication, candidate engineering and product release fields remain false.

## Reproduce without inference

With the original output ZIPs and 17 original input ZIPs already available in the saved `b20_actual_primary_readback` layout, run:

```bash
python3 -SIB research/english_v6_b20_results_20261124/verify_sources_with_git_supplement.py \
  /absolute/path/to/b20_actual_primary_readback

python3 -SIB research/english_v6_b20_results_20261124/verify_arrays_and_diagnostics.py \
  /absolute/path/to/b20_actual_primary_readback \
  --source-authority-sha 8663bbb903b56aaf37ef87e67815d979a3b7146ba1216add33b5cf4d001ba78b
```

The source auditor uses the included saved-only adapter and explicitly blocks model, observer, source-generator and network calls while reconstructing source decisions. The array auditor uses no NumPy, Torch, audio decoder or model runtime. Both perform zero new model/source/observer calls. `FILE_SHA256.json` inventories every retained evidence file except itself.
