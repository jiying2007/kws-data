# B27 engineering and software delivery preparation

This registration is inert. `english_v6_b27_engineering_inputs.json` has
`enabled: false`, null original quality locks, null candidate exports and a
null challenger PT hash. The delivery entry is disabled and has no selected
pipeline commit. No model, source, compiler, pack or installation is invoked by
the preparation or its contracts.

## Required identity and order

The new challenger must be the complete actual B27 TRAIN result, independently
authenticated from all four TRAIN ZIPs, original API evidence, all original
training inputs and the complete original Q1 `NO_PROMOTION`. Its PT identity is
learned only from that actual training lock. The control remains the original
B24 challenger, PT SHA-256
`f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4`.
The old B23 initialization is a different identity and cannot replace it.

Both actual B27 PRIMARY and replica must independently return complete `PASS`
under their frozen assessment. Failure, `STOP`, partial arrays and unknown
consumption keep this stage closed. Both quality reports must bind the same
TRAIN result, pair artifact, model hashes, assessment and design; their full
source inventories must remain independent. The original Q1 result and all
older stop states remain terminal.

| Registration | SHA-256 |
| --- | --- |
| B27 preoptimizer design | `956f0695e6b4e3ac450bc3881c5ef13985d775210cf43cf4da98947a5a1846bf` |
| B27 fresh TRAIN/evaluation assessment | `8c7b4571d2ac08546fd00789fc2809df8c4290ad53a38c1f6b33a993331ae2ee` |
| Original B24 ancestor assessment | `3afdb17e822bb0b4a49757ac116a28101ff9f1447b22429305299047410fe2ad` |
| Original complete Q1 failure report | `9cd5df45c5b9ca22b0eb56f3fdda4788b2420e906a35bb754af9c23eb38be62f` |

After dual quality PASS, the engineer may run exactly the unchanged 14 exposed
originals, two conditions and five backends: 140 actual backend calls, 140
complete raw arrays and 84 original C event CSVs. The original R1 numerical
helpers and 140-call loop are preserved; the new loop differs only in identity
strings. The 13 SDK/DSP sources still come from pipeline commit
`88a043c6c00bb3a1a62f45098a023c7601042ded`. ONNX tolerances remain maximum
absolute error 0.001 and mean absolute error 0.0001; C tolerances remain 0.005
and 0.0005, with every frame token and event sample clock equal.

## Full evidence layout

The engineering runtime inputs contain the three existing exposed-input ZIPs
(`B0`, `engineering_source`, `engineering_screen`), the actual new `pair.zip`,
the complete code/SDK producer, `FETCH.json`, `CONSUMPTION.json`, and:

- `quality/primary/` and `quality/replication/`: each contains exactly the
  original `api/{run,commit,artifacts}.json`, three `originals/{result,producer,consumed}.zip`,
  and the twelve complete `inputs/*.zip` files.
- `proof_roots/primary/training-origin/`: the 41 original files prescribed by
  the frozen PRIMARY proof binding. `trained/` contains `TRAIN_LIVE.json` and
  `TRAIN_{models,producer,receipts,consumed}.zip`; `qualification/no-promotion/`
  contains the exact 18 Q1 originals, and `qualification/training/` contains
  all 18 original TRAIN input ZIPs.
- `proof_roots/replication/training-origin/`: the same complete 41 files,
  reused through regular hardlinks only after every original byte is hashed.
- `proof_roots/replication/primary-quality/`: the same exact 18 original PRIMARY
  files frozen before replica execution, also authenticated before hardlinking.

PRIMARY API bodies are restored from the replica's exact saved original
transport binding. Live API responses are not substituted into that frozen
18-file closure. The full quality readers still reopen and verify every file.
No historical ZIP is nested into any new engineering archive. All four new
engineering archive roles retain the 128,000,000-byte original limit, and the
actual complete producer is conservatively checked before any engineering
budget. The B27 source and external-proof roles keep their separately
registered capacities; no historical role limit is broadened.

## Generate a reviewable engineering input binding

Use the exact code after the B27 quality adapter is frozen and both actual
quality readbacks pass. Every path below is an explicit original evidence root;
`$B27_ENGINEERING_REVIEW` must be a new absolute directory.

```sh
python3 -I -S -B automation/bind_b27_engineering.py inputs \
  --primary "$B27_PRIMARY_ORIGINALS" \
  --primary-lock "$B27_PRIMARY_LOCK" \
  --primary-proof "$B27_PRIMARY_PROOF_ROOT" \
  --replication "$B27_REPLICA_ORIGINALS" \
  --replication-lock "$B27_REPLICA_LOCK" \
  --replication-proof "$B27_REPLICA_PROOF_ROOT" \
  --output "$B27_ENGINEERING_REVIEW"
```

The output includes `engineering_inputs.json`, both complete independently
verified quality reports and `BINDING.json`. It does not edit the real input
registration. After review, committing that exact input document prepares the
single engineering workflow. Activation remains a workflow-only first public
push to `research/en-v6-b27-engineering-once-20261125`, substituting exactly two
`__B27_REVIEWED_ENGINEERING_PARENT__` tokens with its reviewed parent. All
existing contracts must pass before this activation.

## Read back the actual engineering result

Obtain the following original files without repackaging any ZIP:

| File in evidence root | Original source |
| --- | --- |
| `originals/result.zip` | `english-v6-b27-software-sdk-evidence-<run>` |
| `originals/producer.zip` | `english-v6-b27-engineering-producer-<run>` |
| `originals/consumed.zip` | `english-v6-b27-engineering-consumed` |
| `originals/input_proof.zip` | `english-v6-b27-engineering-input-proof-<run>` |
| `api/run.json` | Completed first engineering run API response |
| `api/commit.json`, `api/tree.json` | Activation commit and its complete recursive Git tree |
| `api/artifacts.json` | Complete four-artifact run inventory |
| `api/runs.json`, `api/markers.json` | Exact bytes retained inside the original `input_proof.zip`, not refreshed responses |
| `api/pipeline_commit.json`, `api/pipeline_tree.json` | Frozen SDK commit and its complete recursive tree |
| `inputs/{B0,pair,engineering_source,engineering_screen}.zip` | Exact complete pins from the actual engineering input registration |
| `quality/` and `proof_roots/` | The complete original trees described above |

The independent evidence root differs from the runtime root only by placing
the four exposed/candidate ZIPs in its `inputs/` subdirectory and retaining the
four output ZIPs in `originals/`. The original quality and external proof roots
are retained alongside them.

```sh
python3 -I -S -B automation/bind_b27_engineering.py result \
  "$B27_ENGINEERING_ORIGINALS" "$B27_ENGINEERING_RESULT_REVIEW"
```

This derives `ORIGINAL_LOCK.json` from the actual four ZIPs and eight full API
bodies, then invokes the independent engineering reader. Only a full success
produces `readback/completion.json`, `readback/readback.py` and the result
binding receipt. A failed readback may leave its newly written lock for
inspection; it confers no downstream selection authority.

The readback verifies every array, every CSV, all original exports, serialized
pack tensor bytes, installed SDK bytes, exact Git producer closure and complete
dual-quality/TRAIN/Q1 qualification. Its parent permits only one exact TRAIN
worker and the exact PRIMARY and replica quality workers, each under an
explicit canonical lock and bounded root. Network, model imports and arbitrary
subprocesses remain blocked.

## Prospective readback time budget

No complete proof is replaced by a summary to reduce runtime. Counting the
full TRAIN reader as one complete nested Q1 readback:

| Runtime stage | Complete TRAIN/Q1 reads | B27 panel reads | Time limit |
| --- | ---: | --- | ---: |
| Fetch | 6 | PRIMARY 3, replica 1 | 120 min |
| Preflight | 4 | PRIMARY 2, replica 1 | 90 min |
| Execute, including its independent preflight | 4 | PRIMARY 2, replica 1 | 100 min |

Fetch first verifies PRIMARY proof and quality, then replica proof and quality.
A replica proof includes its own TRAIN and the nested original PRIMARY, which
in turn reopens TRAIN. At the observed approximately eight minutes per complete
Q1 read, Q1 alone accounts for about 48/32/32 minutes respectively; B27 saved
arrays, source audits, hashing, transport and engineering add time. Installation
has a separate 35-minute cap; the job cap is 360 minutes. These are operational
limits only and do not alter the 140-call scientific ledger.

During execute, one explicit primary proof scope spans the whole original
execution. Its preflight must fully verify TRAIN before importing scientific
libraries. The later unchanged `load_candidate` reopens and hashes the same
complete evidence before reusing only the already verified canonical TRAIN
report inside that scope. Exiting the scope destroys the cache.

## Delivery handoff

The matching pipeline runner is `research/english_v6_delivery/once_b27.py`;
`B27.md` there documents the two metadata binder commands. After the independent
engineering readback is reviewed, the pipeline binder prepares a SHA-named
manifest, exact report/verifier copies and a selection draft in a new review
directory. After that exact selection is reviewed and committed, its second
command captures all ten actual Git delivery files and the exact pipeline
commit for the disabled data entry.

The actual delivery copies the candidate pack and installed SDK bytes unchanged,
serializes the original B0 rollback container without another model forward,
installs in a fresh prefix and performs candidate/B0 loader-only checks. It
does not replay audio, initialize a stream, rebuild the SDK or alter an existing
process. Research software installation does not establish physical SSC305,
product promotion, field FRR/FAR or physical latency claims.
