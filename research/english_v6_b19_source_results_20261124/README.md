# B19: bounded source-format feasibility, completed

Original public run **45330168237**, head
`75bd1c05c220ae2e2334fd5c1e06fa87c66630a5`, was the first push attempt
directly after reviewed parent `771a322cee80bb88210cc1c449fd4b79b403c00a`.
The source, Parakeet observer and saved-evidence readback jobs all completed.

The fixed eight new readers produced all **48 whole originals**. All 32
positive originals met the preregistered duration bounds: the actual range was
8,360–19,877 samples at 16 kHz. There were 48 public Piper synthesis requests
and 48 retained native output chunks. Internal neural/ONNX forward counts were
not instrumented and remain null.

All **48 Parakeet attempts returned valid nonempty output**. Baseline,
prospective and ordinary arms each had 16 valid results from 16 originals.
The original `RESULT.json` and `FEASIBILITY.json` are byte-identical, SHA256
`48e10f040b1a631aeee5a335aa8fbe37d1a01fc547550b15507f4aded3a42700`.
Independent strict raw readback reproduces the fixed verdict:
`SOURCE_FORMAT_FEASIBLE_REQUIRES_NEW_PREREGISTRATION`.

**The baseline also passed all 16 originals. No causal repair of B17 or B18
has been established.** Their original STOP outcomes remain unchanged. This
diagnostic supplies no source labels and has no training, KWS evaluation,
replication-opening or promotion authority. Its eight planned canonical
readers and all 48 generated PCM remain permanently excluded from later
training and evaluation sources. A subsequent 1.4/1.8 source experiment is a
new source distribution and requires its own complete preregistration and all
five valid native observers per original under the unchanged admission rule.

## Evidence

`ARTIFACT_LOCKS.json` records the exact three original artifact tuples.
`SOURCE_READBACK.json` preserves complete canary identities and PCM hashes;
`NATIVE_READBACK.json` records the independent saved-only native verification.
The original source `MANIFEST.json`, `HISTORY.json`, `AUTHORIZATION.json`,
`FEASIBILITY.json`, provider ledgers and byte-closure receipt are retained.
The evidence folder includes final run/jobs/artifact/commit API responses and
the six original producer files fetched at the actual run head. The activated
workflow differs from the reviewed template only in its two parent values.

The source readback additionally checked the previously exposed full
`train_context24` originals. Their 24 PCM and 12 group identifiers had not been
included in the older frozen history inventories; their 12 canonical readers
were already excluded. No canary source overlaps them. The explicit B20
history supplement preserves old inventories and adds these originals rather
than rewriting historical evidence.

## Reproduce without any model call

From the project workspace, with the original ZIPs in
`b19_actual_readback/artifacts/` and the unchanged implementation in `kws-data`:

```bash
python -SIB b19_source_evidence_work/research/english_v6_b19_source_results_20261124/verify_saved.py \
  b19_actual_readback kws-data
```

The verifier authenticates all original ZIP bodies, original producer code and
run metadata, then reads full saved source and native evidence. Renderer and
ASR model loading are explicitly disabled. It checks its result against the
retained independent report. No download, regeneration, repeat observation or
KWS inference is performed.
