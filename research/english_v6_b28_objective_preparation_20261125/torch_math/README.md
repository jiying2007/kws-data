# Pure Torch mathematics test draft — not executed

This directory is separate from the fixed objective preparation directory and
from the immutable HF metadata evidence. It copies the original objective and
supporting source/report bytes exactly. No original was edited. This is a test
draft for a possible later public CI run, with **no candidate registration,
training, source evaluation, or promotion authority**.

## Current authoring status

- Torch has not been imported or run during preparation of this draft.
- The only local checks are original-file byte/SHA verification and Python AST
  syntax inspection of the draft. These do not establish a Torch test PASS.
- There is no actual `CI_MATH_RESULT.json` yet. The future runner creates that
  file only when explicitly invoked.
- The complete fit, old synthetic/positive loss function, model/teacher bytes,
  and all existing source/control protocols remain unchanged.

The original Torch source SHA256 is:

```text
1321eef19cf02d8ce8009e307d4fc384eeac7f59a57e2e07c05dbf79d273350a
```

The already executed stdlib report remains the source of the full length-0–9
349,525-path semantics result. This draft does not repeat that exhaustive
census. It uses small weighted path sets as an independent numerical oracle.

## Proposed checks

`test_torch_objective_math.py` contains ten test methods:

1. Frozen actual decoder AST + independent path enumeration for values and
   conditional-occupancy gradients at short moderate sequences.
2. CPU float64 `torch.autograd.gradcheck`, scalar and batched output shapes.
3. Independent central differences of the enumerated Python objective.
4. 120-digit Decimal extreme-value references, including a representable tiny
   nonzero loss/gradient, near-certain event probabilities, and explicitly
   expected underflow below the float64 range.
5. T=0/1/2 exact zero connected to the input graph, unbatched and batched.
6. Batch independence, per-frame shift invariance, and FP32-input / FP64-loss
   / FP32-input-gradient behavior.
7. A 256-frame invented uniform example checked against integer path counts,
   with finite, zero-row-sum gradients.
8. Rejection of invalid shapes, nonfloating values, NaN, infinities, masked
   Torch input and nonrepresentable path-score ranges.
9. Natural-only wrapper value and gradient checks with **an ordinary callable
   returning supplied invented logits**, not an `nn.Module`. Each invocation
   must call that callable once, preserve original FP32 logp, prefix geometry,
   0.6/0.5/tier factors, and zero gradients in the untouched gap. Independent
   path enumeration and the analytic blank-prefix gradient provide the oracle.
10. Unchanged source pins; AST inspection of the original two shared-loss call
    sites, the sole natural-cache call's arguments, preserved external 0.16
    weight, and the draft wrapper's interface; absence of any complete-fit or
    global shared-loss substitution; key blank/repeat decoder edge cases.

All tensors are invented CPU logits. There are no real weights, PCM, acoustic
features, datasets, model objects, optimizer steps, source panels or observers.
Network operations are guarded before Torch import. Model loading/calling,
optimizer construction, subprocess creation and common model/audio artifact
file opens through both `builtins.open` and `io.open` are additionally guarded
during the mathematical tests. These guards support the reviewed source's
scope; they are not a general operating-system sandbox. Blocked attempts make
the run fail; they cannot be converted to a PASS.

## Future explicit CI invocation

The runtime must be independently provisioned as exact `torch==2.9.1+cpu`,
matching the existing reviewed CPU runtime. This draft does not install any
package or add a workflow. With that runtime already present:

```bash
python3 test_torch_objective_math.py --run-pure-math --report CI_MATH_RESULT.json
```

Without `--run-pure-math`, the program rejects execution before importing
Torch. Output is restricted to a fresh `CI_MATH_RESULT.json` in this independent
draft directory, and bytecode-cache writes are disabled before the frozen
objective is imported. The report differentiates toy autograd and invented-callable calls from
real model/training/source operations, and records actual tests run, failures,
skips, runtime, source pins and numeric observations. A missing or incompatible
runtime is an error, never a skipped green test suite.

If a test exposes a required source fix, retain the original pinned bytes and
report the failure and proposed diff first. Do not overwrite the original
objective, weaken the event definition, relax existing scientific thresholds,
or treat a pure-math PASS as model/quality/promotion evidence.
