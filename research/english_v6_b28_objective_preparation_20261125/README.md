# B28 no-event objective preparation draft

This directory is a disposable, local, TRAIN-only mathematical prototype.
It does not register a candidate, modify a ref/PR, alter the complete fit, run
Torch, or authorize any experiment. B27 remains the sole active candidate.
It can be discarded if B27 passes. No B27 SOURCE or PRIMARY result was read.

## Files

- `no_event_reference.py`: independent ordinary-scale probability recurrence,
  four-live-state plus absorbing-HIT log-domain DP, and path predicate.
- `torch_objective_draft.py`: **unexecuted** Torch source, with an explicit
  natural-only wrapper. The verifier parses this source without importing it.
- `verify_stdlib.py`: actual frozen decoder AST oracle, exhaustive paths,
  weighted enumeration, Decimal extreme-value checks, conditional-occupancy
  gradient mathematics, rejection cases, and Torch source structure checks.
- `FROZEN_SOURCE_MANIFEST.json` and `frozen/`: exact source bytes copied from
  Git object `b37932fbbf8579b7efb3f4789acf3f11ca495e87`, including the original
  B27-to-B24Q1 decoder inheritance and the original B23 natural loss interface.
- `STDLIB_VALIDATION.json`: generated only after the standard-library
  verification succeeds, including all code/source pins and zero-call ledger.

Run the verification with Python 3.12, without any third-party package:

```bash
python3 /tmp/b28-event-objective-preparation/verify_stdlib.py \
  --report /tmp/b28-event-objective-preparation/STDLIB_VALIDATION.json
```

The only executed function from a frozen scientific source is the isolated
`ctc_events` AST. Its module and dependencies are never imported. The function
operates on invented integer paths, not model outputs or audio.

## Objective and exact event semantics

For independent frame categorical probabilities, the proposed suffix term is
`-log P(CTC collapse contains no contiguous 1,2,3)`.

CTC first merges adjacent **raw** repeats, then removes blank 0. The live DFA
needs four states because a blank after raw 2 changes how the next raw 2 is
collapsed. In particular, `[1,2,0,2,3]` collapses to `[1,2,2,3]` and has no event.

| State | raw 0 | raw 1 | raw 2 | raw 3 |
|---|---|---|---|---|
| A: no matching suffix | A | B | A | A |
| B: suffix 1 | B | B | C | A |
| C: suffix 12, previous raw 2 | D | B | C | HIT |
| D: suffix 12, previous raw blank | D | B | A | HIT |
| HIT: an event has occurred | HIT | HIT | HIT | HIT |

The log DP retains both no-event mass Z and hit mass H. Its final NLL is
`softplus(log H - log Z)`, equivalent to `-log[Z/(Z+H)]`. Old HIT mass is
multiplied by each row's actual summed probability mass, preserving consistent
normalization rounding. It never subtracts a probability from one or uses an
epsilon/clamp. Unreachable Torch states use `None`, and only nonempty, finite
lists enter `logsumexp`. Input/path representability guards reject values
that cannot be represented instead of silently introducing infinities.

For fewer than three frames the mathematical NLL is exactly zero. The Torch
source returns an empty-slice sum connected to the input graph, without
`input * 0`. This graph property is a source-level design claim; no Torch test
has run. Final NLL values smaller than the float64 subnormal range can still
underflow to zero; the report explicitly tests and describes that limitation.

## Proposed natural-only fit interface

The existing shared `prefix_peak_suffix_loss` is also used for synthetic
ordinary and synthetic positive examples. Replacing that shared function
globally would change more than the stated hypothesis.

The draft `natural_no_event_suffix_loss(model, x, prefix, tier, torch, F,
return_logp=False)` is intended only for the existing `natural_ordinary` loop:

1. Execute the one existing `model(x)` and preserve FP32 `raw.log_softmax(-1)`.
2. Preserve `start = prefix // 160 - 3` and `blank = prefix // 160 - 8`.
3. Compute only the new suffix DP from `raw[:, start:, :]`, converted to
   float64 before its separate log-softmax.
4. Preserve `0.6 * suffix + 0.5 * softplus(prefix blank margin).amax()` and the
   original tier factor `1.0 if silver else 0.75`.
5. Return the original FP32 logp for existing teacher/trace callers. Preserve
   the outer 0.16 natural weight and original gain averaging at the fit caller.

No complete fit has been copied, patched, or executed. No extra model call is
introduced by the proposed interface. The suffix starts in A; this objective
does not penalize events spanning the artificial prefix/suffix boundary.
It concerns event-free *path probability*, not greedy event rate or expected
event count, and it permits nonblank outputs that avoid contiguous 1,2,3.

Before any possible registration, independent Torch numerical/autograd and
integration checks, a new natural-objective trace/readback schema, complete
proof closure, prospective source metadata, and the actual B27 terminal gate
would still be required. This draft establishes none of those authorizations.
