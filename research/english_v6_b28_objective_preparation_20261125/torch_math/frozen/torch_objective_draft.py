"""UNEXECUTED Torch source draft; preparation only, not a training integration.

No Torch import or model runs happen on import. A caller would explicitly pass
the original torch/F modules. The validation script parses this file but does
not import it, construct tensors, call a model, or run Torch/autograd.

Only the natural_ordinary loop may use natural_no_event_suffix_loss. The shared
synthetic/positive prefix_peak_suffix_loss and complete fit remain untouched.
The outer natural weight 0.16 and original gain average stay at the fit caller.
"""
from __future__ import annotations


# Kept explicit here so the proposed Torch recurrence can be reviewed without
# executing or importing another objective implementation.
_A, _B, _C, _D, _HIT = range(5)
_TRANSITIONS = (
    (_A, _B, _A, _A),
    (_B, _B, _C, _A),
    (_D, _B, _C, _HIT),
    (_D, _B, _A, _HIT),
)


def _combine(terms, torch):
    # None denotes structural unreachability, not a tensor filled with -inf.
    # All entries in a nonempty list are finite by the input/path range checks.
    if not terms:
        return None
    if len(terms) == 1:
        return terms[0]
    return torch.logsumexp(torch.stack(terms, dim=0), dim=0)


def no_event_nll_torch(logits, torch):
    """Proposed NLL for finite floating logits [T,4] or [B,T,4].

    Return a scalar for [T,4] or a [B] tensor for [B,T,4]. Only this new
    objective converts its input to float64. Masked -inf inputs are rejected;
    deterministic masks are exercised in the separate stdlib reference.

    Values whose finite path log scores cannot be represented in float64 are
    rejected, never clamped. Normal finite FP32 model outputs are far inside
    this representability bound at the frozen training lengths.
    """
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise TypeError("floating Torch logits are required")
    if logits.ndim not in (2, 3) or logits.shape[-1] != 4:
        raise ValueError("logits must have shape [T,4] or [B,T,4]")
    scalar_result = logits.ndim == 2
    work = logits.unsqueeze(0) if scalar_result else logits
    if work.shape[0] < 1:
        raise ValueError("an empty batch is not supported")
    work = work.to(dtype=torch.float64)
    if not bool(torch.isfinite(work).all()):
        raise ValueError("only finite logits are accepted")
    # Empty slicing preserves a graph connection without multiplication by
    # zero, so even a large finite input cannot produce inf * 0 -> NaN.
    zero = work[:, :0, :].sum(dim=(-2, -1))
    frames = work.shape[1]
    if frames < 3:
        return zero[0] if scalar_result else zero

    logp = work.log_softmax(-1)
    if not bool(torch.isfinite(logp).all()):
        raise ValueError("finite float64 log probabilities are required")
    minimum = -torch.finfo(torch.float64).max / (frames + 1)
    if not bool((logp >= minimum).all()):
        raise ValueError("complete path log masses must remain representable")
    live, hit = (zero, None, None, None), None
    for frame in range(frames):
        row = logp[:, frame, :]
        buckets = [[] for _ in range(5)]
        if hit is not None:
            # Current rows are finite: this cannot be all -inf logsumexp.
            buckets[_HIT].append(hit + torch.logsumexp(row, dim=-1))
        for state, mass in enumerate(live):
            if mass is None:
                continue
            for token in range(4):
                buckets[_TRANSITIONS[state][token]].append(mass + row[:, token])
        combined = tuple(_combine(terms, torch) for terms in buckets)
        live, hit = combined[:4], combined[_HIT]
    log_z = _combine([value for value in live if value is not None], torch)
    if hit is None:
        # A structural guard, also retaining the graph for future extensions.
        result = zero
    else:
        # softplus(H-Z), without subtraction from one or any epsilon/clamp.
        result = torch.logaddexp(torch.zeros_like(log_z), hit - log_z)
    return result[0] if scalar_result else result


def no_event_log_probability_torch(logits, torch):
    return -no_event_nll_torch(logits, torch)


def natural_no_event_suffix_loss(model, x, prefix, tier, torch, F,
                                 return_logp=False):
    """Draft replacement ONLY for a natural_ordinary loss call in B23.fit.

    Exactly one existing model(x), with original FP32 logp for the old prefix,
    teacher and saved-return interfaces. The suffix starts from DFA state A;
    events spanning the artificial prefix/suffix boundary are outside this
    term, as required by the stated suffix-only hypothesis.
    """
    raw = model(x)
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[-1] != 4:
        raise ValueError("the frozen single-example [1,T,4] forward is required")
    if raw.dtype != torch.float32:
        raise ValueError("the original FP32 forward must remain unchanged")
    logp = raw.log_softmax(-1)
    if not bool(torch.isfinite(logp).all()):
        raise ValueError("the original FP32 logp must be finite")
    start = prefix // 160 - 3
    blank = prefix // 160 - 8
    if not 0 < blank < start < logp.shape[1] - 5:
        raise ValueError("IMMUTABLE_B3_SUFFIX_CTC_START")

    suffix = no_event_nll_torch(raw[:, start:, :], torch)[0]
    silent = logp[:, :blank, :]
    blank_margin = torch.logsumexp(silent[..., 1:], dim=-1) - silent[..., 0]
    loss = 0.6 * suffix + 0.5 * F.softplus(blank_margin).amax()
    weighted = loss * (1.0 if tier == "silver" else 0.75)
    return (weighted, logp) if return_logp else weighted
