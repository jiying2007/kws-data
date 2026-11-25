"""Preparation-only CTC no-event mathematics, using the Python standard library.

There is no model, audio, optimizer, observer, candidate registration, or Torch
import here. Blank is 0; an event is contiguous 1, 2, 3 after the original CTC
collapse (merge adjacent raw repeats, then remove blanks).

The probability recurrence is written independently of the transition-table
log-domain recurrence. The validation script also enumerates paths against the
frozen actual decoder AST, independently of both recurrences.
"""
from __future__ import annotations

import math
import sys


A, B, C, D, HIT = range(5)
STATE_NAMES = ("A", "B", "C", "D", "HIT")
# A: no matching suffix; B: suffix 1; C: suffix 12, previous raw 2;
# D: suffix 12, previous raw blank; HIT: at least one completed event.
TRANSITIONS = (
    (A, B, A, A),
    (B, B, C, A),
    (D, B, C, HIT),
    (D, B, A, HIT),
    (HIT, HIT, HIT, HIT),
)
NEG_INF = float("-inf")


def transition(state: int, token: int) -> int:
    if type(state) is not int or not 0 <= state < 5:
        raise ValueError("state must be one of A, B, C, D, HIT")
    if type(token) is not int or not 0 <= token < 4:
        raise ValueError("raw token must be one of 0, 1, 2, 3")
    return TRANSITIONS[state][token]


def path_has_no_event(path) -> bool:
    state = A
    for token in path:
        state = transition(state, token)
    return state != HIT


def _numbers(row, *, allow_mask: bool) -> tuple[float, ...]:
    values = tuple(row)
    if len(values) != 4:
        raise ValueError("each frame must have exactly four values")
    if any(type(value) not in (int, float) for value in values):
        raise TypeError("values must be ordinary Python int or float values")
    values = tuple(float(value) for value in values)
    if any(not math.isfinite(value) and not (allow_mask and value == NEG_INF)
           for value in values):
        raise ValueError("NaN and positive infinity are not accepted")
    return values


def probability_masses(probabilities) -> tuple[float, float]:
    """Return unnormalized (no-event, hit) masses for normalized input rows.

    This explicit five-variable probability recurrence does not consult
    TRANSITIONS. It is useful as an independent ordinary-scale reference;
    extreme positive masses can underflow in this ordinary-scale calculation.
    """
    a, b, c, d, h = 1.0, 0.0, 0.0, 0.0, 0.0
    for supplied in probabilities:
        p0, p1, p2, p3 = _numbers(supplied, allow_mask=False)
        row_total = math.fsum((p0, p1, p2, p3))
        if (not all(0.0 <= value <= 1.0 for value in (p0, p1, p2, p3))
                or not math.isclose(row_total, 1.0, rel_tol=0.0, abs_tol=1e-12)):
            raise ValueError("probability rows must be nonnegative and sum to one")
        a, b, c, d, h = (
            math.fsum((a * p0, a * p2, a * p3, b * p3, d * p2)),
            math.fsum((a * p1, b * p0, b * p1, c * p1, d * p1)),
            math.fsum((b * p2, c * p2)),
            math.fsum((c * p0, d * p0)),
            math.fsum((h * row_total, c * p3, d * p3)),
        )
    return math.fsum((a, b, c, d)), h


def _logsumexp(values) -> float:
    # An empty or wholly masked set is handled before subtraction. In
    # particular, this never evaluates (-inf) - (-inf).
    finite = tuple(value for value in values if value != NEG_INF)
    if not finite:
        return NEG_INF
    if any(not math.isfinite(value) for value in finite):
        raise ValueError("nonrepresentable log mass")
    largest = max(finite)
    return largest + math.log(math.fsum(math.exp(value - largest) for value in finite))


def logits_to_log_probabilities(logits) -> tuple[tuple[float, ...], ...]:
    """Stable float64 log-softmax; -inf masks are for the stdlib oracle only."""
    result = []
    for supplied in logits:
        row = _numbers(supplied, allow_mask=True)
        largest = max(row)
        if largest == NEG_INF:
            raise ValueError("a frame cannot mask every class")
        shifted = tuple(value - largest if value != NEG_INF else NEG_INF for value in row)
        if any(value != NEG_INF and not math.isfinite(value) for value in shifted):
            raise ValueError("logit differences must be representable in float64")
        # Detect overflow that would otherwise be indistinguishable from an
        # intentional -inf mask after subtracting the largest finite logit.
        if any(original != NEG_INF and value == NEG_INF
               for original, value in zip(row, shifted)):
            raise ValueError("logit differences must be representable in float64")
        denominator = math.log(math.fsum(math.exp(value) for value in shifted))
        normalized = tuple(value - denominator if value != NEG_INF else NEG_INF for value in shifted)
        if any(value != NEG_INF and not math.isfinite(value) for value in normalized):
            raise ValueError("log probabilities must be representable in float64")
        result.append(normalized)
    bound = -sys.float_info.max / (len(result) + 1)
    if any(value != NEG_INF and value < bound for row in result for value in row):
        raise ValueError("complete finite path log masses must be representable")
    return tuple(result)


def log_probability_masses(logits) -> tuple[float, float]:
    """Return (log Z, log H), retaining all live and first-hit path masses.

    HIT is absorbing. The old HIT mass is multiplied by the current row's
    actual summed probability mass, so normalization-rounding is applied to
    live and HIT paths consistently. No 1-P subtraction, epsilon, or clamp is
    used. Finite logits always leave a positive all-blank no-event path.
    """
    live, hit = (0.0, NEG_INF, NEG_INF, NEG_INF), NEG_INF
    for row in logits_to_log_probabilities(logits):
        buckets = [[] for _ in range(5)]
        if hit != NEG_INF:
            buckets[HIT].append(hit + _logsumexp(row))
        for state, mass in enumerate(live):
            if mass == NEG_INF:
                continue
            for token, log_probability in enumerate(row):
                if log_probability != NEG_INF:
                    buckets[TRANSITIONS[state][token]].append(mass + log_probability)
        combined = tuple(_logsumexp(values) for values in buckets)
        live, hit = combined[:4], combined[HIT]
    return _logsumexp(live), hit


def _softplus(value: float) -> float:
    if value >= 0.0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def no_event_nll(logits) -> float:
    """Return -log[Z/(Z+H)] using softplus(log H - log Z)."""
    log_z, log_h = log_probability_masses(logits)
    if log_h == NEG_INF:
        return 0.0
    if log_z == NEG_INF:
        return math.inf
    return _softplus(log_h - log_z)


def no_event_log_probability(logits) -> float:
    return -no_event_nll(logits)
