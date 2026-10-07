#!/usr/bin/env python3
"""Replay 450 invented cases against the exact independently reviewed scorer.

Usage: python scoring/posthoc-v2/review/test_phonetic_distance_oracle.py

The JSON receipt is printed only after all checks pass. This reads only this
test's source and the adjacent scorer source, never evaluation artifacts.
"""
from __future__ import annotations

import datetime
import hashlib
import itertools
import json
from pathlib import Path
import platform
import random
import types


EXPECTED_SCORER_SHA256 = "1bccdd66bca3cd67109df1eff63770b5d7e9ba3a9a10b578a665b7bbcc9575b4"
SEED = 8834
POOL = ("a", "b", "c")
CASES_PER_TARGET_LENGTH = 150


def concrete_distance(left, right):
    """Independent full-table Levenshtein oracle for concrete symbol strings."""
    table = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i in range(len(left) + 1):
        table[i][0] = i
    for j in range(len(right) + 1):
        table[0][j] = j
    for i, a in enumerate(left, 1):
        for j, b in enumerate(right, 1):
            table[i][j] = min(
                table[i - 1][j] + 1,
                table[i][j - 1] + 1,
                table[i - 1][j - 1] + int(a != b),
            )
    return table[-1][-1]


def main():
    test_path = Path(__file__).resolve()
    scorer_path = test_path.parents[1] / "calibrate.py"
    source = scorer_path.read_bytes()
    actual_sha256 = hashlib.sha256(source).hexdigest()
    if actual_sha256 != EXPECTED_SCORER_SHA256:
        raise RuntimeError("Scorer source differs from the independently reviewed SHA256")

    # Compile the same bytes that were hashed; do not create scorer bytecode files.
    scorer = types.ModuleType("reviewed_target_local_v2")
    exec(compile(source, str(scorer_path), "exec"), scorer.__dict__)
    rng = random.Random(SEED)
    checked = 0
    expanded_realizations = 0
    case_digest = hashlib.sha256()
    for n in range(2, 5):
        for _ in range(CASES_PER_TARGET_LENGTH):
            target = [{rng.choice(POOL)} for _ in range(n)]
            window, variants = [], []
            for _ in range(rng.randrange(0, n + 2)):
                is_oov = rng.randrange(4) == 0
                readings = [] if is_oov else rng.sample(POOL, rng.randrange(1, 4))
                window.append({"kind": "oov" if is_oov else "lexical", "readings": readings})
                variants.append(POOL if is_oov else readings)
            concrete_target = [next(iter(reading)) for reading in target]
            costs = [concrete_distance(realization, concrete_target)
                     for realization in itertools.product(*variants)]
            expected = min(costs)
            actual = scorer.possible_phonetic_edit_distance(window, target)
            if actual != expected:
                raise AssertionError({"case": checked, "window": window,
                                      "target": concrete_target,
                                      "actual": actual, "expected": expected})
            case_digest.update((json.dumps({"window": window, "target": concrete_target,
                                           "expected": expected},
                                          sort_keys=True, separators=(",", ":")) + "\n").encode())
            checked += 1
            expanded_realizations += len(costs)
    if checked != 450:
        raise AssertionError("The review contract requires exactly 450 generic cases")

    receipt = {
        "schema_version": "target-local-v2-generic-oracle-review-v1",
        "result": "pass",
        "validation_kind": "generic_fixture_replay_of_independent_review",
        "validated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "scorer_relative_path": "../calibrate.py",
        "scorer_sha256": actual_sha256,
        "test_source_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "cases": checked,
        "expanded_concrete_realizations": expanded_realizations,
        "case_stream_sha256": case_digest.hexdigest(),
        "seed": SEED,
        "target_lengths": [2, 3, 4],
        "cases_per_target_length": CASES_PER_TARGET_LENGTH,
        "window_lengths": "0 through target length + 1, inclusive",
        "symbol_pool": list(POOL),
        "oracle": "Enumerate every token reading and OOV realization; minimize an independent full-table Levenshtein distance",
        "scope": {
            "actual_evaluation_data_accessed": False,
            "actual_model_outputs_accessed": False,
            "actual_human_labels_accessed": False,
            "audio_accessed": False,
            "model_runtime_or_weights_loaded": False,
            "scorer_or_existing_tests_modified": False,
        },
        "limitations": [
            "Invented phonetic symbols only; not a validation of model accuracy or acoustic evidence.",
            "This replay validates the minimum-distance helper; the separate generic suite covers decision and contract behavior.",
            "The target-local v2 scorer remains a post-hoc diagnostic, not untouched-holdout qualification.",
        ],
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
