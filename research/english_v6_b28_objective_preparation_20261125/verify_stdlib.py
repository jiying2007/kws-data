"""Reproducible preparation-only verification; never import or execute Torch.

Only the single, hash-pinned ctc_events function AST is compiled from the
frozen source. The original evaluation/training modules are never imported.
"""
from __future__ import annotations

import argparse
import ast
import copy
from decimal import Decimal, localcontext
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys

import no_event_reference as reference


ROOT = Path(__file__).resolve().parent
COMMIT = "b37932fbbf8579b7efb3f4789acf3f11ca495e87"
MANIFEST_SHA = "d2a00cccf68bc185939536a059a113b69edd2c065d3ca92adba93b95ee61982e"
DECODER_SHA = "0ff27034c441b12b69f8b0bf670262116801cd72e0e134fa04a6b09f13029975"
SEED = 20261125


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def close(left, right, *, absolute=5e-13, relative=5e-13, label="numbers"):
    need(math.isclose(left, right, rel_tol=relative, abs_tol=absolute),
         f"{label}: {left!r} != {right!r}")


def frozen_decoder():
    manifest_raw = (ROOT / "FROZEN_SOURCE_MANIFEST.json").read_bytes()
    need(digest(manifest_raw) == MANIFEST_SHA, "exact original source manifest")
    manifest = json.loads(manifest_raw)
    need(manifest["commit"] == COMMIT and len(manifest["entries"]) == 5,
         "all five original frozen source resources")
    source = {}
    for pin in manifest["entries"]:
        path = ROOT / pin["draft_path"]
        need(path.is_file() and not path.is_symlink(), "ordinary original source file")
        raw = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        need(len(raw) == pin["bytes"] and digest(raw) == pin["sha256"]
             and blob == pin["git_blob_sha1"], "complete original source bytes and Git blob")
        source[Path(pin["repository_path"]).name] = raw.decode("utf-8")
    profile = json.loads(source["english_v6_b27_execution_profile.json"])
    need(profile["fixed_base_sha256"]["english_v6_b24q1_primary_eval.py"] == DECODER_SHA,
         "B27 profile binds this actual preserved decoder source")
    adapter = ast.parse(source["english_v6_b27_primary_eval.py"])
    need(not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == "ctc_events" for node in adapter.body),
         "B27 does not redefine the inherited event decoder")
    preserved = [node for node in ast.walk(adapter) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "preserved"
                 and isinstance(node.func.value, ast.Name) and node.func.value.id == "profile"]
    need(len(preserved) == 1 and len(preserved[0].args) == 1
         and isinstance(preserved[0].args[0], ast.Constant)
         and preserved[0].args[0].value == "english_v6_b24q1_primary_eval",
         "exact original decoder inheritance")
    inherited = [node for node in ast.walk(adapter) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "inherit"
                 and isinstance(node.func.value, ast.Name) and node.func.value.id == "profile"]
    need(len(inherited) == 1 and ast.unparse(inherited[0]) == "profile.inherit(original, globals())",
         "actual frozen B27 inheritance call")

    decoder_text = source["english_v6_b24q1_primary_eval.py"]
    need(digest(decoder_text.encode()) == DECODER_SHA, "actual decoder file pin")
    nodes = [node for node in ast.parse(decoder_text).body
             if isinstance(node, ast.FunctionDef) and node.name == "ctc_events"]
    need(len(nodes) == 1 and not nodes[0].decorator_list, "one undecorated actual decoder function")
    selected = copy.deepcopy(nodes[0])
    need(not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(selected)),
         "the selected decoder cannot import model/evaluation dependencies")
    namespace = {"need": need, "__builtins__": {
        "all": all, "dict": dict, "enumerate": enumerate, "int": int,
        "len": len, "list": list, "max": max, "range": range, "type": type,
    }}
    module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    exec(compile(module, "<frozen-ctc-events-only>", "exec"), namespace)
    function_text = ast.get_source_segment(decoder_text, nodes[0])
    return namespace["ctc_events"], {
        "repository": manifest["repository"], "commit": COMMIT,
        "manifest_sha256": MANIFEST_SHA, "file_sha256": DECODER_SHA,
        "function_source_sha256": digest(function_text.encode()),
        "function_lines": [nodes[0].lineno, nodes[0].end_lineno],
        "full_modules_imported": False, "source_files_verified": len(manifest["entries"]),
    }


def exhaustive_semantics(decoder):
    counts = []
    total = 0
    for length in range(10):
        checked = 0
        for path in itertools.product(range(4), repeat=length):
            original = decoder(list(path))
            actual = reference.path_has_no_event(path)
            need(actual == (original["events"] == 0), f"DFA/actual decoder mismatch: {path!r}")
            checked += 1
        need(checked == 4 ** length, "complete path census at every length")
        counts.append({"frames": length, "paths": checked, "mismatches": 0})
        total += checked
    need(total == 349525, "complete lengths zero through nine")
    cases = [
        ([], 0), ([1, 2, 3], 1), ([1, 1, 2, 2, 3, 3], 1),
        ([1, 0, 2, 0, 3], 1), ([1, 2, 0, 2, 3], 0),
        ([1, 2, 2, 0, 3], 1), ([1, 2, 1, 2, 3], 1),
        ([1, 2, 3, 1, 2, 3], 2), ([1, 2, 3, 0, 3], 1),
        ([1, 0, 1, 2, 3], 1), ([1, 2, 0, 0, 3], 1),
    ]
    detailed = []
    for path, expected in cases:
        original = decoder(path)
        need(original["events"] == expected, "named blank/repeat/multiple-event edge case")
        need(reference.path_has_no_event(path) == (expected == 0), "named DFA edge case")
        detailed.append({"path": path, "collapsed": original["tokens"], "events": expected})
    return {"per_length": counts, "total_paths": total, "mismatches": 0, "edge_cases": detailed}


def enumerated_masses(probabilities, decoder):
    no_event, hit = [], []
    for path in itertools.product(range(4), repeat=len(probabilities)):
        weight = math.prod(probabilities[frame][token] for frame, token in enumerate(path))
        (no_event if decoder(list(path))["events"] == 0 else hit).append(weight)
    return math.fsum(no_event), math.fsum(hit)


def weighted_checks(decoder):
    randomizer = random.Random(SEED)
    max_probability_error = 0.0
    max_nll_error = 0.0
    cases, paths = 0, 0
    for length in range(7):
        for _ in range(3):
            logits = [[randomizer.uniform(-2.0, 2.0) for _ in range(4)] for _ in range(length)]
            logp = reference.logits_to_log_probabilities(logits)
            probabilities = [[math.exp(value) for value in row] for row in logp]
            z_enum, h_enum = enumerated_masses(probabilities, decoder)
            z_prob, h_prob = reference.probability_masses(probabilities)
            log_z, log_h = reference.log_probability_masses(logits)
            z_log = math.exp(log_z)
            h_log = 0.0 if log_h == -math.inf else math.exp(log_h)
            for actual, expected in ((z_prob, z_enum), (h_prob, h_enum), (z_log, z_enum), (h_log, h_enum)):
                close(actual, expected, label="independent weighted path mass")
                max_probability_error = max(max_probability_error, abs(actual - expected))
            close(z_prob + h_prob, 1.0, label="probability conservation")
            expected_nll = -math.log(z_enum / (z_enum + h_enum))
            actual_nll = reference.no_event_nll(logits)
            close(actual_nll, expected_nll, label="normalized no-event NLL")
            max_nll_error = max(max_nll_error, abs(actual_nll - expected_nll))
            offsets = [randomizer.uniform(-1000.0, 1000.0) for _ in logits]
            shifted = [[value + offsets[index] for value in row] for index, row in enumerate(logits)]
            close(reference.no_event_nll(shifted), actual_nll, absolute=2e-12,
                  relative=2e-12, label="per-frame shift invariance")
            cases += 1
            paths += 4 ** length
    return {"seed": SEED, "cases": cases, "enumerated_paths": paths,
            "maximum_probability_absolute_error": max_probability_error,
            "maximum_nll_absolute_error": max_nll_error, "row_shift_invariance": "PASS"}


def decimal_nll(logits, decoder):
    with localcontext() as context:
        context.prec = 120
        rows = []
        for row in logits:
            values = [Decimal(str(value)) for value in row]
            largest = max(values)
            weights = [(value - largest).exp() for value in values]
            normalizer = sum(weights, Decimal(0))
            rows.append([weight / normalizer for weight in weights])
        z, h = Decimal(0), Decimal(0)
        for path in itertools.product(range(4), repeat=len(rows)):
            weight = Decimal(1)
            for frame, token in enumerate(path):
                weight *= rows[frame][token]
            if decoder(list(path))["events"] == 0:
                z += weight
            else:
                h += weight
        return float(((z + h) / z).ln())


def extreme_checks(decoder):
    examples = [
        ("uniform_three_frames", [[0.0] * 4 for _ in range(3)]),
        ("tiny_event_mass_survives_cancellation", [
            [0.0, -30.0, -100.0, -100.0], [0.0, -100.0, -30.0, -100.0],
            [0.0, -100.0, -100.0, -30.0],
        ]),
        ("nearly_certain_event_1000_logit_gap", [
            [-1000.0, 0.0, -1000.0, -1000.0], [-1000.0, -1000.0, 0.0, -1000.0],
            [-1000.0, -1000.0, -1000.0, 0.0],
        ]),
        ("nearly_certain_event_20000_logit_gap", [
            [-10000.0, 10000.0, -10000.0, -10000.0],
            [-10000.0, -10000.0, 10000.0, -10000.0],
            [-10000.0, -10000.0, -10000.0, 10000.0],
        ]),
    ]
    results = []
    for label, logits in examples:
        expected = decimal_nll(logits, decoder)
        actual = reference.no_event_nll(logits)
        need(math.isfinite(actual) and actual >= 0.0, "finite nonnegative extreme NLL")
        close(actual, expected, absolute=0.0, relative=5e-12, label=label)
        if label == "tiny_event_mass_survives_cancellation":
            need(actual > 0.0, "representable rare-event loss must not cancel to zero")
            log_z, _ = reference.log_probability_masses(logits)
            need(1.0 - math.exp(log_z) == 0.0, "this case really exposes naive 1-P cancellation")
        results.append({"case": label, "nll": actual, "decimal_120_digit_nll": expected,
                        "absolute_error": abs(actual - expected)})

    # Here the NLL is smaller than the least positive float64 value. Retain the
    # finite log-H and report unavoidable final-value underflow honestly.
    underflow = [[0.0, -1000.0, -1000.0, -1000.0] for _ in range(3)]
    log_z, log_h = reference.log_probability_masses(underflow)
    close(log_h, -3000.0, label="ultra-small event log mass")
    need(math.isfinite(log_z) and reference.no_event_nll(underflow) == 0.0,
         "unrepresentably tiny NLL underflow is expected, not clamped")

    for frames in (0, 1, 2):
        need(reference.no_event_nll([[0.0, 2.0, -3.0, 4.0] for _ in range(frames)]) == 0.0,
             "fewer than three frames has exactly zero mathematical NLL")
    masked_hit = [[-math.inf, 0.0, -math.inf, -math.inf],
                  [-math.inf, -math.inf, 0.0, -math.inf],
                  [-math.inf, -math.inf, -math.inf, 0.0]]
    need(reference.no_event_nll(masked_hit) == math.inf, "deterministic 123 has infinite NLL")
    masked_blank = [[0.0, -math.inf, -math.inf, -math.inf] for _ in range(6)]
    need(reference.no_event_nll(masked_blank) == 0.0, "deterministic blanks have zero NLL")
    interrupted = [[0.0 if token == chosen else -math.inf for token in range(4)]
                   for chosen in (1, 2, 0, 2, 3)]
    need(reference.no_event_nll(interrupted) == 0.0, "blank-separated 2 interrupts contiguous 123")

    # Independent integer counts of uniform paths, with no floating probability
    # recurrence or decoder enumeration at this longer, synthetic length.
    frames = 4096
    a, b, c, d, h = 1, 0, 0, 0, 0
    for _ in range(frames):
        a, b, c, d, h = 3 * a + b + d, a + 2 * b + c + d, b + c, c + d, 4 * h + c + d
    need(a + b + c + d + h == 4 ** frames, "exact integer path count conservation")
    expected = frames * math.log(4.0) - math.log(a + b + c + d)
    actual = reference.no_event_nll([[0.0] * 4 for _ in range(frames)])
    close(actual, expected, absolute=1e-8, relative=1e-10, label="long uniform exact path counts")
    return {"decimal_examples": results,
            "short_sequences": "PASS for T=0,1,2",
            "deterministic_mask_examples": "PASS (stdlib reference only)",
            "below_float64_range": {"log_hit_mass": log_h, "nll": 0.0,
                "explanation": "Final NLL underflows below float64 range; log mass remains finite; no clamp."},
            "long_uniform": {"frames": frames, "nll": actual,
                "exact_integer_count_nll": expected, "absolute_error": abs(actual - expected)}}


def gradient_check(decoder):
    randomizer = random.Random(SEED + 1)
    frames = 4
    logits = [[randomizer.uniform(-1.5, 1.5) for _ in range(4)] for _ in range(frames)]
    probabilities = [[math.exp(value) for value in row]
                     for row in reference.logits_to_log_probabilities(logits)]
    entries = [[[] for _ in range(4)] for _ in range(frames)]
    kept = []
    for path in itertools.product(range(4), repeat=frames):
        if decoder(list(path))["events"]:
            continue
        weight = math.prod(probabilities[frame][token] for frame, token in enumerate(path))
        kept.append(weight)
        for frame, token in enumerate(path):
            entries[frame][token].append(weight)
    z = math.fsum(kept)
    oracle = [[probabilities[frame][token] - math.fsum(entries[frame][token]) / z
               for token in range(4)] for frame in range(frames)]
    maximum_error, maximum_row_sum = 0.0, 0.0
    epsilon = 1e-5
    for frame in range(frames):
        maximum_row_sum = max(maximum_row_sum, abs(math.fsum(oracle[frame])))
        for token in range(4):
            plus, minus = copy.deepcopy(logits), copy.deepcopy(logits)
            plus[frame][token] += epsilon
            minus[frame][token] -= epsilon
            derivative = (reference.no_event_nll(plus) - reference.no_event_nll(minus)) / (2 * epsilon)
            expected = oracle[frame][token]
            need(-1.0 <= expected <= 1.0, "conditional-occupancy gradient bound")
            close(derivative, expected, absolute=2e-9, relative=2e-7,
                  label="stdlib central difference vs enumerated conditional occupancy")
            maximum_error = max(maximum_error, abs(derivative - expected))
    need(maximum_row_sum < 1e-12, "each logit-gradient row sums to zero")
    return {"frames": frames, "enumerated_paths": 4 ** frames,
            "formula": "p_t(k) - P(a_t=k | no event)", "epsilon": epsilon,
            "maximum_absolute_error": maximum_error,
            "maximum_conditional_gradient_row_sum": maximum_row_sum,
            "autograd_executed": False}


def rejection_checks():
    bad_logits = [
        [[0.0, 0.0, 0.0]], [[math.nan, 0.0, 0.0, 0.0]],
        [[math.inf, 0.0, 0.0, 0.0]], [[-math.inf] * 4],
        [[True, 0.0, 0.0, 0.0]], [[1e308, -1e308, 0.0, 0.0]],
    ]
    cases = [(reference.no_event_nll, value) for value in bad_logits]
    cases += [(reference.probability_masses, [[-0.1, 0.3, 0.4, 0.4]]),
              (reference.probability_masses, [[0.1] * 4])]
    for function, value in cases:
        try:
            function(value)
        except (TypeError, ValueError):
            pass
        else:
            raise ValueError("invalid numeric input was not rejected")
    return {"invalid_inputs_rejected": len(cases)}


def torch_source_checks():
    path = ROOT / "torch_objective_draft.py"
    raw = path.read_bytes()
    tree = ast.parse(raw.decode())
    imported = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    need(all(isinstance(node, ast.ImportFrom) and node.module == "__future__" for node in imported),
         "Torch draft contains no runtime dependency import")
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    wrapper = functions["natural_no_event_suffix_loss"]
    model_calls = [node for node in ast.walk(wrapper) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id == "model"]
    need(len(model_calls) == 1 and ast.unparse(model_calls[0]) == "model(x)",
         "wrapper source has exactly one existing model(x)")
    need(not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr in ("backward", "step", "ctc_loss", "load", "load_state_dict")
                 for node in ast.walk(tree)), "draft does not execute training/loading/old CTC")
    combined = functions["_combine"]
    need(isinstance(combined.body[0], ast.If)
         and ast.unparse(combined.body[0].test) == "not terms"
         and ast.unparse(combined.body[0].body[0]) == "return None",
         "empty structural lists return before any logsumexp")
    core_source = ast.get_source_segment(raw.decode(), functions["no_event_nll_torch"])
    need("work[:, :0, :].sum(dim=(-2, -1))" in core_source
         and "if frames < 3:" in core_source,
         "short path source returns graph-connected empty-slice zero")
    names = {"_A": 0, "_B": 1, "_C": 2, "_D": 3, "_HIT": 4}
    tables = [node for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "_TRANSITIONS" for target in node.targets)]
    need(len(tables) == 1, "one explicit Torch DFA table")
    table = tuple(tuple(names[item.id] for item in row.elts) for row in tables[0].value.elts)
    need(table == reference.TRANSITIONS[:4], "Torch source table equals the validated DFA")
    return {"sha256": digest(raw), "bytes": len(raw), "ast_syntax": "PASS",
            "wrapper_model_call_sites": len(model_calls), "matching_transition_table": True,
            "unreachable_list_guard": True, "short_sequence_zero_source_guard": True,
            "runtime_executed": False, "autograd_executed": False,
            "complete_fit_changed": False,
            "limitation": "Static source checks only; Torch numerical and integration behavior remain untested."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "STDLIB_VALIDATION.json")
    args = parser.parse_args()
    need(not any(name == "torch" or name.startswith("torch.") for name in sys.modules),
         "Torch must not be imported before preparation verification")
    decoder, decoder_pin = frozen_decoder()
    result = {
        "schema": "b28-objective-preparation-stdlib-verification-v1",
        "status": "STDLIB_VERIFIED_TORCH_UNEXECUTED",
        "preparation_only": True, "candidate_registered": False,
        "frozen_decoder": decoder_pin,
        "exhaustive_semantics": exhaustive_semantics(decoder),
        "weighted_probabilities": weighted_checks(decoder),
        "numeric_extremes": extreme_checks(decoder),
        "gradient_mathematics": gradient_check(decoder),
        "rejections": rejection_checks(),
        "torch_source": torch_source_checks(),
        "new_execution": {"model_forwards": 0, "audio_decodes": 0,
            "optimizer_updates": 0, "observer_calls": 0,
            "training_runs": 0, "torch_executions": 0,
            "B27_SOURCE_or_PRIMARY_result_artifacts_read": 0},
        "runtime": {"implementation": sys.implementation.name, "python": sys.version.split()[0]},
    }
    need(not any(name == "torch" or name.startswith("torch.") for name in sys.modules),
         "Torch was not imported during any verification")
    result["draft_files"] = {name: {"bytes": len((ROOT / name).read_bytes()),
        "sha256": digest((ROOT / name).read_bytes())}
        for name in ("no_event_reference.py", "torch_objective_draft.py", "verify_stdlib.py", "README.md")}
    report = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    need(args.report.resolve().parent == ROOT, "report stays in the preparation directory")
    args.report.write_bytes(report)
    print(json.dumps({"status": result["status"], "report": str(args.report),
        "bytes": len(report), "sha256": digest(report),
        "exhaustive_paths": result["exhaustive_semantics"]["total_paths"],
        "semantic_mismatches": result["exhaustive_semantics"]["mismatches"],
        "weighted_maximum_probability_error": result["weighted_probabilities"]["maximum_probability_absolute_error"],
        "conditional_gradient_maximum_error": result["gradient_mathematics"]["maximum_absolute_error"],
        "torch_executed": False, "model_forwards": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
