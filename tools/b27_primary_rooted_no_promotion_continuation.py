"""Inert read-only transport for the one original B27 NO_PROMOTION.

The hash-pinned previous transport retains its original API checks, complete
41 + 18 original files, archive limits, and first-push activation checks. Its
functions receive a private globals dictionary; no original globals or files
are patched. The only certification route calls the separately pinned rooted
adapter once and preserves both its wrapper and the unchanged inner report.

The failed standalone continuation stays failed. The 8 MB bound applies only
to six review outputs, never to original proof acquisition or scientific data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import types

ROOT = Path(__file__).parent.parent
CONFIG = Path(__file__).with_suffix(".json")
CONFIG_SCHEMA = "english-v6-b27-primary-rooted-no-promotion-continuation-config-v1"
BRANCH = "refs/heads/research/en-v6-b27-primary-rooted-no-promotion-ci-once-20261125"
WORKFLOW = ".github/workflows/b27-primary-rooted-no-promotion-continuation.yml"
PARENT_TOKEN = "__B27_REVIEWED_ROOTED_NO_PROMOTION_CONTINUATION_PARENT__"
HELPER_PATH = "tools/b27_primary_continuation.py"
HELPER_SHA = "e755b0e4bea36f2b802abf45ccf31cb1a7bfd0ee654bc170f98fbcae9b5c379e"
READER_PATH = "automation/english_v6_b27_quality_readback.py"
READER_SHA = "87bc31725380cbf70d70d6b46929a724dc6bff656d421c5ae63b54c0724e31aa"
ADAPTER_PATH = "automation/english_v6_b27_primary_no_promotion_readback.py"
ROOTED_SCHEMA = "english-v6-b27-primary-rooted-no-promotion-readback-v1"
ROOTED_PATH = "research/english_v6_b27_primary_results_20261125/ROOTED_READBACK.json"
ORIGINAL = dict(
    primary_run_id=45627775991,
    primary_run_head="dce5b169163039091afcc5ab62beed3211d37552",
    primary_prepared_parent="8981de96d00e4ac6d6a1b2e65aea6a7023d1d3dc",
    evaluator_sha256="0810522c031da4e9e8d4d3eadcfa2cecd0365277c8bec557448c8b8315e97d49",
    inputs_sha256="66b39d8a0372d658e83a97f66db3edc421b954b242ecc9a40ca560951d904bda",
    producer_manifest_sha256="1a174d05f22254009f1d8023cf5e2366edb44941e6825d7bf66c7a1a1b678ebe",
    external_proof_binding_sha256="6c6334a1d8d6452e186c41fe472eca6e072a8af42a02a81d9aac68b30e7d9a28",
    original_helper_sha256=HELPER_SHA, original_reader_sha256=READER_SHA,
)
CONFIG_PINS = (*ORIGINAL, "rooted_adapter_sha256", "rooted_adapter_bytes")
FAILED_CONTINUATION = dict(
    run_id=45643969797, job_id=139310051038, conclusion="failure",
    reason="cached checkout english_v6_b27_profile.HERE mixed with captured-root "
           "english_v6_b24q1_primary_eval during original verify_worker.load",
)
CLOSED_FLAGS = (
    "original_verifier_code_changed", "original_path_or_hash_guard_changed",
    "proof_context_transferred", "checkout_fallback_allowed",
    "original_standalone_continuation_succeeded", "replication_open_allowed",
    "candidate_engineering_validation_allowed", "automatic_next_assessment_allowed",
    "promotion_allowed", "product_released",
)
ZERO_COUNTERS = ("new_model_calls", "new_source_calls", "new_observer_calls", "new_optimizer_updates")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def document(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    need(type(value) is dict, "one exact JSON object")
    return value


def validate_config(config):
    need(type(config) is dict and set(config) == {"schema", "enabled", *CONFIG_PINS}
         and config["schema"] == CONFIG_SCHEMA and type(config["enabled"]) is bool,
         "exact rooted NO_PROMOTION continuation configuration")
    if not config["enabled"]:
        need(all(config[key] is None for key in CONFIG_PINS), "every future pin remains null while disabled")
        raise ValueError("INERT: the separately reviewed rooted adapter is not bound")
    need(all(type(config[key]) is type(value) and config[key] == value
             for key, value in ORIGINAL.items()), "only the one unchanged original B27 PRIMARY and frozen helper/reader")
    need(type(config["rooted_adapter_sha256"]) is str
         and re.fullmatch("[0-9a-f]{64}", config["rooted_adapter_sha256"]) is not None
         and type(config["rooted_adapter_bytes"]) is int
         and 0 < config["rooted_adapter_bytes"] < 128_000_000,
         "one separately reviewed exact adapter hash and length")
    return config


def verified_bytes(path, digest, size=None):
    path = Path(path)
    need(path.is_absolute() and path == path.resolve() and path.is_file()
         and all(not p.is_symlink() for p in (path, *path.parents)), "regular reviewed implementation path")
    raw = path.read_bytes()
    need(sha(raw) == digest and (size is None or len(raw) == size), "exact reviewed implementation bytes " + str(path))
    return raw


def load_bytes(path, raw, name):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    # Execute the bytes actually hashed above, without a loader re-reading them.
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def check_wrapper(rooted, original, config, canonical):
    need(rooted.get("schema") == ROOTED_SCHEMA and rooted.get("panel") == "primary"
         and rooted.get("panel_gate") == "NO_PROMOTION"
         and rooted.get("original_reader_sha256") == READER_SHA
         and rooted.get("import_scope_adapter_sha256") == config["rooted_adapter_sha256"]
         and rooted.get("original_report") == original
         and rooted.get("original_report_sha256") == sha(canonical(original))
         and rooted.get("original_failed_continuation_run_id") == FAILED_CONTINUATION["run_id"],
         "same separately identified rooted wrapper and full untouched inner report")
    need(all(rooted.get(key) is False for key in CLOSED_FLAGS)
         and all(type(rooted.get(key)) is int and rooted[key] == 0 for key in ZERO_COUNTERS),
         "rooted NO_PROMOTION never changes old guards or grants downstream execution")


def bind_private(original, adapter, config):
    """Bind only the pinned transport's own functions into an isolated namespace."""
    config = dict(validate_config(config))
    namespace = dict(vars(original))
    for name, value in vars(original).items():
        if isinstance(value, types.FunctionType) and value.__globals__ is vars(original):
            copied = types.FunctionType(value.__code__, namespace, value.__name__, value.__defaults__, value.__closure__)
            copied.__kwdefaults__ = value.__kwdefaults__
            namespace[name] = copied
    old_gate, old_export = namespace["preliminary_gate"], namespace["make_export"]
    namespace.update(CONFIG=CONFIG, CONFIG_SCHEMA=CONFIG_SCHEMA, CONFIG_PINS=CONFIG_PINS,
        BRANCH=BRANCH, WORKFLOW=WORKFLOW, PARENT_TOKEN=PARENT_TOKEN,
        COMMON_PATHS=(*original.COMMON_PATHS, ROOTED_PATH), PASS_PATHS=(),
        CHUNK_PREFIX="B27_PRIMARY_ROOTED_CONTINUATION_CHUNK ",
        COMPLETE_PREFIX="B27_PRIMARY_ROOTED_CONTINUATION_COMPLETE ")

    def same_config(value):
        validate_config(value)
        need(value == config, "the same reviewed configuration throughout the continuation")
        return value

    def no_promotion_gate(*args):
        gate = old_gate(*args)
        need(gate == "NO_PROMOTION", "this separate route rejects PASS before full-proof acquisition")
        return gate

    def certify_once(gate, lock, proof, output, bound, primary_files, primary_transport):
        need(gate == "NO_PROMOTION", "only original NO_PROMOTION; no alternate certification route")
        rooted = adapter.verify_original(proof / "primary-quality", lock, proof)
        report = rooted["original_report"]
        namespace["check_report"](report, gate, lock, bound)
        canonical = namespace["profile"].canonical
        check_wrapper(rooted, report, config, canonical)
        return canonical(report), {ROOTED_PATH: canonical(rooted)}

    def make_export(files, outcome):
        need(outcome == "NO_PROMOTION" and set(files) == set(namespace["COMMON_PATHS"]),
             "exact original five evidence paths plus ROOTED_READBACK; no PASS or replica export")
        profile = namespace["profile"]
        rooted = profile.document(files[ROOTED_PATH])
        report = profile.document(files[namespace["REPORT_PATH"]])
        check_wrapper(rooted, report, config, profile.canonical)
        receipt = profile.document(files[namespace["RECEIPT_PATH"]])
        need(receipt["primary_outcome"] == "NO_PROMOTION"
             and receipt["candidate_terminal"] is True and receipt["replica_proof_ready_for_review"] is False,
             "the original transport produced only a terminal failure receipt")
        receipt.update(schema="english-v6-b27-primary-rooted-no-promotion-continuation-v1",
            status="B27_PRIMARY_ROOTED_NO_PROMOTION_TERMINAL",
            certification_route="rooted_original_no_promotion_readback",
            original_helper_sha256=HELPER_SHA, original_reader_sha256=READER_SHA,
            import_scope_adapter_sha256=config["rooted_adapter_sha256"],
            rooted_readback_sha256=sha(files[ROOTED_PATH]),
            original_failed_continuation=dict(FAILED_CONTINUATION),
            original_standalone_continuation_succeeded=False)
        files[namespace["RECEIPT_PATH"]] = profile.canonical(receipt)
        inventory = profile.document(files[namespace["INVENTORY_PATH"]])
        inventory["review_files"] = {path: dict(bytes=len(raw), sha256=sha(raw))
            for path, raw in files.items() if path != namespace["INVENTORY_PATH"]}
        files[namespace["INVENTORY_PATH"]] = profile.canonical(inventory)
        return old_export(files, outcome)

    namespace.update(validate_config=same_config, preliminary_gate=no_promotion_gate,
                     certify_once=certify_once, make_export=make_export)
    return namespace


def runtime(config):
    config = validate_config(config)  # Before module imports, env, Git, network, or workspace inspection.
    helper_path, adapter_path = ROOT / HELPER_PATH, ROOT / ADAPTER_PATH
    helper_raw = verified_bytes(helper_path, HELPER_SHA)
    verified_bytes(ROOT / READER_PATH, READER_SHA)
    adapter_raw = verified_bytes(adapter_path, config["rooted_adapter_sha256"], config["rooted_adapter_bytes"])
    original = load_bytes(helper_path, helper_raw, "_b27_frozen_original_continuation")
    adapter = load_bytes(adapter_path, adapter_raw, "_b27_reviewed_rooted_no_promotion_adapter")
    return bind_private(original, adapter, config)


def prepare(config, work):
    config = validate_config(config)
    return runtime(config)["prepare"](config, work)


def main():
    config = validate_config(document(CONFIG.read_bytes()))
    return runtime(config)["main"]()


if __name__ == "__main__":
    main()
