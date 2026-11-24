"""Audit saved B20 sources and original outputs with an explicit Git supplement.

The original producer ZIP omitted .github files. This audit preserves that fact:
all original ZIP members are authenticated first, and precisely 32 extra files
are independently bound to the real producer commit trees in a temporary overlay.
No model, source generator, observer, network operation or gate is executed.
"""
from __future__ import annotations

import argparse
import builtins
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest import mock

EVIDENCE = Path(__file__).resolve().parent
CODE = "87cb205d5bbd5e5ba17bd0a5929b7e123c5b5688894e753d34406535273ca802"
SPEC = "78b0096f2712b92fa4c2988ad5888e46ecd68df69c8d836b1fc53f843bf7b3a7"
PRODUCER = "5eee2261be1135e87850114271f8ab23ee250bb054849f80c9330d1aef1bf4b5"
PARENT = "474f7bf00ccf722cf52b6140847bcef78c5a82a3"
HEAD = "1c03eb9da468c7e267ff32bb4a34f4ee17bd00b9"
RUN = 45337450827


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git_sha(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def load(path, name):
    definition = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(value)
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def protected(source, old):
    stack = ExitStack()
    denied = {"torch", "numpy", "torchaudio", "transformers", "piper", "soundfile",
              "onnxruntime", "qwen_asr", "whisper", "requests", "httpx"}
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        need(name.split(".")[0] not in denied, "No runtime/model/network import: " + name)
        return original(name, *args, **kwargs)

    stack.enter_context(mock.patch("builtins.__import__", side_effect=guard))
    failure = AssertionError("Saved readback prohibits any new model/source/observer/network call")
    for obj, name in ((source.B13, "render_panel"), (source, "observe"),
                      (source.observers.raw_asr, "_load"), (source.observers, "_load_phone"),
                      (source.observers.native, "_load_phone"), (old, "snapshots"),
                      (source.lineage, "gh")):
        stack.enter_context(mock.patch.object(obj, name, side_effect=failure))
    return stack


def main(actual, helper_root):
    actual, helper_root = actual.resolve(), helper_root.resolve()
    original_driver = actual / "producer/primary_producer/automation/english_v6_b20_primary_eval.py"
    need(sha(original_driver.read_bytes()) == CODE, "Original exact scientific driver")
    bootstrap = load(original_driver, "b20_supplement_bootstrap")
    spec = bootstrap.protocol()
    need(sha(bootstrap.SPEC.read_bytes()) == SPEC and sha(bootstrap.PRODUCER.read_bytes()) == PRODUCER,
         "Exact original activated registration")
    identity = dict(run_id=str(RUN), run_head=HEAD, run_attempt=1, prepared_parent=PARENT)
    pins = {row["role"]: {key: row[key] for key in ("id", "run_id", "run_head", "bytes", "sha256")}
            for row in read(EVIDENCE / "ARTIFACT_LOCKS.json")}
    run, commit = read(EVIDENCE / "RUN_FINAL.json"), read(EVIDENCE / "COMMIT_API.json")
    artifact_items = {row["id"]: row for row in read(EVIDENCE / "ARTIFACTS_FINAL.json")["artifacts"]}
    bootstrap.validate_execution_metadata(identity, run, commit)
    originals = {}
    for role, pin in pins.items():
        bootstrap.validate_provenance(pin, dict(run=run, commit=commit, artifact=artifact_items[pin["id"]]))
        with bootstrap.original_zip(actual / "artifacts" / (role + "-original.zip"), pin) as archive:
            members = {entry.filename for entry in archive.infolist() if not entry.is_dir()}
            target = actual / role
            need(members == {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()},
                 "Exact original extraction, without supplemented members: " + role)
            originals[role] = {name: archive.read(name) for name in members}
            need(all((target / name).read_bytes() == raw for name, raw in originals[role].items()),
                 "Original artifact member bytes remain unchanged: " + role)
    supplement = EVIDENCE / "producer_git_supplement"
    missing = read(supplement / "MISSING_FILE_INVENTORY.json")
    need(len(missing) == 32 and all(row["path"].startswith(".github/") for row in missing),
         "Exact separately declared 32 hidden-file omissions")
    primary_tree = read(supplement / "PRIMARY_ACTUAL_TREE_API.json")
    source_tree = read(supplement / "SOURCE_ACTUAL_TREE_API.json")
    fetched = read(EVIDENCE / "FETCH.json")
    source_commit = fetched["source_metadata"]["synth_source"]["commit"]
    trees = {"primary": primary_tree, "source": source_tree}
    for role, bound in (("primary", commit), ("source", source_commit)):
        need(trees[role]["sha"] == bound["commit"]["tree"]["sha"] and trees[role]["truncated"] is False,
             "Complete actual commit-bound producer tree: " + role)
    with tempfile.TemporaryDirectory() as temporary:
        overlay = Path(temporary) / "producer"
        for name, raw in originals["producer"].items():
            target = overlay / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        supplement_paths = set()
        for row in missing:
            role, path = row["role"], row["path"]
            need(row["producer_head"] == (HEAD if role == "primary" else source_commit["sha"])
                 and row["tree_sha"] == trees[role]["sha"], "Supplement commit/tree identity")
            member = role + "_producer/" + path
            need(member not in originals["producer"] and member not in supplement_paths,
                 "Supplement only fills an exact declared original absence")
            tree_rows = [entry for entry in trees[role]["tree"] if entry["path"] == path]
            need(len(tree_rows) == 1 and tree_rows[0]["type"] == "blob"
                 and tree_rows[0]["mode"] in ("100644", "100755"), "Actual Git file kind")
            raw = (supplement / role / path).read_bytes()
            need(len(raw) == row["bytes"] and sha(raw) == row["sha256"]
                 and git_sha(raw) == row["git_blob_sha"] == tree_rows[0]["sha"],
                 "Supplement exact length, SHA256 and original commit Git blob")
            target = overlay / member
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            supplement_paths.add(member)
        root = overlay / "primary_producer"
        primary = load(root / "automation/english_v6_b20_primary_eval.py", "b20_authenticated_primary")
        producer, workflow = primary.producer_contract(spec, PARENT, root=root)
        expected = {"FETCH.json", "CONSUMPTION.json"}
        original_counts = {"primary": 0, "source": 0}
        for role in ("primary", "source"):
            for path, digest in primary.producer_files(spec, producer, workflow, role).items():
                member = role + "_producer/" + path
                expected.add(member)
                need(sha((overlay / member).read_bytes()) == digest,
                     "Every registered producer member identity: " + member)
                original_counts[role] += member in originals["producer"]
        need(expected == set(originals["producer"]) | supplement_paths,
             "Exact registered closure equals original artifact plus declared Git supplement")
        need(fetched["schema"] == primary.SCHEMA and fetched["identity"] == identity
             and fetched["inputs_sha256"] == SPEC and fetched["producer_manifest_sha256"] == PRODUCER
             and fetched["workflow_sha256"] == sha(workflow)
             and fetched["prior_consumed_markers"]["total_count"] == 0,
             "Actual initial input acquisition and original one-shot marker")
        primary.validate_execution_metadata(identity, fetched["primary_run"], fetched["primary_commit"])
        inventory = fetched["original_workflow_run_inventory"]
        need(inventory["total_count"] == len(inventory["workflow_runs"]) == 1
             and inventory["workflow_runs"][0]["id"] == RUN, "Only one original assessment")
        input_pins = primary.archive_pins(spec)
        need(len(input_pins) == 17 and set(input_pins) == set(fetched["source_metadata"]),
             "Complete 17 original input identities")
        inputs = actual / "independent/inputs"
        for key, pin in input_pins.items():
            primary.validate_provenance(pin, fetched["source_metadata"][key])
            with primary.original_zip(inputs / (key + ".zip"), pin):
                pass
        adapter_path = helper_root / "automation/english_v6_b20_primary_readback.py"
        need(sha(adapter_path.read_bytes()) == "24b6e56955a2e98b8704d8f531526c895b9f94c877a27353cb07a90e572e4d0e", "Exact reviewed saved-only source adapter")
        adapter = load(adapter_path, "b20_saved_source_adapter")
        # The bootstrap module points at the original incomplete archive. It is
        # no longer used, and must not be mistaken for the authenticated import root.
        old, decision, source, _ = primary.helpers(spec)
        with protected(source, old):
            checked = adapter.authenticate_sources(primary, inputs, spec, identity, producer, workflow, fetched)
        admission = checked["admission"]
        need(admission == read(EVIDENCE / "ADMISSION.json"), "Exact original complete admission reconstructed")
        consumed = dict(identity=identity, marker_name=primary.MARKER,
            admission_sha256=sha(json.dumps(admission, sort_keys=True, separators=(",", ":")).encode()))
        need(read(EVIDENCE / "CONSUMPTION.json") == consumed
             == json.loads(originals["producer"]["CONSUMPTION.json"]), "Independent original consumption marker")
        need("torch" not in sys.modules and "numpy" not in sys.modules, "No model or array runtime used")
        authority = dict(schema="english-v6-b20-independent-primary-source-authority-v1",
            status="AUTHENTICATED_ORIGINAL_SOURCES_ONLY", actual_identity=identity,
            original_artifacts=pins, registered_inputs_sha256=SPEC, implementation_sha256=CODE,
            producer_manifest_sha256=PRODUCER, input_pins=input_pins, admission=admission,
            selected_rows=checked["selected_rows"], all_original_rows=checked["all_original_rows"],
            primary_source_inventory=checked["primary_source_inventory"],
            training_source_sha256=admission["training_source_sha256"],
            producer_python_counts={role: len(producer[role]["files"]) for role in ("primary", "source")},
            original_producer_archive_complete=False, original_producer_members=len(originals["producer"]),
            separately_git_authenticated_supplement_files=len(supplement_paths),
            supplemented_producer_closure_complete=True,
            producer_original_member_counts=original_counts,
            supplement_inventory_sha256=sha((supplement / "MISSING_FILE_INVENTORY.json").read_bytes()),
            source_adapter_sha256=sha(adapter_path.read_bytes()),
            source_originals=120, actual_training_originals=88, actual_admitted_originals=84,
            model_calls=0, source_calls=0, observer_calls=0, numerical_result_audited=False,
            replication_open_allowed=False, promotion_allowed=False, product_released=False)
        write(EVIDENCE / "SOURCE_AUTHORITY.json", authority)
        report = dict(schema="english-v6-b20-original-plus-git-producer-audit-v1",
            status="PASS_ORIGINAL_INPUTS_AND_SEPARATE_GIT_PRODUCER_SUPPLEMENT",
            source_authority_sha256=sha((EVIDENCE / "SOURCE_AUTHORITY.json").read_bytes()),
            actual_identity=identity, original_producer_archive_complete=False,
            original_producer_members=len(originals["producer"]),
            git_supplement_files=len(supplement_paths), supplemented_producer_closure_complete=True,
            input_archives=17, all_original_source_rows=120, selected_originals=84,
            source_coverage=admission["source_coverage"], new_model_calls=0, new_source_calls=0,
            new_observer_calls=0, original_archives_modified=False, numerical_gate_changed=False,
            replication_open_allowed=False, promotion_allowed=False, product_released=False)
        write(EVIDENCE / "SOURCE_AND_PRODUCER_AUDIT.json", report)
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual", type=Path, help="Original b20_actual_primary_readback directory")
    parser.add_argument("helpers", nargs="?", type=Path, default=EVIDENCE / "audit_helpers",
                        help="Saved-only helper directory; defaults to included audit_helpers")
    args = parser.parse_args()
    main(args.actual, args.helpers)
