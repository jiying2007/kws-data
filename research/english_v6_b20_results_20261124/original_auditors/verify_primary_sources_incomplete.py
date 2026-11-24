"""Independent original B20 primary producer/source audit; no new inference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "independent"
WORK = BASE.parent
CODE = "87cb205d5bbd5e5ba17bd0a592b7e123c5b5688894e753d34406535273ca802"
# Exact value below is deliberately literal and independently checked.
CODE = "87cb205d5bbd5e5ba17bd0a5929b7e123c5b5688894e753d34406535273ca802"
SPEC = "78b0096f2712b92fa4c2988ad5888e46ecd68df69c8d836b1fc53f843bf7b3a7"
PRODUCER = "5eee2261be1135e87850114271f8ab23ee250bb054849f80c9330d1aef1bf4b5"
PARENT = "474f7bf00ccf722cf52b6140847bcef78c5a82a3"
HEAD = "1c03eb9da468c7e267ff32bb4a34f4ee17bd00b9"
RUN = 45337450827


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    actual_root = BASE / "producer/primary_producer"
    driver = actual_root / "automation/english_v6_b20_primary_eval.py"
    need(sha(driver.read_bytes()) == CODE, "original actual scientific implementation")
    primary = module(driver, "independent_actual_b20_primary")
    spec = primary.protocol()
    need(sha(primary.SPEC.read_bytes()) == SPEC and sha(primary.PRODUCER.read_bytes()) == PRODUCER,
         "exact actual source-bound scientific registration")
    pins = {r["role"]: {k: r[k] for k in ("id", "run_id", "run_head", "bytes", "sha256")}
            for r in json.loads((BASE / "ARTIFACT_LOCKS.json").read_bytes())}
    need(pins["result"] == dict(id=14894229987, run_id=RUN, run_head=HEAD, bytes=5916607,
         sha256="9972b33b74facbf1e9c055eec3820b981b8d0f9c829791e52c17468bb4891eb5")
         and pins["producer"] == dict(id=14894438758, run_id=RUN, run_head=HEAD, bytes=1761033,
         sha256="fb1874de1f883d12445272c987c2446f6d45b900b098855072dd9c6100d19f27"), "actual original output artifact locks")
    run = json.loads((BASE / "api/run-final.json").read_bytes())
    commit = json.loads((BASE / "api/commit.json").read_bytes())
    artifacts = {a["id"]: a for a in json.loads((BASE / "api/artifacts-final.json").read_bytes())["artifacts"]}
    identity = dict(run_id=str(RUN), run_head=HEAD, run_attempt=1, prepared_parent=PARENT)
    primary.validate_execution_metadata(identity, run, commit)
    for kind, pin in pins.items():
        primary.validate_provenance(pin, dict(run=run, commit=commit, artifact=artifacts[pin["id"]]))
        with primary.original_zip(BASE / "artifacts" / (kind + "-original.zip"), pin) as archive:
            target = BASE / kind
            members = {x.filename for x in archive.infolist() if not x.is_dir()}
            need(members == {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()},
                 "complete original extracted artifact " + kind)
            for name in members:
                need((target / name).read_bytes() == archive.read(name), "unchanged original member " + name)
    producer, workflow = primary.producer_contract(spec, PARENT, root=actual_root)
    fetched = json.loads((BASE / "producer/FETCH.json").read_bytes())
    need(fetched["schema"] == primary.SCHEMA and fetched["identity"] == identity
         and fetched["inputs_sha256"] == SPEC and fetched["producer_manifest_sha256"] == PRODUCER
         and fetched["workflow_sha256"] == sha(workflow)
         and fetched["prior_consumed_markers"]["total_count"] == 0, "actual first acquisition before model calls")
    primary.validate_execution_metadata(identity, fetched["primary_run"], fetched["primary_commit"])
    inventory = fetched["original_workflow_run_inventory"]
    need(inventory["total_count"] == len(inventory["workflow_runs"]) == 1
         and inventory["workflow_runs"][0]["id"] == RUN, "one actual model assessment only")
    expected = {"FETCH.json", "CONSUMPTION.json"}
    for role in ("primary", "source"):
        for path, digest in primary.producer_files(spec, producer, workflow, role).items():
            relative = role + "_producer/" + path
            expected.add(relative)
            need(sha((BASE / "producer" / relative).read_bytes()) == digest, "actual complete producer byte identity " + relative)
    need(expected == {p.relative_to(BASE / "producer").as_posix() for p in (BASE / "producer").rglob("*") if p.is_file()},
         "exact full actual producer closure")
    inputs = primary.archive_pins(spec)
    need(len(inputs) == 17 and set(inputs) == set(fetched["source_metadata"]), "all seventeen original input identities")
    for key, pin in inputs.items():
        primary.validate_provenance(pin, fetched["source_metadata"][key])
        with primary.original_zip(OUT / "inputs" / (key + ".zip"), pin):
            pass
    adapter = module(WORK / "b20_replication_eval_work/automation/english_v6_b20_primary_readback.py", "independent_source_adapter")
    old, decision, source, _ = primary.helpers(spec)
    forbidden = AssertionError("New model, source or observer calls are forbidden in saved readback")
    with patch.object(source.B13, "render_panel", side_effect=forbidden), \
         patch.object(source, "observe", side_effect=forbidden), \
         patch.object(source.observers.raw_asr, "_load", side_effect=forbidden), \
         patch.object(source.observers, "_load_phone", side_effect=forbidden), \
         patch.object(old, "snapshots", side_effect=forbidden):
        checked = adapter.authenticate_sources(primary, OUT / "inputs", spec, identity, producer, workflow, fetched)
    admission = checked["admission"]
    need(admission == json.loads((BASE / "result/ADMISSION.json").read_bytes()), "independently reconstructed exact actual admission")
    consumed = dict(identity=identity, marker_name=primary.MARKER,
        admission_sha256=sha(json.dumps(admission, sort_keys=True, separators=(",", ":")).encode()))
    need(json.loads((BASE / "producer/CONSUMPTION.json").read_bytes()) == consumed
         == json.loads((BASE / "consumed/CONSUMPTION.json").read_bytes()), "separate original uploaded consumed marker binds admission")
    need("torch" not in sys.modules and "numpy" not in sys.modules, "no model or array runtime import in source audit")
    authority = dict(schema="english-v6-b20-independent-primary-source-authority-v1", status="AUTHENTICATED_ORIGINAL_SOURCES_ONLY",
        actual_identity=identity, original_artifacts=pins, registered_inputs_sha256=SPEC, implementation_sha256=CODE,
        producer_manifest_sha256=PRODUCER, input_pins=inputs, admission=admission,
        selected_rows=checked["selected_rows"], all_original_rows=checked["all_original_rows"],
        primary_source_inventory=checked["primary_source_inventory"], training_source_sha256=admission["training_source_sha256"],
        producer_python_counts={role: len(producer[role]["files"]) for role in ("primary", "source")},
        source_originals=120, actual_training_originals=88, actual_admitted_originals=84,
        model_calls=0, source_calls=0, observer_calls=0, numerical_result_audited=False,
        replication_open_allowed=False, promotion_allowed=False, product_released=False)
    path = OUT / "SOURCE_AUTHORITY.json"
    primary.write(path, authority)
    print(json.dumps(dict(status=authority["status"], selected_rows=len(authority["selected_rows"]),
        all_original_rows=len(authority["all_original_rows"]), source_coverage=admission["source_coverage"],
        file=str(path), sha256=sha(path.read_bytes()), model_calls=0, source_calls=0)))


if __name__ == "__main__":
    main()
