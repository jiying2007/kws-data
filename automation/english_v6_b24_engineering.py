"""Inert B24 software SDK handoff, opened only by two original acoustic PASSes.

No model import, pack creation, compilation, numeric replay or deployment occurs
on the closed path. The unchanged B0 packer is used for geometry only. The SDK
is an integration candidate; no physical SSC305 or field-quality claim is made.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = HERE / "english_v6_b24_engineering_inputs.json"
TEMPLATE = HERE / "english_v6_b24_engineering_workflow_template.yml"
PIPELINE_FILES = HERE / "english_v6_b24_engineering_pipeline.json"
REPO = "jiying2007/kws-data"
PIPELINE_REPO = "jiying2007/kws-pipeline"
PIPELINE_COMMIT = "88a043c6c00bb3a1a62f45098a023c7601042ded"
SCHEMA = "english-v6-b24-dual-pass-software-sdk-v1"
QUALITY_READER = "english_v6_b24_quality_readback"
WORKFLOW = ".github/workflows/english-v6-b24-engineering-once.yml"
BRANCH = "research/en-v6-b24-engineering-once-20261124"
PARENT_TOKEN = "__B24_REVIEWED_ENGINEERING_PARENT__"
MARKER = "english-v6-b24-engineering-consumed"
ASSESSMENT_SHA = "3afdb17e822bb0b4a49757ac116a28101ff9f1447b22429305299047410fe2ad"
DESIGN_SHA = "4fa88cf4050aa276323a7306f21d53163d47c27e3c011c2547d5eeebe6980309"
B0_PT = "1ae8a2b20c8b6991f925dd4c626af2f36b308adee8733a3a7215818716f565d7"
B9_PT = "7dba86a3fcda6e5f05cb80b414dc372c70a5a03b3aee783f168441a4871e7655"
CONTROL_PT = "784def9df6c2a8bfbecdb7ba2d7b70630337a1cda72561c05392327aa980297f"
CANDIDATE_PT = "f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4"
MODEL_COMPARISON = "B24_new_challenger_against_immutable_historical_B23_challenger"
B0_NPZ = "02b7febf9f736fd70f4f9ae04042ffc6d609b15011d03d235b960f69b1d71696"
STEM = "mero43-b24-challenger192"
STEMS = {"control": "control/models/mero43-b23-challenger192", "challenger": "challenger/models/" + STEM}
BACKENDS = ("pytorch", "onnxruntime", "gcc", "clang", "cortex-a32-qemu")
CONDITIONS = ("original", "one_second_zero_prefix")
ORIGINALS = 14
MAX_CALLS = ORIGINALS * len(CONDITIONS) * len(BACKENDS)
PACK_BYTES = 216960
ARM_FLAGS = "-mcpu=cortex-a32 -mfpu=neon-vfpv4 -mfloat-abi=hard -fno-fast-math -ffp-contract=off -fexcess-precision=standard"
FROZEN_INPUTS = {
    "B0": dict(id=14514228406, run_id=44495594627,
        run_head="d0b567fe37147aae6ab5d7bdec3721b879a0b22d", bytes=501276,
        sha256="91b203f05041d92e7ffb1b748f94959ce19af6673b91dbc2144920815058d702"),
    "engineering_source": dict(id=14543254675, run_id=44554345276,
        run_head="3af2027d2ca9d195e18b1cf8da8e07663d841bc6", bytes=646524,
        sha256="530fccdd25abca9e76331dd8dad1f2440181158e140f2279ee952833a8d220f6"),
    "engineering_screen": dict(id=14545584039, run_id=44558582911,
        run_head="e7f3e56819b4156cec4fd965e5f0529adcc99a73", bytes=5586,
        sha256="9fd68b177823a24446c50c163a4838967de7b6e9d3e2be23861d136d93af0c9e"),
}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def valid_hash(value, size=64):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{" + str(size) + "}", value) is not None


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def module(name):
    sys.path.insert(0, str(HERE))
    return importlib.import_module(name)


def valid_pin(pin):
    need(type(pin) is dict and set(pin) == {"id", "run_id", "run_head", "bytes", "sha256"}
         and all(type(pin[key]) is int and pin[key] > 0 for key in ("id", "run_id", "bytes"))
         and valid_hash(pin["run_head"], 40) and valid_hash(pin["sha256"]), "exact original artifact identity")


def protocol():
    doc = json.loads(SPEC.read_bytes())
    need(doc.get("enabled") is True, "INERT: original B24 primary and replication PASSes are not bound")
    need(doc["schema"] == SCHEMA and doc["candidate_role"] == "challenger"
         and doc["assessment_sha256"] == ASSESSMENT_SHA and doc["training_design_sha256"] == DESIGN_SHA
         and doc["pipeline_commit"] == PIPELINE_COMMIT and doc["quality_reader"] == QUALITY_READER
         and doc["prepared_parent"] is None and doc["parent_authority"] == "reviewed_workflow"
         and doc["original_count"] == ORIGINALS and doc["conditions"] == list(CONDITIONS)
         and doc["maximum_backend_calls"] == MAX_CALLS
         and doc["onnx_max_abs"] == .001 and doc["onnx_mean_abs"] == .0001
         and doc["c_max_abs"] == .005 and doc["c_mean_abs"] == .0005
         and doc["same_job_matched_control"] is doc["historical_control_retrained"] is False
         and doc["model_comparison"] == MODEL_COMPARISON
         and all(doc[key] is False for key in ("training_allowed", "threshold_search", "new_source_allowed",
             "cal_locked_access", "physical_ssc305", "product_promotion_allowed", "product_decoder_settings_defined")),
         "same frozen B24 weights and numerical thresholds; software-only output")
    need(sha((HERE / "english_v6_b24_panels/assessment.json").read_bytes()) == ASSESSMENT_SHA
         and sha((HERE / "english_v6_b24_b23_warm_start_design.json").read_bytes()) == DESIGN_SHA,
         "original before-training scientific registration")
    need(valid_hash(doc["quality_reader_sha256"])
         and sha((HERE / (QUALITY_READER + ".py")).read_bytes()) == doc["quality_reader_sha256"],
         "reviewed original-array and producer/source-admission reader")
    need(set(doc["inputs"]) == set(FROZEN_INPUTS) | {"pair"}
         and all(doc["inputs"][key] == pin for key, pin in FROZEN_INPUTS.items()), "only existing exposed engineering PCM")
    for pin in doc["inputs"].values():
        valid_pin(pin)
    need(set(doc["quality_locks"]) == {"primary", "replication"}, "two actual acoustic results")
    for panel, lock in doc["quality_locks"].items():
        need(type(lock) is dict and set(lock) == {"schema", "panel", "result", "producer", "consumed",
             "evaluator_sha256", "inputs_sha256", "producer_manifest_sha256", "original_result_sha256"}
             and lock["schema"] == "english-v6-b24-original-quality-lock-v1" and lock["panel"] == panel,
             "panel-specific full quality readback lock")
        for key in ("result", "producer", "consumed"):
            valid_pin(lock[key])
        need(all(valid_hash(lock[key]) for key in
             ("evaluator_sha256", "inputs_sha256", "producer_manifest_sha256", "original_result_sha256"))
             and len({lock[key]["id"] for key in ("result", "producer", "consumed")}) == 3
             and len({(lock[key]["run_id"], lock[key]["run_head"]) for key in ("result", "producer", "consumed")}) == 1,
             "three distinct same-run originals and frozen actual quality producer")
    models = doc["model_pt_sha256"]
    need(set(models) == {"B0", "B9", "control", "challenger"}
         and all(valid_hash(v) for v in models.values()) and len(set(models.values())) == 4
         and models["B0"] == B0_PT and models["B9"] == B9_PT and models["control"] == CONTROL_PT
         and models["challenger"] == CANDIDATE_PT,
         "four unchanged primary and replication weights, with original historical B23 control")
    need(set(doc["candidate_files"]) == {STEM + "." + ext for ext in ("pt", "npz", "onnx")}, "three original challenger exports")
    for pin in doc["candidate_files"].values():
        need(set(pin) == {"bytes", "sha256"} and type(pin["bytes"]) is int and pin["bytes"] > 10000
             and valid_hash(pin["sha256"]), "full original candidate member bytes")
    need(doc["candidate_files"][STEM + ".pt"]["sha256"] == models["challenger"], "exact selected challenger")
    return doc


def source_check():
    doc = json.loads(SPEC.read_bytes())
    if doc.get("enabled") is True:
        protocol()
        return dict(schema=SCHEMA, status="BOUND_REQUIRES_ORIGINAL_DUAL_PASS_READBACK", model_forwards=0,
                    compiled_programs=0, packs_created=0, physical_ssc305=False, product_promotion_allowed=False)
    need(doc["schema"] == SCHEMA and doc["enabled"] is False
         and doc["quality_reader_sha256"] is None and all(v is None for v in doc["quality_locks"].values())
         and doc["inputs"]["pair"] is None and doc["candidate_files"] is None
         and doc["model_pt_sha256"] == dict(B0=B0_PT, B9=B9_PT, control=CONTROL_PT, challenger=None)
         and doc["same_job_matched_control"] is doc["historical_control_retrained"] is False
         and doc["model_comparison"] == MODEL_COMPARISON
         and PARENT_TOKEN in TEMPLATE.read_text(), "prepared package is explicitly inert")
    return dict(schema=SCHEMA, status="INERT_AWAIT_ORIGINAL_DUAL_QUALITY_PASS",
        pack_bytes=PACK_BYTES, maximum_backend_calls=MAX_CALLS, model_forwards=0,
        compiled_programs=0, packs_created=0, physical_ssc305=False, product_promotion_allowed=False)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def validate_event(prepared, head, parents, changed, env, event):
    ref = "refs/heads/" + BRANCH
    need(valid_hash(prepared, 40) and valid_hash(head, 40) and head != prepared
         and parents == [prepared] and changed == [WORKFLOW], "single reviewed parent; workflow-only activation")
    need(all(env.get(k) == v for k, v in dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY=REPO,
         GITHUB_EVENT_NAME="push", GITHUB_REF=ref, GITHUB_RUN_ATTEMPT="1", GITHUB_SHA=head).items())
         and re.fullmatch(r"[1-9][0-9]*", env.get("GITHUB_RUN_ID", "")), "first designated public push only")
    need(event.get("before") == prepared and event.get("after") == head and event.get("ref") == ref
         and event.get("forced") is event.get("deleted") is False
         and event.get("repository", {}).get("full_name") == REPO and event["repository"].get("private") is False,
         "no replay, force push, private execution or history jump")
    return dict(run_id=int(env["GITHUB_RUN_ID"]), run_head=head, run_attempt=1, prepared_parent=prepared)


def guard(prepared):
    head = git(ROOT, "rev-parse", "HEAD").decode().strip()
    parents = git(ROOT, "rev-list", "--parents", "-n", "1", "HEAD").decode().split()[1:]
    changed = git(ROOT, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").decode().splitlines()
    identity = validate_event(prepared, head, parents, changed, os.environ,
                              json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_bytes()))
    actual = (ROOT / WORKFLOW).read_bytes()
    need(actual == TEMPLATE.read_bytes().replace(PARENT_TOKEN.encode(), prepared.encode()),
         "actual runtime workflow differs only by reviewed parent placeholder")
    return identity


def gh(path):
    return subprocess.run(["gh", "api", "repos/" + REPO + "/" + path, "--allow-escape-sequences"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=180).stdout


def one_execution(identity, marker_present):
    run = json.loads(gh("actions/runs/" + str(identity["run_id"])))
    commit = json.loads(gh("commits/" + identity["run_head"]))
    need(run["head_sha"] == identity["run_head"] and run["run_attempt"] == 1
         and run["event"] == "push" and run["head_branch"] == BRANCH and run["path"] == WORKFLOW
         and run["repository"]["full_name"] == REPO and run["repository"]["private"] is False
         and [p["sha"] for p in commit["parents"]] == [identity["prepared_parent"]]
         and [f["filename"] for f in commit["files"]] == [WORKFLOW], "actual original run and reviewed activation commit")
    runs = json.loads(gh("actions/workflows/" + Path(WORKFLOW).name + "/runs?branch=" + BRANCH + "&event=push&per_page=100"))
    need(runs["total_count"] == len(runs["workflow_runs"]) == 1
         and runs["workflow_runs"][0]["id"] == identity["run_id"], "no earlier engineering execution, even expired")
    markers = json.loads(gh("actions/artifacts?name=" + MARKER + "&per_page=100"))
    need(markers["total_count"] == len(markers["artifacts"]) == int(marker_present), "one consumed engineering experiment")
    if marker_present:
        need(markers["artifacts"][0]["workflow_run"]["id"] == identity["run_id"]
             and markers["artifacts"][0]["workflow_run"]["head_sha"] == identity["run_head"], "current run owns consumed marker")


def original_zip(path, pin):
    valid_pin(pin)
    need(path.is_file() and not path.is_symlink(), "regular original archive")
    raw = path.read_bytes()
    need(len(raw) == pin["bytes"] and len(raw) <= 128_000_000 and sha(raw) == pin["sha256"], "full immutable original archive")
    archive = zipfile.ZipFile(io.BytesIO(raw))
    try:
        infos = archive.infolist()
        need(len(infos) <= 8192 and len(infos) == len({i.filename for i in infos})
             and sum(i.file_size for i in infos) <= 128_000_000, "bounded unique archive")
        for info in infos:
            path = PurePosixPath(info.filename)
            need(path.parts and not path.is_absolute() and ".." not in path.parts and "\\" not in info.filename
                 and not stat.S_ISLNK(info.external_attr >> 16) and not info.is_dir(), "safe regular original archive member")
        need(archive.testzip() is None, "original archive CRC")
        return archive
    except BaseException:
        archive.close(); raise


def authenticate_remote(pin):
    valid_pin(pin)
    run = json.loads(gh("actions/runs/" + str(pin["run_id"])))
    artifact = json.loads(gh("actions/artifacts/" + str(pin["id"])))
    need(run["head_sha"] == pin["run_head"] and run["run_attempt"] == 1 and run["status"] == "completed"
         and run["conclusion"] == "success" and run["event"] == "push"
         and run["repository"]["full_name"] == REPO and run["repository"]["private"] is False
         and artifact["id"] == pin["id"] and artifact["expired"] is False
         and artifact["size_in_bytes"] == pin["bytes"] and artifact["digest"] == "sha256:" + pin["sha256"]
         and artifact["workflow_run"]["id"] == pin["run_id"]
         and artifact["workflow_run"]["head_sha"] == pin["run_head"], "original successful public artifact provenance")
    return dict(run=run, artifact=artifact)


def validate_dual_quality(proofs, doc):
    need(doc['model_pt_sha256']['challenger'] == CANDIDATE_PT
         and doc['model_pt_sha256']['control'] == CONTROL_PT,
         'dual quality belongs to the actual frozen B24 candidate and original B23 control')
    need(set(proofs) == {"primary", "replication"}, "two original panel readbacks")
    for panel, proof in proofs.items():
        need(proof["schema"] == "english-v6-b24-verified-original-quality-v1" and proof["panel"] == panel
             and proof["panel_gate"] == "PASS" and proof["model_pt_sha256"] == doc["model_pt_sha256"]
             and proof["assessment_sha256"] == ASSESSMENT_SHA and proof["training_design_sha256"] == DESIGN_SHA
             and proof["pair_pin"] == doc["inputs"]["pair"]
             and proof["original_result_sha256"] == doc["quality_locks"][panel]["original_result_sha256"]
             and proof["all_original_arrays_verified"] is proof["original_source_admission_recomputed"]
                 is proof["original_producer_git_verified"] is True
             and proof["new_model_calls"] == proof["new_source_calls"] == proof["new_observer_calls"] == 0
             and proof["admission"]["same_job_matched_control"] is proof["admission"]["historical_control_retrained"] is False
             and proof["admission"]["model_comparison"] == MODEL_COMPARISON,
             "genuine original dual PASS of exactly one frozen B24 challenger")
    a, b = (proofs[p]["source_inventory"] for p in ("primary", "replication"))
    for key in ("pcm_sha256", "source_group_id", "canonical_people"):
        need(type(a[key]) is list and type(b[key]) is list and bool(a[key]) and bool(b[key])
             and len(a[key]) == len(set(a[key])) and len(b[key]) == len(set(b[key]))
             and not set(a[key]) & set(b[key]), "all original primary/replication sources independent: " + key)
    need(len(a["pcm_sha256"]) == len(b["pcm_sha256"]) == 120
         and doc["quality_locks"]["primary"]["result"]["run_id"] != doc["quality_locks"]["replication"]["result"]["run_id"],
         "two distinct complete original120 panels")
    return proofs


def quality(inputs, doc, fetch=False):
    # This exact reader independently verifies original arrays, source admission,
    # consumption journals and Git-bound producers. A summary PASS is insufficient.
    reader = module(QUALITY_READER)
    need(type(getattr(reader, "__file__", None)) is str
         and sha(Path(reader.__file__).read_bytes()) == doc["quality_reader_sha256"],
         "the executing original-quality reader matches the reviewed digest")
    proofs = {}
    for panel in ("primary", "replication"):
        directory = inputs / "quality" / panel
        if fetch:
            reader.fetch_quality_original(directory, doc["quality_locks"][panel])
        proofs[panel] = reader.verify_quality_original(directory, doc["quality_locks"][panel])
        need(proofs[panel]["panel_gate"] == "PASS", "original panel reader must pass before another panel is opened")
    return validate_dual_quality(proofs, doc)


def verify_pipeline(pipeline):
    expected = json.loads(PIPELINE_FILES.read_bytes())
    need(expected["commit"] == PIPELINE_COMMIT and expected["repository"] == PIPELINE_REPO
         and git(pipeline, "rev-parse", "HEAD").decode().strip() == PIPELINE_COMMIT, "immutable real C11 source commit")
    snapshot = module("english_v6_b21_producer_snapshot")
    entries = snapshot.tree_entries(git(pipeline, "ls-tree", "-rz", "--full-tree", "HEAD"), set(expected["files"]))
    for path, pin in expected["files"].items():
        local = pipeline / path
        need(local.is_file() and not local.is_symlink() and path in entries
             and local.stat().st_size == pin["bytes"] and blob(local.read_bytes()) == pin["git_blob_sha1"]
             == entries[path]["git_blob_sha1"], "unchanged compiled/installed source " + path)
    return expected


def producer_paths(tracked, root=ROOT):
    """Retain the actual library and all upstream resources, including dynamic reads."""
    root = Path(root)
    paths = {name for name in tracked if name.startswith(("automation/", ".github/workflows/"))}
    original = json.loads((root / "automation/english_v6_b24_source_training_producer.json").read_bytes())
    need(original["schema"] == "english-v6-b24-complete-git-producer-v1" and len(original["files"]) > 306
         and original["original_b23_dependency_files"] == 306,
         "complete original B23-warm-start TRAIN producer and every inherited historical resource")
    paths |= set(original["files"])
    for panel in ("primary", "replication"):
        name = "automation/english_v6_b24_" + panel + "_producer.json"
        value = json.loads((root / name).read_bytes())
        need(value["frozen"] is True, "original dual-PASS producer binding must be preserved: " + panel)
        paths.add(name)
        for role in (panel, "source"):
            paths |= set(value[role]["files"])
            paths |= set(value[role].get("resources", {}))
        paths |= set(value["resources"])
        for upstream in value["upstream_manifests"].values():
            paths |= set(upstream["files"])
    required = {WORKFLOW, "automation/english_v6_b24_engineering.py",
                "automation/test_english_v6_b24_engineering.py", "automation/" + SPEC.name,
                "automation/" + TEMPLATE.name, "automation/" + PIPELINE_FILES.name,
                "automation/" + QUALITY_READER + ".py"}
    need(required <= paths and paths <= set(tracked), "all engineering and original dynamic dependencies are tracked")
    return paths


def producer(inputs, pipeline, identity):
    snapshot = module("english_v6_b21_producer_snapshot")
    entries = snapshot.tree_entries(git(ROOT, "ls-tree", "-rz", "--full-tree", "HEAD"))
    paths = producer_paths(entries)
    out = inputs / "producer"; out.mkdir()
    files = {}
    for path in sorted(paths):
        local = ROOT / path
        need(local.is_file() and not local.is_symlink() and path in entries
             and blob(local.read_bytes()) == entries[path]["git_blob_sha1"], "actual reviewed producer Git blob " + path)
        raw = local.read_bytes(); target = out / "repository" / path
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
        files[path] = dict(**entries[path], bytes=len(raw), sha256=sha(raw))
    foreign = verify_pipeline(pipeline)
    cfiles = {}
    for path, pin in foreign["files"].items():
        raw = (pipeline / path).read_bytes(); target = out / "pipeline" / path
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
        cfiles[path] = dict(**pin, sha256=sha(raw))
    write(out / "PRODUCER.json", dict(schema=SCHEMA + "-producer", **identity,
        git_tree=git(ROOT, "rev-parse", "HEAD^{tree}").decode().strip(), files=files,
        pipeline_commit=PIPELINE_COMMIT, pipeline_files=cfiles, include_hidden_files=True,
        closure_rule="complete_tracked_automation_and_workflows_plus_both_original_quality_producers",
        new_model_forwards=0, packs_created=0))


def verify_producer(inputs, pipeline, identity):
    need(not git(ROOT, "status", "--porcelain", "--untracked-files=no")
         and not git(pipeline, "status", "--porcelain", "--untracked-files=no"),
         "all actual tracked producer bytes remain equal to reviewed Git HEADs")
    saved = json.loads((inputs / "producer/PRODUCER.json").read_bytes())
    need(saved["schema"] == SCHEMA + "-producer"
         and all(saved[key] == value for key, value in identity.items())
         and saved["pipeline_commit"] == PIPELINE_COMMIT and saved["include_hidden_files"] is True,
         "same actual original producer identity")
    tracked = module("english_v6_b21_producer_snapshot").tree_entries(
        git(ROOT, "ls-tree", "-rz", "--full-tree", "HEAD"))
    need(set(saved["files"]) == producer_paths(tracked)
         and set(saved["pipeline_files"]) == set(json.loads(PIPELINE_FILES.read_bytes())["files"]),
         "complete original engineering and C11 producer inventories")
    for label, root, files in (("repository", ROOT, saved["files"]),
                               ("pipeline", pipeline, saved["pipeline_files"])):
        need(type(files) is dict and bool(files), "complete saved producer file map")
        for name, pin in files.items():
            live = root / name; recorded = inputs / "producer" / label / name
            need(not Path(name).is_absolute() and ".." not in Path(name).parts
                 and live.is_file() and not live.is_symlink() and recorded.is_file() and not recorded.is_symlink(),
                 "regular exact producer source")
            raw = live.read_bytes()
            need(recorded.read_bytes() == raw and len(raw) == pin["bytes"]
                 and sha(raw) == pin["sha256"] and blob(raw) == pin["git_blob_sha1"],
                 "executed producer bytes still equal actual Git snapshot " + name)


def fetch(inputs, pipeline, prepared):
    doc = protocol(); identity = guard(prepared); one_execution(identity, False)
    need(inputs.is_absolute() and inputs.parent.is_dir() and not inputs.exists(), "fresh immutable input directory")
    inputs.mkdir(); (inputs / "quality").mkdir()
    producer(inputs, pipeline, identity)
    # Readback may collect original historical model/source ZIPs as evidence;
    # no candidate load, pack or engineering PCM access precedes both PASSes.
    proofs = quality(inputs, doc, fetch=True)
    provenance = {}
    for name, pin in doc["inputs"].items():
        provenance[name] = authenticate_remote(pin)
        path = inputs / (name + ".zip")
        with path.open("xb") as handle:
            handle.write(gh("actions/artifacts/" + str(pin["id"]) + "/zip"))
        with original_zip(path, pin):
            pass
    write(inputs / "FETCH.json", dict(schema=SCHEMA, identity=identity,
        input_spec_sha256=sha(SPEC.read_bytes()), quality=proofs, provenance=provenance,
        producer_manifest_sha256=sha((inputs / "producer/PRODUCER.json").read_bytes())))


def original_pair(inputs, doc):
    need(doc['model_pt_sha256']['challenger'] == CANDIDATE_PT,
         'only the actual completed B24 challenger can enter engineering')
    with original_zip(inputs / "pair.zip", doc["inputs"]["pair"]) as archive:
        receipt = json.loads(archive.read("RESULT.json"))
        primary = module("english_v6_b24_primary_eval")
        need(primary.STEMS == STEMS and primary.MODEL_COMPARISON == MODEL_COMPARISON
             and primary.CONTROL_PT == CONTROL_PT, "same new candidate and immutable historical control contract")
        primary.validate_pair_receipt(receipt, doc)
        need(receipt["design_sha256"] == DESIGN_SHA and receipt["assessment_sha256"] == ASSESSMENT_SHA
             and receipt["challenger"]["files"] == doc["candidate_files"]
             and receipt["same_job_matched_control"] is False
             and receipt["new_control_optimizer_updates"] == 0, "original selected single-challenger TRAIN artifact")
        for arm in ("control", "challenger"):
            stem = STEMS[arm]
            for suffix in ("pt", "npz", "onnx"):
                raw = archive.read(stem + "." + suffix)
                need(receipt[arm]["files"][Path(stem).name + "." + suffix] == dict(bytes=len(raw), sha256=sha(raw)),
                     "exact original B24 challenger or historical B23 control export")
                if suffix == "pt":
                    need(sha(raw) == doc["model_pt_sha256"][arm], "same original acoustic-gate candidate")
        return {suffix: archive.read("challenger/models/" + STEM + "." + suffix) for suffix in ("pt", "npz", "onnx")}


def engineering_sources(inputs, doc):
    old = module("english_v6_mero_frozen_b0_b1_paired_once")
    with tempfile.TemporaryDirectory(prefix="b24-old-engineering-") as temp:
        base = Path(temp)
        for kind in ("source", "screen"):
            name = "engineering_" + kind; target = base / kind; target.mkdir()
            with original_zip(inputs / (name + ".zip"), doc["inputs"][name]) as archive:
                archive.extractall(target)
        rows, admission = old.select(base / "source", base / "screen")
        need(len(rows) == ORIGINALS and sum(r["mono_class"] for r in rows) == 6
             and admission["no_locked_or_calibration"] is True, "same14 previously exposed engineering originals")
        waves = {row["id"]: old.pcm(row, base / "source") for row in rows}
    return rows, waves, admission


def preflight(inputs, pipeline, prepared, write_permit=False):
    doc = protocol(); identity = guard(prepared)
    recorded = json.loads((inputs / "FETCH.json").read_bytes())
    need(recorded["identity"] == identity and recorded["input_spec_sha256"] == sha(SPEC.read_bytes())
         and recorded["producer_manifest_sha256"] == sha((inputs / "producer/PRODUCER.json").read_bytes()),
         "same prepared original inputs and exact before-execution producer")
    verify_producer(inputs, pipeline, identity)
    proofs = quality(inputs, doc)
    need(proofs == recorded["quality"], "independently reread same full original PASS evidence")
    verify_pipeline(pipeline)
    for name, pin in doc["inputs"].items():
        with original_zip(inputs / (name + ".zip"), pin):
            pass
    original_pair(inputs, doc)
    rows, waves, source_admission = engineering_sources(inputs, doc)
    admission = dict(schema=SCHEMA, identity=identity, input_spec_sha256=sha(SPEC.read_bytes()),
        candidate_files=doc["candidate_files"], quality=proofs, source_admission=source_admission,
        pcm_sha256={row["id"]: sha(waves[row["id"]]) for row in rows}, pipeline_commit=PIPELINE_COMMIT,
        original_count=ORIGINALS, maximum_backend_calls=MAX_CALLS, optimizer_updates=0,
        same_job_matched_control=False, historical_control_retrained=False, model_comparison=MODEL_COMPARISON)
    if write_permit:
        one_execution(identity, False)
        write(inputs / "CONSUMPTION.json", dict(identity=identity, marker_name=MARKER,
              admission_sha256=sha(json.dumps(admission, sort_keys=True, separators=(",", ":")).encode())))
    return doc, identity, rows, waves, admission


def ctc(labels):
    need(bool(labels) and all(type(x) is int and 0 <= x < 4 for x in labels), "finite four-class argmax sequence")
    onsets = [(i, token) for i, token in enumerate(labels) if token and (i == 0 or labels[i - 1] != token)]
    tokens = [token for _, token in onsets]
    frames = [onsets[i + 2][0] for i in range(max(0, len(tokens) - 2)) if tokens[i:i + 3] == [1, 2, 3]]
    return dict(tokens=tokens, event_frames=frames, event_samples=[512 + 160 * i for i in frames])


def compare_logits(reference, observed, maximum, mean):
    need(len(reference) == len(observed) and len(reference) > 0 and len(reference) % 16 == 0, "complete equal four-class frame bytes")
    a = list(struct.iter_unpack("<4f", reference)); b = list(struct.iter_unpack("<4f", observed))
    need(all(math.isfinite(x) for row in a + b for x in row), "all raw output logits finite")
    delta = [abs(x - y) for row, other in zip(a, b) for x, y in zip(row, other)]
    aa = [max(range(4), key=row.__getitem__) for row in a]
    bb = [max(range(4), key=row.__getitem__) for row in b]
    need(max(delta) <= maximum and sum(delta) / len(delta) <= mean and aa == bb,
         "predeclared numerical and full-frame token parity")
    need(ctc(aa) == ctc(bb), "full collapsed tokens and every event frame/sample clock")
    return dict(max_abs=max(delta), mean_abs=sum(delta) / len(delta), frames=len(a),
                all_frame_tokens_equal=True, **ctc(aa))


def validate_events(raw, frame_count, expected):
    lines = raw.decode("ascii").splitlines(); observed = []
    need(len(lines) == frame_count, "one native event row per actual frame")
    for index, line in enumerate(lines):
        need(re.fullmatch(r"[0-9]+,[0-9]+,[01]", line), "complete native event CSV")
        frame, sample, hit = map(int, line.split(","))
        need(frame == index and sample == 512 + 160 * index, "exact native frame/sample availability clock")
        if hit:
            observed.append(index)
    need(observed == expected["event_frames"]
         and [512 + 160 * i for i in observed] == expected["event_samples"], "every native event frame equals PyTorch")


def load_candidate(inputs, doc, native, torch, np):
    values = original_pair(inputs, doc)
    state = torch.load(io.BytesIO(values["pt"]), weights_only=True, map_location="cpu")
    old = module("english_v6_mero_frozen_b0_b1_paired_once")
    net = old.v5.build_model(torch); net.load_state_dict(state, strict=True); net.eval()
    need(sum(p.numel() for p in net.parameters()) == 43876, "unchanged full candidate architecture")
    spec = importlib.util.spec_from_file_location("b24_original_b0_geometry", native / "pack_mero43.py")
    packer = importlib.util.module_from_spec(spec); spec.loader.exec_module(packer)
    need(packer.PACK_BYTES == PACK_BYTES and packer.PACK_FLOAT_COUNT == 54236 and packer.MAGIC == b"MERO43V1", "unchanged original C11 binary ABI")
    buffers = []
    with np.load(io.BytesIO(values["npz"]), allow_pickle=False) as encoded:
        need(set(encoded.files) == {key for key, _ in packer.geometry()}, "exact NPZ parameter/frontend geometry")
        for key, shape in packer.geometry():
            value = encoded[key]
            need(value.shape == shape and value.dtype == np.dtype("float32") and np.isfinite(value).all(), "finite original float32 tensor " + key)
            if key in state:
                need(np.array_equal(value, state[key].cpu().numpy()), "PT/NPZ tensor byte identity")
            if key == "train_scale":
                need(bool(np.all(value > 0)), "strictly positive frozen train normalization")
            if key == "mel_filters":
                need(bool(np.all((value >= 0) & (value <= 1)) and np.all(value.sum(axis=0) > 0)), "valid fixed Mel bank")
            buffers.append(np.ascontiguousarray(value, dtype="<f4").tobytes())
        frontend = tuple(encoded[key].copy() for key in ("mel_filters", "train_mean", "train_scale"))
    with original_zip(inputs / "B0.zip", doc["inputs"]["B0"]) as archive:
        raw = archive.read("models/mero-fsmn-ctc-expanded256.npz")
        need(sha(raw) == B0_NPZ, "original immutable B0 frontend container")
        with np.load(io.BytesIO(raw), allow_pickle=False) as original:
            need(all(np.array_equal(value, original[key]) for key, value in zip(("mel_filters", "train_mean", "train_scale"), frontend)), "same untouched B0 frontend and train-only normalization")
    packed = struct.pack("<8sII", b"MERO43V1", 1, 54236) + b"".join(buffers)
    need(len(packed) == PACK_BYTES, "exact 216960-byte B24 pack")
    return net, frontend, packed, values, old


def command(argv, log, timeout=180):
    with log.open("xb") as out:
        result = subprocess.run([str(x) for x in argv], stdout=out, stderr=subprocess.STDOUT, timeout=timeout)
    need(result.returncode == 0, "command failed; retained original output: " + log.name)


def build_sdk(pipeline, output, scratch):
    native = pipeline / "research/english_v5_mero"; executables = {}
    for backend in BACKENDS[2:]:
        arm = backend == "cortex-a32-qemu"
        cc = "arm-linux-gnueabihf-gcc" if arm else backend
        build = scratch / ("build-" + backend); install = output / ("sdk-" + backend)
        flags = ARM_FLAGS if arm else "-fno-fast-math -ffp-contract=off"
        configure = ["cmake", "-S", native, "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER=" + cc,
            "-DCMAKE_C_FLAGS=" + flags, "-DCMAKE_INSTALL_PREFIX=" + str(install), "-DHEY_MERO43_BUILD_TESTS=ON",
            "-DHEY_MERO43_BUILD_HOST_TOOL=ON", "-DHEY_MERO43_BUILD_LIVE_TOOL=ON"]
        if arm:
            configure += ["-DCMAKE_SYSTEM_NAME=Linux", "-DCMAKE_SYSTEM_PROCESSOR=arm", "-DCMAKE_EXE_LINKER_FLAGS=-static", "-DCMAKE_CROSSCOMPILING_EMULATOR=qemu-arm"]
        for label, cmd in (("configure", configure), ("build", ["cmake", "--build", build, "--parallel", "2"]),
                           ("ctest", ["ctest", "--test-dir", build, "--output-on-failure"]),
                           ("install", ["cmake", "--install", build])):
            command(cmd, output / "logs" / (backend + "-" + label + ".txt"))
        inventory = subprocess.check_output(["ctest", "--test-dir", str(build), "-N"], text=True, timeout=30)
        need("Total Tests: 2" in inventory, "both actual synthetic CTest cases were installed and run")
        (output / "logs" / (backend + "-ctest-inventory.txt")).write_text(inventory)
        cli = install / "bin/heym43_host_cli"
        need(cli.is_file() and (install / "lib/libheym43_stream.a").is_file()
             and (install / "include/heym43/mero43_stream.h").is_file()
             and (install / "lib/cmake/HeyMero43/HeyMero43Config.cmake").is_file(), "real installed complete SDK")
        consumer = scratch / ("consumer-" + backend); consumer.mkdir()
        (consumer / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.18)\nproject(B24Consumer C)\nfind_package(HeyMero43 0.1 CONFIG REQUIRED)\nadd_executable(consumer main.c)\ntarget_link_libraries(consumer PRIVATE HeyMero43::stream)\n")
        (consumer / "main.c").write_text('#include "mero43_stream.h"\nint main(void){return en_v5_m43_model_bytes()==0 || en_v5_m43_stream_bytes()==0;}\n')
        configure = ["cmake", "-S", consumer, "-B", consumer / "build", "-DCMAKE_C_COMPILER=" + cc,
                     "-DCMAKE_C_FLAGS=" + flags, "-DCMAKE_PREFIX_PATH=" + str(install)]
        if arm:
            configure += ["-DCMAKE_SYSTEM_NAME=Linux", "-DCMAKE_SYSTEM_PROCESSOR=arm", "-DCMAKE_EXE_LINKER_FLAGS=-static"]
        command(configure, output / "logs" / (backend + "-consumer-configure.txt"))
        command(["cmake", "--build", consumer / "build", "--parallel", "2"], output / "logs" / (backend + "-consumer-build.txt"))
        command((["qemu-arm"] if arm else []) + [consumer / "build/consumer"], output / "logs" / (backend + "-consumer-run.txt"))
        if arm:
            elf = subprocess.check_output(["readelf", "-h", str(cli)], text=True, timeout=20)
            need(all(value in elf for value in ("ELF32", "ARM", "little endian", "hard-float ABI")), "actual Cortex-A32 ARM hard-float executable")
            (output / "logs/arm-elf.txt").write_text(elf)
        else:
            command([sys.executable, "-I", "-B", native / "test_live_input.py", install / "bin/heym43-live"], output / "logs" / (backend + "-live-input.txt"))
        executables[backend] = (["qemu-arm"] if arm else []) + [str(cli)]
    return executables


class Ledger:
    def __init__(self, output):
        self.output = output; self.entered = {}; self.saved = set()

    def begin(self, key, backend):
        need(re.fullmatch(r"[A-Za-z0-9_.-]+", key) and key not in self.entered and len(self.entered) < MAX_CALLS
             and backend in BACKENDS, "one bounded backend call per original/condition, no retry")
        write(self.output / (key + ".json"), dict(key=key, backend=backend, status="consumed_unknown"))
        self.entered[key] = backend

    def complete(self, key):
        need(key in self.entered and key not in self.saved, "one saved raw output per entered call")
        self.saved.add(key)


def execute(inputs, pipeline, output, prepared):
    doc, identity, rows, waves, admission = preflight(inputs, pipeline, prepared)
    permit = json.loads((inputs / "CONSUMPTION.json").read_bytes())
    need(permit == dict(identity=identity, marker_name=MARKER,
         admission_sha256=sha(json.dumps(admission, sort_keys=True, separators=(",", ":")).encode())), "exact uploaded preflight consumption")
    one_execution(identity, True)
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(), "new immutable SDK evidence output")
    output.mkdir()
    for name in ("logs", "attempts", "raw_outputs", "models"):
        (output / name).mkdir()
    ledger = Ledger(output / "attempts"); report = dict(schema=SCHEMA, identity=identity, status="STARTED",
        candidate_files=doc["candidate_files"], pipeline_commit=PIPELINE_COMMIT, input_spec_sha256=sha(SPEC.read_bytes()),
        quality=admission["quality"], original_count=ORIGINALS, conditions=list(CONDITIONS), backends=list(BACKENDS),
        records=[], optimizer_updates=0, new_tts_asr_calls=0, independent_quality_claimed=False,
        same_job_matched_control=False, historical_control_retrained=False, model_comparison=MODEL_COMPARISON,
        physical_ssc305=False, field_frr=None, continuous_far_per_hour=None, word_end_p95_ms=None,
        product_promotion_allowed=False, product_decoder_settings=None, replay_allowed=False)
    write(output / "STARTED.json", report); write(output / "ADMISSION.json", admission)
    try:
        import numpy as np
        import torch
        import onnxruntime as ort
        need(np.__version__ == "1.26.4" and torch.__version__ == "2.9.1+cpu" and ort.__version__ == "1.22.1", "pinned numeric runtime")
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True); torch.backends.mkldnn.enabled = False
        native = pipeline / "research/english_v5_mero"
        net, frontend, packed, values, old = load_candidate(inputs, doc, native, torch, np)
        pack = output / "models/b24-challenger.pack"; pack.write_bytes(packed)
        for suffix, raw in values.items():
            (output / "models" / (STEM + "." + suffix)).write_bytes(raw)
        write(output / "PACK.json", dict(schema=SCHEMA, bytes=len(packed), sha256=sha(packed),
            candidate_files=doc["candidate_files"], pair_pin=doc["inputs"]["pair"], layout="MERO43V1", quality_passes=2))
        options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(str(output / "models" / (STEM + ".onnx")), sess_options=options, providers=["CPUExecutionProvider"])
        need([x.name for x in session.get_inputs()] == ["fbank40"]
             and [x.name for x in session.get_outputs()] == ["frame_ctc_logits"], "original ONNX input/output ABI")
        with tempfile.TemporaryDirectory(prefix="b24-sdk-build-") as temp:
            scratch = Path(temp); binaries = build_sdk(pipeline, output, scratch)
            mel, mean, scale = frontend
            for row in rows:
                for condition in CONDITIONS:
                    signal = (b"\0" * 32000 if condition == CONDITIONS[1] else b"") + waves[row["id"]]
                    need(512 <= len(signal) // 2 < 1000000, "bounded engineering clips below frozen offline decoder gap")
                    case = row["id"] + "__" + condition
                    pcm = scratch / (case + ".pcm"); pcm.write_bytes(signal)
                    feature = old.v5.features(signal, {**row, "samples": len(signal) // 2}, torch, np, mel)
                    x = ((feature - mean) / scale).astype(np.float32)[None, :, :]
                    frames = 1 + (len(signal) // 2 - 512) // 160
                    need(x.shape == (1, frames, 40), "same native center-free frame grid")
                    key = case + "__pytorch"; ledger.begin(key, "pytorch")
                    with torch.inference_mode():
                        y = net(torch.from_numpy(x)).detach().cpu().numpy()
                    need(y.shape == (1, frames, 4) and y.dtype == np.dtype("float32") and np.isfinite(y).all(), "full original PyTorch output")
                    reference = np.ascontiguousarray(y[0], dtype="<f4").tobytes()
                    (output / "raw_outputs" / (key + ".f32")).write_bytes(reference); ledger.complete(key)
                    expected = ctc(y[0].argmax(-1).tolist())
                    record = dict(id=row["id"], condition=condition, source_pcm_sha256=sha(waves[row["id"]]),
                        derived_pcm_sha256=sha(signal), frames=frames, pytorch_sha256=sha(reference), **expected, comparisons={})
                    key = case + "__onnxruntime"; ledger.begin(key, "onnxruntime")
                    y = session.run(["frame_ctc_logits"], {"fbank40": x})[0]
                    need(y.shape == (1, frames, 4) and y.dtype == np.dtype("float32"), "full original ONNX output")
                    raw = np.ascontiguousarray(y[0], dtype="<f4").tobytes()
                    (output / "raw_outputs" / (key + ".f32")).write_bytes(raw); ledger.complete(key)
                    record["comparisons"]["onnxruntime"] = dict(sha256=sha(raw), **compare_logits(reference, raw, .001, .0001))
                    for backend, binary in binaries.items():
                        key = case + "__" + backend; ledger.begin(key, backend)
                        logits = output / "raw_outputs" / (key + ".f32")
                        events = output / "raw_outputs" / (key + ".csv")
                        command(binary + [pack, pcm, logits, events], output / "logs" / (key + ".txt"), timeout=130)
                        raw = logits.read_bytes(); event_raw = events.read_bytes(); ledger.complete(key)
                        comparison = compare_logits(reference, raw, .005, .0005)
                        validate_events(event_raw, frames, expected)
                        need("MERO43_C_PCM_STREAM_OK" in (output / "logs" / (key + ".txt")).read_text(), "actual installed C stream returned")
                        record["comparisons"][backend] = dict(sha256=sha(raw), event_csv_sha256=sha(event_raw), **comparison)
                    report["records"].append(record)
        need(len(report["records"]) == 28 and len(ledger.entered) == len(ledger.saved) == MAX_CALLS, "all 28 complete five-backend engineering cases")
        (output / "LICENSE").write_bytes((pipeline / "LICENSE").read_bytes())
        (output / "README.md").write_text("# B24 Hey Mero software SDK candidate\n\nThis immutable candidate passed original primary and replication acoustic gates and saved PT/ONNX/GCC/Clang/ARM-QEMU numerical replay. See RESULT.json, PACK.json and ADMISSION.json for exact provenance.\n\nUse the installed HeyMero43::stream target and authenticate models/b24-challenger.pack before load. The live adapter requires explicit max-gap/cooldown values; this package defines no product decoder settings. The engineering host CLI uses a long gap and zero cooldown only for offline numerical parity.\n\nSSC305 BSP/ABI, physical AFE/callback, board resources, real-human3–5m FRR, continuous natural FAR/h and physical latency remain NOT_RUN. Ubuntu hard-float/QEMU builds are software evidence. Do not identify this artifact as a physically qualified board model.\n")
        report.update(status="PASS_B24_DUAL_QUALITY_SOFTWARE_SDK_CANDIDATE", pack_sha256=sha(packed), pack_bytes=len(packed),
                      full_frame_tokens_and_event_sample_clocks_equal=True, sdk_installed_and_consumed=True)
    except BaseException as exc:
        report.update(status="STOP_ENGINEERING_NO_RETRY", error_type=type(exc).__name__, error=str(exc)[:2000])
        raise
    finally:
        report.update(entered_backend_calls=len(ledger.entered), saved_backend_outputs=len(ledger.saved),
            consumed_unknown=sorted(set(ledger.entered) - ledger.saved),
            backend_calls={backend: list(ledger.entered.values()).count(backend) for backend in BACKENDS})
        write(output / "RESULT.json", report)
        files = {str(p.relative_to(output)): dict(bytes=p.stat().st_size, sha256=sha(p.read_bytes()))
                 for p in sorted(output.rglob("*")) if p.is_file()}
        write(output / "MANIFEST.json", dict(schema=SCHEMA, status=report["status"], files=files,
              physical_ssc305=False, product_promotion_allowed=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "fetch", "preflight", "execute"))
    parser.add_argument("--inputs", type=Path); parser.add_argument("--pipeline", type=Path)
    parser.add_argument("--output", type=Path); parser.add_argument("--prepared-parent")
    args = parser.parse_args()
    if args.mode == "check":
        print(json.dumps(source_check()))
    else:
        protocol()  # Closed before path reads, imports, network, pack or compiler.
        need(args.inputs is not None and args.pipeline is not None, "original inputs and immutable pipeline checkout")
        if args.mode == "fetch":
            fetch(args.inputs, args.pipeline, args.prepared_parent)
        elif args.mode == "preflight":
            preflight(args.inputs, args.pipeline, args.prepared_parent, write_permit=True)
        else:
            need(args.output is not None, "new output required")
            execute(args.inputs, args.pipeline, args.output, args.prepared_parent)
