"""Authenticate and copy the exact 96 already-consumed B21 synthetic originals.

This is a readback, not a source authorization or a replacement renderer.  The
original source run remains STOP.  A separate registered amendment must govern
any new natural material, observations, scoring or replication access.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPO = "jiying2007/kws-data"
RUN = 45367984097
HEAD = "c262e4b5cf06de8a6950c2c797c7aefcf43498d8"
PARENT = "c89c4742c18ac328e5786fec90219cd0cf04b498"
TREE = "6edd6a410611668c25851736494aea1990a9cdc2"
BRANCH = "research/en-v6-b21-primary-source-fixed-mirror-once-20261124"
WORKFLOW = ".github/workflows/english-v6-b21-primary-source-fixed-mirror-once.yml"
IDENTITY = dict(run_id=str(RUN), run_head=HEAD, run_attempt=1,
                run_branch="refs/heads/" + BRANCH, prepared_parent=PARENT)
SCHEMA = "english-v6-b21-exact-consumed-synthetic-reuse-v1"
HISTORY_SCHEMA = "english-v6-b21-consumed-synthetic-complete-history-v1"
MANIFEST_SHA256 = "d09c3c7032d199a5d4a953545106e9242d87c5da37ddcc366bb676f0f20af0f8"
ROWS_SHA256 = "70ea5b39785736953ae1dcfe469fbc776470e37c641de8f907a2a01ae10a150f"
MEMBERS_SHA256 = "4de8717a63cebb57958a2a4b5ebb6378103b3060b48db78ac3e6ebce2b1a5827"
COMPLETE_HISTORY_SHA256 = "69ce3392945fb09a5874bbcefd90d4269c6d4fbebe0834a3127b822c6266e3b3"
LOG_SHA256 = "3f33c91f70a7500ed31ef6c37301512225973b06025e6d861cdadbd2cbe4716d"
BUNDLE_ARTIFACTS = {
    "producer-original.zip": dict(id=14908620422, run_id=RUN, run_head=HEAD, bytes=5707264,
        sha256="edc6fe1e64fa7a3b36dace3608b81290059321f3dbd78d42ba11c684b5e3af51"),
    "transport-original.zip": dict(id=14908530897, run_id=RUN, run_head=HEAD, bytes=27586,
        sha256="a4b07bb458e4d1dca08afbf3f27ff8cbb1850a792eed89de8a2e830e19cb28d0"),
    "consumed-original.zip": dict(id=14908181309, run_id=RUN, run_head=HEAD, bytes=1583,
        sha256="596cbd3b43cbf124aab89e85814c1d22a304f9aad479c6bd99264196f0cd4f62"),
    "source-original.zip": dict(id=14907672424, run_id=RUN, run_head=HEAD, bytes=5726331,
        sha256="e7a9cb7c22c701222806ec53cdd241b8d1fdd9e2d5ec38fea03dc46d90980bfd"),
    "B20_SOURCE.zip": dict(id=14892539914, run_id=45334413855,
        run_head="249449c8a572c536dfb4e990584b451247d7da17", bytes=8974682,
        sha256="de5c580f7c6cdd99abd3268f5667b60d748d21f15ab5e2451123962b26775505"),
}
SOURCE_PIN = BUNDLE_ARTIFACTS["source-original.zip"]
API_ENDPOINTS = {"api/" + name: "repos/" + REPO + "/" + endpoint for name, endpoint in {
    "run-final.json": "actions/runs/" + str(RUN),
    "jobs-final.json": "actions/runs/" + str(RUN) + "/jobs?filter=all&per_page=100",
    "artifacts-final.json": "actions/runs/" + str(RUN) + "/artifacts?per_page=100",
    "commit.json": "git/commits/" + HEAD,
    "tree.json": "git/trees/" + TREE + "?recursive=1",
    "prepared_main_comparison.json": "compare/" + PARENT + "..." + HEAD,
    "source-job.log": "actions/jobs/138354602305/logs",
    "B20-run.json": "actions/runs/45334413855",
    "B20-artifact.json": "actions/artifacts/14892539914",
}.items()}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def canonical_person(value):
    need(type(value) is str and re.fullmatch(r"p?[0-9]+", value) is not None,
         "explicit numeric canonical reader identity")
    return str(int(value.removeprefix("p")))


def opened(bundle, filename):
    path = Path(bundle) / filename
    need(path.is_file() and not path.is_symlink(), "regular immutable evidence archive")
    raw, pin = path.read_bytes(), BUNDLE_ARTIFACTS[filename]
    need(len(raw) == pin["bytes"] and digest(raw) == pin["sha256"], "exact original archive " + filename)
    archive = zipfile.ZipFile(io.BytesIO(raw))
    items = archive.infolist()
    need(archive.testzip() is None and len(items) == len({i.filename for i in items}),
         "full original ZIP CRC and unique members")
    for item in items:
        path = PurePosixPath(item.filename)
        need(path.parts and not path.is_absolute() and ".." not in path.parts and "\\" not in item.filename
             and not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16), "safe original regular member")
    return archive


def retain(archive, output):
    for item in archive.infolist():
        destination = Path(output) / item.filename
        need(not destination.exists() and not destination.is_symlink(), "never overwrite an original")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(archive.read(item))


def validate_artifact(artifact, pin):
    need(artifact["id"] == pin["id"] and artifact["size_in_bytes"] == pin["bytes"]
         and artifact["digest"] == "sha256:" + pin["sha256"] and artifact["expired"] is False
         and artifact["workflow_run"]["id"] == pin["run_id"]
         and artifact["workflow_run"]["head_sha"] == pin["run_head"], "original public artifact byte identity")


def validate_remote(run, jobs, artifacts):
    """Caller must acquire fresh raw API objects before a separately guarded reuse."""
    need(run["id"] == RUN and run["head_sha"] == HEAD and run["run_attempt"] == 1
         and run["status"] == "completed" and run["conclusion"] == "failure" and run["event"] == "push"
         and run["path"] == WORKFLOW and run["head_branch"] == BRANCH
         and run["repository"]["full_name"] == run["head_repository"]["full_name"] == REPO
         and run["repository"]["private"] is run["head_repository"]["private"] is False,
         "actual consumed source failure; never reclassify this as a zero-call failure")
    need(jobs["total_count"] == len(jobs["jobs"]) == 3
         and artifacts["total_count"] == len(artifacts["artifacts"]) == 4, "complete terminal job and artifact inventories")
    by_name = {j["name"]: j for j in jobs["jobs"]}
    need(set(by_name) == {"sources", "observers", "readback"}
         and all(j["run_id"] == RUN and j["status"] == "completed" for j in jobs["jobs"]), "complete original job graph")
    source = by_name["sources"]
    need(source["id"] == 138354602305 and source["conclusion"] == "failure", "exact failed original source job")
    states = [(n, "success") for n in range(1, 11)] + [
        (11, "failure"), (12, "success"), (23, "skipped"), (24, "success"), (25, "success")]
    need([(s["number"], s["conclusion"]) for s in source["steps"]] == states
         and all(s["status"] == "completed" for s in source["steps"])
         and next(s["name"] for s in source["steps"] if s["number"] == 11)
             == "Generate the original96synthetic and24whole natural sources exactly once",
         "transport and marker succeeded, generation failed, original sources retained")
    for name, ident in (("observers", 138356964398), ("readback", 138356964588)):
        job = by_name[name]
        need(job["id"] == ident and job["conclusion"] == "skipped" and not job["steps"], "native observers and readback never started")
    pins = {p["id"]: p for p in BUNDLE_ARTIFACTS.values() if p["run_id"] == RUN}
    need({a["id"] for a in artifacts["artifacts"]} == set(pins), "all four originals and no observer artifact")
    for artifact in artifacts["artifacts"]:
        validate_artifact(artifact, pins[artifact["id"]])
        need(artifact["workflow_run"]["head_branch"] == BRANCH, "same original source branch")


def verify_producer(bundle, root):
    api = Path(bundle) / "api"
    commit, tree, compare = [read_json(api / (name + ".json")) for name in ("commit", "tree", "prepared_main_comparison")]
    need(commit["sha"] == HEAD and commit["tree"]["sha"] == tree["sha"] == TREE
         and [p["sha"] for p in commit["parents"]] == [PARENT] and tree["truncated"] is False,
         "exact original Git head, single reviewed parent and complete tree")
    need(compare["base_commit"]["sha"] == compare["merge_base_commit"]["sha"] == PARENT
         and compare["ahead_by"] == compare["total_commits"] == 1 and compare["behind_by"] == 0
         and [c["sha"] for c in compare["commits"]] == [HEAD]
         and [(f["filename"], f["status"]) for f in compare["files"]] == [(WORKFLOW, "modified")],
         "exact workflow-only activation of the original consumed source")
    entries = {e["path"]: e for e in tree["tree"]}
    need(len(entries) == len(tree["tree"]), "unique complete original Git paths")
    with opened(bundle, "producer-original.zip") as archive:
        proof = json.loads(archive.read("PRODUCER.json"))
        need(proof["role"] == "source_fixed_mirror" and proof["run_id"] == str(RUN) and proof["run_head"] == HEAD
             and proof["prepared_parent"] == PARENT and proof["git_tree"] == TREE and proof["run_attempt"] == 1
             and proof["run_branch"] == BRANCH and proof["runtime_workflow"] == WORKFLOW
             and len(proof["files"]) == 267, "complete actual original 267-file producer")
        need(set(archive.namelist()) == {"PRODUCER.json"} | {"repository/" + name for name in proof["files"]},
             "all and only original scientific producer payloads")
        for name, pin in proof["files"].items():
            raw, entry = archive.read("repository/" + name), entries[name]
            blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            need(len(raw) == pin["bytes"] and digest(raw) == pin["sha256"]
                 and entry["sha"] == pin["git_blob_sha1"] == blob
                 and entry["mode"] == pin["mode"] in ("100644", "100755"), "Git-authenticated original producer " + name)
            if name != WORKFLOW:
                current = Path(root) / name
                need(current.is_file() and not current.is_symlink() and current.read_bytes() == raw,
                     "unchanged original readback dependency " + name)
        template = archive.read("repository/automation/english_v6_b21_fixed_mirror_source_workflow_template.yml")
        need(archive.read("repository/" + WORKFLOW) == template.replace(b"__B21_SOURCE_FIXED_MIRROR_MAIN_SHA__", PARENT.encode()),
             "retained original workflow equals its reviewed template activation")
    return proof


def old_module(name, root):
    value = importlib.import_module(name)
    need(Path(value.__file__).resolve() == (Path(root) / "automation" / (name + ".py")).resolve(),
         "use the authenticated original readback module")
    return value


def verify_transport(bundle, root, temporary):
    verifier = old_module("english_v6_b21_fixed_mirror_transport", root)
    lineage, target = verifier.lineage, Path(temporary) / "transport"
    with opened(bundle, "transport-original.zip") as archive:
        names = {"TRANSPORT/RESULT.json", "TRANSPORT/STARTED.json", "CORPUS_TRANSPORT.json",
                 "AUTHORIZATION.json", "ORIGINAL_RUN_LIVE.json", "FAILED_CONTINUATION_LIVE.json"}
        names |= {"TRANSPORT/ranges/" + str(i) + "-" + state + ".json"
                  for i in range(8) for state in ("STARTED", "RETURNED")}
        need(set(archive.namelist()) == names, "all 22 original transport evidence members")
        retain(archive, target)
    result = verifier.read_receipt(target)
    permission = read_json(target / "AUTHORIZATION.json")
    zero = lineage.origin.verify()
    need(permission["source_execution"] == IDENTITY and permission["transport_origin"] == zero,
         "original identity and both distinct earlier zero-call failures")
    for filename, reader, key in (("ORIGINAL_RUN_LIVE.json", lineage.original_origin, "original_run_live_sha256"),
                                   ("FAILED_CONTINUATION_LIVE.json", lineage.origin, "failed_continuation_live_sha256")):
        raw = (target / filename).read_bytes()
        need(digest(raw) == permission[key], "original authorization binds earlier live-run metadata")
        live = json.loads(raw)
        reader.validate_remote(live["run"], live["jobs"], live["artifacts"])
    need(read_json(target / "TRANSPORT/STARTED.json") == dict(source_execution=IDENTITY,
         transport_plan_sha256=digest(verifier.PLAN.read_bytes()), transport_origin=zero,
         ranges=verifier.ranges(lineage.CORPUS_BYTES, 8), range_retries=0, source_calls=0, model_calls=0),
         "exact original eight-range transport start")
    with opened(bundle, "consumed-original.zip") as archive:
        need(archive.namelist() == ["CONSUMPTION.json"], "one exact before-generation marker")
        marker = json.loads(archive.read("CONSUMPTION.json"))
        need(marker == dict(schema=lineage.SCHEMA + "-consumption", marker_name=lineage.MARKER,
             source_execution=IDENTITY, transport_origin=zero,
             authorization_sha256=digest((target / "AUTHORIZATION.json").read_bytes()),
             corpus_bytes=lineage.CORPUS_BYTES, corpus_sha256=lineage.CORPUS_SHA,
             transport_receipt_sha256=digest((target / "CORPUS_TRANSPORT.json").read_bytes()),
             actual_tts_calls=0, actual_natural_original_decodes=0, actual_observer_calls=0, actual_kws_forwards=0),
             "durable original marker follows full transport and precedes the actual 96 calls")
    return dict(status=result["status"], complete_ranges=len(result["ranges"]), corpus_bytes=result["bytes"],
        corpus_sha256=result["sha256"], elapsed_seconds=result["elapsed_seconds"], original_marker_verified=True,
        whole_archive_redownloaded_during_reuse=False, earlier_zero_call_failures_authenticated=True)


def verify_stop(bundle, root):
    api = Path(bundle) / "api"
    validate_remote(*[read_json(api / name) for name in ("run-final.json", "jobs-final.json", "artifacts-final.json")])
    raw = (api / "source-job.log").read_bytes()
    need(len(raw) == 56840 and digest(raw) == LOG_SHA256, "exact original source failure log")
    log = raw.decode("utf-8", errors="strict")
    need(log.count("ValueError: fixed reader has fewer than two eligible transcripts; STOP without replacement") == 1
         and log.count("##[error]Process completed with exit code 1.") == 1
         and 'english_v6_b21_fixed_mirror_source.py", line 70, in generate' in log
         and 'english_v6_b21_natural_sources.py", line 147, in extract' in log
         and 'english_v6_b21_natural_sources.py", line 95, in pick' in log
         and 'english_v6_b21_natural_sources.py", line 74, in rank_metadata' in log,
         "original metadata STOP after synthetic generation")
    code = (Path(root) / "automation/english_v6_b21_natural_sources.py").read_text()
    node = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == "extract")
    body = ast.get_source_segment(code, node)
    need(body.index("roster, catalog = pick(archive, design)") < body.index('qa.record(source / "NATURAL_METADATA.json", catalog)')
         < body.index('import soundfile as sf') < body.index('with sf.SoundFile('),
         "authenticated code stopped before natural extraction or decoding")
    return dict(original_source_panel_status="STOP_NATURAL_METADATA_INSUFFICIENT", original_failure_preserved=True,
        actual_public_piper_synthesize_calls=96, actual_natural_original_decodes=0,
        actual_native_observer_calls=0, actual_kws_forwards=0,
        internal_tts_forward_count=None, internal_tts_forwards_instrumented=False,
        complete_source_gate_pass=False, source_replay_allowed=False, exact_original_log_sha256=LOG_SHA256)


def verify_previous(bundle, root):
    previous = old_module("english_v6_b21_source_ancestry", root)
    pin = BUNDLE_ARTIFACTS["B20_SOURCE.zip"]
    need(previous.SOURCE_PIN == pin, "same original complete B20 history checkpoint")
    run, artifact = [read_json(Path(bundle) / "api" / name) for name in ("B20-run.json", "B20-artifact.json")]
    need(run["id"] == pin["run_id"] and run["head_sha"] == pin["run_head"]
         and run["run_attempt"] == 1 and run["status"] == "completed" and run["conclusion"] == "success"
         and run["event"] == "push" and run["path"] == ".github/workflows/english-v6-b20-synthetic-primary-once.yml"
         and run["head_branch"] == "research/en-v6-b20-synthetic-primary-once-20261124"
         and run["repository"]["full_name"] == run["head_repository"]["full_name"] == REPO
         and run["repository"]["private"] is run["head_repository"]["private"] is False,
         "completed first public B20 source remains the original history checkpoint")
    validate_artifact(artifact, pin)
    with opened(bundle, "B20_SOURCE.zip"):
        pass
    return previous.authenticate_previous(bundle)


def source_members(manifest):
    rows = manifest["rows"]
    need(len(rows) == len({r["id"] for r in rows}) == 96 and digest(canonical(rows)) == ROWS_SHA256,
         "only the exact 96 already-consumed rows qualify for reuse")
    names = {"STARTED.json", "SYNTHETIC_MANIFEST.json"}
    for row in rows:
        names |= {"wav/" + row["id"] + ".wav", "starts/" + row["id"] + ".json"}
        names |= {"native/" + row["id"] + "/" + part["file"] for part in row["native_chunks"]}
    need(len(names) == 290, "exact 290 original source members")
    return names


def complete_history(prior, rows):
    history = dict(prior, schema=HISTORY_SCHEMA, original_source_pin=SOURCE_PIN,
        original_source_execution=IDENTITY, original_source_panel_status="STOP_NATURAL_METADATA_INSUFFICIENT",
        original_synthetic_manifest_sha256=MANIFEST_SHA256, all_B21_96_raw_originals_added=True,
        original_public_piper_synthesize_calls=96, original_natural_original_decodes=0,
        original_native_observer_calls=0, original_kws_forwards=0)
    for key, field in (("all_pcm_sha256", "pcm_sha256"), ("all_source_groups", "source_group_id")):
        history[key] = sorted(set(prior[key]) | {row[field] for row in rows})
    people = {canonical_person(row["source_speaker_id"]) for row in rows}
    for key in ("canonical_people", "synthetic_excluded_people", "natural_excluded_people"):
        history[key] = sorted(set(prior[key]) | people)
    validate_complete_history(history)
    return history


def validate_original_members(directory, manifest):
    inventory = {}
    for name in source_members(manifest):
        path = Path(directory) / name
        need(path.is_file() and not path.is_symlink(), "every original source member remains a regular file")
        raw = path.read_bytes()
        inventory[name] = dict(bytes=len(raw), sha256=digest(raw))
    need(digest(canonical(inventory)) == MEMBERS_SHA256,
         "all 290 original member bytes remain unchanged, including every original JSON receipt")


def validate_complete_history(value):
    """Hash-authenticate the full exposure set without reopening any audio."""
    need(digest(canonical(value)) == COMPLETE_HISTORY_SHA256, "exact B20 history plus all 96 consumed B21 originals")
    return tuple(set(value[key]) for key in ("all_pcm_sha256", "all_source_groups", "canonical_people"))


def validate_reused_rows(rows, proof):
    need(proof["schema"] == SCHEMA and proof["source_pin"] == SOURCE_PIN
         and proof["original_identity"] == IDENTITY and proof["synthetic_manifest_sha256"] == MANIFEST_SHA256
         and digest(canonical(proof["rows"])) == digest(canonical(rows)) == ROWS_SHA256 and rows == proof["rows"],
         "exact entire original list only; no role-based or prefix-based history exemption")
    validate_complete_history(proof["history"])
    return rows


def require_new_natural_disjoint(rows, proof):
    validate_reused_rows(proof["rows"], proof)
    seen, groups, people = validate_complete_history(proof["history"])
    need(type(rows) is list and rows and len({r["id"] for r in rows}) == len(rows), "nonempty unique new natural originals")
    new_pcm = set()
    for row in rows:
        person = canonical_person(row["speaker"])
        need(row["cohort"] == "natural" and row["pcm_sha256"] not in seen | new_pcm
             and row["source_group_id"] not in groups and person not in people,
             "every new natural original excludes complete old and consumed synthetic PCM/group/person exposure")
        new_pcm.add(row["pcm_sha256"])
    return rows


def verify(bundle_dir, root=ROOT):
    """Read all original bytes using only the authenticated original parsers."""
    bundle, root = Path(bundle_dir), Path(root)
    verify_producer(bundle, root)
    stop = verify_stop(bundle, root)
    previous = verify_previous(bundle, root)
    with tempfile.TemporaryDirectory(prefix="b21-original-reuse-") as temporary:
        transport = verify_transport(bundle, root, temporary)
        source = Path(temporary) / "source"
        with opened(bundle, "source-original.zip") as archive:
            raw = archive.read("SYNTHETIC_MANIFEST.json")
            need(digest(raw) == MANIFEST_SHA256, "immutable original synthetic manifest bytes")
            manifest = json.loads(raw)
            need(set(archive.namelist()) == source_members(manifest), "all and only original 290 source members")
            retain(archive, source)
        original = old_module("english_v6_b21_source", root)
        rows = original.verify_synthetic(source, manifest)
    people = {canonical_person(row["source_speaker_id"]) for row in rows}
    need(len({r["pcm_sha256"] for r in rows}) == 96 and len(people) == len({r["source_group_id"] for r in rows}) == 24
         and not {r["pcm_sha256"] for r in rows} & set(previous["all_pcm_sha256"])
         and not {r["source_group_id"] for r in rows} & set(previous["all_source_groups"])
         and not people & set(previous["canonical_people"])
         and all(r["observed_class"] is None and r["training_allowed"] is False for r in rows),
         "all 96 original sources remain independent, unobserved and ineligible for training")
    return dict(schema=SCHEMA, original_identity=IDENTITY, source_pin=SOURCE_PIN,
        bundle_artifacts=BUNDLE_ARTIFACTS, synthetic_manifest_sha256=MANIFEST_SHA256, rows_sha256=ROWS_SHA256,
        rows=rows, prior_history=previous, history=complete_history(previous, rows),
        complete_history_sha256=COMPLETE_HISTORY_SHA256, consumed_source_stop=stop, transport_readback=transport,
        original_members=290, original_members_sha256=MEMBERS_SHA256, original_synthetic_slots=96, exact_reuse_only=True,
        zero_new_calls=dict(tts=0, natural_decodes=0, native_observers=0, kws_forwards=0, optimizer_updates=0),
        regeneration_allowed=False, source_labels_admitted=False, training_allowed=False, promotion_allowed=False,
        live_run_refresh_required_before_reuse=True)


def materialize(bundle_dir, fresh_absolute_output, root=ROOT):
    """Copy the authenticated 290 originals, retaining both original JSON files."""
    output = Path(fresh_absolute_output)
    need(output.is_absolute() and output.parent.is_dir() and not output.exists() and not output.is_symlink(),
         "one fresh absolute destination; never overwrite or resume")
    proof = verify(bundle_dir, root)
    output.mkdir()
    with opened(bundle_dir, "source-original.zip") as archive:
        retain(archive, output)
        need(all((output / name).read_bytes() == archive.read(name) for name in archive.namelist()),
             "all 290 copied original members are byte-identical")
    return proof


def validate_reused_source(directory, proof, root=ROOT):
    """Authenticate originals after separate amendment files have been added."""
    directory = Path(directory)
    raw = (directory / "SYNTHETIC_MANIFEST.json").read_bytes()
    need(digest(raw) == MANIFEST_SHA256, "never rewrite the original synthetic manifest")
    manifest = json.loads(raw)
    validate_original_members(directory, manifest)
    rows = old_module("english_v6_b21_source", root).verify_synthetic(directory, manifest)
    return validate_reused_rows(rows, proof)
