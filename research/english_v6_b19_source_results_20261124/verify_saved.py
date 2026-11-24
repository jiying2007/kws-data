"""Recompute B19 source-format feasibility from the three saved original ZIPs.

No archive download, renderer, model initialization, ASR or KWS inference.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent
RUN = 45330168237
HEAD = "75bd1c05c220ae2e2334fd5c1e06fa87c66630a5"
PARENT = "771a322cee80bb88210cc1c449fd4b79b403c00a"
BRANCH = "research/en-v6-b19-source-feasibility-once-20261124"
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def need(condition, why):
    if not condition:
        raise ValueError(why)


def verify(inputs, repository):
    pins = json.loads((HERE / "ARTIFACT_LOCKS.json").read_bytes())
    producer = json.loads((HERE / "PRODUCER_FILES.json").read_bytes())
    for name, pin in producer.items():
        raw = (HERE / "evidence/producer" / name).read_bytes()
        need(sha(raw) == pin["original_run_file_sha256"], "saved original producer file " + name)
        compared = raw.replace(PARENT.encode(), b"__B19_SOURCE_FEASIBILITY_MAIN_SHA__") if name.endswith("once.yml") else raw
        need(sha(compared) == pin["reviewed_template_sha256"], "reviewed exact producer bytes " + name)
        need(sha((repository / name).read_bytes()) == pin["reviewed_template_sha256"], "unchanged local verified implementation " + name)
    run = json.loads((HERE / "evidence/run-final.json").read_bytes())
    jobs = json.loads((HERE / "evidence/jobs-final.json").read_bytes())
    commit = json.loads((HERE / "evidence/commit.json").read_bytes())
    artifacts = json.loads((HERE / "evidence/artifacts-final.json").read_bytes())["artifacts"]
    need(run["id"] == RUN and run["head_sha"] == HEAD and run["head_branch"] == BRANCH
         and run["status"] == "completed" and run["conclusion"] == "success" and run["run_attempt"] == 1
         and run["event"] == "push" and run["repository"]["private"] is False
         and run["repository"]["full_name"] == "jiying2007/kws-data", "actual first public B19 completion")
    need(commit["sha"] == HEAD and len(commit["parents"]) == 1 and commit["parents"][0]["sha"] == PARENT
         and [item["filename"] for item in commit["files"]] == [".github/workflows/english-v6-b19-source-feasibility-once.yml"],
         "only the original exact-parent workflow activation")
    need(jobs["total_count"] == len(jobs["jobs"]) == 3
         and {job["name"] for job in jobs["jobs"]} == {"sources", "observer", "readback"}
         and all(job["run_id"] == RUN and job["head_sha"] == HEAD and job["run_attempt"] == 1
                 and job["status"] == "completed" and job["conclusion"] == "success" for job in jobs["jobs"]),
         "all original producers completed successfully")
    by_id = {item["id"]: item for item in artifacts}
    for role, pin in pins.items():
        item = by_id[pin["id"]]
        need(item["size_in_bytes"] == pin["bytes"] and item["digest"] == "sha256:" + pin["sha256"]
             and item["workflow_run"]["id"] == pin["run_id"] == RUN
             and item["workflow_run"]["head_sha"] == pin["run_head"] == HEAD, "original immutable artifact metadata " + role)
    sys.path.insert(0, str(repository / "automation"))
    import english_v6_b19_source as source
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for role, pin in pins.items():
            path = inputs / "artifacts" / pin["local_name"]
            raw = path.read_bytes()
            need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "original ZIP bytes " + role)
            with zipfile.ZipFile(path) as archive:
                source.previous.lineage.extract_original(archive, root / role)
        with patch.object(source.B13, "render_panel", side_effect=RuntimeError("NO_TTS_IN_READBACK")), \
             patch.object(source.observer.adapter, "_load", side_effect=RuntimeError("NO_MODEL_IN_READBACK")):
            rows = source.read_source(root / "source")
            registration = source.registration(root / "source")
            result = source.observer.readback(root / "source", root / "observer", source.read_source, registration)
        original = (root / "readback/FEASIBILITY.json").read_bytes()
        need(sha(original) == sha((root / "observer/RESULT.json").read_bytes())
             == "48e10f040b1a631aeee5a335aa8fbe37d1a01fc547550b15507f4aded3a42700"
             and json.loads(original) == result, "identical retained native result and independently recomputed feasibility")
        need(registration["run_id"] == str(RUN) and registration["run_head"] == HEAD
             and registration["prepared_parent"] == PARENT and registration["run_attempt"] == 1,
             "actual original source/observer binding")
        need(result["actual_native_attempts"] == result["native_returns"] == result["valid_nonempty_payloads"] == 48
             and result["invalid_empty_payloads"] == 0 and result["prospective_pair_feasible"] is True
             and result["diagnostic_complete"] is True
             and all(result["arms"][arm]["valid_nonempty"] == 16
                     and result["arms"][arm]["whole_audio_bounds_ok"] is True
                     for arm in ("baseline", "prospective", "ordinary"))
             and result["source_admission_eligible"] is result["source_gate_passed"] is result["KWS_evaluation_allowed"]
                 is result["training_allowed"] is result["promotion_allowed"] is result["replication_opened"] is False,
             "fixed source-format feasibility only; baseline also passed and no model authority")
        output = dict(schema="english-v6-b19-independent-saved-native-feasibility-readback-v1",
            run_id=RUN, run_head=HEAD, prepared_parent=PARENT, original_slots=len(rows),
            original_readers=len({r["source_speaker_id"] for r in rows}),
            actual_native_attempts=48, native_returns=48, valid_nonempty_payloads=48, invalid_empty_payloads=0,
            arms=result["arms"], result=result["result"], prospective_pair_feasible=True,
            baseline_also_all_valid=True, causal_fix_proven=False,
            source_manifest_sha256=sha((root / "source/MANIFEST.json").read_bytes()),
            original_FEASIBILITY_sha256=sha(original), original_RESULT_sha256=sha(original),
            source_admission_eligible=False, source_gate_passed=False, KWS_evaluation_allowed=False,
            training_allowed=False, promotion_allowed=False, replication_opened=False,
            new_readback_model_calls=0, new_readback_source_calls=0)
    need(output == json.loads((HERE / "NATIVE_READBACK.json").read_bytes()), "unchanged durable pure readback result")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("repository", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.inputs.resolve(), args.repository.resolve()), indent=2))
