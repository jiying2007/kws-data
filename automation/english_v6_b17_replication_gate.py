"""Fail-closed B17 replication opening from an authenticated successful primary.

Only saved primary evidence is downloaded here. No source waveform, corpus,
checkpoint, model package or replica metadata is opened before this gate passes.
The checked-in opening lock is inert until exact successful evidence is bound.
"""
from __future__ import annotations
import argparse
import ast
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b17_source_ancestry as ancestry
import english_v6_b14_decision as decision

OPENING = Path(__file__).with_name("english_v6_b17_replication_opening.json")
SCHEMA = "english-v6-b17-replication-authorized-by-actual-primary-v1"
PRIMARY_SCHEMA = "english-v6-b17-unseen-primary-four-frozen-models-v1"
ASSESSMENT_SHA = ancestry.ASSESSMENT_SHA
ANCESTRY_SHA = ancestry.ANCESTRY_SHA
DESIGN_SHA = "f9ba0ce59b0deed805b313063e648470e61af6c9122d08f6fe9aa5ea9e324831"
DECISION_SHA = "6aa5f999598e47b41eb0a946acd82fdc5b47d30aea8990871c1c1a01065e4023"
QUALITY_SHA = "944f5df3c7bc3244977a8a33f7e1c393bffff9f7f5c32d2d8cc79e2628e265d9"
PRIMARY_EVAL_SHA = "1bedd7cd5a0ed97fa740379fd2120c292f43380aa219b2316c3be4c4dfeeb479"
B0_PT = "1ae8a2b20c8b6991f925dd4c626af2f36b308adee8733a3a7215818716f565d7"
B9_PT = "7dba86a3fcda6e5f05cb80b414dc372c70a5a03b3aee783f168441a4871e7655"
ARMS = ("B0", "B9", "control", "challenger")
GAINS = ("1.0", "0.75", "0.5")
GATE = {"synthetic_positive": 12, "synthetic_ordinary": 24, "synthetic_paired": 8,
        "natural_originals": 12, "natural_speakers": 8}
REPO = "jiying2007/kws-data"
need = ancestry.need
sha = ancestry.digest


def valid_hash(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def valid_artifact(lock):
    need(type(lock) is dict and set(lock) == {"id", "run_id", "run_head", "bytes", "sha256"}
         and all(type(lock[k]) is int and lock[k] > 0 for k in ("id", "run_id", "bytes"))
         and type(lock["run_head"]) is str and re.fullmatch(r"[0-9a-f]{40}", lock["run_head"])
         and valid_hash(lock["sha256"]), "exact original artifact identity required")


def opening():
    """The inert check must be the first action, before even a GitHub metadata request."""
    raw = OPENING.read_bytes()
    lock = json.loads(raw)
    need(lock.get("enabled") is True, "B17 replication is sealed: no authenticated primary PASS bound")
    assessment, old = ancestry.read_plan()
    need(lock["schema"] == "english-v6-b17-replication-opening-lock-v1"
         and lock["assessment_sha256"] == ASSESSMENT_SHA and lock["source_ancestry_sha256"] == ANCESTRY_SHA
         and lock["primary_quality_gate"] == "PASS" and lock["old_replication_opened"] is False
         and lock["new_training_allowed"] is False and lock["product_promotion_allowed"] is False
         and lock["new_synthetic_replication_indices"] == [742, 765]
         and lock["new_natural_replication_split"] == "dev-other"
         and lock["new_natural_replication_readers"] == 9
         and lock["required_primary_branch"] == "research/en-v6-b17-primary-scoring-once-20261124"
         and lock["required_primary_workflow"] == ".github/workflows/english-v6-b17-model-eval-once.yml",
         "only the already frozen B17 replication may open")
    valid_artifact(lock["primary_result"])
    need(valid_hash(lock["primary_inputs_sha256"]) and lock["primary_eval_code_sha256"] == PRIMARY_EVAL_SHA,
         "completed primary source specification and implementation must be pinned")
    models = lock["model_pt_sha256"]
    need(type(models) is dict and set(models) == set(ARMS) and all(valid_hash(v) for v in models.values())
         and models["B0"] == B0_PT and models["B9"] == B9_PT and models["control"] != models["challenger"],
         "four exact frozen primary weights; no model changes on replication")
    need(type(lock["primary_source_inputs"]) is dict
         and set(lock["primary_source_inputs"]) == {"synth_source", "natural_source"},
         "exact actual B17 primary source archives required for future ancestry exclusion")
    for item in lock["primary_source_inputs"].values():
        valid_artifact(item)
    need(sha(Path(decision.__file__).read_bytes()) == DECISION_SHA
         and sha(decision.ASSESSMENT.read_bytes()) == QUALITY_SHA,
         "unchanged original acoustic quality function and criteria")
    need(assessment["replication"]["natural"]["requested_speakers"] == 9
         and assessment["replication"]["synthetic"]["speaker_indices"] == list(range(742, 766)),
         "replication sources remain identical to the pre-optimizer plan")
    return lock, sha(raw)


def read_npy(raw):
    """Bounded float32 C-order four-class parser, using no model/numeric packages."""
    need(raw[:6] == b"\x93NUMPY" and len(raw) >= 10, "saved frame array is a real NPY member")
    version = tuple(raw[6:8])
    need(version in ((1, 0), (2, 0)), "supported immutable numeric array format")
    width = 2 if version == (1, 0) else 4
    size = int.from_bytes(raw[8:8 + width], "little")
    need(0 < size <= 8192 and len(raw) >= 8 + width + size, "bounded frame-array header")
    header = ast.literal_eval(raw[8 + width:8 + width + size].decode("latin1").strip())
    shape = header.get("shape")
    need(set(header) == {"descr", "fortran_order", "shape"}
         and header["descr"] == "<f4" and header["fortran_order"] is False
         and type(shape) is tuple and len(shape) == 2 and type(shape[0]) is int
         and 1 <= shape[0] <= 1500 and shape[1] == 4, "native bounded four-class float32 frames")
    body = raw[8 + width + size:]
    need(len(body) == shape[0] * 16, "all saved frame bytes present")
    labels = []
    for values in struct.iter_unpack("<4f", body):
        need(all(math.isfinite(value) for value in values), "finite saved logit tensor")
        labels.append(max(range(4), key=values.__getitem__))
    onsets = [(i, value) for i, value in enumerate(labels) if value and (i == 0 or labels[i - 1] != value)]
    tokens = [value for _, value in onsets]
    frames = [onsets[i + 2][0] for i in range(max(0, len(onsets) - 2)) if tokens[i:i + 3] == [1, 2, 3]]
    return body, tokens, frames, shape[0]


def verify_primary(raw, lock):
    pin = lock["primary_result"]
    need(len(raw) == pin["bytes"] and len(raw) <= 100_000_000 and sha(raw) == pin["sha256"],
         "exact successful original primary ZIP, not a synthesized permission document")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        need(archive.namelist().count("RESULT.json") == 1
             and archive.namelist().count("ALL_FRAME_LOGITS.npz") == 1
             and set(archive.namelist()) == {"RESULT.json", "ALL_FRAME_LOGITS.npz"}
             and sum(item.file_size for item in archive.infolist()) <= 120_000_000
             and archive.testzip() is None, "complete original two-member score archive")
        result_bytes = archive.read("RESULT.json")
        arrays_bytes = archive.read("ALL_FRAME_LOGITS.npz")
    result = json.loads(result_bytes)
    need(result["schema"] == PRIMARY_SCHEMA and result["assessment_sha256"] == ASSESSMENT_SHA
         and result["training_design_sha256"] == DESIGN_SHA and result["source_gate"] == "SOURCE_GATE_PASS"
         and result["panel_gate"] == "PASS" and result["replication_open_allowed"] is True
         and result["optimizer_updates"] == 0 and result["installed_software_baseline"] == "B0"
         and all(result[k] is False for k in ("threshold_search", "promotion_allowed", "product_released",
                   "B14_replication_opened", "B15_replication_opened", "B16_replication_opened", "ssc305_physical"))
         and result["model_pt_sha256"] == lock["model_pt_sha256"]
         and sha(arrays_bytes) == result["logits_sha256"] and not result["source_shortfall"],
         "a genuine completed B17 primary acoustic PASS is mandatory")
    records = result["records"]
    need(type(records) is list and 48 <= len(records) <= 120
         and len({r["id"] for r in records}) == len({r["pcm_sha256"] for r in records}) == len(records)
         and result["actual_forwards"] == 12 * len(records), "complete source-distinct primary denominator")
    assessment, lineage = ancestry.read_plan()
    planned = json.loads((ancestry.PLANS / "primary.json").read_bytes())
    synth = {r["id"]: r for r in planned["rows"]}
    natural_speakers = set(assessment["primary"]["natural"]["speakers"])
    groups, people = {}, set()
    positives = ordinary = human = 0
    expected_arrays = set()
    with zipfile.ZipFile(io.BytesIO(arrays_bytes)) as arrays:
        need(len(arrays.namelist()) == len(set(arrays.namelist())) == result["actual_forwards"]
             and sum(item.file_size for item in arrays.infolist()) <= 100_000_000
             and arrays.testzip() is None, "all unique saved primary frame arrays")
        for row in records:
            need(valid_hash(row["pcm_sha256"]) and row["tier"] in ("silver", "weak")
                 and type(row["positive"]) is bool and set(row["gains"]) == set(GAINS),
                 "source admission and the three original frozen gains")
            if row["cohort"] == "synthetic":
                expected = synth[row["id"]]
                need(row["group"] == expected["source_group_id"] and row["cell"] == expected["cell_id"]
                     and row["positive"] == (expected["requested_class"] == 2), "original primary synthetic source identity")
                groups.setdefault(row["group"], set()).add(row["positive"])
                positives += row["positive"]
                ordinary += not row["positive"]
            else:
                match = re.fullmatch(r"b17-natural-(\d+)-(\d+)-(\d+)", row["id"])
                need(row["cohort"] == "natural" and match is not None and not row["positive"]
                     and row["cell"] == "read-speech" and match[1] in natural_speakers
                     and match[1] not in lineage["natural_excluded_speakers"]
                     and row["group"] == "librispeech-test-clean-" + match[1], "unexposed canonical primary natural reader")
                people.add(match[1]); human += 1
            for gain in GAINS:
                need(set(row["gains"][gain]) == set(ARMS), "four frozen primary roles")
                frame_counts = set()
                for arm in ARMS:
                    state = row["gains"][gain][arm]
                    key = row["id"] + "__" + gain + "__" + arm
                    need(state["array"] == key, "original saved array identity")
                    expected_arrays.add(key + ".npy")
                    body, tokens, fire, frames = read_npy(arrays.read(key + ".npy"))
                    need(sha(body) == state["sha256"] and tokens == state["tokens"]
                         and fire == state["event_frames"] and len(fire) == state["events"]
                         and frames == state["frames"], "genuine saved primary CTC event/frame readback")
                    frame_counts.add(frames)
                need(len(frame_counts) == 1, "all primary models scored the same original frame grid")
        need(set(arrays.namelist()) == expected_arrays, "no missing, orphan or unrelated array can open replication")
    counts = dict(synthetic_positive=positives, synthetic_ordinary=ordinary,
                  synthetic_paired=sum(classes == {False, True} for classes in groups.values()),
                  natural_originals=human, natural_speakers=len(people))
    need(counts == result["source_coverage"] and all(counts[k] >= threshold for k, threshold in GATE.items()),
         "recomputed original source minima before replication opening")
    recomputed = decision.decide(records)
    need(recomputed["panel_gate"] == "PASS" and all(result[k] == value for k, value in recomputed.items()),
         "unchanged B14 stringent quality decision independently passes saved native logits")
    return dict(primary_result_sha256=sha(raw), primary_result_json_sha256=sha(result_bytes),
                primary_logits_sha256=sha(arrays_bytes), actual_primary_forwards=result["actual_forwards"],
                primary_source_coverage=counts, model_pt_sha256=result["model_pt_sha256"], primary_quality="PASS")


def gh(endpoint):
    return subprocess.check_output(["gh", "api", "repos/" + REPO + "/" + endpoint])


def execution_guard(prepared_parent, branch, env=None, event=None, parent=None):
    """Only the first public single-parent source push can perform source work."""
    e = os.environ if env is None else env
    allowed = {"refs/heads/research/en-v6-b17-" + kind + "-replication-once-20261124"
               for kind in ("synthetic", "natural")}
    need(branch in allowed and re.fullmatch(r"[0-9a-f]{40}", prepared_parent or "") is not None,
         "an exact reviewed source predecessor; no placeholder can execute")
    need(all(e.get(k) == value for k, value in {
        "GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPO, "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": branch, "GITHUB_RUN_ATTEMPT": "1"}.items())
         and re.fullmatch(r"[0-9a-f]{40}", e.get("GITHUB_SHA", "")) is not None,
         "first designated public source event only")
    if event is None:
        event = json.loads(Path(e["GITHUB_EVENT_PATH"]).read_bytes())
    if parent is None:
        parent = subprocess.check_output(["git", "rev-parse", "HEAD^1"], text=True).strip()
    need(event.get("before") == parent == prepared_parent and event.get("after") == e["GITHUB_SHA"]
         and event.get("ref") == branch and event.get("forced") is False and event.get("deleted") is False
         and event.get("repository", {}).get("full_name") == REPO
         and event.get("repository", {}).get("private") is False,
         "single reviewed public predecessor, no force-push, replay or multi-commit jump")


def fetch(out):
    lock, lock_sha = opening()  # Must precede every external read.
    pin = lock["primary_result"]
    run = json.loads(gh("actions/runs/" + str(pin["run_id"])))
    artifact = json.loads(gh("actions/artifacts/" + str(pin["id"])))
    need(run["status"] == "completed" and run["conclusion"] == "success" and run["run_attempt"] == 1
         and run["head_sha"] == pin["run_head"] and run["event"] == "push"
         and run["head_branch"] == lock["required_primary_branch"]
         and run["path"] == lock["required_primary_workflow"]
         and run["repository"]["full_name"] == REPO and run["repository"]["private"] is False,
         "successful first public original B17 primary score workflow")
    need(artifact["id"] == pin["id"] and artifact["expired"] is False
         and artifact["size_in_bytes"] == pin["bytes"] and artifact["digest"] == "sha256:" + pin["sha256"]
         and artifact["workflow_run"]["id"] == pin["run_id"]
         and artifact["workflow_run"]["head_sha"] == pin["run_head"], "original primary artifact belongs to that exact run")
    raw = gh("actions/artifacts/" + str(pin["id"]) + "/zip")
    checked = verify_primary(raw, lock)
    retained = {}
    for name, expected in (("english_v6_b17_primary_inputs.json", lock["primary_inputs_sha256"]),
                           ("english_v6_b17_primary_eval.py", lock["primary_eval_code_sha256"])):
        response = json.loads(gh("contents/automation/" + name + "?ref=" + pin["run_head"]))
        content = base64.b64decode(response["content"], validate=False)
        need(sha(content) == expected, "exact implementation/input contract used by the completed primary")
        retained[name] = content
    inputs = json.loads(retained["english_v6_b17_primary_inputs.json"])
    need(inputs["schema"] == PRIMARY_SCHEMA and inputs["scoring_allowed"] is True
         and inputs["assessment_sha256"] == ASSESSMENT_SHA and inputs["training_design_sha256"] == DESIGN_SHA
         and inputs["source_gate"] == GATE and inputs["parent_pt_sha256"] == B0_PT
         and {k: inputs["inputs"][k] for k in ("synth_source", "natural_source")} == lock["primary_source_inputs"],
         "authenticated actual primary source archives and unchanged quality specification")
    need(set(inputs["inputs"]) == {"B0", "B9", "pair", "synth_source", "synth_screen", "natural_source", "natural_screen"},
         "the original seven authenticated primary inputs")
    for item in inputs["inputs"].values():
        valid_artifact(item)
    out.mkdir()
    (out / "PRIMARY.zip").write_bytes(raw)
    for name, content in retained.items():
        (out / name).write_bytes(content)
    permit = dict(schema=SCHEMA, opening_lock_sha256=lock_sha, assessment_sha256=ASSESSMENT_SHA,
                  source_ancestry_sha256=ANCESTRY_SHA, primary_result=pin,
                  primary_source_inputs=lock["primary_source_inputs"], **checked,
                  only_new_B17_replication_allowed=True, old_replication_opened=False,
                  training_allowed=False, model_selection_changes_allowed=False, product_promotion_allowed=False)
    (out / "PERMIT.json").write_text(json.dumps(permit, indent=2) + "\n")
    require(out)


def require(directory):
    """Each replica source/observer/screen boundary independently rereads original evidence."""
    lock, lock_sha = opening()
    directory = Path(directory)
    checked = verify_primary((directory / "PRIMARY.zip").read_bytes(), lock)
    permit = json.loads((directory / "PERMIT.json").read_bytes())
    need(permit["schema"] == SCHEMA and permit["opening_lock_sha256"] == lock_sha
         and permit["primary_result"] == lock["primary_result"]
         and permit["primary_source_inputs"] == lock["primary_source_inputs"]
         and permit["assessment_sha256"] == ASSESSMENT_SHA and permit["source_ancestry_sha256"] == ANCESTRY_SHA
         and permit["only_new_B17_replication_allowed"] is True
         and all(permit[k] is False for k in ("old_replication_opened", "training_allowed",
                 "model_selection_changes_allowed", "product_promotion_allowed"))
         and all(permit[k] == value for k, value in checked.items()), "current exact primary authorization, no synthetic gate shortcut")
    for name, expected in (("english_v6_b17_primary_inputs.json", lock["primary_inputs_sha256"]),
                           ("english_v6_b17_primary_eval.py", lock["primary_eval_code_sha256"])):
        need(sha((directory / name).read_bytes()) == expected, "preserved verified primary code and input binding")
    return permit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for command in ("fetch", "verify"):
        item = modes.add_parser(command); item.add_argument("directory", type=Path)
    item = modes.add_parser("guard"); item.add_argument("prepared_parent"); item.add_argument("branch")
    args = parser.parse_args()
    if args.mode == "fetch":
        fetch(args.directory)
    elif args.mode == "verify":
        require(args.directory)
        print("PASS saved successful primary authorizes only the frozen B17 replication sources")
    else:
        execution_guard(args.prepared_parent, args.branch)
        print("PASS first public source execution on the exact reviewed predecessor")
