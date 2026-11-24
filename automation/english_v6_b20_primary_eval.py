"""INERT B20 primary: unchanged four B17 weights and original B14 quality.

This scaffold cannot execute without a new frozen B20 registration, authentic
B19 feasibility proof, exact original source/SCREEN inputs, complete producer
code/workflow locks, and a reviewed one-shot parent. No fallback or replay.
"""
from __future__ import annotations
import argparse
import ast
import base64
from contextlib import ExitStack
import hashlib
import importlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
import wave
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = HERE / "english_v6_b20_primary_inputs.json"
FIXED = HERE / "english_v6_b20_fixed_inputs.json"
PRODUCER = HERE / "english_v6_b20_primary_producer.json"
WORKFLOW_TEMPLATE = HERE / "english_v6_b20_primary_workflow_template.yml"
FIXED_SHA = "43206aace5714ddd28c66c22c8df38fd12b0353ee2ad36866816fcc877b57f91"
SCHEMA = "english-v6-b20-b17-frozen-models-primary-v1"
SOURCE_MODULE = "english_v6_b20_source"
SOURCE_SCHEMA = "english-v6-b20-b17-frozen-new-format-source-v1"
SCREEN_SCHEMA = "english-v6-b20-96-original-strict-source-screen-v1"
REPO = "jiying2007/kws-data"
BRANCH = "refs/heads/research/en-v6-b20-primary-scoring-once-20261124"
WORKFLOW = ".github/workflows/english-v6-b20-primary-model-eval-once.yml"
SOURCE_BRANCH = "research/en-v6-b20-synthetic-primary-once-20261124"
SOURCE_WORKFLOW = ".github/workflows/english-v6-b20-synthetic-primary-once.yml"
PARENT_TOKEN = "__B20_REVIEWED_SCORING_PARENT__"
MARKER = "english-v6-b20-primary-consumed"
TRAINING_ASSESSMENT = "7fdfe9cd47b18f64429fece944468af8adfde6c6ba012a57236a8a33aae1a948"
TRAINING_DESIGN = "f9ba0ce59b0deed805b313063e648470e61af6c9122d08f6fe9aa5ea9e324831"
QUALITY_CODE = "6aa5f999598e47b41eb0a946acd82fdc5b47d30aea8990871c1c1a01065e4023"
QUALITY_ASSESSMENT = "944f5df3c7bc3244977a8a33f7e1c393bffff9f7f5c32d2d8cc79e2628e265d9"
B0_NPZ = "02b7febf9f736fd70f4f9ae04042ffc6d609b15011d03d235b960f69b1d71696"
GAINS = ("1.0", "0.75", "0.5")
ARMS = ("B0", "B9", "control", "challenger")
PAIR_SCHEMA = "english-v6-b17-b0-ordinary-margin-pair-v1"
STEMS = {arm: arm + "/models/mero43-b17-" + arm + "192" for arm in ("control", "challenger")}
IDENTITY_FIELDS = ("id", "group", "cohort", "cell", "positive", "tier", "pcm_sha256")
SOURCE_KEYS = ("synth_source", "synth_screen", "natural_source", "natural_screen")
PROVIDERS = ("qwen", "whisper", "parakeet", "xlsr", "lv60")
INPUT_KEYS = {"B0", "B9", "pair", *SOURCE_KEYS}
GATE = dict(synthetic_positive=12, synthetic_ordinary=24, synthetic_paired=8,
            natural_originals=12, natural_speakers=8)
FLAGS = ("training_allowed", "threshold_search", "promotion_allowed", "product_released",
         "cal_locked_access", "old_replication_opened", "B17_failed_synthetic_used_for_scoring",
         "B18_failed_synthetic_used_for_scoring", "B19_canary_used_for_scoring", "replication_opened",
         "candidate_engineering_validation_allowed")
PRODUCER_ROOTS = ("english_v6_b20_primary_eval", SOURCE_MODULE,
                  "english_v6_b17_primary_eval", "english_v6_b14_decision")
B19_REGISTRATION = "automation/english_v6_b19_panels/assessment.json"
B18_DURABLE_FILES = {"MANIFEST.json", "HISTORY.json", "AUTHORIZATION.json", "SOURCE_READBACK.json",
                     "ARTIFACT_LOCKS.json", "B18_KWS_ASSESSMENT_RUNS.json"}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def valid_hash(value, length=64):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value) is not None


def valid_artifact(pin):
    need(type(pin) is dict and set(pin) == {"id", "run_id", "run_head", "bytes", "sha256"}
         and all(type(pin[k]) is int and pin[k] > 0 for k in ("id", "run_id", "bytes"))
         and valid_hash(pin["run_head"], 40) and valid_hash(pin["sha256"]), "complete actual original artifact pin")


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def fixed():
    raw = FIXED.read_bytes()
    need(sha(raw) == FIXED_SHA, "unchanged original B17 checkpoints and natural source pins")
    return json.loads(raw)


def protocol():
    doc = json.loads(SPEC.read_bytes())
    # This must be the first check; no helper/source read, GitHub call or output
    # directory is reachable while the registration remains a scaffold.
    need(doc.get("scoring_allowed") is True, "INERT: B20 has no completed source/assessment authorization")
    frozen = fixed()
    need(doc["schema"] == SCHEMA and doc["source_module"] == SOURCE_MODULE
         and doc["source_schema"] == SOURCE_SCHEMA and doc["screen_schema"] == SCREEN_SCHEMA
         and valid_hash(doc["assessment_sha256"]) and valid_hash(doc["producer_manifest_sha256"])
         and doc["prepared_parent"] is None and doc["parent_authority"] == "reviewed_workflow"
         and doc["training_assessment_sha256"] == TRAINING_ASSESSMENT
         and doc["training_design_sha256"] == TRAINING_DESIGN
         and doc["pair_schema"] == PAIR_SCHEMA and doc["pair_stems"] == STEMS
         and doc["model_pt_sha256"] == frozen["model_pt_sha256"]
         and set(doc["model_pt_sha256"]) == set(ARMS)
         and doc["parent_pt_sha256"] == frozen["model_pt_sha256"]["B0"]
         and doc["gains"] == list(GAINS) and doc["source_gate"] == GATE
         and type(doc["maximum_model_forwards"]) is int and doc["maximum_model_forwards"] == 1404
         and all(doc[k] is False for k in FLAGS) and set(doc["inputs"]) == INPUT_KEYS,
         "one new source assessment, same four models/three gains/quality, no training or promotion")
    for pin in doc["inputs"].values():
        valid_artifact(pin)
    for role in ("synthetic", "natural"):
        need(set(doc[role + "_native_inputs"]) == set(PROVIDERS), "all five original native proof artifacts")
        for provider, pin in doc[role + "_native_inputs"].items():
            valid_artifact(pin)
            if role == "natural":
                need(pin == frozen["natural_native"][provider], "only the original unchanged B17 natural native proof")
            else:
                need(all(pin[k] == doc["inputs"]["synth_source"][k] for k in ("run_id", "run_head")),
                     "all new synthetic original native proofs belong to the same successful source run")
    need(all(doc["inputs"][k] == value for k, value in {**frozen["models"], **frozen["natural"]}.items()),
         "exact original three checkpoint ZIPs and original natural source/SCREEN only")
    need(all(doc["inputs"]["synth_source"][k] == doc["inputs"]["synth_screen"][k]
             for k in ("run_id", "run_head")), "new synthetic source and strict SCREEN share their actual one-shot run")
    need(len({pin["id"] for pin in archive_pins(doc).values()}) == 17,
         "seventeen distinct original source, SCREEN, native and checkpoint artifacts")
    return doc


def archive_pins(spec):
    return {**spec["inputs"], **{role + "_native_" + provider: pin
            for role in ("synthetic", "natural") for provider, pin in spec[role + "_native_inputs"].items()}}


def source_check():
    doc = json.loads(SPEC.read_bytes())
    return dict(schema=SCHEMA, status="INERT_SCAFFOLD" if doc.get("scoring_allowed") is not True else "BOUND_NOT_EXECUTED",
                model_calls=0, source_calls=0, pcm_reads=0, promotion_allowed=False)


def guard(spec, prepared, env=None, event=None, head=None, parent=None):
    e = os.environ if env is None else env
    need(valid_hash(prepared, 40) and spec["prepared_parent"] is None
         and spec["parent_authority"] == "reviewed_workflow", "reviewed workflow is the sole execution parent authority")
    need(all(e.get(k) == v for k, v in {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPO,
         "GITHUB_EVENT_NAME": "push", "GITHUB_REF": BRANCH, "GITHUB_RUN_ATTEMPT": "1"}.items())
         and valid_hash(e.get("GITHUB_SHA"), 40) and re.fullmatch(r"[1-9][0-9]*", e.get("GITHUB_RUN_ID", "")),
         "first designated public B20 primary event only")
    event = json.loads(Path(e["GITHUB_EVENT_PATH"]).read_bytes()) if event is None else event
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() if head is None else head
    parent = subprocess.check_output(["git", "rev-parse", "HEAD^1"], text=True).strip() if parent is None else parent
    need(head == event.get("after") == e["GITHUB_SHA"] and parent == event.get("before") == prepared
         and head != parent and event.get("ref") == BRANCH
         and event.get("forced") is False and event.get("deleted") is False
         and event.get("repository", {}).get("full_name") == REPO
         and event.get("repository", {}).get("private") is False,
         "actual checkout and single reviewed parent; no replay, force push or multi-commit jump")
    return dict(run_id=e["GITHUB_RUN_ID"], run_head=head, run_attempt=1, prepared_parent=parent)


def static_closure(directory, roots):
    """Freeze local modules, literal Python files and whole first-party packages."""
    directory = Path(directory)
    repository = directory.parent

    def local(name):
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", name):
            return []
        for parent in (directory, repository):
            module = parent / (name + ".py")
            if module.is_file():
                return [module]
            package = parent / name
            if (package / "__init__.py").is_file():
                # Whole local packages also cover their relative imports.
                return sorted(p for p in package.rglob("*.py") if p.is_file())
        return []

    pending, seen = [], set()
    for name in roots:
        path = directory / (name + ".py")
        need(path.is_file(), "complete frozen scientific dependency " + name)
        pending.append(path)
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        need(path.resolve().is_relative_to(repository.resolve()), "local scientific code stays in repository")
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_bytes())):
            names = []
            if isinstance(node, ast.Import):
                names = [value.name.split(".")[0] for value in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                literal = re.fullmatch(r"([a-zA-Z_][a-zA-Z_0-9]*)(?:\.py)?", node.value)
                if literal:
                    names = [literal.group(1)]
            pending.extend(p for name in names for p in local(name) if p not in seen)
    return {str(path.relative_to(repository)) for path in seen}


def producer_contract(spec, prepared, root=ROOT):
    raw = (root / "automation" / PRODUCER.name).read_bytes()
    need(sha(raw) == spec["producer_manifest_sha256"], "reviewed complete producer manifest")
    doc = json.loads(raw)
    need(doc["schema"] == "english-v6-b20-primary-producer-v1" and doc["frozen"] is True,
         "producer scaffold must be independently frozen before activation")
    for role, roots in (("primary", PRODUCER_ROOTS), ("source", (SOURCE_MODULE,))):
        pins = doc[role]["files"]
        need(set(pins) == static_closure(root / "automation", roots)
             and all(valid_hash(value) for value in pins.values()), "complete transitive scientific source closure " + role)
        for name, expected in pins.items():
            need(sha((root / name).read_bytes()) == expected, "local frozen scientific source " + name)
    need(type(doc["resources"]) is dict and {
         "automation/" + FIXED.name, "automation/english_v6_b17_primary_inputs.json",
         "automation/english_v6_b18_primary_inputs.json", "automation/english_v6_b20_panels/assessment.json"
         } <= set(doc["resources"]), "all fixed evaluation/source registration resources")
    for name, expected in doc["resources"].items():
        path = PurePosixPath(name)
        need(path.parts and not path.is_absolute() and ".." not in path.parts and "\\" not in name
             and valid_hash(expected) and sha((root / name).read_bytes()) == expected,
             "frozen scientific registration resource " + name)
    source_resources = doc["source"]["resources"]
    need(type(source_resources) is dict and source_resources
         and all(doc["resources"].get(name) == digest for name, digest in source_resources.items())
         and "automation/english_v6_b20_panels/assessment.json" in source_resources
         and B19_REGISTRATION in source_resources,
         "source producer retains its complete immutable registration resources")
    # These dependencies are referenced inside a frozen JSON registration, not
    # Python literals. The actual historical readback consumes MANIFEST/HISTORY.
    durable = json.loads((root / B19_REGISTRATION).read_bytes())["original_B18_source_proof"]["durable_evidence"]
    need(set(durable) == B18_DURABLE_FILES and all(
         ref["path"] in source_resources and source_resources[ref["path"]] == ref["sha256"]
         and doc["resources"].get(ref["path"]) == ref["sha256"] for ref in durable.values()),
         "complete indirect B18 durable evidence in both actual producer resource closures")
    template = (root / "automation" / WORKFLOW_TEMPLATE.name).read_bytes()
    need(doc["primary"]["workflow_path"] == WORKFLOW and doc["source"]["workflow_path"] == SOURCE_WORKFLOW
         and sha(template) == doc["primary"]["workflow_template_sha256"]
         and template.count(PARENT_TOKEN.encode()) == 2
         and valid_hash(doc["source"]["workflow_sha256"]), "reviewed actual workflow contracts")
    need(valid_hash(prepared, 40), "actual guard-authenticated workflow predecessor")
    expected_workflow = template.replace(PARENT_TOKEN.encode(), prepared.encode())
    need((root / WORKFLOW).read_bytes() == expected_workflow,
         "primary workflow permits only the two reviewed prepared-parent replacements")
    return doc, expected_workflow


def producer_files(spec, producer, workflow, role):
    """The saved actual-head bytes include the original manifest/template too."""
    if role == "primary":
        extra = {**producer["resources"], WORKFLOW: sha(workflow),
                 "automation/" + SPEC.name: sha(SPEC.read_bytes()),
                 "automation/" + PRODUCER.name: spec["producer_manifest_sha256"],
                 "automation/" + WORKFLOW_TEMPLATE.name: producer["primary"]["workflow_template_sha256"]}
    else:
        need(role == "source", "explicit producer role")
        extra = {**producer["source"]["resources"], SOURCE_WORKFLOW: producer["source"]["workflow_sha256"]}
    return {**producer[role]["files"], **extra}


def helpers(spec):
    # Caller authenticates both producer closures before this import boundary.
    sys.path.insert(0, str(HERE))
    old = importlib.import_module("english_v6_b17_primary_eval")
    decision = importlib.import_module("english_v6_b14_decision")
    source = importlib.import_module(SOURCE_MODULE)
    need(sha(Path(decision.__file__).read_bytes()) == QUALITY_CODE
         and sha(decision.ASSESSMENT.read_bytes()) == QUALITY_ASSESSMENT
         and old.ASSESSMENT_SHA == TRAINING_ASSESSMENT and old.DESIGN_SHA == TRAINING_DESIGN
         and old.protocol()["scoring_allowed"] is False
         and source.SCHEMA == SOURCE_SCHEMA and source.BATCH == SCREEN_SCHEMA
         and source.ASSESSMENT_SHA == spec["assessment_sha256"]
         and sha(source.ASSESSMENT.read_bytes()) == spec["assessment_sha256"],
         "frozen B20 source registration, B17 STOP and original B14 criteria")
    b18 = json.loads((HERE / "english_v6_b18_primary_inputs.json").read_bytes())
    need(b18["scoring_allowed"] is False, "B18 source STOP remains sealed")
    plan = source.lineage.read_plan()
    reserve = plan["replication"]["natural"]["speakers"]
    need(type(reserve) is list and len(reserve) >= 8 and len(set(reserve)) == len(reserve)
         and all(re.fullmatch(r"[0-9]+", str(p)) for p in reserve),
         "a new fully registered legal natural reserve is required; null/old reserve cannot authorize primary")
    return old, decision, source, plan


def gh(endpoint):
    return subprocess.check_output(["gh", "api", "repos/" + REPO + "/" + endpoint])


def remote_file(path, head, expected):
    obj = json.loads(gh("contents/" + path + "?ref=" + head))
    raw = base64.b64decode(obj["content"])
    need(sha(raw) == expected, "actual original producer bytes " + path)
    return raw


def validate_provenance(pin, meta):
    valid_artifact(pin)
    run, artifact = meta["run"], meta["artifact"]
    need(type(run["id"]) is int and run["id"] == pin["run_id"] and run["head_sha"] == pin["run_head"]
         and type(run["run_attempt"]) is int and run["run_attempt"] == 1
         and run["status"] == "completed" and run["conclusion"] == "success"
         and run["repository"]["full_name"] == REPO and run["repository"]["private"] is False
         and artifact["id"] == pin["id"] and artifact["expired"] is False
         and artifact["size_in_bytes"] == pin["bytes"] and artifact["digest"] == "sha256:" + pin["sha256"]
         and artifact["workflow_run"]["id"] == pin["run_id"]
         and artifact["workflow_run"]["head_sha"] == pin["run_head"]
         and meta["commit"]["sha"] == pin["run_head"], "authentic first successful public input run/artifact")


def validate_execution_metadata(identity, run, commit):
    need(run["id"] == int(identity["run_id"]) and run["head_sha"] == identity["run_head"]
         and type(run["run_attempt"]) is int and run["run_attempt"] == 1 and run["event"] == "push"
         and run["head_branch"] == BRANCH.removeprefix("refs/heads/") and run["path"] == WORKFLOW
         and run["repository"]["full_name"] == REPO and run["repository"]["private"] is False
         and commit["sha"] == identity["run_head"] and len(commit["parents"]) == 1
         and commit["parents"][0]["sha"] == identity["prepared_parent"]
         and [entry["filename"] for entry in commit["files"]] == [WORKFLOW],
         "actual primary run/head/unique prepared parent and workflow-only opening")


def unused_workflow_run(identity):
    path = "actions/workflows/" + Path(WORKFLOW).name + "/runs?branch=" + BRANCH.removeprefix("refs/heads/") + "&event=push&per_page=100"
    data = json.loads(gh(path))
    need(data["total_count"] == len(data["workflow_runs"]) == 1
         and data["workflow_runs"][0]["id"] == int(identity["run_id"]),
         "no earlier execution of this experiment, even if a consumed artifact expired or disappeared")
    return data


def original_zip(path, pin):
    raw = path.read_bytes()
    need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"] and len(raw) <= 128_000_000, "original complete input ZIP")
    z = zipfile.ZipFile(io.BytesIO(raw))
    try:
        infos = z.infolist()
        need(len(infos) <= 8192 and len(infos) == len({i.filename for i in infos})
             and sum(i.file_size for i in infos) <= 128_000_000, "bounded unique archive members")
        for info in infos:
            name = PurePosixPath(info.filename)
            need(name.parts and not name.is_absolute() and ".." not in name.parts and "\\" not in info.filename
                 and not stat.S_ISLNK(info.external_attr >> 16), "safe original archive member")
        need(z.testzip() is None, "original archive CRC")
        return z
    except BaseException:
        z.close()
        raise


def fetch(inputs, prepared):
    spec = protocol()
    identity = guard(spec, prepared)
    producer, workflow = producer_contract(spec, identity["prepared_parent"])
    need(inputs.is_absolute() and inputs.parent.is_dir() and not inputs.exists(), "new evidence directory only")
    prior = json.loads(gh("actions/artifacts?name=" + MARKER + "&per_page=1"))
    need(prior["total_count"] == 0, "a successful or consumed-unknown B20 primary can never be replayed")
    run_inventory = unused_workflow_run(identity)
    run = json.loads(gh("actions/runs/" + identity["run_id"]))
    commit = json.loads(gh("commits/" + identity["run_head"]))
    validate_execution_metadata(identity, run, commit)
    inputs.mkdir()
    pins = producer_files(spec, producer, workflow, "primary")
    for path, expected in pins.items():
        dest = inputs / "primary_producer" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(remote_file(path, identity["run_head"], expected))
    metadata = {}
    for key, pin in archive_pins(spec).items():
        metadata[key] = {"run": json.loads(gh("actions/runs/" + str(pin["run_id"]))),
                         "artifact": json.loads(gh("actions/artifacts/" + str(pin["id"]))),
                         "commit": json.loads(gh("commits/" + pin["run_head"]))}
        validate_provenance(pin, metadata[key])
        raw = gh("actions/artifacts/" + str(pin["id"]) + "/zip")
        need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "whole authentic input archive " + key)
        (inputs / (key + ".zip")).write_bytes(raw)
    for path, expected in producer_files(spec, producer, workflow, "source").items():
        dest = inputs / "source_producer" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(remote_file(path, spec["inputs"]["synth_source"]["run_head"], expected))
    write(inputs / "FETCH.json", dict(schema=SCHEMA, identity=identity, primary_run=run, primary_commit=commit,
         inputs_sha256=sha(SPEC.read_bytes()), producer_manifest_sha256=spec["producer_manifest_sha256"],
         workflow_sha256=sha(workflow), source_metadata=metadata, prior_consumed_markers=prior,
         original_workflow_run_inventory=run_inventory))


def person(row):
    for key in ("source_speaker_id", "speaker", "speaker_id"):
        if key in row:
            value = str(row[key]).removeprefix("p")
            need(re.fullmatch(r"[0-9]+", value), "canonical numeric reader")
            return str(int(value))
    group = row.get("source_group_id", row.get("group"))
    need(type(group) is str and group, "real source family")
    match = re.search(r"(?:piper-libritts-p|libri-speaker-|librispeech:)([0-9]+)$", group)
    return str(int(match[1])) if match else "family:" + group


def score_identity(row, cohort):
    return dict(id=row["id"], group=row["source_group_id"], cohort=cohort,
                cell=row["cell_id"], positive=row["observed_class"] == 2,
                tier=row["tier"], pcm_sha256=row["pcm_sha256"])


def validate_scored_join(records, expected):
    """Only exact admitted originals in their frozen order can carry logits."""
    need(bool(expected) and all(set(row) == set(IDENTITY_FIELDS) and type(row["positive"]) is bool
                               for row in expected)
         and len(records) == len(expected)
         and all(type(row["positive"]) is bool for row in records)
         and [{key: row[key] for key in IDENTITY_FIELDS} for row in records] == expected,
         "saved rows must be the exact admitted ID/PCM/group/cohort/cell/label/tier sequence")


def ctc_events(labels):
    """Original greedy blank/repeat collapse; clocks are sample availability."""
    need(type(labels) is list and all(type(value) is int and 0 <= value < 4 for value in labels),
         "actual four-class frame argmax labels")
    onsets = [(i, token) for i, token in enumerate(labels)
              if token and (i == 0 or labels[i - 1] != token)]
    tokens = [token for _, token in onsets]
    events = [onsets[i + 2][0] for i in range(max(0, len(tokens) - 2))
              if tokens[i:i + 3] == [1, 2, 3]]
    return dict(tokens=tokens, event_frames=events, event_samples=[512 + 160 * frame for frame in events],
                events=len(events))


def validate_new_format(rows):
    wanted = {"mero-fast": (2, 1.4), "mero-slow": (2, 1.8), "question": (0, 1.0), "request": (0, 1.08)}
    need(len(rows) == 96 and {r["speaker_index"] for r in rows} == set(range(798, 822))
         and len({r["source_speaker_id"] for r in rows}) == len({r["source_group_id"] for r in rows}) == 24,
         "exact new 24-source B20 primary")
    for index in range(798, 822):
        family = [r for r in rows if r["speaker_index"] == index]
        need(len(family) == 4 and {r["cell_id"] for r in family} == set(wanted)
             and len({person(r) for r in family}) == 1
             and all((r["requested_class"], r["length_scale"]) == wanted[r["cell_id"]]
                     and r["role"] == "development" and r["training_allowed"] is False
                     and r["observed_class"] is None for r in family),
             "fixed positive 1.4/1.8 and unchanged ordinary 1.0/1.08; no selected replacement rate")


def exact_admitted_join(original, decisions, cohort):
    need(len(original) == len(decisions) and len({r["id"] for r in original}) == len(original)
         and len({r["pcm_sha256"] for r in original}) == len(original), "complete unique original source/SCREEN denominator")
    admitted = []
    for row, label in zip(original, decisions, strict=True):
        need(row["id"] == label["id"] and row["pcm_sha256"] == label["pcm_sha256"],
             "source and SCREEN must match each exact ID/PCM in original order")
        if cohort == "synthetic":
            need(all(label[k] == value for k, value in row.items() if k != "observed_class"),
                 "SCREEN cannot alter any original source field")
            if label["observed_class"] is None:
                need(label["tier"] == "unknown", "unqualified synthetic source remains UNKNOWN")
                continue
            need(label["observed_class"] == row["requested_class"] in (0, 2)
                 and label["tier"] in ("silver", "weak"), "original blind canonical source admission")
            admitted.append(label)
        else:
            need(label["source_id"] == row["source_id"] and label["group"] == row["source_group_id"],
                 "exact original natural utterance and canonical reader")
            if label["tier"] == "unknown":
                continue
            need(label["tier"] in ("silver", "weak") and label["full_wake_veto"] is False
                 and type(label["exact_votes"]) is int and label["exact_votes"] >= 2,
                 "only unchanged originally admitted natural ordinary")
            admitted.append(dict(row, observed_class=0, cell_id="read-speech", tier=label["tier"]))
    return admitted


def reject_overlap(synthetic, natural, training, history, canary_pcm, canary_people, reserve_people):
    need(len(synthetic) == 96 and len(natural) == 24 and len(training) == 88,
         "all original source and TRAIN denominators, including UNKNOWN")
    all_rows = synthetic + natural
    need(len({r["id"] for r in all_rows}) == len({r["pcm_sha256"] for r in all_rows}) == 120,
         "all 120 original source IDs/PCM are distinct")
    need(len(canary_pcm) == 48 and len(canary_people) == 8
         and canary_pcm <= set(history["all_pcm_sha256"])
         and canary_people <= set(history["canonical_people"]), "all B19 diagnostic originals/readers permanently excluded")
    for key, function in (("all_pcm_sha256", lambda r: r["pcm_sha256"]),
                          ("all_source_groups", lambda r: r["source_group_id"]), ("canonical_people", person)):
        need(not ({function(r) for r in synthetic} & set(history[key])), "new synthetic overlaps complete historical " + key)
    # Natural reuse is authorized only by exact original ZIPs/rows/SCREEN before
    # this call. Its old membership is NOT removed from synthetic exclusions.
    need(not ({person(r) for r in synthetic} & {person(r) for r in natural})
         and not ({person(r) for r in all_rows} & canary_people)
         and not ({r["pcm_sha256"] for r in all_rows} & canary_pcm)
         and not ({person(r) for r in all_rows} & reserve_people), "canonical canary/primary/reserve separation")
    train_pcm = {r.get("pcm_sha256", r.get("source_pcm_sha256")) for r in training}
    train_groups = {r.get("source_group_id", r.get("group")) for r in training}
    need(len(train_pcm) == 88 and None not in train_pcm and None not in train_groups
         and not (train_pcm & {r["pcm_sha256"] for r in all_rows})
         and not (train_groups & {r["source_group_id"] for r in all_rows})
         and not ({person(r) for r in training} & {person(r) for r in all_rows}),
         "all source originals including natural UNKNOWN are independent of actual paired TRAIN")


def preflight(inputs, prepared):
    spec = protocol()
    identity = guard(spec, prepared)
    producer, workflow = producer_contract(spec, identity["prepared_parent"])
    fetched = json.loads((inputs / "FETCH.json").read_bytes())
    need(fetched["schema"] == SCHEMA and fetched["identity"] == identity
         and fetched["inputs_sha256"] == sha(SPEC.read_bytes())
         and fetched["producer_manifest_sha256"] == spec["producer_manifest_sha256"]
         and fetched["workflow_sha256"] == sha(workflow) and fetched["prior_consumed_markers"]["total_count"] == 0,
         "same original authenticated acquisition and no previous consumption")
    validate_execution_metadata(identity, fetched["primary_run"], fetched["primary_commit"])
    inventory = fetched["original_workflow_run_inventory"]
    need(inventory["total_count"] == len(inventory["workflow_runs"]) == 1
         and inventory["workflow_runs"][0]["id"] == int(identity["run_id"]), "retained one-shot run inventory")
    for role in ("primary", "source"):
        for path, expected in producer_files(spec, producer, workflow, role).items():
            need(sha((inputs / (role + "_producer") / path).read_bytes()) == expected,
                 "retained actual-head scientific producer " + path)
    for key, pin in archive_pins(spec).items():
        validate_provenance(pin, fetched["source_metadata"][key])
    old, decision, source, plan = helpers(spec)
    with ExitStack() as stack, tempfile.TemporaryDirectory() as tmp:
        archives = {key: stack.enter_context(original_zip(inputs / (key + ".zip"), pin)) for key, pin in archive_pins(spec).items()}
        pair = archives["pair"]
        receipt = json.loads(pair.read("RESULT.json"))
        train = json.loads(pair.read("SOURCE.json"))
        old.validate_pair_receipt(receipt)
        need(train["parent_pt_sha256"] == spec["parent_pt_sha256"]
             and train["design_sha256"] == TRAINING_DESIGN and train["independent_assessment_sha256"] == TRAINING_ASSESSMENT,
             "original actual TRAIN provenance before Torch")
        need(len(train["base_train"]) == 79 and len(train["natural_train"]) == 9,
             "all original 79 base and 9 natural TRAIN sources before Torch")
        for arm in ("control", "challenger"):
            for ext in ("pt", "npz", "onnx"):
                member = old.STEMS[arm] + "." + ext
                raw = pair.read(member)
                need(receipt[arm]["files"][Path(member).name] == dict(bytes=len(raw), sha256=sha(raw)), "complete original paired model member")
                if ext == "pt":
                    need(sha(raw) == spec["model_pt_sha256"][arm], "same frozen trained arm")
        need(sha(archives["B0"].read("models/mero-fsmn-ctc-expanded256.pt")) == spec["model_pt_sha256"]["B0"]
             and sha(archives["B0"].read("models/mero-fsmn-ctc-expanded256.npz")) == B0_NPZ
             and sha(archives["B9"].read(old.common.b9.MODELS["B9_control"]["path"] + ".pt")) == spec["model_pt_sha256"]["B9"],
             "original baseline/teacher identity before Torch")
        directory = Path(tmp) / "synthetic"
        directory.mkdir()
        archives["synth_source"].extractall(directory)  # original_zip already rejects unsafe members.
        owner = source.evaluation_preflight(directory)  # Complete original B19 proof; no new model/source call.
        synthetic = owner["rows"]
        validate_new_format(synthetic)
        history = owner["history"]
        need(owner["assessment_sha256"] == spec["assessment_sha256"]
             and owner["b19_feasibility_status"] == "SOURCE_FORMAT_FEASIBLE_REQUIRES_NEW_PREREGISTRATION"
             and valid_hash(owner["b19_result_sha256"]), "actual re-read B19 PASS, never a CI-success proxy")
        execution = owner["source_execution"]
        pin = spec["inputs"]["synth_source"]
        meta = fetched["source_metadata"]["synth_source"]
        need(str(execution["run_id"]) == str(pin["run_id"]) and execution["run_head"] == pin["run_head"]
             and execution["run_attempt"] == 1 and execution["run_branch"] == "refs/heads/" + SOURCE_BRANCH
             and meta["run"]["head_branch"] == SOURCE_BRANCH and meta["run"]["path"] == SOURCE_WORKFLOW
             and meta["run"]["event"] == "push" and len(meta["commit"]["parents"]) == 1
             and meta["commit"]["parents"][0]["sha"] == execution["prepared_parent"]
             and [entry["filename"] for entry in meta["commit"]["files"]] == [SOURCE_WORKFLOW],
             "source MANIFEST/AUTH binds actual first public run/head/parent")
        manifest_raw = archives["synth_source"].read("MANIFEST.json")
        manifest = json.loads(manifest_raw)
        screen = json.loads(archives["synth_screen"].read("SCREEN.json"))
        planned = source.source_plan("primary")["rows"]
        need(manifest["rows"] == synthetic and manifest["schema"] == SOURCE_SCHEMA
             and manifest["panel"] == screen["panel"] == "primary" and screen["schema"] == SCREEN_SCHEMA
             and screen["source_manifest_sha256"] == sha(manifest_raw)
             and screen["assessment_sha256"] == spec["assessment_sha256"]
             and screen["coverage"] == dict.fromkeys(source.qa.observers.PROVIDERS, 96)
             and screen["tts_calls_at_screen"] == screen["new_model_forwards"] == screen["optimizer_updates"] == 0
             and screen["training_allowed"] is screen["promotion_allowed"] is screen["replication_opened"] is False
             and len(synthetic) == len(planned) == 96
             and {r["speaker_index"] for r in synthetic} == set(range(798, 822))
             and all(all(row[k] == v for k, v in frozen.items()) for row, frozen in zip(synthetic, planned, strict=True)),
             "all 96 exact new planned originals and original five-by-96 strict SCREEN")
        synthetic_providers = Path(tmp) / "synthetic-providers"
        natural_providers = Path(tmp) / "natural-providers"
        for role, target in (("synthetic", synthetic_providers), ("natural", natural_providers)):
            for provider in PROVIDERS:
                dest = target / provider; dest.mkdir(parents=True)
                archives[role + "_native_" + provider].extractall(dest)
        source.verify_screen_contract(directory, screen, synthetic_providers)
        selected_synth = exact_admitted_join(synthetic, screen["rows"], "synthetic")
        natural_manifest = json.loads(archives["natural_source"].read("MANIFEST.json"))
        natural_screen = json.loads(archives["natural_screen"].read("SCREEN.json"))
        natural = natural_manifest["rows"]
        natural_directory = Path(tmp) / "natural-source"
        natural_directory.mkdir()
        archives["natural_source"].extractall(natural_directory)
        reproduced = Path(tmp) / "original-natural-screen.json"
        old.natural.screen(natural_directory, natural_providers, reproduced)
        need(json.loads(reproduced.read_bytes()) == natural_screen,
             "all original natural native receipts reproduce the unchanged original SCREEN exactly; no new observation")
        selected_natural = exact_admitted_join(natural, natural_screen["decisions"], "natural")
        unchanged, natural_counts = old.verified_human(archives["natural_source"], archives["natural_screen"])
        need(selected_natural == unchanged and natural_counts == dict(natural_originals=21, natural_speakers=11)
             and len(natural) == 24 and sum(r["tier"] == "unknown" for r in natural_screen["decisions"]) == 3,
             "only exact originally admitted 21/11 natural; three UNKNOWN cannot substitute")
        for row in natural:
            raw = archives["natural_source"].read("wav/" + row["id"] + ".wav")
            with wave.open(io.BytesIO(raw)) as audio:
                need(audio.getparams()[:3] == (1, 2, 16000) and audio.getnframes() == row["samples"], "all 24 full original natural PCM")
                pcm = audio.readframes(audio.getnframes())
            need(sha(raw) == row["wav_sha256"] and sha(pcm) == row["pcm_sha256"], "natural UNKNOWN raw bytes are also authenticated")
        reserve = {person(r) for r in source.source_plan("replication")["rows"]}
        reserve.update(str(int(p)) for p in plan["replication"]["natural"]["speakers"])
        reject_overlap(synthetic, natural, train["base_train"] + train["natural_train"], history,
                       set(owner["b19_canary_pcm_sha256"]), set(owner["b19_canary_people"]), reserve)
        groups = {}
        for row in selected_synth:
            groups.setdefault(row["source_group_id"], set()).add(row["observed_class"])
        pairs = sorted(k for k, values in groups.items() if values == {0, 2})
        counts = dict(synthetic_positive=sum(r["observed_class"] == 2 for r in selected_synth),
                      synthetic_ordinary=sum(r["observed_class"] == 0 for r in selected_synth),
                      synthetic_paired=len(pairs), **natural_counts)
        need(screen["paired_source_groups"] == pairs and screen["source_coverage"] == dict(
             positive_originals=counts["synthetic_positive"], ordinary_originals=counts["synthetic_ordinary"],
             paired_source_groups=counts["synthetic_paired"]), "exact original screened source denominators")
        gate, missing = old.source_gate(counts)
        need(gate == "SOURCE_GATE_PASS" and not missing and screen["source_gate"] == gate and not screen["source_shortfall"],
             "INSUFFICIENT_SOURCE: no model work is permitted")
    chosen = [(r, "synthetic") for r in selected_synth] + [(r, "natural") for r in selected_natural]
    admission = dict(schema=SCHEMA, identity=identity, assessment_sha256=spec["assessment_sha256"],
        registered_inputs_sha256=sha(SPEC.read_bytes()), producer_manifest_sha256=spec["producer_manifest_sha256"],
        workflow_sha256=sha(workflow), source_gate=gate, source_shortfall=missing, source_coverage=counts,
        original_source_rows=120, admitted_rows=[score_identity(r, c) for r, c in chosen],
        training_source_sha256=sha(json.dumps(train, sort_keys=True, separators=(",", ":")).encode()),
        model_calls=0, B19_canary_excluded=True, promotion_allowed=False)
    return spec, identity, old, decision, chosen, train, admission


class Ledger:
    """Every actual forward entry consumes one immutable key, even if it fails."""
    def __init__(self, directory):
        self.directory = directory
        self.counts = dict.fromkeys(ARMS, 0)
        self.current = None
        self.consumed = {}

    def entered(self, role):
        need(self.current is not None and self.current["role"] == role, "one explicitly bound actual model call")
        key = self.current["key"]
        need(key not in self.consumed, "successful or consumed-unknown case must never be replayed")
        need(sum(self.counts.values()) < 1404, "hard ceiling before an additional model forward entry")
        self.counts[role] += 1
        self.consumed[key] = "consumed_unknown"
        write(self.directory / (key + ".json"), dict(**self.current, status="consumed_unknown"))

    def saved(self, key):
        need(self.consumed.get(key) == "consumed_unknown", "one returned array per actual forward")
        self.consumed[key] = "saved_array"


def score(inputs, output, prepared):
    spec = protocol()  # Before paths, helper imports and model initialization.
    checked = preflight(inputs, prepared)
    spec, identity, old, decision, chosen, train, admission = checked
    permit = json.loads((inputs / "CONSUMPTION.json").read_bytes())
    need(permit["identity"] == identity and permit["marker_name"] == MARKER
         and permit["admission_sha256"] == sha(json.dumps(admission, sort_keys=True, separators=(",", ":")).encode()),
         "one uploaded consumed marker must bind this exact complete preflight")
    unused_workflow_run(identity)
    remote = json.loads(gh("actions/artifacts?name=" + MARKER + "&per_page=100"))
    need(remote["total_count"] == 1 and len(remote["artifacts"]) == 1
         and remote["artifacts"][0]["workflow_run"]["id"] == int(identity["run_id"])
         and remote["artifacts"][0]["workflow_run"]["head_sha"] == identity["run_head"],
         "exact current-run consumed marker; any older or ambiguous consumption forbids replay")
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(), "new result directory only")
    output.mkdir()
    attempts = output / "attempts"; attempts.mkdir()
    raw_outputs = output / "raw_outputs"; raw_outputs.mkdir()
    ledger = Ledger(attempts)
    report = dict(schema=SCHEMA, **identity, assessment_sha256=spec["assessment_sha256"],
        training_assessment_sha256=TRAINING_ASSESSMENT, training_design_sha256=TRAINING_DESIGN,
        registered_inputs_sha256=sha(SPEC.read_bytes()), implementation_sha256=sha(Path(__file__).read_bytes()),
        producer_manifest_sha256=spec["producer_manifest_sha256"], workflow_sha256=admission["workflow_sha256"],
        source_gate="SOURCE_GATE_PASS", source_coverage=admission["source_coverage"], source_shortfall={},
        panel_gate="STARTED", records=[], actual_forwards=0, saved_array_count=0, raw_output_count=0,
        optimizer_updates=0, model_pt_sha256=spec["model_pt_sha256"], installed_software_baseline="B0",
        replication_open_allowed=False, B17_source_failure_preserved=True, B18_source_failure_preserved=True,
        original_natural_reused_without_reobservation=True, ssc305_physical=False,
        natural_field_far_per_hour=None, field_3_5m_frr=None, **dict.fromkeys(FLAGS, False))
    write(output / "ADMISSION.json", admission)
    write(output / "STARTED.json", report)
    hooks, arrays = [], None
    try:
        import numpy as np
        import torch
        need(np.__version__ == "1.26.4" and torch.__version__ == "2.9.1+cpu", "unchanged pinned CPU scoring runtime")
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True); torch.backends.mkldnn.enabled = False
        models, loaded_train = old.snapshots(inputs, spec, torch, np)
        need(loaded_train == train and {k: v[-1] for k, v in models.items()} == spec["model_pt_sha256"],
             "same preflight TRAIN and all four original models/frontends")
        for role, values in models.items():
            def entered(_module, _args, role=role):
                ledger.entered(role)
            hooks.append(values[0].register_forward_pre_hook(entered))
        arrays = zipfile.ZipFile(output / "ALL_FRAME_LOGITS.npz", "x", compression=zipfile.ZIP_DEFLATED)
        with ExitStack() as stack:
            arcs = {k: stack.enter_context(original_zip(inputs / (k + ".zip"), spec["inputs"][k])) for k in ("synth_source", "natural_source")}
            for row, cohort in chosen:
                record = score_identity(row, cohort); record["gains"] = {}; report["records"].append(record)
                raw = arcs["synth_source" if cohort == "synthetic" else "natural_source"].read("wav/" + row["id"] + ".wav")
                with wave.open(io.BytesIO(raw)) as audio:
                    need(audio.getparams()[:3] == (1, 2, 16000) and audio.getnframes() == row["samples"], "whole admitted PCM")
                    pcm = audio.readframes(audio.getnframes())
                need(sha(raw) == row["wav_sha256"] and sha(pcm) == row["pcm_sha256"], "same preflight original PCM")
                for gain in GAINS:
                    signal = old.common.b9.ref.gain_pcm(pcm, float(gain), np)
                    record["gains"][gain] = {}
                    for role in ARMS:
                        net, mel, mean, scale, _ = models[role]
                        feature = old.common.b9.v5.features(signal, row, torch, np, mel)
                        x = torch.from_numpy(((feature - mean) / scale).astype(np.float32)).unsqueeze(0)
                        key = row["id"] + "__" + gain + "__" + role
                        need(re.fullmatch(r"[A-Za-z0-9_.-]+", key), "safe immutable array key")
                        ledger.current = dict(key=key, role=role, source_id=row["id"], gain=gain)
                        with torch.inference_mode():
                            y = net(x)
                        raw_y = y.detach().cpu().numpy()
                        with (raw_outputs / (key + ".npy")).open("xb") as handle:
                            np.save(handle, raw_y, allow_pickle=False)
                            handle.flush(); os.fsync(handle.fileno())
                        report["raw_output_count"] += 1
                        need(raw_y.shape == (1, len(feature), 4) and raw_y.dtype == np.dtype("float32") and np.isfinite(raw_y).all(),
                             "complete actual finite FP32 frame output")
                        matrix = np.ascontiguousarray(raw_y[0], dtype="<f4")
                        encoded = io.BytesIO(); np.save(encoded, matrix, allow_pickle=False)
                        arrays.writestr(key + ".npy", encoded.getvalue())
                        arrays.fp.flush(); os.fsync(arrays.fp.fileno())
                        report["saved_array_count"] += 1
                        ledger.saved(key)
                        labels = matrix.argmax(-1).tolist()
                        record["gains"][gain][role] = dict(array=key, sha256=sha(matrix.tobytes()), frames=len(matrix),
                            **ctc_events(labels))
        validate_scored_join(report["records"], admission["admitted_rows"])
        need(sum(ledger.counts.values()) == report["saved_array_count"] == report["raw_output_count"] == 12 * len(chosen) <= 1404
             and all(v == 3 * len(chosen) for v in ledger.counts.values()), "all and only the exact admitted original/gain/model calls completed")
        decision_result = decision.decide(report["records"])
        report.update(decision_result, replication_open_allowed=decision_result["panel_gate"] == "PASS")
    except BaseException as exc:
        report.update(panel_gate="STOP", error_type=type(exc).__name__, error=str(exc)[:2000], replication_open_allowed=False)
        raise
    finally:
        for hook in hooks:
            hook.remove()
        if arrays is not None:
            arrays.close()
            report["logits_sha256"] = sha((output / "ALL_FRAME_LOGITS.npz").read_bytes())
        report.update(actual_forwards=sum(ledger.counts.values()), model_forward_calls=ledger.counts,
                      consumed_unknown=[key for key, value in ledger.consumed.items() if value != "saved_array"],
                      replay_allowed=False)
        write(output / "RESULT.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "fetch", "preflight", "score"))
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepared-parent")
    args = parser.parse_args()
    if args.mode == "check":
        print(json.dumps(source_check()))
    else:
        protocol()
        need(args.inputs is not None, "original authenticated input directory required")
        if args.mode == "fetch":
            fetch(args.inputs, args.prepared_parent)
        elif args.mode == "preflight":
            checked = preflight(args.inputs, args.prepared_parent)
            admission = checked[-1]
            write(args.inputs / "CONSUMPTION.json", dict(identity=checked[1], marker_name=MARKER,
                  admission_sha256=sha(json.dumps(admission, sort_keys=True, separators=(",", ":")).encode())))
            print(json.dumps(admission))
        else:
            need(args.output is not None, "new output directory required")
            score(args.inputs, args.output, args.prepared_parent)
