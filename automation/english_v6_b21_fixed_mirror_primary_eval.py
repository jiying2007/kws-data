"""B21 primary scorer routed through the audited fixed-official-mirror source.

The unchanged B14 quality function is applied once to exact source-admitted
originals. Inert until real pair/source/native artifact pins and the complete
producer closure are reviewed. Every forward is consumed before execution.
Both original zero-call source failures and both prior drivers remain retained.
All scientific functions, Ledger and Git-blob fetch are unchanged from the
continuation evaluator. The original unused scoring workflow and marker remain.
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
SPEC = HERE / "english_v6_b21_fixed_mirror_primary_inputs.json"
PRODUCER = HERE / "english_v6_b21_fixed_mirror_primary_producer.json"
WORKFLOW_TEMPLATE = HERE / "english_v6_b21_fixed_mirror_primary_workflow_template.yml"
TRAIN_DESIGN = HERE / "english_v6_b21_competing_margin_design.json"
ASSESSMENT = HERE / "english_v6_b21_panels/assessment.json"
TRAINING_DESIGN = "76a1fd1353de00f712023a2b20256a8e622557551823cacd8b808d3f0549b0a1"
TRAINING_ASSESSMENT = "65ad9c7204755f1cd0b639added163387af39604818f9bd26b9da9fe8bd098fc"
SCHEMA = "english-v6-b21-fresh-primary-four-frozen-models-v1"
SOURCE_MODULE = "english_v6_b21_fixed_mirror_source"
SOURCE_SCHEMA = "english-v6-b21-fresh-combined-fixed-mirror-source-v1"
SCREEN_SCHEMA = "english-v6-b21-120-original-fixed-mirror-screen-v1"
REPO = "jiying2007/kws-data"
BRANCH = "refs/heads/research/en-v6-b21-primary-scoring-once-20261124"
WORKFLOW = ".github/workflows/english-v6-b21-primary-model-eval-once.yml"
SOURCE_BRANCH = "research/en-v6-b21-primary-source-fixed-mirror-once-20261124"
SOURCE_WORKFLOW = ".github/workflows/english-v6-b21-primary-source-fixed-mirror-once.yml"
TRAIN_BRANCH = "research/en-v6-b21-blank-retention-train-once-20261124"
PARENT_TOKEN = "__B21_REVIEWED_SCORING_PARENT__"
MARKER = "english-v6-b21-primary-consumed"
QUALITY_CODE = "6aa5f999598e47b41eb0a946acd82fdc5b47d30aea8990871c1c1a01065e4023"
QUALITY_ASSESSMENT = "944f5df3c7bc3244977a8a33f7e1c393bffff9f7f5c32d2d8cc79e2628e265d9"
B0_PT = "1ae8a2b20c8b6991f925dd4c626af2f36b308adee8733a3a7215818716f565d7"
B9_PT = "7dba86a3fcda6e5f05cb80b414dc372c70a5a03b3aee783f168441a4871e7655"
B0_NPZ = "02b7febf9f736fd70f4f9ae04042ffc6d609b15011d03d235b960f69b1d71696"
GAINS = ("1.0", "0.75", "0.5")
ARMS = ("B0", "B9", "control", "challenger")
PAIR_SCHEMA = "english-v6-b21-positive-competing-margin-pair-v1"
STEMS = {arm: arm + "/models/mero43-b21-" + arm + "192" for arm in ("control", "challenger")}
IDENTITY_FIELDS = ("id", "group", "cohort", "cell", "positive", "tier", "pcm_sha256")
PROVIDERS = ("qwen", "whisper", "parakeet", "xlsr", "lv60")
INPUT_KEYS = {"B0", "B9", "pair", "source", "screen"}
GATE = dict(synthetic_positive=12, synthetic_ordinary=24, synthetic_paired=8,
            natural_originals=12, natural_speakers=8)
MAX_FORWARDS = 1440
FLAGS = ("training_allowed", "threshold_search", "promotion_allowed", "product_released",
         "cal_locked_access", "old_replication_opened", "replication_opened",
         "candidate_engineering_validation_allowed")
PRODUCER_ROOTS = ("english_v6_b21_fixed_mirror_primary_eval", SOURCE_MODULE,
                  "english_v6_four_model_panel", "english_v6_b14_decision")
BASELINES = {'B0': {'id': 14514228406, 'run_id': 44495594627, 'run_head': 'd0b567fe37147aae6ab5d7bdec3721b879a0b22d', 'bytes': 501276, 'sha256': '91b203f05041d92e7ffb1b748f94959ce19af6673b91dbc2144920815058d702'}, 'B9': {'id': 14689218171, 'run_id': 44860855857, 'run_head': '846c745ebcee58a1beb27300ddcd7ec8e2178916', 'bytes': 1001753, 'sha256': '1a8b270200e0c72ab1025f139d16abde43c209f142d72e71a7201b847e06a612'}}

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

def guard(spec, prepared, env=None, event=None, head=None, parent=None):
    e = os.environ if env is None else env
    need(valid_hash(prepared, 40) and spec["prepared_parent"] is None
         and spec["parent_authority"] == "reviewed_workflow", "reviewed workflow is the sole execution parent authority")
    need(all(e.get(k) == v for k, v in {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPO,
         "GITHUB_EVENT_NAME": "push", "GITHUB_REF": BRANCH, "GITHUB_RUN_ATTEMPT": "1"}.items())
         and valid_hash(e.get("GITHUB_SHA"), 40) and re.fullmatch(r"[1-9][0-9]*", e.get("GITHUB_RUN_ID", "")),
         "first designated public B21 primary event only")
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


def protocol():
    doc = json.loads(SPEC.read_bytes())
    need(doc.get("scoring_allowed") is True, "INERT: real B21 training and independently admitted source pins required")
    need(doc["schema"] == SCHEMA and doc["source_module"] == SOURCE_MODULE
         and doc["source_schema"] == SOURCE_SCHEMA and doc["screen_schema"] == SCREEN_SCHEMA
         and doc["prepared_parent"] is None and doc["parent_authority"] == "reviewed_workflow"
         and doc["pair_schema"] == PAIR_SCHEMA and doc["pair_stems"] == STEMS
         and doc["parent_pt_sha256"] == B0_PT and doc["model_pt_sha256"]["B0"] == B0_PT
         and doc["model_pt_sha256"]["B9"] == B9_PT and set(doc["model_pt_sha256"]) == set(ARMS)
         and all(valid_hash(value) for value in doc["model_pt_sha256"].values())
         and len(set(doc["model_pt_sha256"].values())) == 4
         and doc["gains"] == list(GAINS) and doc["source_gate"] == GATE
         and type(doc["maximum_model_forwards"]) is int and doc["maximum_model_forwards"] == MAX_FORWARDS
         and all(doc[key] is False for key in FLAGS) and set(doc["inputs"]) == INPUT_KEYS
         and set(doc["native_inputs"]) == set(PROVIDERS), "fixed B21 four-role/fresh-source experiment only")
    for field in ("assessment_sha256", "training_design_sha256", "producer_manifest_sha256"):
        need(valid_hash(doc[field]), "complete frozen registration digest " + field)
    need(sha(ASSESSMENT.read_bytes()) == doc["assessment_sha256"] == TRAINING_ASSESSMENT
         and sha(TRAIN_DESIGN.read_bytes()) == doc["training_design_sha256"] == TRAINING_DESIGN
         and doc["training_assessment_sha256"] == doc["assessment_sha256"],
         "same prospective assessment frozen before both source and optimizer")
    for pin in archive_pins(doc).values():
        valid_artifact(pin)
    need(all(doc["inputs"][role] == pin for role, pin in BASELINES.items()), "unchanged original B0/B9 artifacts")
    source_pins = [doc["inputs"]["source"], doc["inputs"]["screen"], *doc["native_inputs"].values()]
    need(all(all(pin[key] == source_pins[0][key] for key in ("run_id", "run_head")) for pin in source_pins)
         and len({pin["id"] for pin in archive_pins(doc).values()}) == 10
         and doc["inputs"]["pair"]["run_id"] != source_pins[0]["run_id"],
         "ten distinct originals; fresh source, SCREEN and five native proofs share one successful run")
    return doc


def archive_pins(spec):
    return {**spec["inputs"], **{"native_" + provider: pin for provider, pin in spec["native_inputs"].items()}}


def source_check():
    doc = json.loads(SPEC.read_bytes())
    return dict(schema=SCHEMA, status="INERT_SCAFFOLD" if doc.get("scoring_allowed") is not True else "BOUND_NOT_EXECUTED",
                model_calls=0, source_calls=0, pcm_reads=0, promotion_allowed=False)


def producer_contract(spec, prepared, root=ROOT):
    raw = (root / "automation" / PRODUCER.name).read_bytes()
    need(sha(raw) == spec["producer_manifest_sha256"], "frozen B21 complete producer manifest")
    doc = json.loads(raw)
    need(doc["schema"] == "english-v6-b21-primary-producer-v1" and doc["frozen"] is True,
         "reviewed complete primary/source producer inventory")
    for role, roots in (("primary", PRODUCER_ROOTS), ("source", (SOURCE_MODULE,))):
        pins = doc[role]["files"]
        need(set(pins) == static_closure(root / "automation", roots), "complete scientific Python closure " + role)
        for name, digest in pins.items():
            need(valid_hash(digest) and sha((root / name).read_bytes()) == digest, "frozen scientific code " + name)
    required = {"automation/english_v6_b21_panels/assessment.json",
                "automation/english_v6_b21_competing_margin_design.json",
                "automation/english_v6_b14_panels/assessment.json"}
    need(type(doc["resources"]) is dict and required <= set(doc["resources"]), "complete frozen evaluation registrations")
    for name, digest in doc["resources"].items():
        path = PurePosixPath(name)
        need(path.parts and not path.is_absolute() and ".." not in path.parts and "\\" not in name
             and valid_hash(digest) and sha((root / name).read_bytes()) == digest, "frozen registration resource " + name)
    source_resources = doc["source"]["resources"]
    need(type(source_resources) is dict and "automation/english_v6_b21_panels/assessment.json" in source_resources
         and all(doc["resources"].get(path) == digest for path, digest in source_resources.items()),
         "all actual source resources retained in the primary closure")
    template = (root / "automation" / WORKFLOW_TEMPLATE.name).read_bytes()
    need(doc["primary"]["workflow_path"] == WORKFLOW and doc["source"]["workflow_path"] == SOURCE_WORKFLOW
         and sha(template) == doc["primary"]["workflow_template_sha256"]
         and template.count(PARENT_TOKEN.encode()) == 2
         and b"include-hidden-files: true" in template and valid_hash(doc["source"]["workflow_sha256"]),
         "reviewed workflows explicitly retain hidden producer files")
    need(valid_hash(prepared, 40), "actual uniquely reviewed predecessor")
    workflow = template.replace(PARENT_TOKEN.encode(), prepared.encode())
    need((root / WORKFLOW).read_bytes() == workflow, "only two reviewed parent substitutions in actual scoring workflow")
    return doc, workflow


def producer_files(spec, producer, workflow, role):
    if role == "primary":
        extras = {**producer["resources"], WORKFLOW: sha(workflow),
                  "automation/" + SPEC.name: sha(SPEC.read_bytes()),
                  "automation/" + PRODUCER.name: spec["producer_manifest_sha256"],
                  "automation/" + WORKFLOW_TEMPLATE.name: producer["primary"]["workflow_template_sha256"]}
    else:
        need(role == "source", "explicit frozen source producer role")
        extras = {**producer["source"]["resources"], SOURCE_WORKFLOW: producer["source"]["workflow_sha256"]}
    return {**producer[role]["files"], **extras}


def helpers(spec):
    sys.path.insert(0, str(HERE))
    common = importlib.import_module("english_v6_four_model_panel")
    decision = importlib.import_module("english_v6_b14_decision")
    source = importlib.import_module(SOURCE_MODULE)
    need(sha(Path(decision.__file__).read_bytes()) == QUALITY_CODE
         and sha(decision.ASSESSMENT.read_bytes()) == QUALITY_ASSESSMENT
         and source.SCHEMA == SOURCE_SCHEMA and source.BATCH == SCREEN_SCHEMA
         and source.ASSESSMENT_SHA == spec["assessment_sha256"]
         and sha(source.ASSESSMENT.read_bytes()) == spec["assessment_sha256"],
         "unchanged B14 quality and original preregistered B21 source rules")
    plan = source.lineage.read_plan()
    need(plan["source_gate"] == GATE and plan["gains"] == list(GAINS)
         and plan["quality_assessment_sha256"] == QUALITY_ASSESSMENT,
         "prospective source minima, gains and original numerical quality remain frozen")
    return common, decision, source, plan


def gh(endpoint):
    return subprocess.check_output(["gh", "api", "repos/" + REPO + "/" + endpoint])


def git_blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def git_tree(commit):
    digest = commit["commit"]["tree"]["sha"]
    need(valid_hash(digest, 40), "actual producer commit tree identity")
    tree = json.loads(gh("git/trees/" + digest + "?recursive=1"))
    need(tree["sha"] == digest and tree["truncated"] is False, "complete actual producer Git tree")
    return tree


def tree_member(tree, path):
    rows = [row for row in tree["tree"] if row["path"] == path]
    need(len(rows) == 1 and rows[0]["type"] == "blob" and rows[0]["mode"] in ("100644", "100755"),
         "one real producer Git file " + path)
    return rows[0]["sha"]


def remote_file(path, head, expected, tree):
    blob = tree_member(tree, path)
    obj = json.loads(gh("git/blobs/" + blob))
    need(obj["encoding"] == "base64", "complete producer blob encoding")
    raw = base64.b64decode(obj["content"])
    need(type(obj["size"]) is int and len(raw) == obj["size"]
         and sha(raw) == expected and obj["sha"] == git_blob(raw) == blob,
         "actual producer content bound to SHA256 and exact Git blob " + path)
    return raw


def fetch(inputs, prepared):
    spec = protocol()
    identity = guard(spec, prepared)
    producer, workflow = producer_contract(spec, identity["prepared_parent"])
    need(inputs.is_absolute() and inputs.parent.is_dir() and not inputs.exists(), "fresh immutable evidence directory")
    prior = json.loads(gh("actions/artifacts?name=" + MARKER + "&per_page=1"))
    need(prior["total_count"] == 0, "completed or consumed-unknown B21 primary cannot replay")
    run_inventory = unused_workflow_run(identity)
    run = json.loads(gh("actions/runs/" + identity["run_id"]))
    commit = json.loads(gh("commits/" + identity["run_head"]))
    validate_execution_metadata(identity, run, commit)
    inputs.mkdir()
    metadata = {}
    for key, pin in archive_pins(spec).items():
        metadata[key] = {"run": json.loads(gh("actions/runs/" + str(pin["run_id"]))),
                         "artifact": json.loads(gh("actions/artifacts/" + str(pin["id"]))),
                         "commit": json.loads(gh("commits/" + pin["run_head"]))}
        validate_provenance(pin, metadata[key])
        raw = gh("actions/artifacts/" + str(pin["id"]) + "/zip")
        need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "whole original input " + key)
        (inputs / (key + ".zip")).write_bytes(raw)
    trees = {"primary": git_tree(commit), "source": git_tree(metadata["source"]["commit"])}
    for role, head in (("primary", identity["run_head"]), ("source", spec["inputs"]["source"]["run_head"])):
        for path, expected in producer_files(spec, producer, workflow, role).items():
            target = inputs / (role + "_producer") / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(remote_file(path, head, expected, trees[role]))
    write(inputs / "FETCH.json", dict(schema=SCHEMA, identity=identity, primary_run=run, primary_commit=commit,
         inputs_sha256=sha(SPEC.read_bytes()), producer_manifest_sha256=spec["producer_manifest_sha256"],
         workflow_sha256=sha(workflow), source_metadata=metadata, producer_trees=trees,
         prior_consumed_markers=prior, original_workflow_run_inventory=run_inventory))


def validate_pair_receipt(receipt, spec):
    need(receipt.get("schema") == PAIR_SCHEMA
         and receipt.get("status") == "TRAINED_PAIR_REQUIRES_FRESH_INDEPENDENT_DEV"
         and receipt.get("parent_pt_sha256") == B0_PT
         and type(receipt.get("actual_optimizer_updates")) is int and receipt["actual_optimizer_updates"] == 384
         and receipt.get("trained_model_parameters") == 43876
         and receipt.get("baseline_originals") == 79 and receipt.get("natural_originals") == 9
         and receipt.get("positive_path_combinations") == 120
         and receipt.get("positive_originals") == 43 and receipt.get("positive_families") == 34
         and type(receipt.get("different_trained_float_values")) is int and receipt["different_trained_float_values"] > 0
         and receipt.get("dev_cal_locked_forwards") == receipt.get("actual_new_diagnostic_forwards") == 0
         and receipt.get("promotion_allowed") is receipt.get("numerical_B17_replay_claimed") is False
         and receipt.get("assessment_sha256") == receipt.get("independent_assessment_sha256") == spec["assessment_sha256"]
         and receipt.get("design_sha256") == spec["training_design_sha256"],
         "completed registered B21 TRAIN-only B0-parent pair; no historical numerical replay")
    for field in ("common_draw_sha256", "draw_coverage_sha256", "train_mechanism_sha256"):
        need(valid_hash(receipt.get(field)), "retained real paired training receipt " + field)
    for arm in ("control", "challenger"):
        value = receipt.get(arm, {})
        base = STEMS[arm].split("/")[-1]
        need(value.get("stem") == STEMS[arm] and value.get("optimizer_updates") == 192
             and value.get("parent_pt_sha256") == B0_PT
             and value.get("ordinary_ceiling_weight") == value.get("positive_retention_weight") == value.get("natural_loss_weight") == 0.08
             and value.get("positive_competitors") == ("other_two_nonblank" if arm == "control" else "other_three_including_blank")
             and set(value.get("files", {})) == {base + "." + extension for extension in ("pt", "npz", "onnx")},
             "B21 paired arm, unchanged common losses and one changed positive competitor set")
        for pin in value["files"].values():
            need(type(pin.get("bytes")) is int and pin["bytes"] > 10000 and valid_hash(pin.get("sha256")),
                 "complete actual trained model file pin")


def exact_admitted_join(original, decisions):
    need(len(original) == len(decisions) and len({row["id"] for row in original}) == len(original)
         and len({row["pcm_sha256"] for row in original}) == len(original),
         "complete source and native SCREEN population in original order")
    selected = []
    for row, label in zip(original, decisions, strict=True):
        need(row["observed_class"] is None and row["cohort"] in ("synthetic", "natural")
             and all(label[key] == value for key, value in row.items() if key != "observed_class"),
             "SCREEN cannot alter original identity, waveform, cohort or request")
        if label["observed_class"] is None:
            need(label["tier"] == "unknown", "unqualified native source remains UNKNOWN")
            continue
        need(type(label["observed_class"]) is int and label["observed_class"] == row["requested_class"] in (0, 2)
             and label["tier"] in ("silver", "weak"), "only unchanged originally admitted canonical source")
        if row["cohort"] == "natural":
            need(label["observed_class"] == 0 and label["full_wake_veto"] is False
                 and type(label["exact_votes"]) is int and label["exact_votes"] >= 2,
                 "blind independently admitted natural ordinary speech")
            label = dict(label, cell_id="natural-speech")
        selected.append(label)
    return selected


def source_counts(rows):
    groups = {}
    for row in rows:
        if row["cohort"] == "synthetic":
            groups.setdefault(row["source_group_id"], set()).add(row["observed_class"])
    return dict(synthetic_positive=sum(row["cohort"] == "synthetic" and row["observed_class"] == 2 for row in rows),
                synthetic_ordinary=sum(row["cohort"] == "synthetic" and row["observed_class"] == 0 for row in rows),
                synthetic_paired=sum(value == {0, 2} for value in groups.values()),
                natural_originals=sum(row["cohort"] == "natural" for row in rows),
                natural_speakers=len({person(row) for row in rows if row["cohort"] == "natural"}))


def reject_overlap(originals, training, history, reserve_people):
    need(len(originals) == len({row["id"] for row in originals}) == len({row["pcm_sha256"] for row in originals}) == 120
         and sum(row["cohort"] == "synthetic" for row in originals) == 96
         and sum(row["cohort"] == "natural" for row in originals) == 24 and len(training) == 88,
         "all120 source originals including source rejects and UNKNOWN, all88 TRAIN originals")
    for field, get in (("all_pcm_sha256", lambda row: row["pcm_sha256"]),
                       ("all_source_groups", lambda row: row["source_group_id"]), ("canonical_people", person)):
        need(not ({get(row) for row in originals} & set(history[field])), "all primary sources exclude historical " + field)
    primary_people = {person(row) for row in originals}
    need(len(primary_people) == 36 and not primary_people & set(reserve_people),
         "all new36 canonical readers independent of sealed replication")
    train_pcm = {row.get("pcm_sha256", row.get("source_pcm_sha256")) for row in training}
    train_groups = {row.get("source_group_id", row.get("group")) for row in training}
    need(len(train_pcm) == 88 and None not in train_pcm and None not in train_groups
         and not train_pcm & {row["pcm_sha256"] for row in originals}
         and not train_groups & {row["source_group_id"] for row in originals}
         and not {person(row) for row in training} & primary_people,
         "whole source PCM/groups/canonical readers independent of actual paired TRAIN")


def preflight(inputs, prepared):
    spec = protocol()
    identity = guard(spec, prepared)
    producer, workflow = producer_contract(spec, identity["prepared_parent"])
    fetched = json.loads((inputs / "FETCH.json").read_bytes())
    need(fetched["schema"] == SCHEMA and fetched["identity"] == identity
         and fetched["inputs_sha256"] == sha(SPEC.read_bytes())
         and fetched["producer_manifest_sha256"] == spec["producer_manifest_sha256"]
         and fetched["workflow_sha256"] == sha(workflow) and fetched["prior_consumed_markers"]["total_count"] == 0,
         "exact initial acquisition and original unconsumed experiment")
    validate_execution_metadata(identity, fetched["primary_run"], fetched["primary_commit"])
    prior_runs = fetched["original_workflow_run_inventory"]
    need(prior_runs["total_count"] == len(prior_runs["workflow_runs"]) == 1
         and prior_runs["workflow_runs"][0]["id"] == int(identity["run_id"]), "only original first primary run")
    pins = archive_pins(spec)
    need(set(fetched["source_metadata"]) == set(pins), "exact ten original input metadata receipts")
    for role, bound in (("primary", fetched["primary_commit"]), ("source", fetched["source_metadata"]["source"]["commit"])):
        tree = fetched["producer_trees"][role]
        need(tree["sha"] == bound["commit"]["tree"]["sha"] and tree["truncated"] is False,
             "real complete producer tree " + role)
        expected = producer_files(spec, producer, workflow, role)
        directory = inputs / (role + "_producer")
        need(set(expected) == {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()},
             "all and only complete producer closure including hidden metadata " + role)
        for path, digest in expected.items():
            raw = (directory / path).read_bytes()
            need(sha(raw) == digest and git_blob(raw) == tree_member(tree, path), "actual complete producer file " + path)
    for key, pin in pins.items():
        validate_provenance(pin, fetched["source_metadata"][key])
    common, decision, source, plan = helpers(spec)
    with ExitStack() as stack, tempfile.TemporaryDirectory() as temporary:
        archives = {key: stack.enter_context(original_zip(inputs / (key + ".zip"), pin)) for key, pin in pins.items()}
        pair = archives["pair"]
        receipt = json.loads(pair.read("RESULT.json"))
        validate_pair_receipt(receipt, spec)
        train = json.loads(pair.read("SOURCE.json"))
        pair_meta = fetched["source_metadata"]["pair"]
        need(pair_meta["run"]["event"] == "push" and pair_meta["run"]["head_branch"] == TRAIN_BRANCH
             and str(receipt["run_id"]) == str(spec["inputs"]["pair"]["run_id"])
             and receipt["head"] == spec["inputs"]["pair"]["run_head"]
             and len(pair_meta["commit"]["parents"]) == 1 and receipt["parent"] == pair_meta["commit"]["parents"][0]["sha"]
             and train["parent_pt_sha256"] == B0_PT
             and train["design_sha256"] == spec["training_design_sha256"]
             and train["independent_assessment_sha256"] == spec["assessment_sha256"]
             and train["forbidden_sources_opened"] is train["numerical_B17_replay_claimed"] is False
             and len(train["base_train"]) == 79 and len(train["natural_train"]) == 9,
             "actual TRAIN-only execution identity, all original source rows and preoptimizer registration")
        for name, field in (("COMMON_DRAW.json", "common_draw_sha256"), ("DRAW_COVERAGE.json", "draw_coverage_sha256"),
                            ("TRAIN_MECHANISM.json", "train_mechanism_sha256")):
            need(sha(pair.read(name)) == receipt[field], "actual original paired training evidence " + name)
        for arm in ("control", "challenger"):
            for extension in ("pt", "npz", "onnx"):
                name = STEMS[arm] + "." + extension
                raw = pair.read(name)
                need(receipt[arm]["files"][Path(name).name] == dict(bytes=len(raw), sha256=sha(raw)), "full original trained model " + name)
                if extension == "pt":
                    need(sha(raw) == spec["model_pt_sha256"][arm], "same fixed trained model role " + arm)
        need(sha(archives["B0"].read("models/mero-fsmn-ctc-expanded256.pt")) == B0_PT
             and sha(archives["B0"].read("models/mero-fsmn-ctc-expanded256.npz")) == B0_NPZ
             and sha(archives["B9"].read(common.b9.MODELS["B9_control"]["path"] + ".pt")) == B9_PT,
             "original B0/B9 baseline identities before Torch")
        directory = Path(temporary) / "source"
        directory.mkdir()
        archives["source"].extractall(directory)
        owner = source.evaluation_preflight(directory)
        rows, originals = owner["rows"], owner["all_original_rows"]
        execution, source_meta = owner["source_execution"], fetched["source_metadata"]["source"]
        need(owner["assessment_sha256"] == spec["assessment_sha256"]
             and str(execution["run_id"]) == str(spec["inputs"]["source"]["run_id"])
             and execution["run_head"] == spec["inputs"]["source"]["run_head"]
             and execution["run_attempt"] == 1 and execution["run_branch"] == "refs/heads/" + SOURCE_BRANCH
             and source_meta["run"]["head_branch"] == SOURCE_BRANCH and source_meta["run"]["path"] == SOURCE_WORKFLOW
             and source_meta["run"]["event"] == "push" and len(source_meta["commit"]["parents"]) == 1
             and source_meta["commit"]["parents"][0]["sha"] == execution["prepared_parent"]
             and [entry["filename"] for entry in source_meta["commit"]["files"]] == [SOURCE_WORKFLOW],
             "complete real first public B21 source manifest and workflow-only opening")
        providers = Path(temporary) / "providers"
        for provider in PROVIDERS:
            target = providers / provider
            target.mkdir(parents=True)
            archives["native_" + provider].extractall(target)
        screen = json.loads(archives["screen"].read("SCREEN.json"))
        need(screen["schema"] == SCREEN_SCHEMA and screen["source_execution"] == execution
             and screen["source_manifest_sha256"] == sha(archives["source"].read("MANIFEST.json"))
             and screen["coverage"] == dict.fromkeys(PROVIDERS, len(rows))
             and screen["selected_original_slots"] == 120, "all original native observations and unchanged SCREEN source identity")
        source.verify_screen_contract(directory, screen, providers)
        selected = exact_admitted_join(rows, screen["rows"])
        reserve_people = {person(row) for row in source.source_plan("replication")["rows"]}
        reserve_people.update(str(int(value)) for value in plan["replication"]["natural"]["speakers"])
        reject_overlap(originals, train["base_train"] + train["natural_train"], owner["history"], reserve_people)
        counts = source_counts(selected)
        need(counts == screen["source_coverage"] and screen["frozen_minima"] == GATE
             and all(counts[key] >= minimum for key, minimum in GATE.items())
             and screen["source_gate"] == "SOURCE_GATE_PASS" and not screen["source_shortfall"],
             "source coverage gate before any KWS work")
        need(12 * len(selected) <= MAX_FORWARDS, "complete source population fits preregistered1440 forward ceiling")
    need("torch" not in sys.modules and "numpy" not in sys.modules,
         "all source and original TRAIN admission completes before model/array runtimes are imported")
    chosen = [(row, row["cohort"]) for row in selected]
    admission = dict(schema=SCHEMA, identity=identity, assessment_sha256=spec["assessment_sha256"],
        training_design_sha256=spec["training_design_sha256"], registered_inputs_sha256=sha(SPEC.read_bytes()),
        producer_manifest_sha256=spec["producer_manifest_sha256"], workflow_sha256=sha(workflow),
        source_gate="SOURCE_GATE_PASS", source_shortfall={}, source_coverage=counts,
        original_source_rows=120, observer_eligible_rows=len(rows),
        admitted_rows=[score_identity(row, cohort) for row, cohort in chosen],
        training_source_sha256=sha(json.dumps(train, sort_keys=True, separators=(",", ":")).encode()),
        history_sha256=owner["history_sha256"], model_calls=0, old_development_reused=False,
        replication_opened=False, promotion_allowed=False)
    return spec, identity, common, decision, chosen, train, admission



def snapshots(directory, spec, torch, np, common):
    """Authenticate original B0/B9 and completed B21 checkpoints without any forward."""
    models = {}
    with original_zip(directory / "B0.zip", spec["inputs"]["B0"]) as archive, tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp)
        for name in ["RESULT.json"] + ["models/mero-fsmn-ctc-expanded256." + e for e in ("pt", "npz", "onnx")]:
            dest = target / name
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(archive.read(name))
        model, mel, mean, scale, hashes = common.b0.model(torch, np, target, "B0")
        common.need(hashes["pt"] == B0_PT, "installed B0 identity")
        models["B0"] = (model, mel, mean, scale, hashes["pt"])
    with original_zip(directory / "B9.zip", spec["inputs"]["B9"]):
        prior, _ = common.b9.verify_b9_archive(directory / "B9.zip", torch, np)
    models["B9"] = (*prior["B9_control"], common.b9.MODELS["B9_control"]["pt"])
    common.need(all(np.array_equal(a, b) for a, b in zip(models["B0"][1:4], models["B9"][1:4])),
                "B0/B9 use identical frozen acoustic frontend and normalization")
    with original_zip(directory / "pair.zip", spec["inputs"]["pair"]) as archive:
        receipt = json.loads(archive.read("RESULT.json"))
        validate_pair_receipt(receipt, spec)
        sources = json.loads(archive.read("SOURCE.json"))
        for arm in ("control", "challenger"):
            members = {}
            for ext in ("pt", "npz", "onnx"):
                filename = STEMS[arm] + "." + ext
                raw = archive.read(filename)
                expected = receipt[arm]["files"][Path(filename).name]
                common.need(common.sha(raw) == expected["sha256"] and len(raw) == expected["bytes"],
                            "complete trained B21 artifact member " + filename)
                members[ext] = raw
            weights = torch.load(io.BytesIO(members["pt"]), map_location="cpu", weights_only=True)
            net = common.b9.v5.build_model(torch)
            net.load_state_dict(weights, strict=True)
            net.eval()
            common.need(sum(p.numel() for p in net.parameters()) == 43876 and
                        all(bool(torch.isfinite(p).all()) for p in net.parameters()), "finite actual B21 weights")
            with np.load(io.BytesIO(members["npz"]), allow_pickle=False) as npz:
                common.need(set(npz.files) == set(weights) | {"mel_filters", "train_mean", "train_scale"} and
                            all(np.array_equal(npz[k], t.cpu().numpy()) for k, t in weights.items()),
                            "B21 PT and NPZ learned tensor identity")
                frontend = tuple(npz[k].copy() for k in ("mel_filters", "train_mean", "train_scale"))
            common.need(all(np.array_equal(a, b) for a, b in zip(frontend, models["B0"][1:4])),
                        "candidate retains B0 frozen acoustic frontend")
            models[arm] = (net, *frontend, common.sha(members["pt"]))
    common.need(set(models) == set(common.NAMES), "four explicitly authenticated frozen roles")
    return models, sources

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
        need(sum(self.counts.values()) < MAX_FORWARDS, "hard ceiling before an additional model forward entry")
        self.counts[role] += 1
        self.consumed[key] = "consumed_unknown"
        write(self.directory / (key + ".json"), dict(**self.current, status="consumed_unknown"))

    def saved(self, key):
        need(self.consumed.get(key) == "consumed_unknown", "one returned array per actual forward")
        self.consumed[key] = "saved_array"

def score(inputs, output, prepared):
    spec = protocol()  # Before paths, helper imports and model initialization.
    checked = preflight(inputs, prepared)
    spec, identity, common, decision, chosen, train, admission = checked
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
        training_assessment_sha256=spec["assessment_sha256"], training_design_sha256=spec["training_design_sha256"],
        registered_inputs_sha256=sha(SPEC.read_bytes()), implementation_sha256=sha(Path(__file__).read_bytes()),
        producer_manifest_sha256=spec["producer_manifest_sha256"], workflow_sha256=admission["workflow_sha256"],
        source_gate="SOURCE_GATE_PASS", source_coverage=admission["source_coverage"], source_shortfall={},
        panel_gate="STARTED", records=[], actual_forwards=0, saved_array_count=0, raw_output_count=0,
        optimizer_updates=0, model_pt_sha256=spec["model_pt_sha256"], installed_software_baseline="B0",
        replication_open_allowed=False, B20_failed_primary_preserved=True,
        original_natural_reused_without_reobservation=False, fresh_source_originals=True, ssc305_physical=False,
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
        models, loaded_train = snapshots(inputs, spec, torch, np, common)
        need(loaded_train == train and {k: v[-1] for k, v in models.items()} == spec["model_pt_sha256"],
             "same preflight TRAIN and all four original models/frontends")
        for role, values in models.items():
            def entered(_module, _args, role=role):
                ledger.entered(role)
            hooks.append(values[0].register_forward_pre_hook(entered))
        arrays = zipfile.ZipFile(output / "ALL_FRAME_LOGITS.npz", "x", compression=zipfile.ZIP_DEFLATED)
        with ExitStack() as stack:
            archive = stack.enter_context(original_zip(inputs / "source.zip", spec["inputs"]["source"]))
            for row, cohort in chosen:
                record = score_identity(row, cohort); record["gains"] = {}; report["records"].append(record)
                raw = archive.read("wav/" + row["id"] + ".wav")
                with wave.open(io.BytesIO(raw)) as audio:
                    need(audio.getparams()[:3] == (1, 2, 16000) and audio.getnframes() == row["samples"], "whole admitted PCM")
                    pcm = audio.readframes(audio.getnframes())
                need(sha(raw) == row["wav_sha256"] and sha(pcm) == row["pcm_sha256"], "same preflight original PCM")
                for gain in GAINS:
                    signal = common.b9.ref.gain_pcm(pcm, float(gain), np)
                    record["gains"][gain] = {}
                    for role in ARMS:
                        net, mel, mean, scale, _ = models[role]
                        feature = common.b9.v5.features(signal, row, torch, np, mel)
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
        need(sum(ledger.counts.values()) == report["saved_array_count"] == report["raw_output_count"] == 12 * len(chosen) <= MAX_FORWARDS
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
