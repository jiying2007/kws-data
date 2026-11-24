"""B20 source-only preregistration and complete original exposure readback.

Authentic B19 feasibility is necessary but never grants a source label or a KWS
call. A first public exact-parent event is additionally required to render the
new primary. Every previous replication and the new reserve stay unopened.
"""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import english_v6_b19_source as canary

previous, qa = canary.lineage, canary.qa
PLANS = Path(__file__).with_name("english_v6_b20_panels")
ASSESSMENT = PLANS / "assessment.json"
ASSESSMENT_SHA = "22a238a7c773912bd4bc1429e8b3efbd18147a793a68a798fa064417713afd87"
BASE_ANCESTRY_SHA = previous.BASE_ANCESTRY_SHA
SCHEMA = "english-v6-b20-complete-original-source-history-v1"
REPO = "jiying2007/kws-data"
BRANCH = "refs/heads/research/en-v6-b20-synthetic-primary-once-20261124"
PROOF_FILES = ("B19_RUN.json", "B19_COMMIT.json", "B19_JOBS.json", "B19_ARTIFACTS.json", "B19_CODE.json")
CONTEXT24 = dict(id=14790239936, run_id=45113854481,
    run_head="75d26ca922f393019451e91627dd163d5edace7c", bytes=5320773,
    sha256="321d0a22d7bc0c15fea8181d0bce1544d8f2bce17ca35f551e00544b99c844f7")
CONTEXT24_MANIFEST_SHA = "e7405f712d74284524a5022a6e988877da2f6ab7ccd285e58dc5139f48013e00"
sha, need, canonical = previous.sha, qa.need, previous.canonical


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def draft_plan():
    """Read only prospective local metadata; this grants no execution authority."""
    design = json.loads(ASSESSMENT.read_bytes())
    old, diagnostic = previous.previous.lineage.read_plan(), previous.read_plan()
    need(design["schema"] == "english-v6-b20-frozen-models-new-source-format-assessment-v1"
         and design["B19_assessment_sha256"] == canary.ASSESSMENT_SHA
         and design["training_assessment_sha256"] == old["training_assessment_sha256"]
         and design["training_design_sha256"] == old["training_design_sha256"]
         and design["base_source_ancestry_sha256"] == BASE_ANCESTRY_SHA
         and design["frozen_model_artifacts"] == old["frozen_model_artifacts"]
         and design["frozen_model_pt_sha256"] == old["frozen_model_pt_sha256"]
         and design["reused_natural_primary"] == old["reused_natural_primary"]
         and design["source_gate"] == old["source_gate"] and design["gains"] == old["gains"]
         and design["quality_assessment_sha256"] == old["quality_assessment_sha256"]
         and design["quality_criteria"] == old["quality_criteria"]
         and design["source_format"]["positive_length_scales"] == [1.4, 1.8]
         and design["source_format"]["ordinary_length_scales"] == {"question": 1.0, "request": 1.08}
         and design["maximum_source_calls"] == design["maximum_public_Piper_synthesize_calls"] == 96
         and design["maximum_native_observer_calls"] == 480
         and all(design[k] is False for k in ("new_training_allowed", "threshold_search", "replication_opened",
             "promotion_allowed", "product_released", "B19_canary_source_admission_allowed", "B19_canary_KWS_evaluation_allowed"))
         and all(design[k] is True for k in ("B17_source_stop_retained", "B18_source_stop_retained", "all_prior_replication_sealed"))
         and design["new_natural_source_calls"] == design["new_natural_observer_calls"] == 0,
         "new complete source-format protocol preserves frozen models and old gates")
    for role, key, first, last in (("primary", "future_primary", 798, 822), ("replication", "future_replication", 822, 846)):
        plan = design[role]["synthetic"]
        need(plan["speaker_indices"] == list(range(first, last)) and plan["original_slots"] == 96
             and plan["speaker_id_map"] == diagnostic["source_maps"][key], "previously fixed new canonical source identities")
    reserve = design["replication"]["natural"]
    need(reserve == json.loads((PLANS / "natural-reserve.json").read_bytes())
         and reserve["corpus_split"] == "train-clean-100" and reserve["project_role"] == "development"
         and reserve["training_allowed"] is reserve["opened"] is False
         and reserve["requested_speakers"] == len(set(reserve["speakers"])) == len(reserve["speakers"]) == 9
         and reserve["source_slots"] == 18 and reserve["minimum_approved_speakers"] == 8
         and reserve["minimum_approved_utterances"] == 12,
         "new independent nine-reader reserve stays sealed under the unchanged minima")
    validate_natural_reserve(design)
    return design


def validate_natural_reserve(design):
    """Freeze readers from already saved official metadata; never read audio."""
    reserve = design["replication"]["natural"]
    raw = (PLANS / "SPEAKERS.TXT").read_bytes()
    need(sha(raw) == reserve["metadata_speakers_sha256"]
         == "3350f093382f8baa1d589c9f5a5cee8ec3705e656f521d6eafbaa31a5fcd8f76", "exact official speaker census")
    pool = set()
    for line in raw.decode("utf-8", errors="strict").splitlines():
        if not line.strip() or line.startswith(";"): continue
        fields = [part.strip() for part in line.split("|")]
        if fields[2] == "train-clean-100": pool.add(fields[0])
    _, _, actual_people = previous.registered_history()
    _, old = previous.previous.previous.lineage.read_plan()
    b18 = previous.previous.lineage.read_plan()
    excluded = set(old["natural_excluded_speakers"]) | actual_people
    excluded.update(b18["replication"]["natural"]["speakers"])
    excluded.update(p.removeprefix("p") for p in b18["replication"]["synthetic"]["speaker_id_map"].values())
    excluded.update(design["B19_all_source_exclusions"]["canonical_people"])
    for panel in ("primary", "replication"):
        excluded.update(p.removeprefix("p") for p in design[panel]["synthetic"]["speaker_id_map"].values())
    eligible = pool - excluded
    ranked = sorted(eligible, key=lambda person: (sha((reserve["salt"] + "/speaker/" + person).encode()), person))
    need(len(pool) == 251 and len(eligible) == 239 and reserve["available_census"] == sorted(eligible, key=int)
         and reserve["excluded_canonical_speakers"] == sorted(excluded, key=int) and reserve["speakers"] == ranked[:9]
         and reserve["selected_utterance_ids"] is None
         and reserve["utterance_selection"]["rank_before_audio"] is True
         and reserve["utterance_selection"]["no_replacement_after_quality_rejection"] is True
         and reserve["utterance_selection"]["whole_original_only"] is True
         and reserve["old_dev_other_nine_stay_sealed"] == sorted(b18["replication"]["natural"]["speakers"], key=int)
         and reserve["upstream"]["bytes"] == 6387309499
         and reserve["upstream"]["sha256"] == "d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2"
         and reserve["upstream"]["md5"] == "2a93770f6d5c6c964bc36631d331a522",
         "new fixed reserve readers, untouched old nine, metadata-only whole-utterance rule and exact future archive")


def read_plan():
    need(valid_hash(ASSESSMENT_SHA), "B20_INERT_NO_AUTHENTICATED_B19_PASS_OR_FINAL_ASSESSMENT")
    need(sha(ASSESSMENT.read_bytes()) == ASSESSMENT_SHA, "exact separately frozen B20 assessment")
    design = draft_plan()
    proof = design["B19_original_proof"]
    need(design["execution_enabled"] is True and type(proof["run_id"]) is int
         and re.fullmatch(r"[0-9a-f]{40}", proof["run_head"] or "") is not None
         and re.fullmatch(r"[0-9a-f]{40}", proof["prepared_parent"] or "") is not None
         and valid_hash(proof["feasibility_result_sha256"])
         and set(proof["artifacts"]) == {"source", "observer", "readback"}, "actual successful canary proof required")
    for pin in proof["artifacts"].values():
        need(type(pin) is dict and type(pin["id"]) is int and pin["id"] > 0
             and type(pin["bytes"]) is int and pin["bytes"] > 0 and valid_hash(pin["sha256"])
             and pin["run_id"] == proof["run_id"] and pin["run_head"] == proof["run_head"],
             "no null, invented or mixed original B19 artifact locks")
    exclusions = design["B19_all_source_exclusions"]
    need(exclusions["original_slots"] == len(exclusions["all_PCM_sha256"]) == len(set(exclusions["all_PCM_sha256"])) == 48
         and all(valid_hash(x) for x in exclusions["all_PCM_sha256"])
         and len(exclusions["canonical_people"]) == 8 and design["actual_complete_history_inventory"] is not None
         and design["observer_code_contract"] is not None,
         "all actual canary originals and a complete new study must be frozen before any call")
    return design


def validate_identity(value):
    need(set(value) == {"run_id", "run_head", "run_attempt", "run_branch", "prepared_parent"}
         and isinstance(value["run_id"], str) and re.fullmatch(r"[1-9][0-9]*", value["run_id"]) is not None
         and all(re.fullmatch(r"[0-9a-f]{40}", value[k]) is not None for k in ("run_head", "prepared_parent"))
         and value["run_head"] != value["prepared_parent"] and value["run_branch"] == BRANCH
         and type(value["run_attempt"]) is int and value["run_attempt"] == 1, "one actual first B20 source run")
    return value


def execution_guard(prepared_parent, env=None, event=None, parent=None):
    read_plan()
    e = os.environ if env is None else env
    need(all(e.get(k) == v for k, v in {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPO,
         "GITHUB_REF": BRANCH, "GITHUB_EVENT_NAME": "push", "GITHUB_RUN_ATTEMPT": "1"}.items()),
         "only the separately registered first public B20 source event")
    identity = validate_identity(dict(run_id=e.get("GITHUB_RUN_ID", ""), run_head=e.get("GITHUB_SHA", ""),
        run_attempt=1, run_branch=BRANCH, prepared_parent=prepared_parent))
    if event is None: event = json.loads(Path(e["GITHUB_EVENT_PATH"]).read_bytes())
    if parent is None: parent = subprocess.check_output(["git", "rev-parse", "HEAD^1"], text=True).strip()
    need(event.get("before") == parent == prepared_parent and event.get("after") == identity["run_head"]
         and event.get("ref") == BRANCH and event.get("forced") is event.get("deleted") is False
         and event.get("repository", {}).get("full_name") == REPO and event["repository"].get("private") is False,
         "one reviewed exact predecessor; no replay, force push or multi-commit jump")
    return identity


def proof_locks():
    return read_plan()["B19_original_proof"]["artifacts"]


def required_archives():
    pins = proof_locks()
    return tuple(dict.fromkeys((*previous.required_archives(), str(CONTEXT24["id"]), *(str(p["id"]) for p in pins.values()))))


def validate_producer(directory):
    proof = read_plan()["B19_original_proof"]
    root = Path(directory)
    run, commit, jobs, artifacts, code = [json.loads((root / name).read_bytes()) for name in PROOF_FILES]
    need(run["id"] == proof["run_id"] and run["head_sha"] == proof["run_head"]
         and run["head_branch"] == proof["run_branch"] and run["status"] == "completed"
         and run["conclusion"] == "success" and run["run_attempt"] == 1 and run["event"] == "push"
         and run["repository"]["full_name"] == REPO and run["repository"]["private"] is False,
         "authenticated first public completed B19 run, never a diagnostic workflow shortcut")
    workflow = ".github/workflows/english-v6-b19-source-feasibility-once.yml"
    need(commit["sha"] == proof["run_head"] and len(commit["parents"]) == 1
         and commit["parents"][0]["sha"] == proof["prepared_parent"]
         and [row["filename"] for row in commit["files"]] == [workflow], "B19 exact reviewed producer predecessor")
    need(jobs["total_count"] == len(jobs["jobs"]) == 3
         and {j["name"] for j in jobs["jobs"]} == {"sources", "observer", "readback"}
         and all(j["run_id"] == proof["run_id"] and j["head_sha"] == proof["run_head"]
                 and j["run_attempt"] == 1 and j["status"] == "completed" and j["conclusion"] == "success"
                 for j in jobs["jobs"]), "all original source, native observer and diagnostic readback producers succeeded")
    need(set(artifacts) == set(proof_locks()), "all three original artifacts authenticated")
    for role, pin in proof_locks().items():
        item = artifacts[role]
        need(item["id"] == pin["id"] and item["size_in_bytes"] == pin["bytes"]
             and item["digest"] == "sha256:" + pin["sha256"] and item["expired"] is False
             and item["workflow_run"]["id"] == pin["run_id"] and item["workflow_run"]["head_sha"] == pin["run_head"],
             "original B19 source/native/readback artifact metadata")
    need(code == proof["producer_file_sha256"], "exact original B19 code and parent-only workflow change")


def verify_b19(directory):
    """Authenticate original archive bytes and recompute feasibility, without models."""
    design = read_plan()
    validate_producer(directory)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for role, pin in proof_locks().items():
            with previous.previous.lineage.opened(directory, pin) as archive:
                previous.previous.lineage.extract_original(archive, root / role)
        rows = canary.read_source(root / "source")
        reg = canary.registration(root / "source")
        proof = design["B19_original_proof"]
        need(reg["run_id"] == str(proof["run_id"]) and reg["run_head"] == proof["run_head"]
             and reg["prepared_parent"] == proof["prepared_parent"], "actual B19 source generation matches its artifact run")
        result = canary.observer.readback(root / "source", root / "observer", canary.read_source, reg)
        raw = (root / "readback/FEASIBILITY.json").read_bytes()
        need(sha(raw) == proof["feasibility_result_sha256"] and json.loads(raw) == result
             and result["prospective_pair_feasible"] is result["diagnostic_complete"] is True
             and result["actual_native_attempts"] == result["native_returns"] == 48
             and all(result["arms"][arm]["whole_audio_bounds_ok"] is True for arm in ("baseline", "prospective", "ordinary"))
             and all(result["arms"][arm]["valid_nonempty"] == 16 for arm in ("prospective", "ordinary"))
             and result["source_admission_eligible"] is result["source_gate_passed"] is result["KWS_evaluation_allowed"] is False
             and result["training_allowed"] is result["promotion_allowed"] is result["replication_opened"] is False,
             "48 real returns and fixed paired-format feasibility; no diagnostic source label or KWS authority")
        actual = (set(row["pcm_sha256"] for row in rows), set(row["source_group_id"] for row in rows),
                  set(row["source_speaker_id"].removeprefix("p") for row in rows))
        expected = design["B19_all_source_exclusions"]
        need(len(rows) == len(actual[0]) == 48 and tuple(map(len, actual)) == (48, 8, 8)
             and all(values == set(expected[key]) for values, key in zip(actual,
                 ("all_PCM_sha256", "source_groups", "canonical_people"), strict=True)),
             "all original canary slots and planned readers permanently excluded, including invalid baseline empties")
        return actual


def context24_increment(directory):
    design = read_plan()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "context24"
        with previous.previous.lineage.opened(directory, CONTEXT24) as archive:
            previous.previous.lineage.extract_original(archive, root)
        raw = (root / "MANIFEST.json").read_bytes()
        need(sha(raw) == CONTEXT24_MANIFEST_SHA, "original full context24 manifest")
        meta = json.loads(raw); rows = meta["rows"]
        need(meta["role"] == "train" and len(rows) == len({r["id"] for r in rows}) == 24
             and len({r["speaker"] for r in rows}) == 12 and meta["raw_audio_decodes"] == 24
             and meta["rejections"] == [], "all previously exposed context24 originals, never just admitted or reviewed rows")
        for row in rows:
            measured = qa.pcm_tools.inspect_wav(root / "wav" / (row["id"] + ".wav"))
            need(all(row[k] == value for k, value in measured.items())
                 and qa.sha(root / "flac" / (row["id"] + ".flac")) == row["flac_sha256"]
                 and row["full_original_audio"] is True and row["source_group_id"] == "librispeech-" + row["speaker"],
                 "retained full context24 WAV/PCM/native FLAC, including all UNKNOWN originals")
        actual = ({r["pcm_sha256"] for r in rows}, {r["source_group_id"] for r in rows}, {r["speaker"] for r in rows})
        supplement = design["historical_context24_supplement"]
        need(supplement["source_artifact"] == CONTEXT24 and supplement["manifest_sha256"] == CONTEXT24_MANIFEST_SHA
             and all(values == set(supplement[key]) for values, key in zip(actual,
                 ("all_pcm_sha256", "all_source_groups", "canonical_people"), strict=True)),
             "all raw context24 identities match the separately recorded historical supplement")
        return actual


def validate_sets(values):
    design = read_plan()
    expected = design["actual_complete_history_inventory"]
    for key, valueset in zip(("all_pcm_sha256", "all_source_groups", "canonical_people"), values, strict=True):
        need(len(valueset) == expected[key]["count"]
             and sha(canonical(sorted(valueset))) == expected[key]["sorted_canonical_json_sha256"],
             "complete B20 original ancestry set: " + key)
    return values


def history(directory):
    read_plan()
    base = previous.history(Path(directory))
    context = context24_increment(directory)
    need(not base[0] & context[0] and not base[1] & context[1] and context[2] <= base[2],
         "explicit context24 full-PCM/group supplement; older frozen history remains unchanged")
    new = verify_b19(directory)
    combined = tuple(base[i] | context[i] for i in range(3))
    need(all(not combined[i] & new[i] for i in range(3)), "B19 all48 originals and eight people were source-independent")
    return validate_sets(tuple(combined[i] | new[i] for i in range(3)))


def history_receipt(directory):
    seen, groups, people = history(directory)
    archives = {key: dict(bytes=(Path(directory) / (key + ".zip")).stat().st_size,
                          sha256=qa.sha(Path(directory) / (key + ".zip"))) for key in required_archives()}
    return dict(schema=SCHEMA, assessment_sha256=ASSESSMENT_SHA, original_archives=archives,
        all_pcm_sha256=sorted(seen), all_source_groups=sorted(groups), canonical_people=sorted(people),
        B19_complete_canary_excluded=True, full_context24_originals_added=True,
        B17_B18_source_STOP_preserved=True, all_old_replication_sealed=True)


def validate_history(receipt):
    read_plan()
    need(receipt["schema"] == SCHEMA and receipt["assessment_sha256"] == ASSESSMENT_SHA
         and set(receipt["original_archives"]) == set(required_archives())
         and all(receipt[k] is True for k in ("B19_complete_canary_excluded", "full_context24_originals_added",
                  "B17_B18_source_STOP_preserved", "all_old_replication_sealed")), "all old44, context24 and B19 original archives")
    old = previous.previous.lineage.read_plan()
    known = dict(old["base_original_archive_sha256"])
    known.update({str(p["id"]): p["sha256"] for p in previous.previous.lineage.proof_locks().values()})
    known.update({str(p["id"]): p["sha256"] for p in previous.proof_locks().values()})
    extra = [CONTEXT24, *proof_locks().values()]
    known.update({str(p["id"]): p["sha256"] for p in extra})
    need(set(known) == set(required_archives()), "no missing historical original archive pin")
    for key, digest in known.items():
        value = receipt["original_archives"][key]
        need(value["sha256"] == digest and type(value["bytes"]) is int and value["bytes"] > 0,
             "exact historical ZIP SHA and positive archive length")
    for pin in extra:
        need(receipt["original_archives"][str(pin["id"])]["bytes"] == pin["bytes"], "exact original context/canary ZIP length")
    values = tuple(set(receipt[key]) for key in ("all_pcm_sha256", "all_source_groups", "canonical_people"))
    need(all(len(value) == len(receipt[key]) for value, key in zip(values,
             ("all_pcm_sha256", "all_source_groups", "canonical_people"), strict=True)), "no duplicated ancestry inventory")
    return validate_sets(values)


def gh(endpoint):
    return subprocess.check_output(["gh", "api", "repos/" + REPO + "/" + endpoint])


def authorize(out, prepared_parent):
    identity = execution_guard(prepared_parent)  # Closed before all remote metadata access.
    proof = read_plan()["B19_original_proof"]
    need(out.is_absolute() and not out.exists() and out.parent.is_dir(), "fresh source authorization directory")
    run = json.loads(gh("actions/runs/" + str(proof["run_id"])))
    commit = json.loads(gh("commits/" + proof["run_head"]))
    jobs = json.loads(gh("actions/runs/" + str(proof["run_id"]) + "/attempts/1/jobs?per_page=100"))
    artifacts = {role: json.loads(gh("actions/artifacts/" + str(pin["id"]))) for role, pin in proof_locks().items()}
    code = {}
    for filename, digest in proof["producer_file_sha256"].items():
        item = json.loads(gh("contents/" + filename + "?ref=" + proof["run_head"]))
        need(item["encoding"] == "base64" and item["type"] == "file" and item["path"] == filename, "original producer file")
        raw = base64.b64decode(item["content"])
        if filename.endswith("english-v6-b19-source-feasibility-once.yml"):
            need(raw.count(proof["prepared_parent"].encode()) == 2, "only two original B19 parent placeholders")
            raw = raw.replace(proof["prepared_parent"].encode(), b"__B19_SOURCE_FEASIBILITY_MAIN_SHA__")
        need(sha(raw) == digest, "exact reviewed original B19 producer code")
        code[filename] = sha(raw)
    out.mkdir()
    for filename, value in zip(PROOF_FILES, (run, commit, jobs, artifacts, code), strict=True):
        qa.record(out / filename, value)
    validate_producer(out)
    for pin in proof_locks().values():
        raw = gh("actions/artifacts/" + str(pin["id"]) + "/zip")
        need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "original B19 artifact body")
        (out / (str(pin["id"]) + ".zip")).write_bytes(raw)
    verify_b19(out)
    qa.record(out / "AUTHORIZATION.json", dict(schema=SCHEMA, assessment_sha256=ASSESSMENT_SHA,
        source_execution=identity, B19_original_artifacts=proof_locks(),
        B19_feasibility_result_sha256=proof["feasibility_result_sha256"], B19_format_feasibility_only=True,
        B19_producer_file_sha256={name: qa.sha(out / name) for name in PROOF_FILES},
        B19_canary_admission_allowed=False, old_replication_opened=False, new_KWS_forwards=0))


def require_authorization(directory):
    design = read_plan()  # No source/old proof file is opened while inert.
    directory = Path(directory)
    receipt = json.loads((directory / "AUTHORIZATION.json").read_bytes())
    validate_identity(receipt["source_execution"])
    need(receipt["schema"] == SCHEMA and receipt["assessment_sha256"] == ASSESSMENT_SHA
         and receipt["B19_original_artifacts"] == proof_locks()
         and receipt["B19_feasibility_result_sha256"] == design["B19_original_proof"]["feasibility_result_sha256"]
         and receipt["B19_format_feasibility_only"] is True and receipt["B19_canary_admission_allowed"] is False
         and receipt["old_replication_opened"] is False and receipt["new_KWS_forwards"] == 0,
         "same original canary prerequisite authorizes only this new full source protocol")
    proof = directory / "B19_PROOF" if (directory / "B19_PROOF").is_dir() else directory
    need(receipt["B19_producer_file_sha256"] == {name: qa.sha(proof / name) for name in PROOF_FILES},
         "original producer receipt byte closure")
    verify_b19(proof)
    return receipt


def retain_proof(directory, output):
    read_plan()
    output.mkdir()
    for name in (*PROOF_FILES, *(str(pin["id"]) + ".zip" for pin in proof_locks().values())):
        shutil.copyfile(Path(directory) / name, output / name)
