"""Offline, metadata-only binding of the preregistered B21 natural amendment.

The rule was frozen before the metadata diagnostic. This module never reads
audio, imports scientific runtimes, invokes a model, downloads data, or opens
either source panel. A source executor must use the committed, verified roster
and obtain its own separately registered execution authorization.
"""
from __future__ import annotations
import argparse
import ast
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import types
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RULE = "automation/english_v6_b21_natural_amendment_registration.json"
RULE_SHA = "d680ac1b4491efbbd47222b527898236b6b171dc0bf2827c2082484f67882caf"
INPUTS = "automation/english_v6_b21_natural_amendment_inputs.json"
EVIDENCE = "automation/english_v6_b21_natural_amendment_evidence"
SCHEMA = "english-v6-b21-natural-amendment-binding-v1"
BUNDLE_SCHEMA = "english-v6-b21-natural-amendment-original-metadata-inputs-v1"
REPO = "jiying2007/kws-data"
METADATA_BRANCH = "research/en-v6-b21-natural-metadata-once-20261124"
METADATA_WORKFLOW = ".github/workflows/english-v6-b21-natural-metadata-once.yml"
METADATA_TEMPLATE = "automation/english_v6_b21_natural_metadata_workflow_template.yml"
METADATA_TOKEN = "__B21_NATURAL_METADATA_MAIN_SHA__"
METADATA_CODE = "automation/english_v6_b21_natural_metadata.py"
METADATA_CODE_SHA = "56808478c13335fcbd2a696d4c63b68b7f8814baf494b42854731cef22a8883d"
METADATA_TEST = "automation/test_english_v6_b21_natural_metadata.py"
METADATA_TEST_SHA = "95d4a69b8a537b176a8228ef70fbb15a5f750ccca897f7221f5db075558face3"
METADATA_TEMPLATE_SHA = "2bbfd312b33e7597d3e852777f2928d85f0f0b3c685555687e82111309b523a0"
ZERO = dict(new_source_calls=0, natural_decodes=0, audio_members_extracted=0,
    audio_headers_interpreted=0, observer_calls=0, kws_forwards=0,
    optimizer_updates=0, replacement_reader_draws=0, source_panel_openings=0)
TRANSCRIPT = re.compile(r"LibriSpeech/train-clean-100/(\d+)/(\d+)/(\d+)-(\d+)\.trans\.txt")
FROZEN_PATHS = {
    "assessment_sha256": "automation/english_v6_b21_panels/assessment.json",
    "ancestry_sha256": "automation/english_v6_b21_panels/ancestry.json",
    "primary_natural_sha256": "automation/english_v6_b21_panels/natural-primary.json",
    "replication_natural_sha256": "automation/english_v6_b21_panels/natural-replication.json",
    "primary_natural_code_sha256": "automation/english_v6_b21_natural_sources.py",
    "replication_natural_code_sha256": "automation/english_v6_b21_replication_natural_sources.py",
    "speakers_sha256": "automation/english_v6_b21_panels/SPEAKERS.TXT",
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def person(value):
    need(type(value) is str and re.fullmatch(r"[0-9]+", value) is not None,
         "canonical numeric person identity")
    return str(int(value))


def load_rule(root=None):
    root = ROOT if root is None else Path(root)
    raw = (root / RULE).read_bytes()
    need(digest(raw) == RULE_SHA, "unchanged rule registered before the metadata result")
    rule = json.loads(raw)
    for field, path in FROZEN_PATHS.items():
        need(digest((root / path).read_bytes()) == rule["frozen_inputs"][field],
             "unchanged original input: " + path)
    assessment = read_json(root / FROZEN_PATHS["assessment_sha256"])
    ancestry = read_json(root / FROZEN_PATHS["ancestry_sha256"])
    need(rule["catalog_binding"]["bound"] is False
         and rule["panels"]["primary"]["selected_speakers"] is None
         and rule["panels"]["replication"]["selected_speakers"] is None
         and not any(rule[k] for k in ("execution_authorized", "source_panels_opened",
                                      "training_allowed", "threshold_search", "promotion_allowed")),
         "prospective rule is immutable and cannot itself authorize execution")
    for role in ("primary", "replication"):
        design = read_json(root / FROZEN_PATHS[role + "_natural_sha256"])
        need(design == assessment[role]["natural"]
             and design["salt"] == rule["panels"][role]["salt"], "original role-specific salt")
    return rule, assessment, ancestry


def frozen_rank(panel, root=None):
    """Compile only the original, hash-checked pure rank_metadata function."""
    root = ROOT if root is None else Path(root)
    rule, _, _ = load_rule(root)
    need(panel in ("primary", "replication"), "registered panel only")
    path = root / FROZEN_PATHS[panel + "_natural_code_sha256"]
    tree = ast.parse(path.read_bytes())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "rank_metadata"]
    need(len(nodes) == 1, "one original pure metadata rank function")
    namespace = dict(qa=types.SimpleNamespace(need=need), digest=digest,
        CORPUS_SPLIT="train-clean-100", defaultdict=defaultdict, re=re,
        BLOCK=set(rule["text_predicate"]["blocked_exact_tokens"]))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["rank_metadata"]


def predicate(words):
    """Exactly the original primary and replication lexical predicate."""
    BLOCK = set("HEY HAY MERO MEERO MIRO MARROW MERROW".split())
    return (4 <= len(words) <= 14 and not BLOCK.intersection(words)
            and all(re.fullmatch(r"[A-Z]+(?:'[A-Z]+)*", word) for word in words))


def validate_catalog(catalog, rule):
    need(catalog["schema"] == rule["catalog_schema"]
         and catalog["corpus_bytes"] == rule["corpus"]["bytes"]
         and catalog["corpus_sha256"] == rule["corpus"]["sha256"]
         and catalog["speakers_sha256"] == rule["frozen_inputs"]["speakers_sha256"]
         and digest(catalog["speakers_text"].encode("utf-8")) == catalog["speakers_sha256"]
         and catalog["metadata_only"] is True
         and all(type(catalog.get(k)) is int and catalog[k] == v for k, v in ZERO.items()),
         "complete original corpus text metadata with zero audio or scientific calls")
    census = set()
    for line in catalog["speakers_text"].splitlines():
        if not line.strip() or line.startswith(";"):
            continue
        fields = [part.strip() for part in line.split("|")]
        need(len(fields) >= 3, "original census fields")
        if fields[2] == "train-clean-100":
            need(person(fields[0]) == fields[0] and fields[0] not in census,
                 "unique canonical census people")
            census.add(fields[0])
    need(type(catalog["transcripts"]) is list and catalog["transcripts"], "full transcript catalog")
    paths = [item["path"] for item in catalog["transcripts"]]
    need(paths == sorted(set(paths)), "unique canonically ordered transcript members")
    eligible, all_people, seen = defaultdict(list), set(), set()
    text_bytes = len(catalog["speakers_text"].encode("utf-8"))
    for item in catalog["transcripts"]:
        match = TRANSCRIPT.fullmatch(item["path"])
        raw = item["text"].encode("utf-8")
        need(match is not None and type(item["bytes"]) is int
             and len(raw) == item["bytes"] and digest(raw) == item["sha256"],
             "exact original text bytes and canonical transcript member")
        speaker, chapter, sp2, ch2 = match.groups()
        need((speaker, chapter) == (sp2, ch2) and speaker in census,
             "original corpus speaker/chapter identity")
        all_people.add(speaker)
        text_bytes += len(raw)
        for line in item["text"].splitlines():
            parts = line.split(" ", 1)
            need(len(parts) == 2, "original identifier and transcript")
            ident, text = parts
            need(re.fullmatch(speaker + "-" + chapter + r"-\d+", ident) is not None
                 and ident not in seen, "globally unique original utterance")
            seen.add(ident)
            if predicate(text.split(" ")):
                eligible[speaker].append(dict(speaker=speaker, chapter=chapter,
                    source_id=ident, transcript=text, transcript_file_sha256=item["sha256"],
                    transcript_member=item["path"]))
    need(all_people == census and catalog["metadata_bytes"] == text_bytes,
         "all official corpus readers and exact complete metadata byte count")
    return census, eligible


def excluded_people(rule, assessment, ancestry):
    people = set()
    for values in (ancestry["natural_excluded_people"], ancestry["canonical_people"],
                   assessment["primary"]["natural"]["excluded_canonical_speakers"],
                   assessment["replication"]["natural"]["excluded_canonical_speakers"],
                   rule["base_exclusion_rule"]["original_failed_primary12"],
                   rule["base_exclusion_rule"]["original_sealed_replication12"]):
        people.update(person(value) for value in values)
    for role in ("primary", "replication"):
        people.update(person(value.removeprefix("p"))
                      for value in assessment[role]["synthetic"]["speaker_id_map"].values())
    return people


def select_catalog(catalog, root=None):
    """Pure metadata recomputation; no roster is opened, replaced or executed."""
    root = ROOT if root is None else Path(root)
    rule, assessment, ancestry = load_rule(root)
    census, eligible = validate_catalog(catalog, rule)
    excluded = excluded_people(rule, assessment, ancestry)
    selected, result = set(), {}
    for role in rule["selection_sequence"]:
        blocked = excluded | selected
        available = census - blocked
        pool = {speaker for speaker in available if len(eligible[speaker]) >= 2}
        salt = rule["panels"][role]["salt"]
        ranked = sorted(pool, key=lambda speaker:
                        (digest((salt + "/speaker/" + speaker).encode()), speaker))
        need(len(ranked) >= 12, "insufficient eligible people; STOP without partial panel or new salt")
        speakers = ranked[:12]
        design = dict(assessment[role]["natural"], speakers=speakers,
            excluded_canonical_speakers=sorted(blocked, key=int),
            available_census=sorted(available, key=int),
            eligible_census=sorted(pool, key=int), opened=False)
        original_catalog = dict(
            schema="english-v6-b21-" + role + "-natural-original-metadata-v1",
            speakers_text=catalog["speakers_text"],
            transcripts=[dict(path=item["path"], sha256=item["sha256"], text=item["text"])
                         for item in catalog["transcripts"]
                         if TRANSCRIPT.fullmatch(item["path"]).group(1) in speakers])
        roster = frozen_rank(role, root)(original_catalog, design)
        need(len(roster) == len({r["source_id"] for r in roster}) == 24
             and [r["speaker"] for r in roster] == [sp for sp in speakers for _ in range(2)],
             "original unchanged rank function yields precisely twelve reader pairs")
        design["selected_utterance_ids"] = [r["source_id"] for r in roster]
        result[role] = dict(design=design, roster=roster, source_opened=False,
            ranked_eligible_people=ranked, metadata_only=True)
        selected.update(speakers)
    need(len(selected) == 24 and not selected & excluded, "new independent primary and sealed reserve")
    return dict(schema="english-v6-b21-natural-amendment-exact-selection-v1",
        rule_sha256=RULE_SHA, base_excluded_people=sorted(excluded, key=int),
        corpus_census_people=len(census),
        eligible_transcript_counts={speaker: len(eligible[speaker])
                                    for speaker in sorted(census, key=int)},
        panels=result, execution_authorized=False, source_panels_opened=False,
        training_allowed=False, promotion_allowed=False, new_source_calls=0)


def read_zip(raw):
    archive = zipfile.ZipFile(io.BytesIO(raw))
    names = archive.namelist()
    need(len(names) == len(set(names)) and archive.testzip() is None,
         "original archive CRC and unique members")
    for name in names:
        path = Path(name)
        need(not path.is_absolute() and ".." not in path.parts
             and not archive.getinfo(name).is_dir(), "safe ordinary original archive member")
    return archive


def authenticate_originals(bundle, root=None):
    """Read saved original ZIPs and GitHub API records; never contact a service."""
    root = ROOT if root is None else Path(root)
    rule, _, _ = load_rule(root)
    need(bundle["schema"] == BUNDLE_SCHEMA
         and set(bundle["archives"]) == {"metadata", "producer", "consumed"}
         and set(bundle["api"]) == {"run", "artifacts", "commit", "tree"},
         "complete explicit original metadata binding bundle")
    apis = {key: read_json(path) for key, path in bundle["api"].items()}
    run, commit, tree = apis["run"], apis["commit"], apis["tree"]
    head = run["head_sha"]
    need(run["event"] == "push" and run["run_attempt"] == 1
         and run["status"] == "completed" and run["conclusion"] == "success"
         and run["head_branch"] == METADATA_BRANCH and run["path"] == METADATA_WORKFLOW
         and run["repository"]["full_name"] == REPO and run["repository"]["private"] is False
         and run["head_repository"]["full_name"] == REPO
         and run["head_repository"]["private"] is False,
         "actual successful first public metadata-only execution")
    need(commit["sha"] == head and len(commit["parents"]) == 1
         and [entry["filename"] for entry in commit["files"]] == [METADATA_WORKFLOW]
         and tree["sha"] == commit["commit"]["tree"]["sha"] and tree["truncated"] is False,
         "original workflow-only activation and complete exact Git tree")
    parent = commit["parents"][0]["sha"]
    execution = dict(run_id=str(run["id"]), run_head=head, run_attempt=1,
        run_branch="refs/heads/" + METADATA_BRANCH, prepared_parent=parent)
    old = rule["original_failed_source"]
    old_pin = old["source_artifact"]
    failure = dict(run_id=old["run_id"], run_head=old["run_head"],
        source_artifact_id=old_pin["id"], source_artifact_bytes=old_pin["bytes"],
        source_artifact_sha256=old_pin["sha256"])
    entries = {item["path"]: item for item in tree["tree"]}
    need(entries[RULE]["type"] == "blob" and entries[RULE]["sha"] == blob((root / RULE).read_bytes()),
         "this exact prospective rule existed in the metadata execution Git tree")
    artifacts = apis["artifacts"]["artifacts"]
    need(apis["artifacts"]["total_count"] == len(artifacts), "complete original artifact list")
    names = {"metadata": "english-v6-b21-natural-metadata-" + str(run["id"]),
             "producer": "english-v6-b21-natural-metadata-producer-" + str(run["id"]),
             "consumed": "english-v6-b21-natural-metadata-consumed"}
    pins, archives = {}, {}
    for role, name in names.items():
        hits = [a for a in artifacts if a["name"] == name]
        need(len(hits) == 1, "one original artifact for " + role)
        artifact = hits[0]
        raw = Path(bundle["archives"][role]).read_bytes()
        need(artifact["expired"] is False and artifact["workflow_run"]["id"] == run["id"]
             and artifact["workflow_run"]["head_sha"] == head
             and artifact["size_in_bytes"] == len(raw)
             and artifact["digest"] == "sha256:" + digest(raw),
             "original artifact ZIP identity: " + role)
        pins[role] = dict(id=artifact["id"], run_id=run["id"], run_head=head,
                         bytes=len(raw), sha256=digest(raw))
        archives[role] = read_zip(raw)
    try:
        producer = archives["producer"]
        proof = json.loads(producer.read("PRODUCER.json"))
        need(proof["schema"] == "english-v6-b21-natural-metadata-producer-v1"
             and proof["execution"] == execution
             and proof["preregistered_amendment_rule_sha256"] == RULE_SHA
             and proof["failed_source"] == failure
             and proof["original_failed_source_status"] == "STOP"
             and all(type(proof.get(k)) is int and proof[k] == v for k, v in ZERO.items()),
             "exact original metadata-only producer")
        paths = [f["path"] for f in proof["files"]]
        frozen = {path: rule["frozen_inputs"][field] for field, path in FROZEN_PATHS.items()}
        frozen.update({
            RULE: RULE_SHA,
            "automation/english_v6_b21_fixed_mirror_transport.py":
                "eca61c7239562aac0fdb08ef5bf93dec93a017dcf771f78e19b857c0b4d35db4",
            "automation/english_v6_b21_fixed_mirror_transport.json":
                "82f915a9ea4355c545dc3c8aabeeca25460896021a4303dfc1f3280597cd3850"})
        closure = set(frozen) | {METADATA_CODE, METADATA_TEST, METADATA_TEMPLATE, METADATA_WORKFLOW}
        need(paths == sorted(closure) and proof["frozen_inputs"] == frozen
             and len(closure) == 14, "exact reviewed fourteen-file metadata producer closure")
        need(set(producer.namelist()) == {"PRODUCER.json"} | {"repository/" + p for p in paths},
             "no unaccounted producer members")
        for item in proof["files"]:
            raw = producer.read("repository/" + item["path"])
            need(item["bytes"] == len(raw) and item["sha256"] == digest(raw)
                 and item["git_blob_sha1"] == blob(raw)
                 and entries[item["path"]]["type"] == "blob"
                 and entries[item["path"]]["sha"] == blob(raw),
                 "original producer bytes match exact Git tree: " + item["path"])
        need(digest(producer.read("repository/" + METADATA_CODE)) == METADATA_CODE_SHA
             and digest(producer.read("repository/" + METADATA_TEST)) == METADATA_TEST_SHA
             and digest(producer.read("repository/" + METADATA_TEMPLATE)) == METADATA_TEMPLATE_SHA
             and producer.read("repository/" + RULE) == (root / RULE).read_bytes(),
             "exact reviewed metadata diagnostic and prospectively frozen rule")
        template = producer.read("repository/" + METADATA_TEMPLATE)
        need(producer.read("repository/" + METADATA_WORKFLOW) ==
             template.decode().replace(METADATA_TOKEN, parent).encode(),
             "only original reviewed predecessor fills the activation token")
        for path, expected in proof["frozen_inputs"].items():
            need(digest(producer.read("repository/" + path)) == expected,
                 "every captured immutable metadata input")
        metadata = archives["metadata"]
        result = json.loads(metadata.read("RESULT.json"))
        catalog_raw = metadata.read("FULL_METADATA_CATALOG.json")
        diagnostic_raw = metadata.read("ORIGINAL_READER_DIAGNOSTIC.json")
        authorization = json.loads(metadata.read("AUTHORIZATION.json"))
        consumed = json.loads(archives["consumed"].read("CONSUMPTION.json"))
        need(result["schema"] == "english-v6-b21-natural-metadata-execution-v1"
             and result["status"] == "METADATA_DIAGNOSTIC_COMPLETE"
             and result["execution"] == execution
             and result["failed_source"] == failure
             and result["catalog_sha256"] == digest(catalog_raw)
             and result["diagnostic_sha256"] == digest(diagnostic_raw)
             and result["original_failed_source_status"] == "STOP"
             and result["metadata_only"] is True
             and all(type(result.get(k)) is int and result[k] == v for k, v in ZERO.items()),
             "complete original catalog result with zero scientific calls")
        need(authorization["execution"] == execution
             and authorization["failed_source"] == failure
             and authorization["producer_sha256"] == digest(producer.read("PRODUCER.json"))
             and authorization["metadata_only"] is authorization["source_panels_remain_closed"] is True
             and authorization["maximum_whole_corpus_transfers"] == 1
             and authorization["maximum_ranges"] == 8 and authorization["range_retries"] == 0
             and all(type(authorization.get(k)) is int and authorization[k] == v for k, v in ZERO.items())
             and consumed == dict(authorization, status="METADATA_TRANSPORT_CONSUMED_UNKNOWN",
                                  audio_source_budget_authorized=False),
             "durable metadata-only consumption before original transport")
        transport = json.loads(metadata.read("CORPUS_TRANSPORT.json"))
        need(transport["status"] == "COMPLETE_VERIFIED"
             and transport["bytes"] == rule["corpus"]["bytes"]
             and transport["sha256"] == rule["corpus"]["sha256"]
             and transport["range_retries"] == 0
             and sorted(r["index"] for r in transport["ranges"]) == list(range(8))
             and transport == json.loads(metadata.read("TRANSPORT/RESULT.json")),
             "one complete original corpus transport before text parsing")
        diagnostic = json.loads(diagnostic_raw)
        need(diagnostic["schema"] == "english-v6-b21-original-natural-reader-diagnostic-v1"
             and diagnostic["failed_source"] == failure
             and diagnostic["original_failed_source_status"] == "STOP"
             and diagnostic["audio_or_source_continuation_authorized"] is False
             and diagnostic["source_panels_remain_closed"] is True
             and all(type(diagnostic.get(k)) is int and diagnostic[k] == v for k, v in ZERO.items()),
             "retrospective diagnostic leaves both original panels closed")
        need(diagnostic["panels"]["primary"]["original_readers"] ==
             rule["base_exclusion_rule"]["original_failed_primary12"]
             and diagnostic["panels"]["replication"]["original_readers"] ==
             rule["base_exclusion_rule"]["original_sealed_replication12"],
             "whole original failed and sealed panels remain exposed in evidence")
        catalog = json.loads(catalog_raw)
        selection = select_catalog(catalog, root)
        return dict(execution=execution, pins=pins, catalog_sha256=digest(catalog_raw),
            diagnostic_sha256=digest(diagnostic_raw), selection=selection,
            selection_sha256=digest(encode(selection)))
    finally:
        for archive in archives.values():
            archive.close()


def inert_inputs():
    return dict(schema=SCHEMA, rule_sha256=RULE_SHA, bound=False,
        metadata_artifact_pin=None, metadata_producer_pin=None, metadata_consumed_pin=None,
        metadata_execution=None, catalog_sha256=None, diagnostic_sha256=None,
        selection_sha256=None, selection=None, evidence_files=None,
        execution_authorized=False, source_panels_opened=False, training_allowed=False,
        promotion_allowed=False, live_metadata_refresh_required=True)


def bound_document(verified, evidence_files):
    return dict(inert_inputs(), bound=True, metadata_artifact_pin=verified["pins"]["metadata"],
        metadata_producer_pin=verified["pins"]["producer"],
        metadata_consumed_pin=verified["pins"]["consumed"],
        metadata_execution=verified["execution"], catalog_sha256=verified["catalog_sha256"],
        diagnostic_sha256=verified["diagnostic_sha256"],
        selection_sha256=verified["selection_sha256"], selection=verified["selection"],
        evidence_files=evidence_files)


def evidence_bundle(root):
    directory = Path(root) / EVIDENCE
    return dict(schema=BUNDLE_SCHEMA,
        archives={role: str(directory / (role + "-original.zip"))
                  for role in ("metadata", "producer", "consumed")},
        api={role: str(directory / "api" / (role + ".json"))
             for role in ("run", "artifacts", "commit", "tree")})


def verify_registration(root=None):
    root = ROOT if root is None else Path(root)
    rule, _, _ = load_rule(root)
    inputs = read_json(root / INPUTS)
    need(type(inputs.get("bound")) is bool, "explicit closed or bound metadata registration")
    if inputs["bound"]:
        files = inputs["evidence_files"]
        need(type(files) is dict and files, "complete original metadata evidence")
        expected = {EVIDENCE + "/" + role + "-original.zip"
                    for role in ("metadata", "producer", "consumed")}
        expected.update(EVIDENCE + "/api/" + role + ".json"
                        for role in ("run", "artifacts", "commit", "tree"))
        need(set(files) == expected, "exact original ZIP/API evidence closure")
        for name, pin in files.items():
            path = root / name
            raw = path.read_bytes()
            need(not path.is_symlink() and pin == dict(bytes=len(raw), sha256=digest(raw)),
                 "unchanged committed original metadata evidence")
        verified = authenticate_originals(evidence_bundle(root), root)
        need(inputs == bound_document(verified, files),
             "committed exact selection equals original catalog and preregistered rule")
    else:
        need(inputs == inert_inputs(), "all future pins, IDs and gates remain inert")
    selection = inputs["selection"]
    return dict(rule=rule, rule_sha256=RULE_SHA, bound=inputs["bound"],
        primary_design=None if selection is None else selection["panels"]["primary"]["design"],
        replication_design=None if selection is None else selection["panels"]["replication"]["design"],
        primary_roster=None if selection is None else selection["panels"]["primary"]["roster"],
        replication_roster=None if selection is None else selection["panels"]["replication"]["roster"],
        metadata_artifact_pin=inputs["metadata_artifact_pin"],
        metadata_producer_pin=inputs["metadata_producer_pin"],
        metadata_consumed_pin=inputs["metadata_consumed_pin"],
        metadata_execution=inputs["metadata_execution"], catalog_sha256=inputs["catalog_sha256"],
        descriptor_sha256=digest((root / INPUTS).read_bytes()),
        source_panels_opened=False, execution_authorized=False,
        live_metadata_refresh_required=True)


def resource_paths(root=None):
    root = ROOT if root is None else Path(root)
    verify_registration(root)
    inputs = read_json(root / INPUTS)
    return sorted({RULE, INPUTS} | set(inputs.get("evidence_files") or {}))


def bind(root, bundle_path, output):
    root, output = Path(root), Path(output)
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(),
         "fresh absolute metadata-binding output, never overwrite registration")
    need(read_json(root / INPUTS) == inert_inputs(), "bind only the still-inert registration")
    bundle = read_json(bundle_path)
    verified = authenticate_originals(bundle, root)
    files = {}
    for role, source in bundle["archives"].items():
        files[EVIDENCE + "/" + role + "-original.zip"] = Path(source)
    for role, source in bundle["api"].items():
        files[EVIDENCE + "/api/" + role + ".json"] = Path(source)
    output.mkdir()
    pins = {}
    for name, source in sorted(files.items()):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = source.read_bytes()
        target.write_bytes(raw)
        pins[name] = dict(bytes=len(raw), sha256=digest(raw))
    (output / INPUTS).write_bytes(encode(bound_document(verified, pins)))
    return dict(status="EXACT_METADATA_ROSTERS_BOUND_FOR_REVIEW_ONLY",
        rule_sha256=RULE_SHA, catalog_sha256=verified["catalog_sha256"],
        selection_sha256=verified["selection_sha256"], source_panels_opened=False,
        execution_authorized=False, new_source_calls=0, new_model_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    check = sub.add_parser("check")
    check.add_argument("--root", type=Path, default=ROOT)
    binding = sub.add_parser("bind")
    binding.add_argument("repository", type=Path)
    binding.add_argument("bundle", type=Path)
    binding.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.mode == "check":
        value = verify_registration(args.root)
        print(json.dumps({k: value[k] for k in ("bound", "rule_sha256", "source_panels_opened",
                                              "execution_authorized")}, sort_keys=True))
    else:
        print(json.dumps(bind(args.repository, args.bundle, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
