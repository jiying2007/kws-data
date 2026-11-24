"""One public, text-only diagnostic of the frozen whole LibriSpeech corpus.

The original failed panel remains stopped. This program neither opens audio
members nor imports source/model implementations. It retains the complete
speaker census and transcript bytes, and diagnoses only the registered readers.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
REPO = "jiying2007/kws-data"
BRANCH = "refs/heads/research/en-v6-b21-natural-metadata-once-20261124"
WORKFLOW = ".github/workflows/english-v6-b21-natural-metadata-once.yml"
TEMPLATE = "automation/english_v6_b21_natural_metadata_workflow_template.yml"
TOKEN = "__B21_NATURAL_METADATA_MAIN_SHA__"
MARKER = "english-v6-b21-natural-metadata-consumed"
CORPUS_BYTES = 6387309499
CORPUS_SHA = "d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2"
OFFICIAL = (
    "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "https://openslr.trmal.net/resources/12/train-clean-100.tar.gz",
    "https://openslr.elda.org/resources/12/train-clean-100.tar.gz",
    "https://openslr.magicdatatech.com/resources/12/train-clean-100.tar.gz",
)
FROZEN = {
    "automation/english_v6_b21_natural_amendment_registration.json": "d680ac1b4491efbbd47222b527898236b6b171dc0bf2827c2082484f67882caf",
    "automation/english_v6_b21_fixed_mirror_transport.py": "eca61c7239562aac0fdb08ef5bf93dec93a017dcf771f78e19b857c0b4d35db4",
    "automation/english_v6_b21_fixed_mirror_transport.json": "82f915a9ea4355c545dc3c8aabeeca25460896021a4303dfc1f3280597cd3850",
    "automation/english_v6_b21_natural_sources.py": "e271b75276a1b8cd3928f4f6cd25be8bfb74481063ae55f4e3c9cf5d3fef3fda",
    "automation/english_v6_b21_replication_natural_sources.py": "ff5189ee2fecca22e2dd7588868e4db5ccf83b9b5fa1cf221ff4f42f55ebb24e",
    "automation/english_v6_b21_panels/assessment.json": "65ad9c7204755f1cd0b639added163387af39604818f9bd26b9da9fe8bd098fc",
    "automation/english_v6_b21_panels/natural-primary.json": "ae52045c4d6b8159e221fe4e0c8a3140fdb337c8fa86e573bf4866fa2f925f3c",
    "automation/english_v6_b21_panels/natural-replication.json": "45b8e6cee46f7a9da6eb4a8e4f52cc64ee6fea6c6909eafb53072bf7ba49d063",
    "automation/english_v6_b21_panels/SPEAKERS.TXT": "3350f093382f8baa1d589c9f5a5cee8ec3705e656f521d6eafbaa31a5fcd8f76",
    "automation/english_v6_b21_panels/ancestry.json": "26cf48813b5e712bc8820a2429fc0917db9f6e3eaa809a008d98c533714be2b0",
}
OWN = (
    "automation/english_v6_b21_natural_metadata.py",
    "automation/test_english_v6_b21_natural_metadata.py",
    TEMPLATE,
    WORKFLOW,
)
FAILURE = dict(run_id=45367984097, run_head="c262e4b5cf06de8a6950c2c797c7aefcf43498d8",
    source_artifact_id=14907672424, source_artifact_bytes=5726331,
    source_artifact_sha256="e7a9cb7c22c701222806ec53cdd241b8d1fdd9e2d5ec38fea03dc46d90980bfd")
ZERO = dict(new_source_calls=0, natural_decodes=0, audio_members_extracted=0,
    audio_headers_interpreted=0, observer_calls=0, kws_forwards=0, optimizer_updates=0,
    replacement_reader_draws=0, source_panel_openings=0)
BLOCK = set("HEY HAY MERO MEERO MIRO MARROW MERROW".split())
TRANSCRIPT = re.compile(r"LibriSpeech/train-clean-100/(\d+)/(\d+)/(\d+)-(\d+)\.trans\.txt")


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def record(path, value):
    path = Path(path)
    with path.open("xb") as stream:
        stream.write((json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def frozen(root=ROOT):
    root = Path(root)
    for name, expected in FROZEN.items():
        need(digest((root / name).read_bytes()) == expected, "unchanged frozen diagnostic input: " + name)
    assessment = json.loads((root / "automation/english_v6_b21_panels/assessment.json").read_bytes())
    for panel in ("primary", "replication"):
        original = json.loads((root / ("automation/english_v6_b21_panels/natural-" + panel + ".json")).read_bytes())
        need(original == assessment[panel]["natural"] and original["opened"] is False
             and original["selected_utterance_ids"] is None and original["training_allowed"] is False,
             "original unopened assessment panel, no new panel")
    return assessment


def primitive_namespace(root=ROOT):
    """Compile only five hash-authenticated transport definitions, no imports."""
    path = Path(root) / "automation/english_v6_b21_fixed_mirror_transport.py"
    raw = path.read_bytes()
    need(digest(raw) == FROZEN[str(path.relative_to(root))], "exact reviewed transport implementation")
    names = ("ranges", "validate_headers", "final_url_allowed", "OfficialRedirect", "transfer")
    tree = ast.parse(raw)
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    need(tuple(node.name for node in nodes) == names, "exact stdlib transport primitive subset")
    namespace = dict(need=need, record=record, OFFICIAL=OFFICIAL, urllib=__import__("urllib"),
                     hashlib=hashlib, time=time, os=os)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def eligibility_predicate(panel="primary", root=ROOT):
    """Use the exact frozen predicate without importing an audio/source module."""
    name = "english_v6_b21_natural_sources.py" if panel == "primary" else "english_v6_b21_replication_natural_sources.py"
    path = Path(root) / "automation" / name
    raw = path.read_bytes()
    need(panel in ("primary", "replication") and digest(raw) == FROZEN["automation/" + name], "frozen natural predicate")
    tree = ast.parse(raw)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "rank_metadata")
    tests = [node.test for node in ast.walk(function) if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp)
             and any(isinstance(child, ast.Name) and child.id == "BLOCK" for child in ast.walk(node.test))]
    expected = ast.parse("4 <= len(words) <= 14 and not BLOCK.intersection(words) and all(re.fullmatch(r\"[A-Z]+(?:'[A-Z]+)*\", word) for word in words)", mode="eval")
    need(len(tests) == 1 and ast.dump(tests[0]) == ast.dump(expected.body), "exact original word-count, veto and lexical filters")
    code = compile(ast.Expression(body=tests[0]), str(path), "eval")
    return lambda words: eval(code, dict(BLOCK=BLOCK, re=re), dict(words=words))


def validate_identity(value):
    need(set(value) == {"run_id", "run_head", "run_attempt", "run_branch", "prepared_parent"}
         and re.fullmatch(r"[1-9][0-9]*", value["run_id"] or "") is not None
         and all(re.fullmatch(r"[0-9a-f]{40}", value[k] or "") is not None for k in ("run_head", "prepared_parent"))
         and value["run_head"] != value["prepared_parent"] and value["run_attempt"] == 1
         and value["run_branch"] == BRANCH, "first registered public metadata-only execution")
    return value


def execution_guard(prepared_parent, env=None, event=None, parent=None, root=ROOT):
    e = os.environ if env is None else env
    need(all(e.get(k) == v for k, v in {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPO,
         "GITHUB_REF": BRANCH, "GITHUB_EVENT_NAME": "push", "GITHUB_RUN_ATTEMPT": "1"}.items()),
         "public first push on separately registered metadata-only branch")
    identity = validate_identity(dict(run_id=e.get("GITHUB_RUN_ID", ""), run_head=e.get("GITHUB_SHA", ""),
        run_attempt=1, run_branch=BRANCH, prepared_parent=prepared_parent))
    if event is None:
        event = json.loads(Path(e["GITHUB_EVENT_PATH"]).read_bytes())
    if parent is None:
        parent = subprocess.check_output(["git", "rev-parse", "HEAD^1"], cwd=root, text=True).strip()
    need(event.get("before") == parent == prepared_parent and event.get("after") == identity["run_head"]
         and event.get("ref") == BRANCH and event.get("forced") is event.get("deleted") is False
         and event.get("repository", {}).get("full_name") == REPO and event["repository"].get("private") is False,
         "exact reviewed parent, one activation push, no private run, force push or retry")
    return identity


def verify_git(identity, root=ROOT):
    root = Path(root)
    need(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() == identity["run_head"], "exact checkout")
    parents = subprocess.check_output(["git", "show", "-s", "--format=%P", "HEAD"], cwd=root, text=True).strip().split()
    changed = subprocess.check_output(["git", "diff", "--name-only", identity["prepared_parent"], "HEAD"], cwd=root, text=True).splitlines()
    need(parents == [identity["prepared_parent"]] and changed == [WORKFLOW], "single workflow-only activation from reviewed preparation")
    expected = (root / TEMPLATE).read_text().replace(TOKEN, identity["prepared_parent"]).encode()
    need((root / WORKFLOW).read_bytes() == expected, "exact metadata-only workflow with only parent binding")
    files = []
    for name in sorted(set(FROZEN) | set(OWN)):
        raw = (root / name).read_bytes()
        original = subprocess.check_output(["git", "cat-file", "blob", "HEAD:" + name], cwd=root)
        need(raw == original and not (root / name).is_symlink(), "exact Git-bound producer input: " + name)
        files.append(dict(path=name, bytes=len(raw), sha256=digest(raw),
            git_blob_sha1=hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()))
    return files


def gh(endpoint):
    return json.loads(subprocess.check_output(["gh", "api", "repos/" + REPO + "/" + endpoint]))


def one_execution(identity, consumed, api=gh):
    branch = BRANCH.removeprefix("refs/heads/")
    runs = api("actions/workflows/" + WORKFLOW.rsplit("/", 1)[1] + "/runs?branch=" + branch + "&event=push&per_page=100")
    need(runs["total_count"] == 1 and len(runs["workflow_runs"]) == 1, "only one metadata diagnostic execution ever")
    run = runs["workflow_runs"][0]
    need(str(run["id"]) == identity["run_id"] and run["head_sha"] == identity["run_head"]
         and run["head_branch"] == branch and run["run_attempt"] == 1 and run["event"] == "push"
         and run["repository"]["full_name"] == REPO and run["repository"]["private"] is False,
         "same public current metadata execution")
    artifacts = api("actions/artifacts?name=" + MARKER + "&per_page=100")
    need(artifacts["total_count"] == int(consumed) and len(artifacts["artifacts"]) == int(consumed), "metadata-only budget marker, no repeat")
    if consumed:
        marker = artifacts["artifacts"][0]
        need(marker["name"] == MARKER and marker["expired"] is False
             and str(marker["workflow_run"]["id"]) == identity["run_id"]
             and marker["workflow_run"]["head_sha"] == identity["run_head"], "this run owns the single metadata budget")
    return dict(runs=runs, artifacts=artifacts)


def previous_failure(api=gh):
    run = api("actions/runs/" + str(FAILURE["run_id"]))
    artifact = api("actions/artifacts/" + str(FAILURE["source_artifact_id"]))
    need(run["id"] == FAILURE["run_id"] and run["head_sha"] == FAILURE["run_head"]
         and run["run_attempt"] == 1 and run["status"] == "completed" and run["conclusion"] == "failure"
         and run["repository"]["full_name"] == REPO and run["repository"]["private"] is False,
         "actual failed source remains failed; diagnostic cannot redefine its result")
    need(artifact["id"] == FAILURE["source_artifact_id"] and artifact["size_in_bytes"] == FAILURE["source_artifact_bytes"]
         and artifact["digest"] == "sha256:" + FAILURE["source_artifact_sha256"] and artifact["expired"] is False
         and artifact["workflow_run"]["id"] == FAILURE["run_id"] and artifact["workflow_run"]["head_sha"] == FAILURE["run_head"],
         "exact retained original partial source archive; never regenerate its synthetic originals")
    return dict(run=run, source_artifact=artifact)


def prepare(directory, prepared_parent):
    frozen()
    identity = execution_guard(prepared_parent)
    files = verify_git(identity)
    live = one_execution(identity, False)
    old = previous_failure()
    directory = Path(directory)
    need(directory.is_absolute() and directory.parent.is_dir() and not directory.exists(), "exclusive fresh metadata-only destination")
    directory.mkdir()
    for sub in ("producer", "receipts", "consumed"):
        (directory / sub).mkdir()
    producer = directory / "producer"
    for item in files:
        target = producer / "repository" / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / item["path"], target)
    proof = dict(schema="english-v6-b21-natural-metadata-producer-v1", execution=identity, files=files,
        frozen_inputs=FROZEN, failed_source=FAILURE, original_failed_source_status="STOP",
        preregistered_amendment_rule_sha256=FROZEN["automation/english_v6_b21_natural_amendment_registration.json"],
        amendment_rule_executed=False, **ZERO)
    record(producer / "PRODUCER.json", proof)
    receipt = dict(schema="english-v6-b21-natural-metadata-authorization-v1", execution=identity,
        producer_sha256=digest((producer / "PRODUCER.json").read_bytes()), failed_source=FAILURE,
        original_failed_source_status="STOP", maximum_whole_corpus_transfers=1, maximum_ranges=8,
        range_retries=0, metadata_only=True, source_panels_remain_closed=True, **ZERO)
    record(directory / "receipts/AUTHORIZATION.json", receipt)
    record(directory / "receipts/CURRENT_EXECUTION_LIVE.json", live)
    record(directory / "receipts/FAILED_SOURCE_LIVE.json", old)
    record(directory / "consumed/CONSUMPTION.json", dict(receipt,
        status="METADATA_TRANSPORT_CONSUMED_UNKNOWN", audio_source_budget_authorized=False))
    return receipt


def transport_spec():
    value = json.loads((HERE / "english_v6_b21_fixed_mirror_transport.json").read_bytes())
    need(digest((HERE / "english_v6_b21_fixed_mirror_transport.json").read_bytes()) == FROZEN["automation/english_v6_b21_fixed_mirror_transport.json"], "unchanged registered transport parameters")
    need(value["urls"] == [OFFICIAL[3]] and value["bytes"] == CORPUS_BYTES and value["sha256"] == CORPUS_SHA
         and value["parallel_ranges"] == 8 and value["range_retries"] == 0 and value["fallback_urls"] == []
         and value["maximum_elapsed_seconds"] == 2700 and value["socket_timeout_seconds"] == 60,
         "one whole corpus, eight fixed official ranges, no retries or alternatives")
    return value


def download(corpus, directory, spec=None, primitives=None):
    """Called only after the metadata controller has consumed its separate budget."""
    spec = transport_spec() if spec is None else spec
    primitives = primitive_namespace() if primitives is None else primitives
    corpus, directory = Path(corpus), Path(directory)
    partial = corpus.with_name(corpus.name + ".partial")
    transport = directory / "TRANSPORT"
    need(corpus.is_absolute() and corpus.parent.is_dir() and not corpus.exists() and not partial.exists()
         and not transport.exists(), "fresh full transport only; no resume or retry")
    need(os.statvfs(corpus.parent).f_bavail * os.statvfs(corpus.parent).f_frsize > spec["bytes"] + 536870912, "sufficient whole archive space")
    transport.mkdir()
    receipts = transport / "ranges"
    receipts.mkdir()
    parts = primitives["ranges"](spec["bytes"], spec["parallel_ranges"])
    record(transport / "STARTED.json", dict(ranges=parts, maximum_whole_corpus_transfers=1,
        transport_helper_sha256=FROZEN["automation/english_v6_b21_fixed_mirror_transport.py"], **ZERO))
    started, stop = time.monotonic(), threading.Event()
    deadline = started + spec["maximum_elapsed_seconds"]
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    report = dict(schema="english-v6-b21-natural-metadata-transport-v1", status="STOP", urls=spec["urls"],
        bytes=spec["bytes"], sha256=spec["sha256"], ranges=[], range_retries=0, **ZERO)
    try:
        os.ftruncate(fd, spec["bytes"])
        with ThreadPoolExecutor(max_workers=spec["parallel_ranges"]) as executor:
            futures = [executor.submit(primitives["transfer"], part, spec, fd, receipts, stop, deadline,
                urllib.request.build_opener(primitives["OfficialRedirect"](spec["urls"][0]))) for part in parts]
            try:
                for future in as_completed(futures, timeout=spec["maximum_elapsed_seconds"]):
                    report["ranges"].append(future.result())
            except BaseException:
                stop.set()
                raise
        os.fsync(fd)
        need(time.monotonic() <= deadline and partial.stat().st_size == spec["bytes"], "complete archive within transfer budget")
        with partial.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        need(actual == spec["sha256"] and time.monotonic() <= deadline, "full original SHA before any tar parsing")
        partial.rename(corpus)
        report.update(status="COMPLETE_VERIFIED", ranges=sorted(report["ranges"], key=lambda item: item["index"]))
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        os.close(fd)
        report["elapsed_seconds"] = time.monotonic() - started
        record(transport / "RESULT.json", report)
    record(directory / "CORPUS_TRANSPORT.json", report)
    return report


def scan_text_metadata(corpus, expected_bytes=CORPUS_BYTES, expected_sha=CORPUS_SHA):
    """Traverse the gzip/tar stream; extract ONLY census/transcript text members."""
    corpus = Path(corpus)
    need(corpus.is_file() and not corpus.is_symlink() and corpus.stat().st_size == expected_bytes,
         "complete ordinary corpus file before metadata scan")
    with corpus.open("rb") as stream:
        need(hashlib.file_digest(stream, "sha256").hexdigest() == expected_sha, "whole original compressed SHA before tar opening")
    speakers, transcripts, seen = None, [], set()
    entries, text_bytes, ignored_audio_members = 0, 0, 0
    with tarfile.open(corpus, "r|gz") as archive:
        for member in archive:
            entries += 1
            need(entries <= 100000, "bounded complete archive scan")
            match = TRANSCRIPT.fullmatch(member.name)
            if member.name != "LibriSpeech/SPEAKERS.TXT" and match is None:
                ignored_audio_members += int(member.name.endswith(".flac"))
                continue
            need(member.isfile() and member.name not in seen and 0 < member.size <= 8388608,
                 "unique bounded ordinary metadata member")
            seen.add(member.name)
            text_bytes += member.size
            need(text_bytes <= 67108864, "bounded total text metadata")
            with archive.extractfile(member) as stream:
                raw = stream.read()
            need(len(raw) == member.size, "complete original text member")
            text = raw.decode("utf-8", errors="strict")
            if member.name == "LibriSpeech/SPEAKERS.TXT":
                need(speakers is None, "one official speaker census")
                speakers = text
            else:
                need(match.group(1) == match.group(3) and match.group(2) == match.group(4), "canonical speaker/chapter transcript path")
                transcripts.append(dict(path=member.name, bytes=len(raw), sha256=digest(raw), text=text))
    need(speakers is not None and transcripts, "complete original text metadata is present")
    return dict(schema="english-v6-b21-full-train-clean-100-text-metadata-v1", corpus_bytes=expected_bytes,
        corpus_sha256=expected_sha, speakers_text=speakers, speakers_sha256=digest(speakers.encode()),
        transcripts=sorted(transcripts, key=lambda item: item["path"]), metadata_bytes=text_bytes,
        archive_entries_seen=entries, ignored_audio_members=ignored_audio_members, metadata_only=True, **ZERO)


def diagnose(catalog, assessment, root=ROOT):
    need(catalog["speakers_sha256"] == assessment["metadata_speakers_sha256"]
         and digest(catalog["speakers_text"].encode()) == catalog["speakers_sha256"], "same original speaker census")
    census = set()
    for line in catalog["speakers_text"].splitlines():
        if not line.strip() or line.startswith(";"):
            continue
        fields = [part.strip() for part in line.split("|")]
        need(len(fields) >= 3, "official census fields")
        if fields[2] == "train-clean-100":
            census.add(fields[0])
    paths = [item["path"] for item in catalog["transcripts"]]
    need(paths == sorted(set(paths)), "complete unique sorted transcript metadata")
    rows, seen, chapters = defaultdict(list), set(), defaultdict(set)
    predicate = eligibility_predicate(root=root)
    replication_predicate = eligibility_predicate("replication", root=root)
    for item in catalog["transcripts"]:
        match = TRANSCRIPT.fullmatch(item["path"])
        need(match is not None and digest(item["text"].encode()) == item["sha256"], "unchanged original transcript bytes")
        speaker, chapter, sp2, ch2 = match.groups()
        need((speaker, chapter) == (sp2, ch2) and speaker in census, "original corpus canonical reader")
        chapters[speaker].add(chapter)
        for line in item["text"].splitlines():
            fields = line.split(" ", 1)
            need(len(fields) == 2, "original transcript identifier and text")
            ident, text = fields
            need(re.fullmatch(speaker + "-" + chapter + r"-\d+", ident) is not None and ident not in seen,
                 "globally unique original transcript source identity")
            seen.add(ident)
            words = text.split(" ")
            reasons = []
            if len(words) < 4:
                reasons.append("fewer_than_4_words")
            if len(words) > 14:
                reasons.append("more_than_14_words")
            if BLOCK.intersection(words):
                reasons.append("blocked_wake_token")
            if not all(re.fullmatch(r"[A-Z]+(?:'[A-Z]+)*", word) for word in words):
                reasons.append("noncanonical_token")
            need(predicate(words) == replication_predicate(words) == (not reasons), "diagnostic exactly matches both frozen predicates")
            rows[speaker].append(dict(source_id=ident, transcript=text, word_count=len(words),
                eligible=not reasons, rejection_reasons=reasons, transcript_member=item["path"],
                transcript_file_sha256=item["sha256"]))
    need(set(rows) == census, "full official train-clean-100 transcript census, not a partial reader catalog")
    panels = {}
    for panel in ("primary", "replication"):
        design = assessment[panel]["natural"]
        fixed = design["speakers"]
        need(len(fixed) == len(set(fixed)) == 12 and set(fixed) <= census
             and not set(fixed) & set(design["excluded_canonical_speakers"]), "only the original twelve fixed readers")
        diagnostics = []
        for speaker in fixed:
            originals = rows[speaker]
            eligible = [row for row in originals if row["eligible"]]
            eligible.sort(key=lambda row: (digest((design["salt"] + "/utterance/" + row["source_id"]).encode()), row["source_id"]))
            diagnostics.append(dict(speaker=speaker, transcript_members=len(chapters[speaker]),
                total_transcripts=len(originals), eligible_transcripts=len(eligible),
                rejection_reason_counts=dict(sorted(Counter(reason for row in originals for reason in row["rejection_reasons"]).items())),
                original_two_slot_requirement_met=len(eligible) >= 2,
                ranked_pair_status="RETROSPECTIVE_METADATA_ONLY_NOT_A_FROZEN_SOURCE_ROSTER",
                original_metadata_ranked_pair=[row["source_id"] for row in eligible[:2]] if len(eligible) >= 2 else None,
                originals=sorted(originals, key=lambda row: row["source_id"])))
        failed = [row["speaker"] for row in diagnostics if not row["original_two_slot_requirement_met"]]
        panels[panel] = dict(original_readers=fixed, readers=diagnostics, ineligible_readers=failed,
            first_original_stop_reader=failed[0] if failed else None,
            original_metadata_rank_status="STOP" if failed else "ELIGIBLE_METADATA_ONLY",
            source_panel_opened=False, replacement_readers=None)
    return dict(schema="english-v6-b21-original-natural-reader-diagnostic-v1", original_assessment_sha256=FROZEN["automation/english_v6_b21_panels/assessment.json"],
        corpus_sha256=catalog["corpus_sha256"], full_corpus_readers=len(census), full_corpus_transcripts=len(seen),
        failed_source=FAILURE, original_failed_source_status="STOP", panels=panels,
        original_synthetic_regeneration_allowed=False, audio_or_source_continuation_authorized=False,
        metadata_only=True, source_panels_remain_closed=True, **ZERO)


def collect(directory, corpus, prepared_parent):
    assessment = frozen()
    identity = execution_guard(prepared_parent)
    verify_git(identity)
    directory, corpus = Path(directory), Path(corpus)
    receipts = directory / "receipts"
    authorization = json.loads((receipts / "AUTHORIZATION.json").read_bytes())
    need(authorization["execution"] == identity and authorization["failed_source"] == FAILURE
         and authorization["metadata_only"] is True and authorization["source_panels_remain_closed"] is True
         and all(authorization[key] == value for key, value in ZERO.items())
         and authorization["producer_sha256"] == digest((directory / "producer/PRODUCER.json").read_bytes()),
         "same captured metadata-only producer and budget")
    one_execution(identity, True)
    need(corpus.name == "train-clean-100.tar.gz", "exact whole corpus destination")
    result = dict(schema="english-v6-b21-natural-metadata-execution-v1", execution=identity,
        failed_source=FAILURE, status="STOP", metadata_only=True, original_failed_source_status="STOP", **ZERO)
    try:
        download(corpus, receipts)
        record(receipts / "SCAN_STARTED.json", dict(corpus_sha256=CORPUS_SHA,
            only_extract_members=["LibriSpeech/SPEAKERS.TXT", "LibriSpeech/train-clean-100/*/*/*.trans.txt"], **ZERO))
        catalog = scan_text_metadata(corpus)
        record(receipts / "FULL_METADATA_CATALOG.json", catalog)
        diagnostic = diagnose(catalog, assessment)
        record(receipts / "ORIGINAL_READER_DIAGNOSTIC.json", diagnostic)
        result.update(status="METADATA_DIAGNOSTIC_COMPLETE",
            catalog_sha256=digest((receipts / "FULL_METADATA_CATALOG.json").read_bytes()),
            diagnostic_sha256=digest((receipts / "ORIGINAL_READER_DIAGNOSTIC.json").read_bytes()),
            original_panel_metadata_status={panel: diagnostic["panels"][panel]["original_metadata_rank_status"] for panel in ("primary", "replication")})
    except BaseException as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record(receipts / "RESULT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initial = commands.add_parser("prepare")
    initial.add_argument("directory")
    initial.add_argument("prepared_parent")
    execute = commands.add_parser("collect")
    execute.add_argument("directory")
    execute.add_argument("corpus")
    execute.add_argument("prepared_parent")
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.directory, args.prepared_parent)
    else:
        value = collect(args.directory, args.corpus, args.prepared_parent)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
