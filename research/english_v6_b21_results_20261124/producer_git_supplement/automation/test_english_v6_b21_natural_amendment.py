"""Metadata-only boundary fixtures; all transcript payloads here are synthetic."""
from __future__ import annotations
import ast
import copy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b21_natural_amendment as module


def fixture_catalog(root=module.ROOT):
    rule, _, _ = module.load_rule(root)
    speakers = (Path(root) / module.FROZEN_PATHS["speakers_sha256"]).read_text()
    census = [line.split("|")[0].strip() for line in speakers.splitlines()
              if line.strip() and not line.startswith(";")
              and line.split("|")[2].strip() == "train-clean-100"]
    transcripts = []
    for speaker in census:
        text = (speaker + "-1-0000 THE QUICK BROWN FOX\n" +
                speaker + "-1-0001 THESE ARE ONLY TEST WORDS\n")
        raw = text.encode()
        transcripts.append(dict(path=f"LibriSpeech/train-clean-100/{speaker}/1/{speaker}-1.trans.txt",
                                bytes=len(raw), sha256=module.digest(raw), text=text))
    return dict(schema=rule["catalog_schema"], corpus_bytes=rule["corpus"]["bytes"],
        corpus_sha256=rule["corpus"]["sha256"], speakers_text=speakers,
        speakers_sha256=module.digest(speakers.encode()), metadata_only=True,
        transcripts=sorted(transcripts, key=lambda row: row["path"]),
        metadata_bytes=len(speakers.encode()) + sum(row["bytes"] for row in transcripts),
        archive_entries_seen=1000, ignored_audio_members=500, **module.ZERO)


def replace_text(catalog, speaker, text):
    item = next(item for item in catalog["transcripts"]
                if module.TRANSCRIPT.fullmatch(item["path"]).group(1) == speaker)
    before = item["bytes"]
    raw = text.encode()
    item.update(text=text, bytes=len(raw), sha256=module.digest(raw))
    catalog["metadata_bytes"] += len(raw) - before


def zip_bytes(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(files.items()):
            archive.writestr(name, raw)
    return stream.getvalue()


def original_fixture(directory, catalog, root=module.ROOT):
    """Synthetic original-evidence envelope, never real source evidence."""
    directory, root = Path(directory), Path(root)
    rule, _, _ = module.load_rule(root)
    head, parent, tree_sha, run_id = "a" * 40, "b" * 40, "c" * 40, 42
    execution = dict(run_id=str(run_id), run_head=head, prepared_parent=parent,
                     run_attempt=1, run_branch="refs/heads/" + module.METADATA_BRANCH)
    old, old_pin = rule["original_failed_source"], rule["original_failed_source"]["source_artifact"]
    failure = dict(run_id=old["run_id"], run_head=old["run_head"],
        source_artifact_id=old_pin["id"], source_artifact_bytes=old_pin["bytes"],
        source_artifact_sha256=old_pin["sha256"])
    tree = ast.parse((root / module.METADATA_CODE).read_bytes())
    frozen = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "FROZEN"
                          for target in node.targets))
    paths = sorted(set(frozen) | {module.METADATA_CODE, module.METADATA_TEMPLATE,
        module.METADATA_WORKFLOW, "automation/test_english_v6_b21_natural_metadata.py"})
    raw_files = {name: (root / name).read_bytes() for name in paths}
    raw_files[module.METADATA_WORKFLOW] = raw_files[module.METADATA_TEMPLATE].decode().replace(
        module.METADATA_TOKEN, parent).encode()
    files = [dict(path=name, bytes=len(raw), sha256=module.digest(raw), git_blob_sha1=module.blob(raw))
             for name, raw in sorted(raw_files.items())]
    proof = dict(schema="english-v6-b21-natural-metadata-producer-v1",
        execution=execution, files=files, frozen_inputs=frozen, failed_source=failure,
        preregistered_amendment_rule_sha256=module.RULE_SHA,
        original_failed_source_status="STOP", **module.ZERO)
    proof_raw = module.encode(proof)
    producer_raw = zip_bytes({"PRODUCER.json": proof_raw,
                             **{"repository/" + name: raw for name, raw in raw_files.items()}})
    authorization = dict(execution=execution, failed_source=failure,
        producer_sha256=module.digest(proof_raw), metadata_only=True,
        source_panels_remain_closed=True, maximum_whole_corpus_transfers=1,
        maximum_ranges=8, range_retries=0, **module.ZERO)
    consumed_raw = zip_bytes({"CONSUMPTION.json": module.encode(dict(authorization,
        status="METADATA_TRANSPORT_CONSUMED_UNKNOWN", audio_source_budget_authorized=False))})
    diagnostic = dict(schema="english-v6-b21-original-natural-reader-diagnostic-v1",
        failed_source=failure, original_failed_source_status="STOP",
        audio_or_source_continuation_authorized=False, source_panels_remain_closed=True,
        panels={role: {"original_readers": rule["base_exclusion_rule"][
            "original_failed_primary12" if role == "primary" else "original_sealed_replication12"]}
            for role in ("primary", "replication")}, **module.ZERO)
    catalog_raw, diagnostic_raw = module.encode(catalog), module.encode(diagnostic)
    transport = dict(status="COMPLETE_VERIFIED", bytes=rule["corpus"]["bytes"],
        sha256=rule["corpus"]["sha256"], range_retries=0,
        ranges=[{"index": number} for number in range(8)])
    result = dict(schema="english-v6-b21-natural-metadata-execution-v1",
        status="METADATA_DIAGNOSTIC_COMPLETE", execution=execution, failed_source=failure,
        catalog_sha256=module.digest(catalog_raw), diagnostic_sha256=module.digest(diagnostic_raw),
        original_failed_source_status="STOP", metadata_only=True, **module.ZERO)
    metadata_raw = zip_bytes({
        "RESULT.json": module.encode(result), "FULL_METADATA_CATALOG.json": catalog_raw,
        "ORIGINAL_READER_DIAGNOSTIC.json": diagnostic_raw,
        "AUTHORIZATION.json": module.encode(authorization),
        "CORPUS_TRANSPORT.json": module.encode(transport),
        "TRANSPORT/RESULT.json": module.encode(transport)})
    artifacts, archives = [], {}
    names = {"metadata": "english-v6-b21-natural-metadata-42",
             "producer": "english-v6-b21-natural-metadata-producer-42",
             "consumed": "english-v6-b21-natural-metadata-consumed"}
    for ident, (role, raw) in enumerate(
            [("metadata", metadata_raw), ("producer", producer_raw), ("consumed", consumed_raw)], 1):
        path = directory / (role + ".zip")
        path.write_bytes(raw)
        archives[role] = str(path)
        artifacts.append(dict(id=ident, name=names[role], expired=False,
            size_in_bytes=len(raw), digest="sha256:" + module.digest(raw),
            workflow_run={"id": run_id, "head_sha": head}))
    apis = {
        "run": dict(id=run_id, head_sha=head, event="push", run_attempt=1,
            status="completed", conclusion="success", head_branch=module.METADATA_BRANCH,
            path=module.METADATA_WORKFLOW, repository={"full_name": module.REPO, "private": False},
            head_repository={"full_name": module.REPO, "private": False}),
        "artifacts": dict(total_count=3, artifacts=artifacts),
        "commit": dict(sha=head, parents=[{"sha": parent}],
            files=[{"filename": module.METADATA_WORKFLOW}], commit={"tree": {"sha": tree_sha}}),
        "tree": dict(sha=tree_sha, truncated=False, tree=[
            {"path": name, "type": "blob", "sha": module.blob(raw)}
            for name, raw in sorted(raw_files.items())])}
    api_paths = {}
    for role, value in apis.items():
        path = directory / (role + ".json")
        path.write_bytes(module.encode(value))
        api_paths[role] = str(path)
    bundle = dict(schema=module.BUNDLE_SCHEMA, archives=archives, api=api_paths)
    (directory / "bundle.json").write_bytes(module.encode(bundle))
    return bundle


class NaturalAmendmentTests(unittest.TestCase):
    def setUp(self):
        self.catalog = fixture_catalog()

    def test_inert_registration_has_no_rosters_or_execution_authority(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in [module.RULE, *module.FROZEN_PATHS.values()]:
                dst = root / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(module.ROOT / name, dst)
            (root / module.INPUTS).write_bytes(module.encode(module.inert_inputs()))
            value = module.verify_registration(root)
            self.assertFalse(value["bound"])
            self.assertIsNone(value["primary_roster"])
            self.assertIsNone(value["replication_roster"])
            self.assertFalse(value["execution_authorized"])
            self.assertEqual(module.resource_paths(root), sorted([module.RULE, module.INPUTS]))
            altered = module.inert_inputs()
            altered["metadata_artifact_pin"] = {"id": 123}
            (root / module.INPUTS).write_bytes(module.encode(altered))
            with self.assertRaises(ValueError):
                module.verify_registration(root)

    def test_frozen_predicate_edges_and_original_rank_function(self):
        cases = [("ONE TWO THREE", False), ("ONE TWO THREE FOUR", True),
                 (" ".join(["WORD"] * 14), True), (" ".join(["WORD"] * 15), False),
                 ("DON'T CHANGE THESE WORDS", True), ("HEY KEEP THESE WORDS", False),
                 ("Keep THESE FOUR WORDS", False), ("KEEP THESE FOUR WORDS.", False),
                 ("KEEP  THESE FOUR WORDS", False)]
        for text, expected in cases:
            self.assertEqual(module.predicate(text.split(" ")), expected, text)
        selected = module.select_catalog(self.catalog)
        for role in ("primary", "replication"):
            panel = selected["panels"][role]
            self.assertEqual(len(panel["roster"]), 24)
            self.assertEqual(len(panel["design"]["selected_utterance_ids"]), 24)
            self.assertFalse(panel["source_opened"])

    def test_predicate_ast_matches_both_original_scientific_filters(self):
        source = ast.parse(Path(module.__file__).read_bytes())
        predicate = next(node for node in source.body
                         if isinstance(node, ast.FunctionDef) and node.name == "predicate")
        actual = next(node.value for node in predicate.body if isinstance(node, ast.Return))
        for role in ("primary", "replication"):
            original = ast.parse((module.ROOT / module.FROZEN_PATHS[
                role + "_natural_code_sha256"]).read_bytes())
            rank = next(node for node in original.body
                        if isinstance(node, ast.FunctionDef) and node.name == "rank_metadata")
            expected = [node.test for node in ast.walk(rank) if isinstance(node, ast.If)
                        and any(isinstance(child, ast.Name) and child.id == "BLOCK"
                                for child in ast.walk(node.test))]
            self.assertEqual(len(expected), 1)
            self.assertEqual(ast.dump(actual), ast.dump(expected[0]))

    def test_full_exclusions_original_salts_and_sequential_reserve(self):
        selected = module.select_catalog(self.catalog)
        rule, assessment, ancestry = module.load_rule()
        excluded = module.excluded_people(rule, assessment, ancestry)
        chosen = set()
        for role in ("primary", "replication"):
            panel = selected["panels"][role]
            design = panel["design"]
            pool = set(design["eligible_census"])
            salt = assessment[role]["natural"]["salt"]
            expected = sorted(pool, key=lambda sp: (module.digest((salt + "/speaker/" + sp).encode()), sp))[:12]
            self.assertEqual(design["speakers"], expected)
            self.assertEqual(design["salt"], salt)
            self.assertFalse(set(expected) & (excluded | chosen))
            self.assertTrue(chosen <= set(design["excluded_canonical_speakers"]))
            chosen.update(expected)
        self.assertEqual(len(chosen), 24)
        self.assertFalse(chosen & set(rule["base_exclusion_rule"]["original_failed_primary12"]))
        self.assertFalse(chosen & set(rule["base_exclusion_rule"]["original_sealed_replication12"]))

    def test_ineligible_people_filtered_before_rank_not_replaced_after_selection(self):
        first = module.select_catalog(self.catalog)["panels"]["primary"]["design"]["speakers"][0]
        replace_text(self.catalog, first, first + "-1-0000 ONLY THREE WORDS\n")
        selected = module.select_catalog(self.catalog)
        self.assertNotIn(first, selected["panels"]["primary"]["ranked_eligible_people"])
        self.assertNotIn(first, selected["panels"]["replication"]["ranked_eligible_people"])

    def test_insufficient_pool_stops_without_partial_or_new_salt(self):
        rule, assessment, ancestry = module.load_rule()
        excluded = module.excluded_people(rule, assessment, ancestry)
        allowed = sorted(set(module.validate_catalog(self.catalog, rule)[0]) - excluded, key=int)[:23]
        for item in list(self.catalog["transcripts"]):
            speaker = module.TRANSCRIPT.fullmatch(item["path"]).group(1)
            if speaker not in allowed:
                replace_text(self.catalog, speaker, speaker + "-1-0000 ONLY THREE WORDS\n")
        with self.assertRaisesRegex(ValueError, "STOP"):
            module.select_catalog(self.catalog)

    def test_duplicate_missing_or_modified_transcript_metadata_rejected(self):
        for mutation in ("duplicate", "missing", "sha", "bytes"):
            catalog = copy.deepcopy(self.catalog)
            if mutation == "duplicate":
                catalog["transcripts"].append(catalog["transcripts"][0])
            elif mutation == "missing":
                gone = catalog["transcripts"].pop()
                catalog["metadata_bytes"] -= gone["bytes"]
            elif mutation == "sha":
                catalog["transcripts"][0]["sha256"] = "0" * 64
            else:
                catalog["transcripts"][0]["bytes"] += 1
            with self.assertRaises(ValueError, msg=mutation):
                module.select_catalog(catalog)

    def test_catalog_cannot_claim_audio_or_different_corpus(self):
        for key, value in (("natural_decodes", 1), ("metadata_only", False),
                           ("corpus_sha256", "0" * 64), ("new_source_calls", False)):
            catalog = copy.deepcopy(self.catalog)
            catalog[key] = value
            with self.assertRaises(ValueError, msg=key):
                module.select_catalog(catalog)

    def test_original_zip_binding_and_committed_readback_keep_all_gates_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            bundle = original_fixture(directory, self.catalog)
            verified = module.authenticate_originals(bundle)
            self.assertEqual(verified["catalog_sha256"], module.digest(module.encode(self.catalog)))
            inert = directory / "inert-repository"
            for name in [module.RULE, *module.FROZEN_PATHS.values()]:
                destination = inert / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(module.ROOT / name, destination)
            (inert / module.INPUTS).write_bytes(module.encode(module.inert_inputs()))
            output = directory / "output"
            result = module.bind(inert, directory / "bundle.json", output)
            self.assertFalse(result["source_panels_opened"])
            self.assertFalse(result["execution_authorized"])
            for name in [module.RULE, *module.FROZEN_PATHS.values()]:
                destination = output / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(module.ROOT / name, destination)
            value = module.verify_registration(output)
            self.assertTrue(value["bound"])
            self.assertFalse(value["execution_authorized"])
            self.assertTrue(value["live_metadata_refresh_required"])
            self.assertEqual(len(value["primary_roster"]), 24)
            self.assertEqual(len(value["replication_roster"]), 24)
            self.assertEqual(len(module.resource_paths(output)), 9)
            with self.assertRaises(ValueError):
                module.bind(inert, directory / "bundle.json", output)

    def test_original_zip_byte_change_rejected_before_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = original_fixture(temp, self.catalog)
            path = Path(bundle["archives"]["metadata"])
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "artifact ZIP"):
                module.authenticate_originals(bundle)

    def test_rule_must_be_in_original_execution_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = original_fixture(temp, self.catalog)
            path = Path(bundle["api"]["tree"])
            tree = module.read_json(path)
            next(item for item in tree["tree"] if item["path"] == module.RULE)["sha"] = "0" * 40
            path.write_bytes(module.encode(tree))
            with self.assertRaisesRegex(ValueError, "prospective rule"):
                module.authenticate_originals(bundle)

    def test_retry_private_failed_or_wrong_activation_rejected(self):
        for key, value in (("run_attempt", 2), ("conclusion", "failure"),
                           ("head_branch", "main"), ("event", "workflow_dispatch")):
            with tempfile.TemporaryDirectory() as temp:
                bundle = original_fixture(temp, self.catalog)
                path = Path(bundle["api"]["run"])
                run = module.read_json(path)
                run[key] = value
                path.write_bytes(module.encode(run))
                with self.assertRaises(ValueError, msg=key):
                    module.authenticate_originals(bundle)

    def test_unsafe_original_zip_member_rejected(self):
        with self.assertRaises(ValueError):
            module.read_zip(zip_bytes({"../outside.json": b"{}"}))

    def test_helper_imports_no_audio_model_network_or_executor_runtime(self):
        tree = ast.parse(Path(module.__file__).read_bytes())
        imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                   for alias in node.names}
        imports.update(node.module.split(".")[0] for node in ast.walk(tree)
                       if isinstance(node, ast.ImportFrom) and node.module)
        self.assertFalse(imports & {"torch", "numpy", "soundfile", "wave", "piper", "onnxruntime",
                                    "requests", "urllib", "subprocess", "socket"})
        self.assertTrue(imports <= {"__future__", "argparse", "ast", "collections", "hashlib", "io",
                                    "json", "pathlib", "re", "types", "zipfile"})


if __name__ == "__main__":
    unittest.main()
