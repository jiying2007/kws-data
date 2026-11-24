"""Offline invented provenance fixtures; never generate or score a source."""
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

HERE = Path(__file__).resolve().parent
definition = importlib.util.spec_from_file_location("b21_fixed_mirror_binding", HERE / "bind_b21_fixed_mirror_primary.py")
binding = importlib.util.module_from_spec(definition)
definition.loader.exec_module(binding)


@contextmanager
def snapshot_fixture(primary, role="source", workflow=None, comparison=False):
    """Build a real small producer ZIP and content-bound saved Git tree."""
    workflow = workflow or (".github/workflows/english-v6-b21-blank-retention-train-once.yml"
                            if role == "train" else primary.SOURCE_WORKFLOW)
    parent, head, tree_sha = "1" * 40, "2" * 40, "3" * 40
    token = (b"__B21_REVIEWED_TRAIN_PARENT__" if role == "train"
             else b"__B21_NATURAL_AMENDMENT_AUTH_MAIN_SHA__")
    template = b"name: saved-producer-fixture\nparent: " + token + b"\nbefore: " + token + b"\n"
    template_path = (workflow if role == "train" else
                     "automation/english_v6_b21_natural_amendment_source_auth_workflow_template.yml")
    code_path = "automation/saved_producer_fixture.py"
    raw_files = {code_path: b'"""Saved stdlib-only fixture; no execution entrypoint."""\n',
                 workflow: template.replace(token, parent.encode())}
    resources = []
    if role == "source":
        raw_files[template_path] = template
        resources.append(template_path)
    files = {name: dict(mode="100644", bytes=len(raw), sha256=binding.sha(raw),
                       git_blob_sha1=primary.git_blob(raw)) for name, raw in raw_files.items()}
    document = dict(schema="english-v6-b21-original-git-producer-snapshot-v1",
        role="source_natural_amendment" if role == "source" else role,
        run_id=123, run_head=head, run_attempt=1,
        run_branch=primary.TRAIN_BRANCH if role == "train" else primary.SOURCE_BRANCH,
        runtime_workflow=workflow, git_tree=tree_sha, prepared_parent=parent,
        runtime_workflow_preserved_hidden=True, source_code_reads_only=True,
        network_calls=0, new_source_calls=0, new_model_forwards=0, optimizer_updates=0,
        files=files, python_files=[code_path], resource_files=resources)
    meta = dict(run=dict(id=123, head_sha=head, path=workflow),
        commit=dict(sha=head, commit=dict(tree=dict(sha=tree_sha)), parents=[dict(sha=parent)],
                    files=[dict(filename=workflow)]),
        tree=dict(sha=tree_sha, truncated=False, tree=[dict(path=name, type="blob",
                    mode=pin["mode"], size=pin["bytes"], sha=pin["git_blob_sha1"])
                    for name, pin in files.items()]))
    if comparison:
        del meta["commit"]["files"]
        meta["comparison"] = dict(base_commit=dict(sha=parent), head=head, total_commits=1,
                                  ahead_by=1, behind_by=0, files=[dict(filename=workflow)])
    with tempfile.TemporaryDirectory() as temporary:
        repository = Path(temporary)
        for name, raw in raw_files.items():
            path = repository / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        (repository / template_path).write_bytes(template)
        yield repository, document, meta, raw_files


def original_zip_fixture(document, raw_files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("PRODUCER.json", binding.serialized(document))
        for name, raw in raw_files.items():
            archive.writestr("repository/" + name, raw)
    stream.seek(0)
    return zipfile.ZipFile(stream)


def proof_fixture():
    first = dict(original_run_id=45352507569,
        original_run_head="d6db2375632adc8a627d094fcf23c5e87b22eddd",
        original_prepared_parent="60f1e6507216ae605d601f3f09ab15177b46aae2",
        original_job_id=138302172086, original_run_attempt=1,
        source_job_log_sha256="c9ecd7399ef7b7b243ea243becfa9b07fbd1ecba402195fc61019581c7db62ac",
        actual_tts_calls=0, natural_original_decodes=0, observer_calls=0, kws_forwards=0,
        source_generation_started=False, transport_continuation=True, original_opened_marker_preserved=True)
    return dict(first, original_transport_origin=first,
        failed_continuation_run_id=45365292922,
        failed_continuation_run_head="2da5cbe0d616f96b21a1bb5665613b9ce8d1c788",
        failed_continuation_prepared_parent="3144028b5db5a7f1314654858b8d4f1343b8cf40",
        failed_continuation_job_id=138345323477,
        failed_continuation_manifest_sha256="a1a0d2acf630fd8b23e773e1e7906401787a8a75cbca81fca2de08dd1d54c4ca",
        failed_continuation_producer_files=246,
        failed_continuation_actual_tts_calls=0, failed_continuation_natural_original_decodes=0,
        failed_continuation_observer_calls=0, failed_continuation_kws_forwards=0,
        failed_continuation_consumption_started=False, failed_continuation_source_generation_started=False,
        failed_continuation_preserved=True, live_refresh_required_for_both_failed_runs=True)


class BindingBoundaryTests(unittest.TestCase):
    def test_driver_pin_and_amended_source_artifact_routing(self):
        driver = HERE / "english_v6_b21_fixed_mirror_primary_eval.py"
        self.assertEqual(hashlib.sha256(driver.read_bytes()).hexdigest(), binding.DRIVER_SHA)
        self.assertEqual(binding.artifact_name("source", 123), "english-v6-b21-natural-amendment-source-123")
        self.assertEqual(binding.artifact_name("source_producer", 123), "english-v6-b21-natural-amendment-source-producer-123")
        self.assertEqual(binding.artifact_name("screen", 123), "english-v6-b21-natural-amendment-frozen-screen-123")
        self.assertEqual(binding.artifact_name("native_qwen", 123), "b21-natural-amendment-dev-observer-qwen")
        self.assertEqual(binding.artifact_name("pair", 123), "english-v6-b21-competing-margin-weights-123")

    def test_requires_both_exact_failure_identities(self):
        proof = proof_fixture()
        self.assertEqual(binding.validate_transport_origin(proof, copy.deepcopy(proof)), proof)
        first = proof["original_transport_origin"]
        with self.assertRaisesRegex(ValueError, "both original failure identities"):
            binding.validate_transport_origin(first, copy.deepcopy(first))
        changed = copy.deepcopy(proof)
        changed["failed_continuation_run_id"] += 1
        with self.assertRaises(ValueError):
            binding.validate_transport_origin(changed, copy.deepcopy(changed))

    def test_saved_owner_must_equal_independently_verified_complete_proof(self):
        proof = proof_fixture()
        changed = copy.deepcopy(proof)
        changed["unverified_extra"] = "cannot be silently ignored"
        with self.assertRaisesRegex(ValueError, "matches both independently authenticated"):
            binding.validate_transport_origin(changed, proof)
        changed = copy.deepcopy(proof)
        changed["original_transport_origin"]["actual_tts_calls"] = 1
        with self.assertRaises(ValueError):
            binding.validate_transport_origin(changed, copy.deepcopy(changed))

    def test_unknown_or_nonzero_consumption_cannot_authorize_binding(self):
        for field in ("failed_continuation_actual_tts_calls", "failed_continuation_natural_original_decodes",
                      "failed_continuation_observer_calls", "failed_continuation_kws_forwards"):
            for value in (1, False, None):
                proof = proof_fixture()
                proof[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    binding.validate_transport_origin(proof, copy.deepcopy(proof))
        for field in ("failed_continuation_consumption_started", "failed_continuation_source_generation_started"):
            proof = proof_fixture()
            proof[field] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                binding.validate_transport_origin(proof, copy.deepcopy(proof))


class SnapshotRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with binding.source_protection():
            cls.primary = binding.load(HERE / "english_v6_b21_fixed_mirror_primary_eval.py",
                                       "b21_snapshot_fixture_primary")

    def verify(self, repository, document, meta, raw_files, role="source"):
        with original_zip_fixture(document, raw_files) as archive, binding.source_protection():
            return binding.snapshot(archive, role, meta, self.primary, repository)

    def test_new_auth_source_zip_manifest_and_git_tree_are_accepted(self):
        with snapshot_fixture(self.primary) as fixture:
            result = self.verify(*fixture)
            self.assertEqual(result["runtime_workflow"], self.primary.SOURCE_WORKFLOW)
            self.assertEqual(result, fixture[1])

    def test_new_auth_source_accepts_saved_parent_comparison(self):
        with snapshot_fixture(self.primary, comparison=True) as fixture:
            self.assertEqual(self.verify(*fixture), fixture[1])

    def test_old_source_workflow_is_rejected_even_with_consistent_zip_and_tree(self):
        old = ".github/workflows/english-v6-b21-natural-amendment-sources-once.yml"
        with snapshot_fixture(self.primary, workflow=old) as fixture:
            with self.assertRaisesRegex(ValueError, "actual pre-execution Git-bound producer snapshot source"):
                self.verify(*fixture)

    def test_train_snapshot_keeps_its_original_workflow(self):
        with snapshot_fixture(self.primary, role="train") as fixture:
            result = self.verify(*fixture, role="train")
            self.assertEqual(result["runtime_workflow"],
                             ".github/workflows/english-v6-b21-blank-retention-train-once.yml")

    def test_source_run_cannot_claim_old_path_while_zip_claims_auth_path(self):
        with snapshot_fixture(self.primary) as fixture:
            fixture[2]["run"]["path"] = ".github/workflows/english-v6-b21-natural-amendment-sources-once.yml"
            with self.assertRaisesRegex(ValueError, "actual pre-execution Git-bound producer snapshot source"):
                self.verify(*fixture)

    def test_source_git_tree_must_authenticate_actual_zip_members(self):
        with snapshot_fixture(self.primary) as fixture:
            fixture[2]["tree"]["tree"][0]["sha"] = "0" * 40
            with self.assertRaisesRegex(ValueError, "actual authenticated Git tree member"):
                self.verify(*fixture)

    def test_source_zip_bytes_must_match_manifest(self):
        with snapshot_fixture(self.primary) as fixture:
            fixture[3]["automation/saved_producer_fixture.py"] += b"# changed\n"
            with self.assertRaisesRegex(ValueError, "original producer content/Git blob identity"):
                self.verify(*fixture)


if __name__ == "__main__":
    unittest.main()
