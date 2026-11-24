"""Offline identity/authorization regressions. No model, audio or network work."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent
loader = importlib.util.spec_from_file_location("b20_evaluator_under_test", HERE / "english_v6_b20_primary_eval.py")
module = importlib.util.module_from_spec(loader)
loader.loader.exec_module(module)
m = module


def digest(text):
    return m.sha(str(text).encode())


def input_pin(index=1):
    return dict(id=800000 + index, run_id=800001, run_head="a" * 40,
                bytes=100 + index, sha256=digest(index))


def active_fixture():
    spec = json.loads(m.SPEC.read_bytes())
    fixed = m.fixed()
    spec.update(scoring_allowed=True, assessment_sha256="a" * 64,
                producer_manifest_sha256="b" * 64, prepared_parent=None, parent_authority="reviewed_workflow")
    spec["inputs"].update(fixed["natural"])
    spec["inputs"].update(synth_source=input_pin(), synth_screen=input_pin(2))
    spec["synthetic_native_inputs"] = {provider: input_pin(i + 3) for i, provider in enumerate(m.PROVIDERS)}
    spec["natural_native_inputs"] = copy.deepcopy(fixed["natural_native"])
    return spec


def environment():
    env = dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY=m.REPO, GITHUB_EVENT_NAME="push",
               GITHUB_REF=m.BRANCH, GITHUB_RUN_ATTEMPT="1", GITHUB_SHA="d" * 40, GITHUB_RUN_ID="700001")
    event = dict(after="d" * 40, before="c" * 40, ref=m.BRANCH, forced=False, deleted=False,
                 repository=dict(full_name=m.REPO, private=False))
    return env, event


def source_rows():
    rows = []
    for index in range(798, 822):
        for cell, cls, rate in (("mero-fast", 2, 1.4), ("mero-slow", 2, 1.8),
                                ("question", 0, 1.0), ("request", 0, 1.08)):
            ident = f"b20-primary-{index}-{cell}"
            rows.append(dict(id=ident, pcm_sha256=digest(ident), source_group_id=f"piper-libritts-p{index}",
                             source_speaker_id=f"p{index}", speaker_index=index, cell_id=cell,
                             requested_class=cls, length_scale=rate, role="development",
                             training_allowed=False, observed_class=None))
    return rows


def natural_rows():
    return [dict(id=f"original-natural-{i}", pcm_sha256=digest("natural" + str(i)),
                 source_group_id=f"librispeech-{5000+i//2}", speaker=str(5000+i//2),
                 source_id=f"{5000+i//2}-100-{i:04d}") for i in range(24)]


def natural_labels(rows):
    return [dict(id=row["id"], pcm_sha256=row["pcm_sha256"], source_id=row["source_id"],
                 group=row["source_group_id"], tier="silver" if i < 21 else "unknown",
                 full_wake_veto=False, exact_votes=3 if i < 21 else 0) for i, row in enumerate(rows)]


def overlap_fixture():
    synthetic, natural = source_rows(), natural_rows()
    training = [dict(id=f"train-{i}", source_pcm_sha256=digest("train" + str(i)),
                     group=f"librispeech:{6000+i}") for i in range(88)]
    canary_pcm = {digest("canary" + str(i)) for i in range(48)}
    canary_people = {str(7000+i) for i in range(8)}
    history = dict(all_pcm_sha256=sorted(canary_pcm | {r["pcm_sha256"] for r in natural}),
                   all_source_groups=[r["source_group_id"] for r in natural],
                   canonical_people=sorted(canary_people | {m.person(r) for r in natural}))
    return synthetic, natural, training, history, canary_pcm, canary_people, {"9000"}


class Contracts(unittest.TestCase):
    def test_inert_first_before_all_io_or_helpers(self):
        # Exercise the disabled state independently of whether the reviewed
        # repository SPEC has subsequently received authentic source pins.
        disabled = json.loads(m.SPEC.read_bytes())
        disabled["scoring_allowed"] = False
        with tempfile.TemporaryDirectory() as tmp, patch.object(m, "SPEC", Path(tmp) / "disabled.json"), \
             patch.object(m, "fixed", side_effect=AssertionError("fixed read")), \
             patch.object(m, "gh", side_effect=AssertionError("network")), \
             patch.object(m, "helpers", side_effect=AssertionError("model helpers")), \
             patch.object(m, "guard", side_effect=AssertionError("environment")):
            m.SPEC.write_text(json.dumps(disabled))
            for entry, arguments in ((m.protocol, ()), (m.fetch, (None, None)),
                                     (m.preflight, (None, None)), (m.score, (None, None, None))):
                with self.subTest(entry=entry.__name__), self.assertRaisesRegex(ValueError, "INERT"):
                    entry(*arguments)
        self.assertEqual(m.source_check()["model_calls"], 0)
        self.assertNotIn("torch", m.sys.modules)
        self.assertNotIn("numpy", m.sys.modules)

    def test_false_activation_and_changed_original_identity_rejected(self):
        spec = active_fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(m, "SPEC", Path(tmp) / "spec.json"):
            m.SPEC.write_text(json.dumps(spec))
            self.assertEqual(m.protocol(), spec)
            bad_cases = []
            a = copy.deepcopy(spec); a["inputs"]["synth_source"] = None; bad_cases.append(a)
            a = copy.deepcopy(spec); a["inputs"]["natural_source"]["sha256"] = "0" * 64; bad_cases.append(a)
            a = copy.deepcopy(spec); a["model_pt_sha256"]["challenger"] = "0" * 64; bad_cases.append(a)
            a = copy.deepcopy(spec); a["synthetic_native_inputs"]["parakeet"]["run_id"] += 1; bad_cases.append(a)
            a = copy.deepcopy(spec); a["natural_native_inputs"]["qwen"]["id"] += 1; bad_cases.append(a)
            a = copy.deepcopy(spec); a["pair_schema"] = "B9-parent"; bad_cases.append(a)
            a = copy.deepcopy(spec); a["threshold_search"] = True; bad_cases.append(a)
            for bad in bad_cases:
                with self.subTest(changed=bad), self.assertRaises(ValueError):
                    m.SPEC.write_text(json.dumps(bad)); m.protocol()

    def test_actual_head_parent_public_attempt(self):
        env, event = environment()
        spec = dict(prepared_parent=None, parent_authority="reviewed_workflow")
        result = m.guard(spec, "c" * 40, env, event, "d" * 40, "c" * 40)
        self.assertEqual(result["run_head"], "d" * 40)
        for key, value in (("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_REF", "refs/heads/main")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.guard(spec, "c" * 40, dict(env, **{key: value}), event, "d" * 40, "c" * 40)
        for bad in (dict(event, forced=True), dict(event, before="e" * 40),
                    dict(event, repository=dict(full_name=m.REPO, private=True))):
            with self.subTest(event=bad), self.assertRaises(ValueError):
                m.guard(spec, "c" * 40, env, bad, "d" * 40, "c" * 40)

    def test_actual_run_and_workflow_only_parent(self):
        identity = dict(run_id="8", run_head="a" * 40, prepared_parent="b" * 40)
        run = dict(id=8, head_sha="a" * 40, run_attempt=1, event="push", path=m.WORKFLOW,
                   head_branch=m.BRANCH.removeprefix("refs/heads/"), repository=dict(full_name=m.REPO, private=False))
        commit = dict(sha="a" * 40, parents=[dict(sha="b" * 40)], files=[dict(filename=m.WORKFLOW)])
        m.validate_execution_metadata(identity, run, commit)
        for bad in (dict(commit, parents=[dict(sha="c" * 40)]),
                    dict(commit, files=[dict(filename="automation/english_v6_b14_decision.py")])):
            with self.assertRaises(ValueError):
                m.validate_execution_metadata(identity, run, bad)

    def test_input_artifact_run_hash_and_attempt_must_match(self):
        pin = input_pin()
        run = dict(id=pin["run_id"], head_sha=pin["run_head"], run_attempt=1, status="completed", conclusion="success",
                   repository=dict(full_name=m.REPO, private=False))
        artifact = dict(id=pin["id"], expired=False, size_in_bytes=pin["bytes"], digest="sha256:" + pin["sha256"],
                        workflow_run=dict(id=pin["run_id"], head_sha=pin["run_head"]))
        meta = dict(run=run, artifact=artifact, commit=dict(sha=pin["run_head"]))
        m.validate_provenance(pin, meta)
        for target, key, value in (("run", "run_attempt", 2), ("run", "run_attempt", True),
                                   ("run", "head_sha", "b" * 40), ("artifact", "digest", "sha256:" + "0" * 64)):
            bad = copy.deepcopy(meta); bad[target][key] = value
            with self.subTest(target=target, key=key), self.assertRaises(ValueError):
                m.validate_provenance(pin, bad)

    def test_preexisting_run_blocks_even_without_consumed_marker(self):
        inventory = dict(total_count=2, workflow_runs=[dict(id=7), dict(id=8)])
        with patch.object(m, "gh", return_value=json.dumps(inventory).encode()), self.assertRaises(ValueError):
            m.unused_workflow_run(dict(run_id="8"))

    def test_static_closure_includes_transitive_and_literal_dynamic_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "automation"; directory.mkdir()
            (directory / "entry.py").write_text('import first\nimport importlib\nimportlib.import_module("dynamic")\nload_path("by_filename.py")\nload_path("missing.py")\nload_path("../escaped.py")\nload_path("not_python.json")\n')
            (directory / "first.py").write_text("from second import value\n")
            (directory / "second.py").write_text("value = 1\n")
            (directory / "dynamic.py").write_text("import second\n")
            (directory / "by_filename.py").write_text("import indirect_filename\n")
            (directory / "indirect_filename.py").write_text("value = 2\n")
            (directory / "escaped.py").write_text("import poison\n")
            (directory / "poison.py").write_text("value = 3\n")
            (directory / "not_python.json").write_text('{"module": "poison"}\n')
            self.assertEqual(m.static_closure(directory, ("entry",)),
                             {"automation/" + name + ".py" for name in
                              ("entry", "first", "second", "dynamic", "by_filename", "indirect_filename")})

    def test_static_closure_covers_first_party_package_relative_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); directory = root / "automation"; directory.mkdir()
            (directory / "entry.py").write_text("from contracts.identity import Identity\n")
            package = root / "contracts"; package.mkdir()
            (package / "__init__.py").write_text("from .identity import Identity\n")
            (package / "identity.py").write_text("from .helper import value\nclass Identity: pass\n")
            (package / "helper.py").write_text("value = 1\n")
            (package / "not_code.json").write_text("{}\n")
            self.assertEqual(m.static_closure(directory, ("entry",)),
                             {"automation/entry.py", "contracts/__init__.py", "contracts/identity.py", "contracts/helper.py"})

    def test_producer_code_resource_and_only_parent_workflow_bytes(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(m, "PRODUCER_ROOTS", ("entry",)), \
             patch.object(m, "SOURCE_MODULE", "source"):
            root = Path(tmp); directory = root / "automation"; directory.mkdir()
            for name, text in {"entry": "import source\nimport downstream\n", "source": "value = 1\n",
                               "downstream": "contract = 1\n"}.items():
                (directory / (name + ".py")).write_text(text)
            names = ("automation/" + m.FIXED.name, "automation/english_v6_b17_primary_inputs.json",
                     "automation/english_v6_b18_primary_inputs.json", "automation/english_v6_b20_panels/assessment.json")
            resources = {}
            for name in names:
                target = root / name; target.parent.mkdir(exist_ok=True); target.write_text("{}\n")
                resources[name] = m.sha(target.read_bytes())
            durable = {}
            for name in m.B18_DURABLE_FILES:
                relative = "research/history/" + name
                target = root / relative; target.parent.mkdir(parents=True, exist_ok=True); target.write_text("{}\n")
                resources[relative] = m.sha(target.read_bytes())
                durable[name] = dict(path=relative, sha256=resources[relative])
            b19 = root / m.B19_REGISTRATION; b19.parent.mkdir(parents=True)
            b19.write_text(json.dumps(dict(original_B18_source_proof=dict(durable_evidence=durable))))
            resources[m.B19_REGISTRATION] = m.sha(b19.read_bytes())
            template = ("parent=" + m.PARENT_TOKEN + "\nenv=" + m.PARENT_TOKEN + "\nno-model-change\n").encode()
            (directory / m.WORKFLOW_TEMPLATE.name).write_bytes(template)
            actual = template.replace(m.PARENT_TOKEN.encode(), b"c" * 40)
            workflow = root / m.WORKFLOW; workflow.parent.mkdir(parents=True); workflow.write_bytes(actual)
            doc = dict(schema="english-v6-b20-primary-producer-v1", frozen=True, resources=resources,
                       primary=dict(files={name: m.sha((root / name).read_bytes())
                                    for name in m.static_closure(directory, ("entry",))},
                                    workflow_path=m.WORKFLOW, workflow_template_sha256=m.sha(template)),
                       source=dict(files={"automation/source.py": m.sha((directory / "source.py").read_bytes())},
                                   resources=dict(resources), workflow_path=m.SOURCE_WORKFLOW,
                                   workflow_sha256="a" * 64))
            manifest = directory / m.PRODUCER.name
            manifest.write_text(json.dumps(doc))
            spec = dict(producer_manifest_sha256=m.sha(manifest.read_bytes()), prepared_parent=None,
                        parent_authority="reviewed_workflow")
            self.assertEqual(m.producer_contract(spec, "c" * 40, root)[1], actual)
            for path in (directory / "downstream.py", root / names[-1], workflow):
                raw = path.read_bytes(); path.write_bytes(raw + b"changed\n")
                with self.subTest(path=path), self.assertRaises(ValueError): m.producer_contract(spec, "c" * 40, root)
                path.write_bytes(raw)
            history_path = durable["HISTORY.json"]["path"]
            old_digest = doc["source"]["resources"].pop(history_path)
            manifest.write_text(json.dumps(doc)); spec["producer_manifest_sha256"] = m.sha(manifest.read_bytes())
            with self.assertRaisesRegex(ValueError, "indirect B18 durable evidence"):
                m.producer_contract(spec, "c" * 40, root)
            doc["source"]["resources"][history_path] = old_digest
            del doc["primary"]["files"]["automation/downstream.py"]
            manifest.write_text(json.dumps(doc)); spec["producer_manifest_sha256"] = m.sha(manifest.read_bytes())
            with self.assertRaises(ValueError): m.producer_contract(spec, "c" * 40, root)

    def test_constructible_two_commit_activation_without_self_hash(self):
        """A real local Git predecessor freezes SPEC; only the workflow opens it."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.check_output(["git", "-c", "user.name=B20 offline fixture",
                    "-c", "user.email=b20-fixture@example.invalid", "-c", "commit.gpgsign=false",
                    *args], cwd=root, text=True).strip()
            git("init", "-q")
            spec_path = root / "automation" / m.SPEC.name; spec_path.parent.mkdir()
            spec = dict(prepared_parent=None, parent_authority="reviewed_workflow")
            spec_raw = json.dumps(spec).encode(); spec_path.write_bytes(spec_raw)
            workflow = root / m.WORKFLOW; workflow.parent.mkdir(parents=True)
            template = ("guard=" + m.PARENT_TOKEN + "\nenv=" + m.PARENT_TOKEN + "\n").encode()
            workflow.write_bytes(template)
            git("add", "."); git("commit", "-qm", "freeze scientific registration")
            prepared = git("rev-parse", "HEAD")
            workflow.write_bytes(template.replace(m.PARENT_TOKEN.encode(), prepared.encode()))
            git("add", m.WORKFLOW); git("commit", "-qm", "open only the reviewed workflow")
            head = git("rev-parse", "HEAD"); parent = git("rev-parse", "HEAD^1")
            self.assertEqual(spec_path.read_bytes(), spec_raw)
            env, event = environment()
            env["GITHUB_SHA"] = head; event.update(after=head, before=prepared)
            identity = m.guard(spec, prepared, env, event, head, parent)
            run = dict(id=int(identity["run_id"]), head_sha=head, run_attempt=1, event="push", path=m.WORKFLOW,
                       head_branch=m.BRANCH.removeprefix("refs/heads/"), repository=dict(full_name=m.REPO, private=False))
            def commit_evidence():
                return dict(sha=head, parents=[dict(sha=parent)],
                            files=[dict(filename=name) for name in git("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines()])
            m.validate_execution_metadata(identity, run, commit_evidence())
            with self.assertRaises(ValueError):
                m.guard(spec, "e" * 40, env, event, head, parent)
            # An amended opening which also edits scientific SPEC must be rejected.
            spec_path.write_bytes(spec_raw + b"\n")
            git("add", "automation/" + m.SPEC.name); git("commit", "--amend", "--no-edit", "-q")
            head = git("rev-parse", "HEAD"); run["head_sha"] = head; identity["run_head"] = head
            with self.assertRaises(ValueError):
                m.validate_execution_metadata(identity, run, commit_evidence())

    def test_fixed_rates_no_replacement_or_reserved_source(self):
        rows = source_rows()
        m.validate_new_format(rows)
        for key, value in (("length_scale", 1.2), ("speaker_index", 822), ("observed_class", 2),
                           ("training_allowed", True)):
            bad = copy.deepcopy(rows); bad[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.validate_new_format(bad)

    def test_synthetic_source_screen_exact_join_and_unknown_exclusion(self):
        rows = source_rows()
        decisions = [dict(row, tier="silver", observed_class=row["requested_class"]) for row in rows]
        decisions[0].update(observed_class=None, tier="unknown")
        self.assertEqual(len(m.exact_admitted_join(rows, decisions, "synthetic")), 95)
        for key, value in (("id", "substitution"), ("pcm_sha256", "0" * 64),
                           ("source_group_id", "unrelated"), ("cell_id", "another-cell"), ("tier", "gold")):
            bad = copy.deepcopy(decisions); bad[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.exact_admitted_join(rows, bad, "synthetic")

    def test_natural_exact_utterance_not_merely_same_reader(self):
        rows = natural_rows(); decisions = natural_labels(rows)
        selected = m.exact_admitted_join(rows, decisions, "natural")
        self.assertEqual(len(selected), 21)
        self.assertEqual(len({m.person(row) for row in selected}), 11)
        for key, value in (("source_id", "5000-999999999-0000"), ("group", "librispeech-5001"),
                           ("pcm_sha256", "0" * 64)):
            bad = copy.deepcopy(decisions); bad[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.exact_admitted_join(rows, bad, "natural")
        with self.assertRaises(ValueError):
            m.exact_admitted_join(rows, list(reversed(decisions)), "natural")

    def test_saved_rows_reject_unknown_substitution_or_changed_identity(self):
        rows = natural_rows(); selected = m.exact_admitted_join(rows, natural_labels(rows), "natural")
        expected = [m.score_identity(row, "natural") for row in selected]
        records = [dict(row, gains={}) for row in expected]
        m.validate_scored_join(records, expected)
        for key, value in (("id", rows[23]["id"]), ("pcm_sha256", rows[23]["pcm_sha256"]),
                           ("positive", 0), ("tier", "unknown"), ("group", "librispeech-999")):
            bad = copy.deepcopy(records); bad[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.validate_scored_join(bad, expected)

    def test_all_originals_history_train_canary_and_reserve(self):
        values = overlap_fixture()
        m.reject_overlap(*values)  # Exact natural originals may already belong to history.
        for kind in ("history_pcm", "history_person", "unknown_train", "unknown_canary", "reserve"):
            synthetic, natural, train, history, pcm, people, reserve = copy.deepcopy(values)
            if kind == "history_pcm": history["all_pcm_sha256"].append(synthetic[0]["pcm_sha256"])
            if kind == "history_person": history["canonical_people"].append(m.person(synthetic[0]))
            if kind == "unknown_train": train[0]["source_pcm_sha256"] = natural[23]["pcm_sha256"]
            if kind == "unknown_canary": natural[23]["pcm_sha256"] = next(iter(pcm))
            if kind == "reserve": reserve.add(m.person(natural[23]))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                m.reject_overlap(synthetic, natural, train, history, pcm, people, reserve)

    def test_ctc_blank_repeat_multiple_phrase_events_and_sample_clock(self):
        cases = (([], [], []), ([1, 1, 0, 1, 2, 2, 3, 0, 1, 2, 3], [1, 1, 2, 3, 1, 2, 3], [6, 10]),
                 ([0, 1, 0, 2, 0, 3], [1, 2, 3], [5]), ([1, 2, 0, 2, 3], [1, 2, 2, 3], []),
                 ([1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], [2, 5]))
        for labels, tokens, frames in cases:
            with self.subTest(labels=labels):
                actual = m.ctc_events(labels)
                self.assertEqual(actual, dict(tokens=tokens, event_frames=frames,
                    event_samples=[512 + 160 * frame for frame in frames], events=len(frames)))

    def test_each_forward_entry_consumed_even_without_saved_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = m.Ledger(Path(tmp))
            ledger.current = dict(key="original__1.0__B0", role="B0", source_id="original", gain="1.0")
            ledger.entered("B0")
            self.assertEqual(ledger.counts["B0"], 1)
            self.assertEqual(ledger.consumed[ledger.current["key"]], "consumed_unknown")
            with self.assertRaises(ValueError): ledger.entered("B0")
            ledger.saved(ledger.current["key"])
            with self.assertRaises(ValueError): ledger.entered("B0")
            self.assertEqual(ledger.counts["B0"], 1)
            self.assertEqual(len(list(Path(tmp).glob("*.json"))), 1)
            ledger.counts["B0"] = 1404
            ledger.current = dict(key="different", role="B0", source_id="another", gain="1.0")
            with self.assertRaises(ValueError): ledger.entered("B0")

    def test_complete_zip_sha_and_unsafe_members_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.zip"
            for name in ("SAFE.json", "../escape.json"):
                out = io.BytesIO()
                with zipfile.ZipFile(out, "w") as archive: archive.writestr(name, b"{}")
                path.write_bytes(out.getvalue())
                pin = dict(bytes=path.stat().st_size, sha256=m.sha(path.read_bytes()))
                if name == "SAFE.json":
                    with m.original_zip(path, pin) as archive: self.assertEqual(archive.read(name), b"{}")
                    with self.assertRaises(ValueError): m.original_zip(path, dict(pin, sha256="0" * 64))
                else:
                    with self.assertRaises(ValueError): m.original_zip(path, pin)


if __name__ == "__main__":
    unittest.main()
