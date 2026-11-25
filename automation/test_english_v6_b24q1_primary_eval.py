"""No model imports or network: B24Q1 primary boundary and consumption fixtures."""
import ast
import base64
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent
definition = importlib.util.spec_from_file_location("b24q1_primary_contract", HERE / "english_v6_b24q1_primary_eval.py")
primary = importlib.util.module_from_spec(definition)
definition.loader.exec_module(primary)


def row(index, cohort="synthetic", positive=False):
    result = dict(id="row-" + str(index), cohort=cohort, observed_class=None,
                  requested_class=2 if positive else 0, pcm_sha256=f"{index:064x}",
                  wav_sha256=f"{index + 1000:064x}", samples=16000,
                  source_group_id="group-" + str(index), role="development")
    if cohort == "synthetic":
        result.update(source_speaker_id="p" + str(index), cell_id="mero-fast" if positive else "request")
    else:
        result.update(speaker=str(index), source_id=str(index) + "-a", transcript="A plain sentence")
    return result


def label(original, unknown=False):
    value = dict(original, observed_class=None if unknown else original["requested_class"],
                 tier="unknown" if unknown else "silver")
    if original["cohort"] == "natural":
        value.update(exact_votes=3, full_wake_veto=False)
    return value


def policy():
    return dict(schema="english-v6-b24q1-source-policy-v1", experiment_id="B24Q1",
        proposal_sha256=primary.APPROVED_PROPOSAL_SHA, approval_sha256=primary.APPROVAL_SHA,
        orchestration_contract_sha256=primary.ORCHESTRATION_SHA, quarantine_helper_sha256="a"*64,
        observer_sha256="b"*64, asr_adapter_sha256="c"*64, parser_contract_sha256="d"*64,
        orchestration_implementation_sha256="e"*64)


def quarantine_fixture():
    originals = [row(i, positive=i<12) for i in range(96)] + [row(i, "natural") for i in range(96,120)]
    rows = originals[:118]
    decisions = [label(r, unknown=r["id"] in ("row-1", "row-2", "row-3")) for r in rows]
    for item in decisions:
        if item["id"] in ("row-1", "row-2"):item["reason"] = primary.QUARANTINE_REASON
    empty = {"row-1": {"whisper": {"fixture": True}, "parakeet": {"fixture": True}},
             "row-2": {"whisper": {"fixture": True}}}
    invalid = {p: sum(p in v for v in empty.values()) for p in primary.PROVIDERS}
    per = {p: dict(planned=118, attempted=118, returned=118, valid=118-invalid[p],
                   invalid_empty=invalid[p], fatal=0, unknown=0) for p in primary.PROVIDERS}
    screen = dict(rows=decisions, source_policy=policy(), source_rows=118, selected_original_slots=120,
        original_quality_rejections=2, coverage={p: v["valid"] for p,v in per.items()},
        completed_coverage=dict.fromkeys(primary.PROVIDERS,118), invalid_empty_coverage=invalid,
        quarantined_ids=sorted(empty), quarantine=empty, attrition={"fixture_only": True},
        partition=dict(pre_observer_rejected=2, quarantined_invalid_empty=2, five_valid_unqualified=1, qualified=115),
        native_slot_accounting=dict(per_provider=per, total={k:sum(v[k] for v in per.values()) for k in next(iter(per.values()))}, maximum_per_panel=600))
    return screen, rows, originals


class BoundaryTests(unittest.TestCase):
    def audit(self, fixture=None):
        screen, rows, originals = fixture or quarantine_fixture()
        return primary.source_audit(screen, rows, originals, policy(), json.dumps(screen).encode())

    def test_all120_fixed_n_and_multi_provider_union_remain_distinct(self):
        result = self.audit()
        self.assertEqual(result["observer_eligible_rows"],118)
        self.assertEqual(result["original_source_rows"],120)
        self.assertEqual(result["qualified_rows"],115)
        self.assertEqual(result["partition"]["quarantined_invalid_empty"],2)
        self.assertEqual(result["native_slot_accounting"]["total"]["invalid_empty"],3)
        self.assertEqual(result["native_slot_accounting"]["total"]["returned"],590)
        self.assertEqual(result["quarantined_kws_forwards"],0)

    def test_denominator_cannot_shrink_or_fill_the_unused600_ceiling(self):
        for mutate in (
            lambda s:s.update(source_rows=116),
            lambda s:s["native_slot_accounting"]["total"].update(planned=600),
            lambda s:s["completed_coverage"].update(parakeet=117),
            lambda s:s["native_slot_accounting"]["per_provider"]["qwen"].update(attempted=119),
            lambda s:s["native_slot_accounting"]["per_provider"]["qwen"].update(fatal=1)):
            f=quarantine_fixture();mutate(f[0])
            with self.assertRaises(ValueError):self.audit(f)

    def test_full_exposure_and_eligible_original_identity_are_required(self):
        s,rows,originals=quarantine_fixture()
        with self.assertRaises(ValueError):self.audit((s,rows,originals[:-1]))
        rows=copy.deepcopy(rows);rows[0]["pcm_sha256"]="f"*64
        with self.assertRaises(ValueError):self.audit((s,rows,originals))
        s,rows,originals=quarantine_fixture();s["partition"]["pre_observer_rejected"]=0
        with self.assertRaises(ValueError):self.audit((s,rows,originals))

    def test_quarantine_cannot_hide_empty_payloads_or_enter_quality(self):
        for mutate in (
            lambda s:s["quarantine"]["row-1"].pop("parakeet"),
            lambda s:s["quarantine"].pop("row-2"),
            lambda s:s["rows"][1].update(observed_class=2,tier="silver"),
            lambda s:s["rows"][1].update(reason="ordinary_unknown"),
            lambda s:s["quarantined_ids"].append("row-3")):
            f=quarantine_fixture();mutate(f[0])
            with self.assertRaises(ValueError):self.audit(f)

    def test_approved_policy_is_explicit_and_version_bound(self):
        primary.validate_source_policy(policy())
        for key in ("proposal_sha256","approval_sha256","orchestration_contract_sha256"):
            value=policy();value[key]="f"*64
            with self.assertRaises(ValueError):primary.validate_source_policy(value)
        value=policy();value.pop("orchestration_implementation_sha256")
        with self.assertRaises(ValueError):primary.validate_source_policy(value)

    def test_api_preserves_control_and_non_utf8_bytes_without_terminal_output(self):
        raw = b"PK\x03\x04\x1b[31m\xff\x00\x80\r\n"
        with patch.object(primary.subprocess, "run") as run:
            run.return_value.stdout = raw
            self.assertEqual(primary.gh("actions/artifacts/123/zip"), raw)
            run.assert_called_once_with(["gh", "api", "repos/" + primary.REPO + "/actions/artifacts/123/zip",
                                         "--allow-escape-sequences"], stdout=primary.subprocess.PIPE,
                                        stderr=primary.subprocess.PIPE, check=True)



    def test_original_proof_zip_larger_than_contents_limit_uses_git_blob(self):
        raw = b"original-failure-proof" * 65000
        blob = primary.git_blob(raw)
        tree = dict(tree=[dict(path="research/original-producer.zip", type="blob", mode="100644", sha=blob)])
        response = dict(sha=blob, encoding="base64", size=len(raw), content=base64.b64encode(raw).decode())
        with patch.object(primary, "gh", return_value=json.dumps(response).encode()) as fetch:
            self.assertEqual(primary.remote_file("research/original-producer.zip", "a" * 40, primary.sha(raw), tree), raw)
            fetch.assert_called_once_with("git/blobs/" + blob)
        response["size"] -= 1
        with patch.object(primary, "gh", return_value=json.dumps(response).encode()):
            with self.assertRaises(ValueError): primary.remote_file("research/original-producer.zip", "a" * 40, primary.sha(raw), tree)

    def test_inert_is_before_dependencies_or_any_remote_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = Path(temporary) / "inert.json"
            spec.write_text('{"scoring_allowed":false}')
            with patch.object(primary, "SPEC", spec), patch.object(primary, "helpers", side_effect=AssertionError("helper")), \
                 patch.object(primary, "gh", side_effect=AssertionError("network")):
                with self.assertRaisesRegex(ValueError, "INERT"):
                    primary.fetch(Path(temporary) / "inputs", "a" * 40)
                with self.assertRaisesRegex(ValueError, "INERT"):
                    primary.preflight(Path(temporary) / "inputs", "a" * 40)
                with self.assertRaisesRegex(ValueError, "INERT"):
                    primary.score(Path(temporary) / "inputs", Path(temporary) / "result", "a" * 40)
                self.assertFalse((Path(temporary) / "inputs").exists())
                self.assertFalse((Path(temporary) / "result").exists())

    def test_source_join_preserves_unknown_and_natural_identity(self):
        originals = [row(1, positive=True), row(2), row(3, "natural")]
        decisions = [label(originals[0]), label(originals[1], unknown=True), label(originals[2])]
        selected = primary.exact_admitted_join(originals, decisions)
        self.assertEqual([value["id"] for value in selected], ["row-1", "row-3"])
        self.assertEqual(selected[1]["cell_id"], "natural-speech")
        changed = copy.deepcopy(decisions)
        changed[0]["source_group_id"] = "substituted"
        with self.assertRaises(ValueError):
            primary.exact_admitted_join(originals, changed)
        with self.assertRaises(ValueError):
            primary.exact_admitted_join(originals, list(reversed(decisions)))

    def test_natural_veto_and_boolean_class_are_rejected(self):
        natural = row(5, "natural")
        actual = label(natural)
        actual["full_wake_veto"] = True
        with self.assertRaises(ValueError):
            primary.exact_admitted_join([natural], [actual])
        ordinary = row(6)
        actual = label(ordinary)
        actual["observed_class"] = False
        with self.assertRaises(ValueError):
            primary.exact_admitted_join([ordinary], [actual])

    def test_full_source_rejects_are_in_overlap_denominator(self):
        originals = [dict(row(i + 100), source_speaker_id="p" + str(2000 + i // 4)) for i in range(96)]
        originals += [dict(row(i + 500, "natural"), speaker=str(3000 + i // 2)) for i in range(24)]
        training = [row(i + 1000) for i in range(88)]
        history = dict(all_pcm_sha256=[], all_source_groups=[], canonical_people=[])
        primary.reject_overlap(originals, training, history, {"99999"})
        leaked = copy.deepcopy(history)
        leaked["all_pcm_sha256"] = [originals[-1]["pcm_sha256"]]
        with self.assertRaises(ValueError):
            primary.reject_overlap(originals, training, leaked, {"99999"})
        with self.assertRaises(ValueError):
            primary.reject_overlap(originals, training, history, {originals[-1]["speaker"]})

    def test_consumption_precedes_forward_and_cannot_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = primary.Ledger(Path(temporary))
            ledger.current = dict(key="row1__1.0__control", role="control", source_id="row1", gain="1.0")
            ledger.entered("control")
            saved = json.loads((Path(temporary) / "row1__1.0__control.json").read_text())
            self.assertEqual(saved["status"], "consumed_unknown")
            self.assertEqual(ledger.counts["control"], 1)
            with self.assertRaises(ValueError):
                ledger.entered("control")
            ledger.saved(ledger.current["key"])
            with self.assertRaises(ValueError):
                ledger.entered("control")

    def test1440_hard_limit_is_before_new_consumption(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = primary.Ledger(Path(temporary))
            ledger.counts = dict.fromkeys(primary.ARMS, 360)
            ledger.current = dict(key="forbidden", role="B0", source_id="row1", gain="1.0")
            with self.assertRaisesRegex(ValueError, "hard ceiling"):
                ledger.entered("B0")
            self.assertEqual(sum(ledger.counts.values()), 1440)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_original_raw_archive_safety(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary) / "unsafe.zip"
            data = io.BytesIO()
            with zipfile.ZipFile(data, "w") as archive:
                archive.writestr("../outside", b"data")
            p.write_bytes(data.getvalue())
            pin = dict(id=1, run_id=1, run_head="a" * 40,
                       bytes=p.stat().st_size, sha256=primary.sha(p.read_bytes()))
            with self.assertRaisesRegex(ValueError, "safe original archive member"):
                primary.original_zip(p, pin)

    def test_only_explicit_source_receives_composite_envelope_before_file_read(self):
        roles = [None, *primary.INPUT_KEYS, "train_producer", "source_producer", "result", "producer", "consumed",
                 *("native_" + provider for provider in primary.PROVIDERS)]
        for role in roles:
            maximum = 768_000_000 if role == "source" else 128_000_000
            self.assertEqual(primary.archive_limit(role), maximum)
            pin = dict(id=1, run_id=1, run_head="a" * 40, bytes=maximum, sha256="b" * 64)
            primary.valid_artifact(pin, role=role)
            with patch.object(Path, "read_bytes", side_effect=AssertionError("must reject before reading")):
                with self.assertRaises(ValueError):
                    primary.original_zip(Path("absent.zip"), dict(pin, bytes=maximum + 1), role=role)
        with self.assertRaises(ValueError):
            primary.archive_limit("arbitrary")

    def test_large_expanded_source_is_bounded_and_other_roles_stay_small(self):
        # The original STOP ZIPs together exceed128MB. This harmless compressed
        # fixture exercises the real expanded-size boundary without models/data.
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "composite.zip"
            manifest = dict(schema=primary.SOURCE_SCHEMA, panel="primary",
                            assessment_sha256=primary.EVALUATION_ASSESSMENT)
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("MANIFEST.json", json.dumps(manifest))
                with archive.open("ORIGINAL_B24_STOP/fixture.bin", "w") as member:
                    remaining = 128_000_001
                    block = b"\0" * (1024 * 1024)
                    while remaining:
                        count = min(remaining, len(block))
                        member.write(block[:count])
                        remaining -= count
            pin = dict(id=1, run_id=1, run_head="a" * 40,
                       bytes=path.stat().st_size, sha256=primary.sha(path.read_bytes()))
            with primary.original_zip(path, pin, role="source") as archive:
                self.assertGreater(sum(item.file_size for item in archive.infolist()), 128_000_000)
            for role in (None, "pair", "source_producer", "native_xlsr"):
                with self.assertRaisesRegex(ValueError, "bounded unique archive members"):
                    primary.original_zip(path, pin, role=role)

    def test_source_envelope_cannot_be_used_for_another_panel_or_assessment(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.zip"
            original = dict(schema=primary.SOURCE_SCHEMA, panel="primary",
                            assessment_sha256=primary.EVALUATION_ASSESSMENT)
            for key, value in (("schema", "other-source"), ("panel", "replication"),
                               ("assessment_sha256", primary.TRAINING_ASSESSMENT)):
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("MANIFEST.json", json.dumps(dict(original, **{key: value})))
                pin = dict(id=1, run_id=1, run_head="a" * 40,
                           bytes=path.stat().st_size, sha256=primary.sha(path.read_bytes()))
                with self.assertRaisesRegex(ValueError, "composite envelope"):
                    primary.original_zip(path, pin, role="source")

    def test_ctc_event_clocks_keep_original_greedy_rule(self):
        result = primary.ctc_events([0, 1, 1, 0, 2, 0, 3, 3, 0])
        self.assertEqual(result, dict(tokens=[1, 2, 3], event_frames=[6], event_samples=[1472], events=1))

    def test_first_public_single_parent_event_required(self):
        env = dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY=primary.REPO, GITHUB_EVENT_NAME="push",
                   GITHUB_REF=primary.BRANCH, GITHUB_RUN_ATTEMPT="1", GITHUB_SHA="b" * 40, GITHUB_RUN_ID="123")
        event = dict(before="a" * 40, after="b" * 40, ref=primary.BRANCH, forced=False, deleted=False,
                     repository=dict(full_name=primary.REPO, private=False))
        spec = dict(prepared_parent=None, parent_authority="reviewed_workflow")
        identity = primary.guard(spec, "a" * 40, env, event, "b" * 40, "a" * 40)
        self.assertEqual(identity["run_id"], "123")
        for key, value in (("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_EVENT_NAME", "workflow_dispatch")):
            with self.assertRaises(ValueError):
                primary.guard(spec, "a" * 40, dict(env, **{key: value}), event, "b" * 40, "a" * 40)

    def test_real_git_blob_required_for_hidden_producer_file(self):
        raw = b"hidden source metadata\n"
        tree = dict(tree=[dict(path=".github/triggers/a.json", type="blob", mode="100644", sha=primary.git_blob(raw))])
        self.assertEqual(primary.tree_member(tree, ".github/triggers/a.json"), primary.git_blob(raw))
        with self.assertRaises(ValueError):
            primary.tree_member(tree, "missing")

    def test_workflow_retains_all_hidden_original_producer_files(self):
        template = primary.WORKFLOW_TEMPLATE.read_text()
        self.assertEqual(template.count(primary.PARENT_TOKEN), 2)
        self.assertIn("include-hidden-files: true", template)
        self.assertIn("primary_producer/", template)
        self.assertIn("source_producer/", template)
        self.assertIn("english-v6-b24q1-primary-consumed", template)
        self.assertNotIn("workflow_dispatch", template)


    def test_b24q1_preserves_numerical_rules_and_historical_control_scope(self):
        source = (HERE / "english_v6_b21_fixed_mirror_primary_eval.py").read_bytes()
        def bodies(raw):
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(raw).body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        old, new = bodies(source), bodies(Path(primary.__file__).read_bytes())
        for name in ("ctc_events", "Ledger", "source_counts", "score_identity", "validate_scored_join"):
            self.assertEqual(old[name], new[name], name)
        self.assertEqual(primary.STEMS["control"], "control/models/mero43-b23-challenger192")
        self.assertEqual(primary.STEMS["challenger"], "challenger/models/mero43-b24-challenger192")
        self.assertEqual(primary.GAINS, ("1.0", "0.75", "0.5"))
        self.assertEqual(primary.MAX_FORWARDS, 1440)
        self.assertEqual(primary.QUALITY_CODE, primary.sha((HERE / "english_v6_b14_decision.py").read_bytes()))
        original = json.loads((HERE / "english_v6_b23_primary_inputs.json").read_bytes())
        self.assertEqual(primary.BASELINES, {name: original["inputs"][name] for name in ("B0", "B9")})
        self.assertEqual(primary.INITIAL_PT, primary.CONTROL_PT)
        self.assertNotEqual(primary.INITIAL_PT, primary.B0_PT)

    def test_score_preserves_original_fixed_numerical_execution(self):
        def score(path):
            return next(node for node in ast.parse(path.read_bytes()).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'score')
        original = score(HERE / "english_v6_b24r1_primary_eval.py")
        current = score(Path(primary.__file__))
        additions = {}
        for node in ast.walk(current):
            if isinstance(node, ast.Call):
                for field in list(node.keywords):
                    if field.arg in ('source_audit', 'B24R1_original_source_stop_preserved',
                                     'original_whole_panel_failure_contract_unchanged'):
                        additions[field.arg] = ast.dump(field.value, include_attributes=False)
                        node.keywords.remove(field)
        self.assertEqual(additions, {
            'source_audit': ast.dump(ast.parse('admission["source_audit"]', mode='eval').body, include_attributes=False),
            'B24R1_original_source_stop_preserved': 'Constant(value=True)',
            'original_whole_panel_failure_contract_unchanged': 'Constant(value=False)'})
        self.assertEqual(ast.dump(current, include_attributes=False), ast.dump(original, include_attributes=False))

    def test_registration_inert_until_twelve_original_artifacts_bound(self):
        spec = json.loads(primary.SPEC.read_bytes())
        self.assertIsNone(spec["prepared_parent"])
        self.assertTrue(all(spec[key] is False for key in primary.FLAGS))
        self.assertFalse(spec["same_job_matched_control"])
        self.assertFalse(spec["historical_control_retrained"])
        self.assertEqual(spec["model_pt_sha256"]["control"], primary.CONTROL_PT)
        self.assertEqual(set(primary.archive_pins(spec)), primary.INPUT_KEYS | {"native_" + p for p in primary.PROVIDERS} | {"train_producer", "source_producer"})
        if spec["scoring_allowed"]:
            self.assertEqual(primary.protocol(), spec)
        else:
            self.assertIsNone(spec["producer_manifest_sha256"])
            self.assertIsNone(spec["model_pt_sha256"]["challenger"])
            self.assertTrue(all(pin is None for pin in spec["upstream_producers"].values()))

    def test_new_train_receipt_rejects_false_matched_control_and_legacy_pair(self):
        spec = json.loads(primary.SPEC.read_bytes())
        spec["model_pt_sha256"]["challenger"] = primary.CANDIDATE_PT
        design = json.loads(primary.TRAIN_DESIGN.read_bytes())
        control = dict(design["historical_control"], output_stem=primary.STEMS["control"])
        challenger = dict(stem=primary.STEMS["challenger"], optimizer_updates=192, parent_pt_sha256=primary.INITIAL_PT,
            optimization_forwards=1844, checkpoint_reload_forwards=2, onnx_export_forwards=1, actual_new_model_forwards=1847,
            optimizer_initialization='fresh_AdamW', restored_optimizer_state=False, exact_optimizer_continuation=False,
            old_B23_optimizer_updates_reexecuted=0)
        for arm, value in (("control", control), ("challenger", challenger)):
            base = primary.STEMS[arm].split("/")[-1]
            value["files"] = {base + "." + ext: dict(bytes=20000, sha256=spec["model_pt_sha256"][arm] if ext == "pt" else "f" * 64)
                              for ext in ("pt", "npz", "onnx")}
        receipt = dict(schema=primary.PAIR_SCHEMA, status="TRAINED_B23_WARM_START_CANDIDATE_REQUIRES_NEW_INDEPENDENT_PRIMARY",
            actual_optimizer_updates=192, actual_new_model_forwards=1847, new_control_optimizer_updates=0,
            new_control_forwards=0, new_teacher_forwards=0, reused_ordinary_teacher_arrays=264,
            original_retention_conditions=120, union_retention_conditions=122, dev_cal_locked_forwards=0,
            new_tts_asr_calls=0, promotion_allowed=False, historical_numerical_replay_claimed=False,
            same_job_matched_control=False, assessment_sha256=spec["training_assessment_sha256"],
            design_sha256=spec["training_design_sha256"], recipe_sha256=spec["training_design_sha256"],
            model_pt_sha256=spec["model_pt_sha256"], control=control, challenger=challenger,
            phase_initialization_pt_sha256=primary.INITIAL_PT, optimizer_initialization='fresh_AdamW',
            restored_optimizer_state=False, exact_optimizer_continuation=False, old_B23_optimizer_updates_reexecuted=0,
            **dict.fromkeys(("common_draw_sha256", "source_sha256", "union_evidence_sha256", "old120_references_sha256", "assessment_binding_sha256"), "e" * 64))
        primary.validate_pair_receipt(receipt, spec)
        for key, value in (("same_job_matched_control", True), ("actual_optimizer_updates", 384),
                           ("schema", "english-v6-b21-positive-competing-margin-pair-v1"), ("new_control_forwards", 1),
                           ("phase_initialization_pt_sha256", primary.B0_PT), ("optimizer_initialization", "restored_AdamW"),
                           ("restored_optimizer_state", True), ("exact_optimizer_continuation", True),
                           ("old_B23_optimizer_updates_reexecuted", 192)):
            with self.assertRaises(ValueError): primary.validate_pair_receipt(dict(receipt, **{key: value}), spec)
        for key, value in (("parent_pt_sha256", primary.B0_PT), ("restored_optimizer_state", True),
                           ("optimizer_initialization", "restored_AdamW"), ("exact_optimizer_continuation", True),
                           ("old_B23_optimizer_updates_reexecuted", 192)):
            changed = copy.deepcopy(receipt)
            changed['challenger'][key] = value
            with self.assertRaises(ValueError): primary.validate_pair_receipt(changed, spec)

    def test_flat_training_evidence_requires_actual_prior_failure_binding(self):
        # Isolated adapter fixture: receipt-budget validation is tested above.
        # These in-memory dummy bytes are never an execution or quality grant.
        parent, head = 'a' * 40, 'b' * 40
        spec = json.loads(primary.SPEC.read_bytes())
        spec['inputs']['pair'] = dict(run_id=123, run_head=head)
        meta = dict(run=dict(event='push', head_branch=primary.TRAIN_BRANCH, path=primary.TRAIN_WORKFLOW),
                    commit=dict(parents=[dict(sha=parent)], files=[dict(filename=primary.TRAIN_WORKFLOW)]))
        lock = dict(original_result_sha256='c' * 64)
        opening = dict(schema='english-v6-b24-after-original-b23-no-promotion-v1',
                       condition='only_after_actual_complete_B23_primary_NO_PROMOTION', enabled=True,
                       baseline_challenger_pt_sha256=primary.CONTROL_PT, quality_lock=lock)
        prior = dict(schema='english-v6-b24-original-b23-no-promotion-v1', panel='primary', panel_gate='NO_PROMOTION',
                     model_pt_sha256=dict(challenger=primary.CONTROL_PT), original_result_sha256=lock['original_result_sha256'],
                     original_lock_sha256=primary.sha(json.dumps(lock, sort_keys=True, separators=(',', ':')).encode()),
                     all_original_arrays_verified=True, original_source_admission_recomputed=True,
                     original_producer_git_verified=True, authorization_granted=False,
                     replication_open_allowed=False, candidate_engineering_validation_allowed=False)
        train = dict(schema='english-v6-b24-authenticated-train-source-v1', originals=88, synthetic_originals=79,
                     natural_originals=9, natural_readers=8, ordinary_teachers_reused=264,
                     same_common_draw=True, prospective_assessment_frozen=True, new_model_forwards=0,
                     new_source_calls=0, new_dev_model_forwards=0, dev_rows_used_for_training=0,
                     base_train=[None] * 79, natural_train=[None] * 9, positive_retention=[None] * 120,
                     retention_additions=[None] * 2, previous_failure=prior, saved_B23_dev_readback_for_authorization=True)
        binding = json.loads((HERE / 'english_v6_b24_assessment_binding.json').read_bytes())
        origin = json.loads((HERE / 'english_v6_b24_panels/b23-source-origin.json').read_bytes())
        members = origin['original_metadata_sha256']
        train['initialization'] = json.loads(primary.TRAIN_DESIGN.read_bytes())['initialization']
        train['prospective_history_reexport'] = dict(schema='english-v6-b24-original-source-metadata-reexport-v1',
            status='VERIFIED_ORIGINAL_METADATA_REEXPORT', source_artifact=origin['source_artifact'],
            original_metadata_sha256=members, original_manifest_sha256=members['MANIFEST.json'],
            original_history_sha256=members['HISTORY.json'],
            published_files={name: binding['files'][name] for name in binding['files']
                             if name.startswith('automation/english_v6_b24_panels/b23-source-')},
            original_rows=120, prior_panel_people_excluded=72, complete_history_validated='PASS_METADATA_ONLY',
            original_source_identity_verified=True, whole_zip_sha256_verified=True, metadata_member_crc_verified=True,
            audio_members_interpreted=0, observer_receipts_read=0, model_calls=0, new_source_calls=0,
            quality_results_read=False, scientific_execution_authorized=False)
        train['previous_source_metadata_members'] = {'ORIGINAL_B23_SOURCE_' + name:
            dict(bytes=size, sha256=members[name], original_member=name)
            for name, size in (('MANIFEST.json', 480267), ('HISTORY.json', 240561))}
        receipt = dict(identity=dict(run_id='123', run_head=head, run_attempt=1,
                                     run_branch='refs/heads/' + primary.TRAIN_BRANCH, prepared_parent=parent),
                       original_B23_NO_PROMOTION_reauthenticated=True, prefix_blank_reduction='amax')
        base = {'DESIGN.json': primary.TRAIN_DESIGN.read_bytes(),
                'ASSESSMENT_BINDING.json': (HERE / 'english_v6_b24_assessment_binding.json').read_bytes(),
                'COMMON_DRAW.json': b'{}', 'UNION_EVIDENCE.json': b'{}', 'OLD120_POSITIVE_REFERENCES.json': b'{}'}
        for arm in ('control', 'challenger'):
            receipt[arm] = dict(files={})
            for extension in ('pt', 'npz', 'onnx'):
                name = primary.STEMS[arm] + '.' + extension
                base[name] = ('dummy-fixture-' + name).encode()
                receipt[arm]['files'][Path(name).name] = dict(bytes=len(base[name]), sha256=primary.sha(base[name]))
        def check(current_train, current_opening, current_receipt):
            payload = dict(base, **{'SOURCE.json': json.dumps(current_train).encode(),
                                   'OPENING.json': json.dumps(current_opening).encode()})
            value = copy.deepcopy(current_receipt)
            for name, field in (('SOURCE.json', 'source_sha256'), ('COMMON_DRAW.json', 'common_draw_sha256'),
                                ('UNION_EVIDENCE.json', 'union_evidence_sha256'), ('OLD120_POSITIVE_REFERENCES.json', 'old120_references_sha256'),
                                ('ASSESSMENT_BINDING.json', 'assessment_binding_sha256'), ('OPENING.json', 'opening_sha256')):
                value[field] = primary.sha(payload[name])
            payload['RESULT.json'] = json.dumps(value).encode()
            data = io.BytesIO()
            with zipfile.ZipFile(data, 'w') as archive:
                for name, raw in payload.items(): archive.writestr(name, raw)
            with zipfile.ZipFile(io.BytesIO(data.getvalue())) as archive, patch.object(primary, 'validate_pair_receipt') as budget:
                result = primary.validate_training_archive(archive, spec, meta)
                budget.assert_called_once_with(value, spec)
                return result
        self.assertEqual(check(train, opening, receipt), train)
        for key, value in (('panel_gate', 'PASS'), ('all_original_arrays_verified', False),
                           ('original_result_sha256', 'd' * 64), ('authorization_granted', True)):
            changed = copy.deepcopy(train)
            changed['previous_failure'][key] = value
            with self.assertRaises(ValueError): check(changed, opening, receipt)
        with self.assertRaises(ValueError): check(train, dict(opening, enabled=False), receipt)
        with self.assertRaises(ValueError): check(train, opening, dict(receipt, prefix_blank_reduction='mean'))
        for key, value in (('original_rows', 119), ('prior_panel_people_excluded', 36),
                           ('whole_zip_sha256_verified', False), ('quality_results_read', True),
                           ('original_manifest_sha256', 'f' * 64), ('scientific_execution_authorized', True)):
            changed = copy.deepcopy(train)
            changed['prospective_history_reexport'][key] = value
            with self.assertRaises(ValueError): check(changed, opening, receipt)
        changed = copy.deepcopy(train)
        changed['initialization']['pt_sha256'] = primary.B0_PT
        with self.assertRaises(ValueError): check(changed, opening, receipt)
        changed = copy.deepcopy(train)
        changed['previous_source_metadata_members']['ORIGINAL_B23_SOURCE_MANIFEST.json']['sha256'] = 'f' * 64
        with self.assertRaises(ValueError): check(changed, opening, receipt)

    def test_every_original_dynamic_source_member_is_required(self):
        parent, head = "a" * 40, "b" * 40
        template = ("parent: " + primary.SOURCE_PARENT_TOKEN + "\nagain: " + primary.SOURCE_PARENT_TOKEN + "\n").encode()
        files = {primary.SOURCE_TEMPLATE: template, primary.SOURCE_WORKFLOW: template.replace(primary.SOURCE_PARENT_TOKEN.encode(), parent.encode()),
                 "automation/dynamically_loaded.py": b"value = 1\n"}
        pins = {name: dict(mode="100644", git_blob_sha1=primary.git_blob(raw), bytes=len(raw), sha256=primary.sha(raw))
                for name, raw in files.items()}
        manifest = dict(schema="english-v6-b24q1-complete-source-producer-v1", run_id="123", run_head=head,
            run_attempt=1, run_branch=primary.SOURCE_BRANCH, prepared_parent=parent, runtime_workflow=primary.SOURCE_WORKFLOW,
            git_tree="c" * 40, files=pins, source_code_reads_only=True, new_model_forwards=0, optimizer_updates=0, new_source_calls=0)
        meta = dict(run=dict(id=123, head_sha=head, path=primary.SOURCE_WORKFLOW),
            commit=dict(sha=head, commit=dict(tree=dict(sha="c" * 40)), parents=[dict(sha=parent)], files=[dict(filename=primary.SOURCE_WORKFLOW)]),
            tree=dict(sha="c" * 40, truncated=False, tree=[dict(path=name, mode="100644", type="blob", sha=pin["git_blob_sha1"]) for name, pin in pins.items()]))
        def archive(exclude=None):
            output=io.BytesIO()
            with zipfile.ZipFile(output,"w") as z:
                z.writestr("PRODUCER.json", json.dumps(manifest))
                for name, raw in files.items():
                    if name != exclude: z.writestr("repository/" + name, raw)
            return zipfile.ZipFile(io.BytesIO(output.getvalue()))
        with archive() as z: self.assertEqual(primary.snapshot_contract(z,"source",meta), manifest)
        with archive("automation/dynamically_loaded.py") as z:
            with self.assertRaisesRegex(ValueError,"entire original producer"):
                primary.snapshot_contract(z,"source",meta)

    def test_train_snapshot_requires_all_306_and_exact_warm_start_workflow(self):
        # Synthetic Git/ZIP fixture: validates inventory and activation shape,
        # and cannot serve as an actual artifact or authorize execution.
        parent, head = 'a' * 40, 'b' * 40
        previous = json.loads((HERE / 'english_v6_b23_source_training_producer.json').read_bytes())
        self.assertEqual(len(previous['files']), 306)
        names = set(previous['files']) | set(json.loads((HERE / 'english_v6_b24_assessment_binding.json').read_bytes())['files'])
        names |= {'automation/english_v6_b24_b23_warm_start.py', 'automation/english_v6_b24_b23_warm_start_design.json',
            'automation/english_v6_b24_producer.py', 'automation/english_v6_b24_b23_failure.py',
            'automation/english_v6_b24_opening.json', 'automation/english_v6_b24_assessment_binding.json',
            'automation/english_v6_b24_training_workflow_template.yml', primary.TRAIN_WORKFLOW}
        files = {name: ('fixture: ' + name).encode() for name in names}
        template = (HERE / 'english_v6_b24_training_workflow_template.yml').read_bytes()
        files['automation/english_v6_b24_training_workflow_template.yml'] = template
        files[primary.TRAIN_WORKFLOW] = template.replace(b'__B24_REVIEWED_B23_WARM_START_PARENT__', parent.encode())
        def verify(payloads, count):
            pins = {name: dict(mode='100644', git_blob_sha1=primary.git_blob(raw), bytes=len(raw), sha256=primary.sha(raw))
                    for name, raw in payloads.items()}
            manifest = dict(schema='english-v6-b24-complete-git-producer-v1',
                identity=dict(run_id='123', run_head=head, run_attempt=1,
                              run_branch='refs/heads/' + primary.TRAIN_BRANCH, prepared_parent=parent),
                runtime_workflow=primary.TRAIN_WORKFLOW, git_tree='c' * 40, files=pins,
                source_code_reads_only=True, new_model_forwards=0, optimizer_updates=0, new_source_calls=0,
                recipe_sha256=primary.TRAINING_DESIGN, original_b23_dependency_files=count)
            meta = dict(run=dict(id=123, head_sha=head, path=primary.TRAIN_WORKFLOW),
                commit=dict(sha=head, commit=dict(tree=dict(sha='c' * 40)), parents=[dict(sha=parent)],
                            files=[dict(filename=primary.TRAIN_WORKFLOW)]),
                tree=dict(sha='c' * 40, truncated=False,
                          tree=[dict(path=name, mode='100644', type='blob', sha=pin['git_blob_sha1']) for name, pin in pins.items()]))
            output = io.BytesIO()
            with zipfile.ZipFile(output, 'w') as archive:
                archive.writestr('PRODUCER.json', json.dumps(manifest))
                for name, raw in payloads.items(): archive.writestr('repository/' + name, raw)
            with zipfile.ZipFile(io.BytesIO(output.getvalue())) as archive:
                return primary.snapshot_contract(archive, 'train', meta)
        self.assertEqual(verify(files, 306)['original_b23_dependency_files'], 306)
        with self.assertRaisesRegex(ValueError, 'all actual306'):
            verify(files, 264)
        missing = dict(files)
        del missing['automation/english_v6_b23_prefix_peak.py']
        with self.assertRaisesRegex(ValueError, 'every actual B23 dependency'):
            verify(missing, 306)
        changed = dict(files)
        changed[primary.TRAIN_WORKFLOW] = template.replace(b'__B24_REVIEWED_B23_WARM_START_PARENT__', b'd' * 40)
        with self.assertRaisesRegex(ValueError, 'workflow-only activation'):
            verify(changed, 306)


if __name__ == "__main__":
    unittest.main()
