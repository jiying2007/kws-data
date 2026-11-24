"""B18 boundaries using offline fixtures only; no model or external data calls."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b18_primary_eval as subject


class Archive:
    def __init__(self, members=None):
        self.members = members or {}

    def read(self, name):
        return self.members[name]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FrozenAssessment(unittest.TestCase):
    def test_registered_models_and_old_quality_are_frozen(self):
        doc = subject.protocol()
        self.assertFalse(doc["promotion_allowed"])
        self.assertFalse(doc["training_allowed"])
        self.assertFalse(doc["cal_locked_access"])
        self.assertEqual(doc["source_gate"], subject.GATE)
        self.assertEqual(doc["gains"], list(subject.old.GAINS))
        self.assertEqual(doc["training_assessment_sha256"], subject.old.ASSESSMENT_SHA)
        self.assertFalse(subject.old.protocol()["scoring_allowed"])
        if not doc["scoring_allowed"]:
            self.assertIsNone(doc["inputs"]["synth_screen"])

    def test_cannot_substitute_model_training_or_natural_evidence(self):
        original = subject.protocol()
        for field, wrong in (("training_assessment_sha256", "0" * 64),
                             ("threshold_search", True),
                             ("maximum_model_forwards", 1500),
                             ("replication_opened", True),
                             ("model_pt_sha256", dict(original["model_pt_sha256"], challenger="0" * 64))):
            changed = copy.deepcopy(original)
            changed[field] = wrong
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "inputs.json"
                path.write_text(json.dumps(changed))
                with patch.object(subject, "SPEC", path), self.assertRaises(ValueError):
                    subject.protocol()
        changed = copy.deepcopy(original)
        changed["inputs"]["natural_screen"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inputs.json"
            path.write_text(json.dumps(changed))
            with patch.object(subject, "SPEC", path), self.assertRaises(ValueError):
                subject.protocol()

    def test_inert_fetch_creates_nothing_and_cannot_call_network(self):
        doc = copy.deepcopy(subject.protocol())
        doc["scoring_allowed"] = False
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "unopened"
            with patch.object(subject, "guard", return_value={}), \
                 patch.object(subject, "protocol", return_value=doc), \
                 patch.object(subject.prior_replica.opening, "gh", side_effect=AssertionError("network forbidden")):
                with self.assertRaises(ValueError):
                    subject.fetch(dest, "a" * 40)
            self.assertFalse(dest.exists())

    def test_source_shortfall_has_zero_model_calls_and_no_torch_import(self):
        doc = copy.deepcopy(subject.protocol())
        doc["scoring_allowed"] = True
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result"
            with patch.object(subject, "guard", return_value={}), \
                 patch.object(subject, "protocol", return_value=doc), \
                 patch.object(subject.old.common, "opened", return_value=Archive()), \
                 patch.object(subject, "verified_synthetic", return_value=([], dict(synthetic_positive=11, synthetic_ordinary=24, synthetic_paired=8), [])), \
                 patch.object(subject, "verified_natural", return_value=([], dict(natural_originals=21, natural_speakers=11), [])), \
                 patch.object(subject.old, "snapshots", side_effect=AssertionError("model load forbidden")), \
                 patch.dict(sys.modules, {"torch": None}):
                subject.score(Path(tmp), output, "a" * 40)
            result = json.loads((output / "RESULT.json").read_bytes())
            self.assertEqual(result["panel_gate"], "INSUFFICIENT_SOURCE")
            self.assertEqual(result["actual_forwards"], 0)
            self.assertEqual(result["saved_array_count"], 0)
            self.assertFalse(result["replication_open_allowed"])
            self.assertEqual(sorted(p.name for p in output.iterdir()), ["RESULT.json"])


class ExecutionGuard(unittest.TestCase):
    def fixture(self):
        parent, head = "a" * 40, "b" * 40
        env = dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY=subject.REPO,
                   GITHUB_EVENT_NAME="push", GITHUB_REF=subject.BRANCH,
                   GITHUB_RUN_ATTEMPT="1", GITHUB_RUN_ID="123", GITHUB_SHA=head)
        event = dict(before=parent, after=head, ref=subject.BRANCH, forced=False, deleted=False,
                     repository=dict(private=False, full_name=subject.REPO))
        return parent, env, event

    def test_exact_first_public_child_commit(self):
        parent, env, event = self.fixture()
        self.assertEqual(subject.guard(parent, env, event, parent)["run_head"], env["GITHUB_SHA"])
        for field, bad in (("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_EVENT_NAME", "workflow_dispatch"),
                           ("GITHUB_REF", "refs/heads/main"), ("GITHUB_REPOSITORY", "other/repo")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                subject.guard(parent, dict(env, **{field: bad}), event, parent)
        for field, bad in (("before", "0" * 40), ("forced", True), ("deleted", True),
                           ("repository", dict(private=True, full_name=subject.REPO))):
            with self.subTest(field=field), self.assertRaises(ValueError):
                subject.guard(parent, env, dict(event, **{field: bad}), parent)


class AllOriginalBoundaries(unittest.TestCase):
    def fixtures(self):
        synthetic = [dict(id="s" + str(i), pcm_sha256="s-pcm" + str(i),
                          source_group_id="s-group" + str(i // 4), source_speaker_id="p" + str(1000 + i // 4))
                     for i in range(96)]
        natural = [dict(id="n" + str(i), pcm_sha256="n-pcm" + str(i),
                        source_group_id="n-group" + str(i // 2), speaker=str(2000 + i // 2))
                   for i in range(24)]
        training = dict(base_train=[dict(id=str(i), group="t-group" + str(i), pcm_sha256="t-pcm" + str(i))
                                    for i in range(79)],
                        natural_train=[dict(id=str(i), source_group_id="t-group" + str(i),
                                            source_pcm_sha256="t-pcm" + str(i)) for i in range(79, 88)])
        return synthetic, natural, training

    def test_unknown_original_cannot_be_removed_to_hide_train_overlap(self):
        synthetic, natural, training = self.fixtures()
        subject.validate_originals(synthetic, natural, training)
        with self.assertRaises(ValueError):
            subject.validate_originals(synthetic, natural[:-1], training)
        natural[-1]["pcm_sha256"] = "t-pcm87"
        with self.assertRaises(ValueError):
            subject.validate_originals(synthetic, natural, training)

    def test_cross_cohort_canonical_person_and_duplicate_pcm_rejected(self):
        synthetic, natural, training = self.fixtures()
        natural[-1]["speaker"] = "1000"
        with self.assertRaises(ValueError):
            subject.validate_originals(synthetic, natural, training)
        natural[-1]["speaker"] = "2011"
        natural[-1]["pcm_sha256"] = synthetic[0]["pcm_sha256"]
        with self.assertRaises(ValueError):
            subject.validate_originals(synthetic, natural, training)

    def test_source_authorization_cannot_name_a_different_artifact_run(self):
        saved = dict(source_execution=dict(run_id="123", run_head="a" * 40, run_attempt=1,
                                          run_branch=subject.synth.lineage.BRANCH))
        with patch.object(subject.prior_replica, "extract_source"), \
             patch.object(subject.synth, "read_source", return_value=[]), \
             patch.object(subject.synth.lineage, "require_authorization", return_value=saved), \
             self.assertRaises(ValueError):
            subject.verified_synthetic(Archive(), Archive(), dict(run_id=124, run_head="a" * 40))

    def natural_archive(self):
        rows, members = [], {}
        for i in range(24):
            buf = io.BytesIO()
            pcm = int(i).to_bytes(2, "little", signed=True) * 16
            with wave.open(buf, "wb") as out:
                out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                out.writeframes(pcm)
            raw = buf.getvalue()
            row = dict(id="n" + str(i), samples=16, wav_sha256=subject.sha(raw), pcm_sha256=subject.sha(pcm))
            rows.append(row)
            members["wav/" + row["id"] + ".wav"] = raw
        members["MANIFEST.json"] = json.dumps(dict(rows=rows)).encode()
        return Archive(members), rows

    def test_reused_natural_keeps_24_raw_slots_and_verifies_unknown_pcm(self):
        source, rows = self.natural_archive()
        with patch.object(subject.old, "verified_human", return_value=(rows[:21], dict(natural_originals=21, natural_speakers=11))):
            selected, counts, originals = subject.verified_natural(source, Archive())
            self.assertEqual((len(selected), len(originals)), (21, 24))
            source.members["wav/n23.wav"] = source.members["wav/n0.wav"]
            with self.assertRaises(ValueError):
                subject.verified_natural(source, Archive())

    def test_new_synthetic_label_may_only_add_observed_class_and_tier(self):
        plan = subject.synth.source_plan("primary")
        originals = [dict(row, observed_class=None, pcm_sha256=str(i)) for i, row in enumerate(plan["rows"])]
        manifest = dict(schema=subject.synth.SCHEMA, panel="primary")
        raw = json.dumps(manifest).encode()
        labels = [dict(row, observed_class=row["requested_class"], tier="silver") for row in originals]
        pairs = sorted({r["source_group_id"] for r in labels})
        assessment = subject.synth.lineage.read_plan()
        lab = dict(schema=subject.synth.BATCH, panel="primary", source_manifest_sha256=subject.sha(raw),
                   assessment_sha256=subject.synth.ASSESSMENT_SHA,
                   registered_plan_file_sha256=subject.sha((subject.synth.PLANS / "primary.json").read_bytes()),
                   coverage=dict.fromkeys(subject.synth.qa.observers.PROVIDERS, 96), new_model_forwards=0,
                   optimizer_updates=0, tts_calls_at_screen=0, training_allowed=False, promotion_allowed=False,
                   replication_opened=False, rows=labels, unknown_count=0, paired_source_groups=pairs,
                   source_coverage=dict(positive_originals=sum(r["requested_class"] == 2 for r in labels),
                                        ordinary_originals=sum(r["requested_class"] == 0 for r in labels),
                                        paired_source_groups=len(pairs)))
        lab.update(source_ancestry_sha256=subject.synth.lineage.BASE_ANCESTRY_SHA,
                   parser_contract_sha256=assessment["parser_contract_sha256"],
                   observer_code_contract=assessment["observer_code_contract"],
                   strict_raw_receipt_readback=True, B17_source_stop_retained=True,
                   all_prior_replication_sealed=True,
                   original_B17_source_proof=assessment["original_B17_source_proof"]["source_inputs"],
                   frozen_minima=dict(positive_originals=12, ordinary_originals=24, paired_source_groups=8),
                   source_gate="SOURCE_GATE_PASS", source_shortfall={})
        with patch.object(subject.prior_replica, "extract_source"), \
             patch.object(subject.synth, "read_source", return_value=originals):
            chosen, counts, raw_rows = subject.verified_synthetic(Archive({"MANIFEST.json": raw}),
                                                                Archive({"SCREEN.json": json.dumps(lab).encode()}))
            self.assertEqual((len(chosen), len(raw_rows)), (96, 96))
            self.assertEqual(counts["synthetic_paired"], 24)
            lab["rows"][0]["pcm_sha256"] = "tampered"
            with self.assertRaises(ValueError):
                subject.verified_synthetic(Archive({"MANIFEST.json": raw}), Archive({"SCREEN.json": json.dumps(lab).encode()}))


if __name__ == "__main__":
    unittest.main()
