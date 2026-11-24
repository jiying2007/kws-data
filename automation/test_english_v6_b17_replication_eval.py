"""Fixture-only replica gates: no source audio, model, network or opening operation."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b17_replication_eval as replica


class ReplicaContract(unittest.TestCase):
    def test_spec_keeps_quality_and_no_release(self):
        doc = replica.protocol()
        self.assertIs(type(doc["scoring_allowed"]), bool)
        self.assertFalse(doc["promotion_allowed"])
        self.assertFalse(doc["product_released"])
        self.assertFalse(doc["old_replication_opened"])
        self.assertEqual(doc["source_gate"], replica.primary.GATE)
        if not doc["scoring_allowed"]:
            self.assertIsNone(doc["opening_lock_sha256"])
            self.assertIsNone(doc["inputs"]["pair"])

    def test_inert_spec_fails_before_opening_saved_primary(self):
        closed = dict(replica.protocol(), scoring_allowed=False)
        with patch.object(replica.opening, "require") as req:
            with self.assertRaises(ValueError):
                replica.bind_primary(closed, Path("not-read"))
            req.assert_not_called()

    def test_canonical_cross_format_identity(self):
        self.assertEqual(replica.canonical_person({"source_speaker_id": "p123"}), "123")
        self.assertEqual(replica.canonical_person({"speaker": 123}), "123")
        with self.assertRaises(ValueError):
            replica.canonical_person({"source_speaker_id": "voice123"})

    def test_history_or_primary_person_pcm_and_group_all_excluded(self):
        row = dict(id="replica", pcm_sha256="a" * 64,
                   source_group_id="piper-libritts-p123", source_speaker_id="p123")
        replica.reject_ancestry([row], set(), set(), set())
        for old_pcm, old_group, people in (({"a" * 64}, set(), set()),
                                             (set(), {row["source_group_id"]}, set()),
                                             (set(), set(), {"123"})):
            with self.assertRaises(ValueError):
                replica.reject_ancestry([row], old_pcm, old_group, people)
        with self.assertRaises(ValueError):
            replica.reject_ancestry([row, dict(row)], set(), set(), set())

    def test_opening_identity_cannot_change_between_source_and_score(self):
        permit = dict(primary_result_sha256="p", opening_lock_sha256="o",
                      primary_source_inputs={"synth_source": "s", "natural_source": "n"})
        doc = dict(panel="replication", assessment_sha256=replica.primary.ASSESSMENT_SHA,
                   source_ancestry_sha256=replica.primary.synth.lineage.ANCESTRY_SHA,
                   primary_result_sha256="p", opening_lock_sha256="o",
                   primary_source_inputs=permit["primary_source_inputs"], old_replication_opened=False)
        replica.permission_fields(doc, permit)
        for altered in (dict(doc, primary_result_sha256="different"),
                        dict(doc, opening_lock_sha256="different"), dict(doc, old_replication_opened=True),
                        dict(doc, panel="primary")):
            with self.assertRaises(ValueError):
                replica.permission_fields(altered, permit)

    def test_same_primary_model_archive_required(self):
        pin = dict(id=1, run_id=2, run_head="a" * 40, bytes=3, sha256="b" * 64)
        inputs = dict.fromkeys(replica.INPUT_KEYS, pin)
        permit = dict(primary_result_sha256="p", opening_lock_sha256="o", model_pt_sha256={"B0": "b"})
        spec = dict(scoring_allowed=True, opening_lock_sha256="o", primary_result_sha256="p",
                    model_pt_sha256=permit["model_pt_sha256"], inputs=inputs)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "english_v6_b17_primary_inputs.json").write_text(json.dumps({"inputs": inputs}))
            with patch.object(replica.opening, "require", return_value=permit):
                self.assertEqual(replica.bind_primary(spec, root), permit)
                for key in ("B0", "B9", "pair"):
                    altered = dict(spec, inputs={**inputs, key: {**pin, "id": 999}})
                    with self.assertRaises(ValueError):
                        replica.bind_primary(altered, root)

    def test_no_archive_path_escape(self):
        for name in ("../bad", "/absolute", "nested/../../bad", "windows\\bad"):
            payload = io.BytesIO()
            with zipfile.ZipFile(payload, "w") as z:
                z.writestr(name, "fixture")
            with zipfile.ZipFile(io.BytesIO(payload.getvalue())) as z, tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    replica.extract_source(z, Path(tmp) / "source")

    def test_unchanged_quality_still_rejects_added_ordinary_event(self):
        def event_row(uid, group, positive, values, cohort="synthetic"):
            return dict(id=uid, group=group, positive=positive, cohort=cohort,
                gains={g: {arm: {"events": count} for arm, count in zip(replica.common.NAMES, values)}
                       for g in replica.primary.GAINS})
        rows = [event_row("p", "p", True, (0, 1, 0, 1)),
                event_row("n1", "n1", False, (0, 1, 0, 0), "natural"),
                event_row("n2", "n2", False, (0, 1, 0, 0), "natural")]
        self.assertEqual(replica.primary.old_decision.decide(rows)["panel_gate"], "PASS")
        rows.append(event_row("added", "new", False, (0, 0, 0, 1)))
        self.assertEqual(replica.primary.old_decision.decide(rows)["panel_gate"], "NO_PROMOTION")


if __name__ == "__main__":
    unittest.main()
