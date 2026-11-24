"""Actual contract checks for prospective B17 evaluation; no model/data calls."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b17_primary_eval as subject


class SourceAndModelContract(unittest.TestCase):
    def test_original_source_minima_are_not_satisfied_by_gain_copies(self):
        self.assertEqual(subject.source_gate(dict(subject.GATE)), ("SOURCE_GATE_PASS", {}))
        for key in subject.GATE:
            counts = dict(subject.GATE)
            counts[key] -= 1
            gate, missing = subject.source_gate(counts)
            self.assertEqual(gate, "INSUFFICIENT_SOURCE")
            self.assertEqual(set(missing), {key})

    def test_all_source_dimensions_and_integer_counts_required(self):
        with self.assertRaises(ValueError):
            subject.source_gate({"synthetic_positive": 1000})
        for bad in (True, 12.5, -1):
            counts = dict(subject.GATE, natural_originals=bad)
            with self.assertRaises(ValueError):
                subject.source_gate(counts)

    def test_frozen_protocol_is_inert_before_original_artifacts(self):
        doc = subject.protocol()
        self.assertFalse(doc["promotion_allowed"])
        self.assertFalse(doc["physical_ssc305"])
        self.assertEqual(doc["parent_pt_sha256"], subject.B0_PT)
        self.assertEqual(doc["source_gate"], subject.GATE)
        # The source-only publication has null locks. A later reviewed execution
        # commit may populate them, but cannot change the frozen assessment.
        if not doc["scoring_allowed"]:
            self.assertIsNone(doc["inputs"]["pair"])

    def test_parent_cannot_silently_remain_b9(self):
        receipt = dict(schema=subject.PAIR_SCHEMA, parent_pt_sha256=subject.B0_PT,
                       actual_optimizer_updates=384, trained_model_parameters=43876,
                       dev_cal_locked_forwards=0, promotion_allowed=False,
                       assessment_sha256=subject.ASSESSMENT_SHA, design_sha256=subject.DESIGN_SHA)
        for arm in ("control", "challenger"):
            receipt[arm] = dict(optimizer_updates=192, stem=subject.STEMS[arm])
        subject.validate_pair_receipt(receipt)
        for field, bad in (("parent_pt_sha256", "7dba86a3fcda6e5f05cb80b414dc372c70a5a03b3aee783f168441a4871e7655"),
                           ("actual_optimizer_updates", 192), ("dev_cal_locked_forwards", 1),
                           ("promotion_allowed", True), ("assessment_sha256", "0" * 64)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                subject.validate_pair_receipt(dict(receipt, **{field: bad}))

    def test_natural_roster_rejects_duplicate_omitted_and_historical_readers(self):
        people = [str(i) for i in range(100, 112)]
        rows = [dict(source_id=f"{person}-{slot}", speaker=person)
                for person in people for slot in (1, 2)]
        roster = {"primary": rows}
        assessment = {"primary": {"natural": {"speakers": people}}}
        ancestry = {"natural_excluded_speakers": ["1462"]}
        doc = {"rows": copy.deepcopy(rows), "source_rejections": []}
        subject.validate_natural_roster(doc, roster, assessment, ancestry)
        rejected = copy.deepcopy(doc)
        last = rejected["rows"].pop()
        rejected["source_rejections"].append(dict(source_id=last["source_id"], reason="duration"))
        subject.validate_natural_roster(rejected, roster, assessment, ancestry)
        for missing, replaced, contaminated in ((True, False, False), (False, True, False), (False, False, True)):
            changed = copy.deepcopy(doc)
            excluded = copy.deepcopy(ancestry)
            if missing:
                changed["rows"].pop()
            if replaced:
                changed["rows"][0] = copy.deepcopy(changed["rows"][1])
            if contaminated:
                excluded["natural_excluded_speakers"].append(people[0])
            with self.assertRaises(ValueError):
                subject.validate_natural_roster(changed, roster, assessment, excluded)
        duplicate = copy.deepcopy(roster)
        duplicate["primary"][0] = copy.deepcopy(duplicate["primary"][1])
        with self.assertRaises(ValueError):
            subject.validate_natural_roster(doc, duplicate, assessment, ancestry)

    def test_final_boundary_rejects_actual_train_group_and_pcm(self):
        training = dict(base_train=[dict(id=str(i), group="speaker-"+str(i), pcm_sha256="sha-"+str(i))
                                    for i in range(79)],
                        natural_train=[dict(id=str(i), source_group_id="speaker-"+str(i),
                                            source_pcm_sha256="sha-"+str(i)) for i in range(79, 88)])
        safe = [dict(pcm_sha256="fresh", source_group_id="fresh-group")]
        subject.reject_train_overlap(safe, training)
        with self.assertRaises(ValueError):
            subject.reject_train_overlap([dict(pcm_sha256="sha-80", source_group_id="fresh-group")], training)
        with self.assertRaises(ValueError):
            subject.reject_train_overlap([dict(pcm_sha256="fresh", source_group_id="speaker-12")], training)
        malformed = copy.deepcopy(training)
        malformed["base_train"].pop()
        with self.assertRaises(ValueError):
            subject.reject_train_overlap(safe, malformed)


def row(ident, positive, cohort, values):
    return dict(id=ident, group=ident, positive=positive, cohort=cohort,
                gains={gain: {arm: {"events": value} for arm, value in values.items()}
                       for gain in subject.GAINS})


class FrozenDecision(unittest.TestCase):
    def passing_rows(self):
        return [row("positive", True, "synthetic", dict(B0=0, B9=1, control=0, challenger=1)),
                row("reader1", False, "natural", dict(B0=0, B9=1, control=0, challenger=0)),
                row("reader2", False, "natural", dict(B0=0, B9=1, control=0, challenger=0))]

    def test_strict_pass_does_not_authorize_product_promotion(self):
        decision = subject.old_decision.decide(self.passing_rows())
        self.assertEqual(decision["panel_gate"], "PASS")
        self.assertNotIn("promotion_allowed", decision)

    def test_identical_false_event_totals_do_not_hide_swapped_errors(self):
        rows = self.passing_rows() + [
            row("old-error", False, "synthetic", dict(B0=1, B9=1, control=1, challenger=0)),
            row("new-error", False, "synthetic", dict(B0=0, B9=0, control=0, challenger=1))]
        decision = subject.old_decision.decide(rows)
        self.assertEqual(decision["panel_gate"], "NO_PROMOTION")
        self.assertTrue(decision["added_ordinary_events"])

    def test_b0_natural_guard_remains_required_at_each_gain(self):
        rows = self.passing_rows()
        rows[1]["gains"]["0.5"] = {arm: {"events": value}
                                         for arm, value in dict(B0=0, B9=2, control=1, challenger=1).items()}
        decision = subject.old_decision.decide(rows)
        self.assertEqual(decision["panel_gate"], "NO_PROMOTION")
        self.assertEqual(decision["natural_above_B0_gains"], ["0.5"])


if __name__ == "__main__":
    unittest.main()
