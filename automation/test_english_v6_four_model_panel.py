"""Source-only regressions for the registered B11 and B12 panel decisions."""

from __future__ import annotations

import copy
from contextlib import ExitStack
import io
import json
import zipfile
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_four_model_panel as scoring


def row(source_id, group, positive, *, faster=False):
    events = int(positive)
    return {"id": source_id, "group": group, "positive": positive, "verified_faster": faster,
            "gains": {g: {name: {"events": events} for name in scoring.NAMES}
                      for g in scoring.GAINS}}


def change(r, gain, **events):
    for name, value in events.items():
        r["gains"][gain][name]["events"] = value


def b11_rows():
    rows = []
    for i in range(4):
        neg = row(f"ordinary-{i}", f"voice-{i}", False)
        pos = row(f"target-{i}", f"voice-{i}", True)
        if i < 2:
            change(neg, "1.0", B0=1, B9=1, control=1)
        rows.extend((neg, pos))
    return rows


def b12_rows():
    rows = []
    for i in range(4):
        neg = row(f"ordinary-{i}", f"voice-{i}", False)
        pos = row(f"target-{i}", f"voice-{i}", True, faster=i < 2)
        if i < 2:
            change(pos, "0.5", B9=0, control=0)
        rows.extend((neg, pos))
    return rows


class FrozenPanelDecision(unittest.TestCase):
    def test_frozen_conditions(self):
        self.assertEqual(scoring.GAINS, ("1.0", "0.75", "0.5"))
        self.assertEqual(scoring.NAMES, ("B0", "B9", "control", "challenger"))
        with self.assertRaises(ValueError):
            scoring.decision(b11_rows(), "B13")

    def test_b11_counts_source_groups_at_original_gain_against_both_arms(self):
        baseline = b11_rows()
        result = scoring.decision(baseline, "B11")
        self.assertEqual(result["panel_gate"], "PASS")
        self.assertEqual(result["improved_groups"], ["voice-0", "voice-1"])
        for comparator in ("B9", "control"):
            rows = copy.deepcopy(baseline)
            change(rows[2], "1.0", **{comparator: 0})
            with self.subTest(tie_with=comparator):
                self.assertEqual(scoring.decision(rows, "B11")["panel_gate"], "NO_PROMOTION")
        for gain in ("0.75", "0.5"):
            rows = copy.deepcopy(baseline)
            change(rows[2], "1.0", B9=0, control=0)
            change(rows[2], gain, B9=1, control=1)
            with self.subTest(other_gain=gain):
                self.assertEqual(scoring.decision(rows, "B11")["improved_groups"], ["voice-0"])

    def test_repeated_successes_in_one_family_cannot_supply_second_family(self):
        for hypothesis, rows in (("B11", b11_rows()), ("B12", b12_rows())):
            for r in rows:
                if r["group"] == "voice-1":
                    r["group"] = "voice-0"
            with self.subTest(hypothesis=hypothesis):
                result = scoring.decision(rows, hypothesis)
                self.assertEqual(result["improved_groups"], ["voice-0"])
                self.assertEqual(result["panel_gate"], "NO_PROMOTION")

    def test_gain_conditions_do_not_inflate_original_denominators(self):
        for hypothesis, rows in (("B11", b11_rows()), ("B12", b12_rows())):
            result = scoring.decision(rows, hypothesis)
            for gain in scoring.GAINS:
                with self.subTest(hypothesis=hypothesis, gain=gain):
                    r = result["summary"]["challenger"][gain]
                    self.assertEqual(r["positive_originals"], 4)
                    self.assertEqual(r["ordinary_originals"], 4)

    def test_b12_requires_verified_faster_clean_hit_and_half_gain_recovery(self):
        baseline = b12_rows()
        self.assertEqual(scoring.decision(baseline, "B12")["panel_gate"], "PASS")
        for change_type, arm in (("slow", None), ("clean", "B9"), ("clean", "control"),
                                 ("clean", "challenger"), ("half_already", "control"),
                                 ("half_already", "B9"), ("half_miss", "challenger")):
            rows = copy.deepcopy(baseline)
            r = rows[3]
            if change_type == "slow":
                r["verified_faster"] = False
            elif change_type == "clean":
                change(r, "1.0", **{arm: 0})
            else:
                change(r, "0.5", **{arm: int(change_type == "half_already")})
            with self.subTest(condition=change_type, arm=arm):
                result = scoring.decision(rows, "B12")
                self.assertEqual(result["improved_groups"], ["voice-0"])
                self.assertEqual(result["panel_gate"], "NO_PROMOTION")

    def test_ordinary_regression_against_either_comparator_vetoes_each_gain(self):
        for hypothesis, maker in (("B11", b11_rows), ("B12", b12_rows)):
            for gain in scoring.GAINS:
                for clean_arm in ("B9", "control"):
                    rows = maker()
                    r = rows[-2]
                    change(r, gain, challenger=1, control=1, B9=1)
                    change(r, gain, **{clean_arm: 0})
                    with self.subTest(hypothesis=hypothesis, gain=gain, clean_arm=clean_arm):
                        result = scoring.decision(rows, hypothesis)
                        self.assertEqual(result["panel_gate"], "NO_PROMOTION")
                        self.assertIn([r["id"], gain], result["new_ordinary_false_or_extra"])

    def test_positive_group_lost_against_b0_at_any_gain(self):
        for hypothesis, maker in (("B11", b11_rows), ("B12", b12_rows)):
            for gain in scoring.GAINS:
                rows = maker()
                r = rows[-1]
                change(r, gain, B0=1, control=0, challenger=0)
                with self.subTest(hypothesis=hypothesis, gain=gain):
                    result = scoring.decision(rows, hypothesis)
                    self.assertEqual(result["panel_gate"], "NO_PROMOTION")
                    self.assertIn([r["id"], gain], result["diagnostic_lost_positive_original_conditions"])

    def test_b12_positive_group_lost_against_matched_control_at_any_gain(self):
        for gain in scoring.GAINS:
            rows = b12_rows()
            r = rows[-1]
            change(r, gain, B0=0, control=1, challenger=0)
            with self.subTest(gain=gain):
                result = scoring.decision(rows, "B12")
                self.assertEqual(result["panel_gate"], "NO_PROMOTION")
                self.assertIn([r["id"], gain], result["diagnostic_lost_positive_original_conditions"])

    def test_global_clean_ordinary_cohort_supplies_no_b11_benefit(self):
        rows = b11_rows()
        for r in rows:
            if not r["positive"]:
                change(r, "1.0", B0=0, B9=0, control=0)
        self.assertEqual(scoring.decision(rows, "B11")["panel_gate"], "NO_PROMOTION")

    def test_positive_original_loss_with_sibling_hit_is_logged_while_family_remains_correct(self):
        for hypothesis, maker in (("B11", b11_rows), ("B12", b12_rows)):
            rows = maker()
            r = rows[-1]
            rows.append(row("another-target", r["group"], True))
            change(r, "0.75", challenger=0)
            change(rows[1], "1.0", challenger=2)
            result = scoring.decision(rows, hypothesis)
            with self.subTest(hypothesis=hypothesis):
                self.assertEqual(result["panel_gate"], "PASS")
                self.assertIn([r["id"], "0.75"], result["diagnostic_lost_positive_original_conditions"])
                self.assertIn([rows[1]["id"], "1.0"], result["diagnostic_extra_positive_events"])
                self.assertNotIn([r["group"], "0.75", "B0"], result["lost_positive_group_conditions"])

    def test_b11_group_recall_and_b12_preservation_have_distinct_meanings(self):
        for hypothesis, maker, expected in (("B11", b11_rows, "PASS"), ("B12", b12_rows, "NO_PROMOTION")):
            rows = maker()
            # Equal aggregate gain-0.75 recall, exchanging one lost family and one newly detected.
            change(rows[5], "0.75", B0=1, control=1, challenger=0)
            change(rows[7], "0.75", B0=0, control=0, challenger=1)
            with self.subTest(hypothesis=hypothesis):
                result = scoring.decision(rows, hypothesis)
                self.assertEqual(result["panel_gate"], expected)
                self.assertIn([rows[5]["group"], "0.75", "B0"], result["lost_positive_group_conditions"])


class SourceBinding(unittest.TestCase):
    def fixtures(self):
        plan = scoring.panel.source_plan("primary")
        plan_sha = scoring.sha(scoring.panel.b10.canon(plan))
        originals = [dict(r, pcm_sha256=scoring.sha(r["id"].encode()), whole_native_resample=True,
                          crop_calls=0, samples=100 if r["cell_id"] == "mero-fast" else 120)
                     for r in plan["rows"]]
        decisions = [dict(r, observed_class=r["requested_class"], tier="silver") for r in originals]
        return (dict(schema=scoring.panel.SCHEMA, panel="primary", plan_sha256=plan_sha,
                     tts_calls=48, new_kws_forwards=0, human_gold=False, rows=originals),
                dict(plan_sha256=plan_sha, frozen_before_kws=True, kws_forwards=0, optimizer_updates=0,
                     human_gold=False, promotion_allowed=False, rows=decisions,
                     paired_source_groups=sorted({r["source_group_id"] for r in originals}),
                     meets_four_pair_minimum=True))

    def select(self, source, review, *, name="primary", corrupt_digest=False):
        # Same bytes that the saved independent screen bound at source time.
        raw = json.dumps(source).encode()
        review["source_manifest_sha256"] = "0" * 64 if corrupt_digest else scoring.sha(raw)
        with ExitStack() as stack:
            def archive(filename, content):
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w") as encoded:
                    encoded.writestr(filename, content)
                return stack.enter_context(zipfile.ZipFile(buf))
            return scoring.select(archive("MANIFEST.json", raw),
                                  archive("SCREEN.json", json.dumps(review).encode()), name)

    def test_admission_has_measured_tempo_unique_originals_and_excludes_unknown(self):
        src, lab = self.fixtures()
        key = src["rows"][0]["source_group_id"]
        src["rows"][0]["samples"] = 150
        lab["rows"][0]["samples"] = 150
        lab["rows"][2].update(observed_class=None, tier="unknown")
        selected, groups, faster = self.select(src, lab)
        self.assertEqual(len(selected), 47)
        self.assertEqual(len(groups), 12)
        self.assertEqual(len(faster), 11)
        self.assertNotIn(key, faster)
        self.assertNotIn(lab["rows"][2]["id"], {r["id"] for r in selected})

    def test_frozen_input_mutations_fail_closed(self):
        for reason in ("roster_duplicate_pcm", "duplicate_review_id", "role", "altered_screen_row",
                       "leaky_postscore_label", "unknown_truth", "sha_mismatch", "wrong_panel", "insufficient"):
            src, lab = self.fixtures()
            if reason == "roster_duplicate_pcm":
                src["rows"][1]["pcm_sha256"] = src["rows"][0]["pcm_sha256"]
                lab["rows"][1]["pcm_sha256"] = src["rows"][0]["pcm_sha256"]
            elif reason == "duplicate_review_id":
                lab["rows"][1]["id"] = lab["rows"][0]["id"]
            elif reason == "role":
                src["rows"][0]["role"] = lab["rows"][0]["role"] = "train"
            elif reason == "altered_screen_row":
                lab["rows"][1]["samples"] += 1
            elif reason == "leaky_postscore_label":
                lab["kws_forwards"] = 1
            elif reason == "unknown_truth":
                lab["rows"][1].update(observed_class=None, tier="weak")
            elif reason == "insufficient":
                keep = set(lab["paired_source_groups"][:3])
                for r in lab["rows"]:
                    if r["source_group_id"] not in keep and r["observed_class"] == 2:
                        r.update(observed_class=None, tier="unknown")
                lab["paired_source_groups"] = sorted(keep)
                lab["meets_four_pair_minimum"] = False
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                self.select(src, lab, corrupt_digest=reason == "sha_mismatch",
                            name="replication" if reason == "wrong_panel" else "primary")


if __name__ == "__main__":
    unittest.main()
