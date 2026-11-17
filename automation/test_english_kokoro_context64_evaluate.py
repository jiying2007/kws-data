"""Pure in-memory control/evidence fixtures; no speech runtime or real inputs."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_kokoro_context64_evaluate as evaluator

PLAN = json.loads((HERE / "english_kokoro_context64_plan.json").read_bytes())
RUN = {"repository": "jiying2007/kws-data", "commit": "a" * 40, "run_id": "1", "run_attempt": "1", "event": "push"}
TOKENS = {
    "mero-direct-eh": ["h", "eɪ", "m", "ɛ", "ɹ", "oʊ"],
    "mero-direct-i": ["h", "eɪ", "m", "i", "ɹ", "oʊ"],
    "vosen-direct-s": ["h", "eɪ", "v", "oʊ", "s", "ə", "n"],
    "vosen-direct-z": ["h", "eɪ", "v", "oʊ", "z", "ə", "n"],
    "contrast-mero-ae": ["h", "eɪ", "m", "æ", "ɹ", "oʊ"],
    "contrast-mero-ih": ["h", "eɪ", "m", "ɪ", "ɹ", "oʊ"],
    "contrast-vosen-b": ["h", "eɪ", "b", "oʊ", "s", "ə", "n"],
    "contrast-vosen-ih": ["h", "eɪ", "v", "oʊ", "z", "ɪ", "n"],
}
ALL_TOKENS = sorted({t for values in TOKENS.values() for t in values} | {"k", "ʊ", "d", "w"})
VOCAB = {token: i + 4 for i, token in enumerate(ALL_TOKENS)}
BY_ID = {i: "unused" + str(i) for i in range(392)} | {i: token for token, i in VOCAB.items()}


def counts(number):
    return {"declared_count": number, "complete": True, "actual_forward_attempts": number,
        "native_forward_returns": number, "completed": number,
        "forward_attempts_lower_bound": number, "forward_attempts_upper_bound": number}


def phone_row(parent, tokens):
    samples = parent["pcm_samples"]
    frames = (samples - 400) // 320 + 1
    ids = [0] * frames
    ids[0:3] = [VOCAB[t] for t in ("k", "ʊ", "d")]
    for index, token in enumerate(tokens, frames - len(tokens) - 1):
        ids[index] = VOCAB[token]
    return {"status": "success", "capture": {"in_memory_fixture": True}, "frame_ids": ids,
        "phone_tokens": ["k", "ʊ", "d"] + tokens, "vocab": VOCAB, "by_id": BY_ID}


def fixture():
    parents, children = [], []
    specs = []
    for kind, field in (("positive", "positive_cells"), ("contrast", "contrast_cells"), ("ordinary", "ordinary_cells")):
        specs.extend((kind, cell) for cell in PLAN["design"][field])
    for preset in PLAN["design"]["presets"]:
        for kind, cell in specs:
            identity = preset + "/" + cell["cell_id"]
            component = "kokoro-v1_0-preset:" + preset
            parent = {"id": "parent-" + identity, "kind": kind, "preset_id": preset, "preset_component": component,
                "source_group_id": component, "role": "train", "cell_id": cell["cell_id"],
                "pronunciation_hypothesis": cell.get("pronunciation_hypothesis", cell["cell_id"]),
                "requested_class": cell.get("requested_class", 0), "requested_text": cell["requested_text"],
                "generation_status": "success", "pcm_samples": 128000, "pcm_sha256": "1" * 64,
                "target_start_parent16k": 112400}
            parents.append(parent)
            for outside in ([False, True] if kind == "positive" else [False]):
                child = dict(parent)
                child.update(id=("outside-" if outside else "main-") + identity, parent_id=parent["id"],
                    kind="outside" if outside else kind, exact_parent_pcm_slice=True,
                    child_start_parent16k=40000 if outside else 110000, child_end_parent16k=60000 if outside else 128000,
                    pcm_samples=20000 if outside else 18000, pcm_sha256="2" * 64)
                if outside:
                    child.update(requested_class=0, requested_text=PLAN["design"]["carrier"]["boundary_control"]["text"],
                        pronunciation_hypothesis="ordinary-negative-v1", cell_id="outside-" + cell["cell_id"])
                g = evaluator.whole_support_geometry(child["pcm_samples"])
                child.update(whole_support_windows=g["whole_support_windows"], geometry_usable_positive=g["geometry_usable_positive"])
                children.append(child)
    text_results = {}
    for provider in evaluator.TEXT_PROVIDERS:
        rows = []
        for child in children:
            text = child["requested_text"]
            if child["kind"] == "contrast":
                text = "Hey Mero." if "mero" in child["cell_id"] else "Hey Vosen."
            rows.append({"id": child["id"], "parent_id": child["parent_id"], "status": "complete", "raw_text": text,
                "capture_sha256": "3" * 64, "reason": "STDLIB_IN_MEMORY_FIXTURE"})
        text_results[provider] = counts(64) | {"provider": provider, "rows": rows}
    phone_results = {}
    for provider in evaluator.PHONE_PROVIDERS:
        phone_results[provider] = {"summary": counts(48) | {"provider": provider},
            "rows": {p["id"]: phone_row(p, TOKENS.get(p["cell_id"], ["k", "ʊ", "d"])) for p in parents}}
    return {"run": RUN, "parents": parents, "children": children, "generation_summary": counts(48)}, text_results, phone_results


def change_phone(manifest, phone_results, parent_id, tokens, provider="phoneme"):
    parent = next(p for p in manifest["parents"] if p["id"] == parent_id)
    phone_results[provider]["rows"][parent_id] = phone_row(parent, tokens)


class CandidateTests(unittest.TestCase):
    def texts(self, text="Hey Mero."):
        return {p: text for p in evaluator.TEXT_PROVIDERS}

    def classify(self, fixed=None, parent=None, texts=None, windows=3):
        fixed = TOKENS["mero-direct-eh"] if fixed is None else fixed
        parent = ["k", "ʊ", "d"] + fixed if parent is None else parent
        return evaluator.candidate_from_evidence(self.texts() if texts is None else texts, fixed, parent, windows)

    def test_classification_derives_class_and_hypothesis_without_intent(self):
        value = self.classify()
        self.assertEqual((value["candidate_class"], value["candidate_hypothesis"], value["candidate_tier"]), (2, "mero-us-eh-v1", "context_supported_weak"))
        self.assertFalse(value["training_allowed"])
        self.assertEqual(value["training_weight"], 0.0)

    def test_two_votes_cannot_be_upgraded(self):
        texts = self.texts(); texts["whisper"] = "I heard something."
        self.assertFalse(self.classify(texts=texts)["candidate_positive"])

    def test_third_contradictory_core_veto(self):
        texts = self.texts("Hey Vosen."); texts["parakeet"] = "Hay Rosen."
        value = self.classify(fixed=TOKENS["vosen-direct-z"], texts=texts)
        self.assertFalse(value["candidate_positive"])
        self.assertEqual(value["reason"], "FROZEN_CONTRADICTORY_WAKE_VETO")

    def test_aliases_and_stress_rhotics_are_exact_frozen_rules(self):
        value = self.classify(fixed=["h", "eɪ", "m", "ˈiː", "ɹ", "oʊ"], texts=self.texts("Hay Meero."))
        self.assertEqual(value["candidate_hypothesis"], "mero-us-i-v1")

    def test_wrong_core_and_tail_are_not_normalized_into_a_target(self):
        for tokens in (["h", "eɪ", "m", "e", "ɹ", "oʊ"], TOKENS["contrast-mero-ae"], TOKENS["contrast-mero-ih"],
            ["h", "eɪ", "m", "ɛ", "ɹ", "ɔ"], ["h", "eɪ", "m", "ɛ", "ɹ", "aʊ"]):
            with self.subTest(tokens=tokens):
                self.assertFalse(self.classify(fixed=tokens)["candidate_positive"])

    def test_region_and_terminal_must_be_same_hypothesis(self):
        self.assertFalse(self.classify(parent=TOKENS["mero-direct-i"])["candidate_positive"])

    def test_geometry_floor_is_three_and_bool_is_not_integer(self):
        for value in (0, 1, 2, True, None):
            self.assertFalse(self.classify(windows=value)["candidate_positive"])
        self.assertTrue(self.classify(windows=3)["candidate_positive"])

    def test_missing_h_is_allowed_only_with_all_three_votes(self):
        self.assertTrue(self.classify(fixed=TOKENS["mero-direct-eh"][1:])["candidate_positive"])
        texts = self.texts(); texts["qwen"] = None
        self.assertFalse(self.classify(fixed=TOKENS["mero-direct-eh"][1:], texts=texts)["candidate_positive"])

    def test_both_fixed_region_boundaries_are_applied(self):
        region = evaluator.fixed_child_region([4, 5, 6, 7], BY_ID, 321, 1040, 1600)
        self.assertEqual((region["frame_start"], region["frame_stop_exclusive"]), (2, 3))
        self.assertEqual(region["phone_tokens"], [BY_ID[6]])

    def test_ctc_blank_does_not_delete_a_second_terminal_n(self):
        ids = [VOCAB["n"], 0, VOCAB["n"], 0]
        region = evaluator.fixed_child_region(ids, BY_ID, 0, 1600, 1600)
        self.assertEqual(region["phone_tokens"], ["n", "n"])
        value = self.classify(fixed=TOKENS["vosen-direct-z"] + ["n"], texts=self.texts("Hey Vosen."))
        self.assertFalse(value["candidate_positive"])


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.texts, self.phones = fixture()

    def evaluate(self, plan=None):
        return evaluator.evaluate_evidence(self.manifest, self.texts, self.phones, PLAN if plan is None else plan, RUN)

    def child(self, kind, cell=None, preset=None):
        return next(c for c in self.manifest["children"] if c["kind"] == kind and (cell is None or c["cell_id"] == cell)
            and (preset is None or c["preset_id"] == preset))

    def set_texts(self, child_id, text, providers=evaluator.TEXT_PROVIDERS):
        for provider in providers:
            next(r for r in self.texts[provider]["rows"] if r["id"] == child_id)["raw_text"] = text

    def test_complete_control_fixture_passes_but_is_never_trainable(self):
        value = self.evaluate()
        self.assertEqual(value["status"], "PASS_CONTEXT_CONTRACT_FEASIBILITY_ONLY")
        self.assertEqual(value["counts"]["actual_high_level_calls"], 336)
        self.assertEqual(len(value["rows"]), 64)
        self.assertTrue(all(r["training_weight"] == 0 and not r["training_allowed"] for r in value["rows"]))
        self.assertFalse(value["training_snapshot_allowed"])
        self.assertFalse(value["automatic_expansion"])
        self.assertFalse(value["high_quality_model_passed"])

    def test_requested_class_does_not_help_blind_candidate(self):
        before = [r["candidate"] for r in self.evaluate()["rows"]]
        for row in self.manifest["children"]:
            row["requested_class"] = 2 if row["requested_class"] == 0 else 0
            row["pronunciation_hypothesis"] = "not-a-classification-hint"
        after = [r["candidate"] for r in self.evaluate()["rows"]]
        self.assertEqual(before, after)

    def test_nearmiss_spelled_as_positive_still_tests_phone_veto(self):
        value = self.evaluate()
        contrast = [r for r in value["rows"] if r["kind"] == "contrast"]
        self.assertEqual(len(contrast), 16)
        self.assertTrue(all(r["control"]["machine_corroborated"] for r in contrast))
        self.assertTrue(all(not r["candidate"]["candidate_positive"] for r in contrast))

    def test_positive_hallucination_in_a_contrast_control_stops(self):
        child = self.child("contrast", "contrast-mero-ae")
        change_phone(self.manifest, self.phones, child["parent_id"], TOKENS["mero-direct-eh"])
        value = self.evaluate()
        self.assertEqual(value["status"], "STOP_CONTEXT_CONTRACT_VALIDATION_FAILED")
        self.assertFalse(value["gates"]["all48_controls_candidate_positive_count_zero"])
        self.assertFalse(value["gates"]["all16_contrast_controls_reference_pattern_corroborated"])

    def test_ordinary_false_target_cannot_hide_in_no_match_average(self):
        child = self.child("ordinary")
        self.set_texts(child["id"], "Hey Mero.")
        change_phone(self.manifest, self.phones, child["parent_id"], TOKENS["mero-direct-eh"])
        value = self.evaluate()
        self.assertFalse(value["context_contract_control_test_passed"])
        self.assertFalse(value["gates"]["all48_controls_candidate_positive_count_zero"])

    def test_two_exact_ordinary_transcripts_are_not_three(self):
        child = self.child("ordinary")
        self.set_texts(child["id"], "Something else.", ("qwen",))
        self.assertFalse(self.evaluate()["gates"]["all16_ordinary_controls_all3_exact_text_and_no_target_in_fixed_region"])

    def test_outside_end_must_be_before_terminal_target(self):
        child = self.child("outside")
        child.update(child_start_parent16k=110000, child_end_parent16k=128000, pcm_samples=18000)
        geometry = evaluator.whole_support_geometry(child["pcm_samples"])
        child.update(whole_support_windows=geometry["whole_support_windows"], geometry_usable_positive=geometry["geometry_usable_positive"])
        value = self.evaluate()
        self.assertFalse(value["gates"]["all16_outside_controls_all3_exact_text_no_target_and_before_terminal"])

    def test_outside_parent_two_supported_components_is_not_four_required(self):
        for parent in self.manifest["parents"]:
            if parent["cell_id"] == "mero-direct-eh" and parent["preset_id"] in ("af_bella", "af_sarah"):
                change_phone(self.manifest, self.phones, parent["id"], TOKENS["contrast-mero-ae"])
        value = self.evaluate()
        self.assertTrue(value["context_contract_control_test_passed"])
        evidence = next(r for r in value["outside_parent_support"] if r["hypothesis"] == "mero-us-eh-v1")
        self.assertEqual(evidence["supported_count"], 2)

    def test_one_supported_component_cannot_meet_two(self):
        for parent in self.manifest["parents"]:
            if parent["cell_id"] == "mero-direct-eh" and parent["preset_id"] != "af_bella":
                change_phone(self.manifest, self.phones, parent["id"], TOKENS["contrast-mero-ae"])
        value = self.evaluate()
        self.assertFalse(value["gates"]["positive_each_hypothesis_distinct_preset_minimum"])
        self.assertFalse(value["gates"]["outside_each_parent_hypothesis_distinct_preset_minimum"])

    def test_lv60_cannot_rescue_reference_or_select_hypothesis(self):
        for parent in self.manifest["parents"]:
            if parent["cell_id"] == "mero-direct-eh":
                change_phone(self.manifest, self.phones, parent["id"], TOKENS["contrast-mero-ae"])
        value = self.evaluate()
        self.assertFalse(value["context_contract_control_test_passed"])
        self.assertTrue(all(not r["candidate"]["candidate_positive"] for r in value["rows"] if r["kind"] == "positive" and r["cell_id"] == "mero-direct-eh"))

    def test_missing_final_text_ledger_is_unknown_with_full_denominator(self):
        provider = self.texts["qwen"]
        provider.update(complete=False, actual_forward_attempts=None, native_forward_returns=None, completed=None,
            forward_attempts_lower_bound=0, forward_attempts_upper_bound=64)
        for row in provider["rows"]:
            row.update(status="UNAVAILABLE", raw_text=None, capture_sha256=None)
        value = self.evaluate()
        self.assertEqual(value["status"], "INCOMPLETE_DIAGNOSTIC_NO_EXPANSION")
        self.assertIsNone(value["counts"]["actual_high_level_calls"])
        self.assertEqual((value["counts"]["calls_lower_bound"], value["counts"]["calls_upper_bound"]), (272, 336))
        self.assertEqual(len(value["rows"]), 64)
        self.assertFalse(value["gates"]["all16_contrast_controls_reference_pattern_corroborated"])

    def test_returned_but_capture_failed_preserves_consumed_call(self):
        child = self.child("ordinary")
        self.set_texts(child["id"], None, ("qwen",))
        self.texts["qwen"].update(complete=False, completed=63)
        value = self.evaluate()
        self.assertEqual(value["status"], "INCOMPLETE_DIAGNOSTIC_NO_EXPANSION")
        self.assertEqual(value["counts"]["actual_high_level_calls"], 336)
        self.assertEqual(value["counts"]["high_level_returns"], 336)
        self.assertEqual(value["counts"]["complete_saved_outputs"], 335)

    def test_diagnostic_provider_is_required_in_declared96_denominator(self):
        summary = self.phones["phoneme_lv60"]["summary"]
        summary.update(complete=False, actual_forward_attempts=None, native_forward_returns=None, completed=None,
            forward_attempts_lower_bound=0, forward_attempts_upper_bound=48)
        for value in self.phones["phoneme_lv60"]["rows"].values():
            value.update(capture=None, frame_ids=None, phone_tokens=None, vocab=None, by_id=None)
        result = self.evaluate()
        self.assertEqual(result["status"], "INCOMPLETE_DIAGNOSTIC_NO_EXPANSION")
        self.assertFalse(result["gates"]["all96_parent_phone_captures_complete"])

    def test_self_reported_geometry_and_preset_component_are_not_trusted(self):
        child = self.child("positive")
        child["whole_support_windows"] += 1
        with self.assertRaisesRegex(ValueError, "INTEGER_GEOMETRY"):
            self.evaluate()
        self.manifest, self.texts, self.phones = fixture()
        self.child("positive")["preset_component"] = "invented-independent-preset"
        with self.assertRaisesRegex(ValueError, "TYPED_COMPONENT"):
            self.evaluate()

    def test_declared_minimum_cannot_be_relaxed_after_observation(self):
        plan = copy.deepcopy(PLAN)
        plan["design"]["validation_gates"]["minimum_distinct_preset_components_per_positive_hypothesis"] = 1
        with self.assertRaisesRegex(ValueError, "FROZEN_BEFORE_CALLS"):
            self.evaluate(plan)

    def test_source_only_preflight_keeps_old_reference_exact(self):
        result = evaluator.source_preflight()
        self.assertEqual(result["source_check"], "PASS")
        self.assertEqual(result["new_model_calls"], 0)
        self.assertFalse(result["training_allowed"])

    def test_public_entry_joins_only_pinned_pure_readers_and_full_rosters(self):
        plan = copy.deepcopy(PLAN)
        plan["evaluate_code_sha256"] = evaluator.sha(Path(evaluator.__file__).read_bytes())
        source_root = Path("mock-source-no-real-input")
        roots = {p: Path("mock-" + p) for p in evaluator.TEXT_PROVIDERS + evaluator.PHONE_PROVIDERS}
        source_readback = Mock(return_value=self.manifest)
        text_readback = Mock(side_effect=lambda root, children, plan, run, *, provider: self.texts[provider])
        phone_readback = Mock(side_effect=lambda **kwargs: self.phones[kwargs["provider"]])
        worker = SimpleNamespace(source_readback=source_readback, read_text_provider=text_readback)
        phone = SimpleNamespace(readback=phone_readback)
        with patch.object(evaluator, "module_at", side_effect=[worker, phone]) as loader:
            result = evaluator.evaluate(source_root, roots, plan, RUN)
        self.assertEqual(loader.call_count, 2)
        source_readback.assert_called_once_with(source_root, plan, allow_partial=True)
        self.assertEqual(text_readback.call_count, 3)
        for provider in evaluator.TEXT_PROVIDERS:
            text_readback.assert_any_call(roots[provider], self.manifest["children"], plan, RUN, provider=provider)
        self.assertEqual(phone_readback.call_count, 2)
        for provider in evaluator.PHONE_PROVIDERS:
            phone_readback.assert_any_call(provider=provider, observer_root=roots[provider], source_root=source_root,
                plan=plan, run_identity=RUN, parent_rows=self.manifest["parents"])
        self.assertEqual(result["declared_children"], 64)
        self.assertEqual(result["counts"]["actual_high_level_calls"], 336)
        self.assertEqual(result["new_model_calls_during_evaluation"], 0)

    def test_public_entry_rejects_wrong_source_pin_before_readers(self):
        plan = copy.deepcopy(PLAN)
        plan["evaluate_code_sha256"] = "0" * 64
        roots = {p: Path("mock-" + p) for p in evaluator.TEXT_PROVIDERS + evaluator.PHONE_PROVIDERS}
        with patch.object(evaluator, "module_at") as loader, self.assertRaisesRegex(ValueError, "EVALUATOR_SOURCE_PIN"):
            evaluator.evaluate(Path("mock-source-no-real-input"), roots, plan, RUN)
        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
