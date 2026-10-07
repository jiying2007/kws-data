#!/usr/bin/env python3
"""Offline checks. Constructed transcript strings are tests, not ASR evidence."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
import freeze_lexicons as freeze

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "frozen"
SCORER = Path("/workspace/shared/kws-pipeline-restored/research/asr_evidence/calibrate.py")


class FrozenOfficialLexiconTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = freeze.load_scorer(SCORER)
        cls.source = freeze.read_archive(ROOT / "upstream/pypinyin-0.55.0.tar.gz")
        cls.report = json.loads((FROZEN / "freeze.json").read_bytes())
        cls.config = json.loads((FROZEN / "rules.json").read_bytes())
        cls.characters = json.loads((FROZEN / "characters.json").read_bytes())
        cls.phrases = json.loads((FROZEN / "phrases.json").read_bytes())
        cls.reverse = json.loads((FROZEN / "reading-roundtrip.json").read_bytes())
        cls.rules_sha = freeze.sha256((FROZEN / "rules.json").read_bytes())
        cls.rules = cls.scorer.Rules(cls.config, cls.rules_sha, cls.characters, cls.phrases, freeze.TARGETS)

    def row(self, text, **changes):
        result = {"raw_text": text, "status": "success", "completeness": "complete", "quality_flags": []}
        result.update(changes)
        return result

    def test_pinned_source_and_scorer(self):
        self.assertEqual(freeze.sha256(SCORER.read_bytes()), freeze.SCORER_SHA256)
        self.assertEqual(len((ROOT / "upstream/pypinyin-0.55.0.tar.gz").read_bytes()), freeze.ARCHIVE_BYTES)
        for name, expected in freeze.SOURCE_HASHES.items():
            self.assertEqual(freeze.sha256(self.source[name]), expected)
            self.assertEqual((ROOT / "upstream/python-pinyin" / name).read_bytes(), self.source[name])

    def test_every_retained_reference_hash_and_size(self):
        for index in ["upstream-references.json", "unicode-notice-references.json"]:
            for item in json.loads((ROOT / "upstream" / index).read_bytes()):
                raw = (ROOT / item["path"]).read_bytes()
                self.assertEqual(len(raw), item["bytes"])
                self.assertEqual(freeze.sha256(raw), item["sha256"])

    def test_pinned_data_submodules_reconstruct_shipped_dictionaries(self):
        characters = {}
        for line in (ROOT / "upstream/pinyin-data/pinyin.txt").read_text().splitlines():
            clean = line.split("#", 1)[0].strip()
            if clean:
                codepoint, reading = clean.split(":", 1)
                key = str(int(codepoint.removeprefix("U+"), 16))
                self.assertNotIn(key, characters)
                characters[key] = reading.strip()
        phrases = {}
        for line in (ROOT / "upstream/phrase-pinyin-data/pinyin.txt").read_text().splitlines():
            clean = line.split("#", 1)[0].strip()
            if clean:
                key, reading = clean.split(":", 1)
                key, rows = key.strip(), [[r] for r in reading.split()]
                if key not in phrases:
                    phrases[key] = rows
                else:
                    self.assertEqual(len(rows), len(phrases[key]))
                    for i, values in enumerate(rows):
                        for reading in values:
                            if reading not in phrases[key][i]:
                                phrases[key][i].append(reading)
        self.assertEqual(characters, json.loads(self.source["pypinyin/pinyin_dict.json"]))
        self.assertEqual(phrases, json.loads(self.source["pypinyin/phrases_dict.json"]))

    def test_all_frozen_artifact_hashes_and_sizes(self):
        for name, spec in self.report["artifacts"].items():
            raw = (FROZEN / name).read_bytes()
            self.assertEqual(len(raw), spec["bytes"])
            self.assertEqual(freeze.sha256(raw), spec["sha256"])

    def test_full_entry_counts_no_filter(self):
        self.assertEqual(len(self.characters), 41923)
        self.assertEqual(len(self.phrases), 47111)
        self.assertEqual(self.report["filtered_entries"], 0)
        self.assertEqual(self.report["removed_readings"], 0)

    def test_character_roundtrip_every_reading_and_order(self):
        restored = {k: ",".join(self.reverse[r] for r in values.split(",")) for k, values in self.characters.items()}
        self.assertEqual(restored, json.loads(self.source["pypinyin/pinyin_dict.json"]))

    def test_phrase_roundtrip_every_reading_and_order(self):
        restored = {k: [[self.reverse[r] for r in values] for values in rows] for k, rows in self.phrases.items()}
        self.assertEqual(restored, json.loads(self.source["pypinyin/phrases_dict.json"]))

    def test_reproducible_export_bytes(self):
        _, _, characters, phrases, reverse, checks = freeze.export(self.source, self.scorer)
        self.assertGreater(checks, 30)
        for name, value in [("characters.json", characters), ("phrases.json", phrases), ("reading-roundtrip.json", reverse)]:
            self.assertEqual(freeze.json_bytes(value), (FROZEN / name).read_bytes())

    def test_reading_map_injective(self):
        self.assertEqual(len(self.reverse), 1559)
        self.assertEqual(len(set(self.reverse.values())), 1559)
        for encoded, original in self.reverse.items():
            self.assertEqual(freeze.tone_digits(original), encoded)
            self.assertEqual(self.scorer.toneless(original), self.scorer.toneless(encoded))

    def test_special_combining_readings_preserved(self):
        for original, encoded in [("m̀", "m4"), ("m̄", "m1"), ("ê̄", "ê1"), ("ê̌", "ê3"), ("ḿ", "m2"), ("ế", "ê2")]:
            self.assertEqual(freeze.tone_digits(original), encoded)
            self.assertEqual(self.reverse[encoded], original)

    def test_umlaut_and_circumflex_are_not_discarded(self):
        self.assertEqual(freeze.tone_digits("ǘ"), "ü2")
        self.assertNotEqual(freeze.tone_digits("lǜ"), freeze.tone_digits("lù"))
        self.assertEqual(freeze.tone_digits("lǜ"), "lü4")
        self.assertNotEqual(freeze.tone_digits("ế"), freeze.tone_digits("é"))

    def test_unmarked_neutral_not_invented(self):
        for reading in ["de", "ma", "n", "ê"]:
            self.assertEqual(freeze.tone_digits(reading), reading)

    def test_targets_resolve_unambiguously(self):
        expected = [["ni3", "hao3", "xiao3", "wo1"], ["xiao3", "wo1", "xiao3", "wo1"]]
        for target, pronunciation in zip(freeze.TARGETS, expected):
            tokens, _ = self.rules.tokenize(target)
            self.assertEqual([t["readings"] for t in tokens], [[r] for r in pronunciation])
            self.assertTrue(all(t["kind"] == "lexical" for t in tokens))
        self.assertEqual(self.rules.tokenize(freeze.TARGETS[0])[1][0]["text"], "你好")

    def test_required_markers_are_frozen_and_block(self):
        self.assertEqual(self.config["uncertainty_markers"], list(self.scorer.REQUIRED_MARKERS))
        for marker in self.scorer.REQUIRED_MARKERS:
            derived = self.rules.derive(self.row("你好小窝" + marker))
            self.assertEqual([t["presence"] for t in derived["targets"]], ["unknown", "unknown"])

    def test_literal_targets(self):
        for i, target in enumerate(freeze.TARGETS):
            self.assertEqual(self.rules.derive(self.row(target))["targets"][i]["presence"], "positive")

    def test_polyphonic_character_still_ambiguous(self):
        self.assertEqual(self.characters[str(ord("好"))], "hao3,hao4")
        result = self.rules.derive(self.row("好"))
        self.assertTrue(all(t["presence"] == "unknown" for t in result["targets"]))
        self.assertTrue(all("lexicon_ambiguous_reading" in t["blockers"] for t in result["targets"]))

    def test_boundary_partial_and_homophone_candidates_abstain(self):
        for text in ["你好 小窝", "你好小", "你好小涡", "小窝 小窝"]:
            i = 1 if text.startswith("小窝") else 0
            self.assertEqual(self.rules.derive(self.row(text))["targets"][i]["presence"], "unknown")

    def test_oov_still_abstains(self):
        result = self.rules.derive(self.row("abcxyz"))
        self.assertTrue(all(t["presence"] == "unknown" for t in result["targets"]))

    def test_failures_and_unknown_completeness_abstain(self):
        for changes in [{"status": "error"}, {"completeness": "unknown"}, {"quality_flags": ["decoding_warning"]}]:
            result = self.rules.derive(self.row("你好小窝小窝小窝", **changes))
            self.assertEqual([t["presence"] for t in result["targets"]], ["unknown", "unknown"])

    def test_raw_source_is_rejected_not_silently_dropped(self):
        source_characters = json.loads(self.source["pypinyin/pinyin_dict.json"])
        for item in self.report["unmodified_source_contract_rejections"]:
            with self.assertRaises(self.scorer.ContractError):
                self.scorer.validate_readings(source_characters[item["codepoint"]].split(","))
        self.assertEqual(len(self.report["unmodified_source_contract_rejections"]), 4)

    def test_missing_mandatory_marker_rejected(self):
        config = deepcopy(self.config)
        config["uncertainty_markers"].pop()
        with self.assertRaises(self.scorer.ContractError):
            self.scorer.Rules(config, self.rules_sha, self.characters, self.phrases, freeze.TARGETS)

    def test_source_archive_tampering_fails(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path = Path(temp) / "tampered.tar.gz"
            path.write_bytes(b"not the official archive")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                freeze.read_archive(path)

    def test_no_pypinyin_execution(self):
        self.assertFalse(any(name == "pypinyin" or name.startswith("pypinyin.") for name in sys.modules))

    def test_cli_accepts_full_lexicons_with_not_run_test_envelopes(self):
        # Names, IDs and hashes in this temporary harness are invented. They
        # cannot claim actual ASR execution, and no test transcript is imported.
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            work = Path(temp)
            models = [{"model_id": "lexicon-contract-check-a", "revision": "a" * 40, "run_id": "not-run-contract-check-a"},
                      {"model_id": "lexicon-contract-check-b", "revision": "b" * 40, "run_id": "not-run-contract-check-b"}]
            row = {"recording_id": "invented-not-run-contract-check", "wav_sha256": "c" * 64,
                   "human_target_presence": ["unknown", "unknown"]}
            manifest = {"schema_version": "kws-asr-evidence-manifest-v1", "probe_set_id": "invented-lexicon-contract-test-only",
                        "execution_kind": "not_run_template", "target_order": freeze.TARGETS, "models": models, "records": [row]}
            manifest_bytes = freeze.json_bytes(manifest)
            (work / "manifest.json").write_bytes(manifest_bytes)
            for i, name in enumerate(["model-a.json", "model-b.json"]):
                envelope = {"schema_version": "kws-asr-evidence-input-v1", "execution_kind": "not_run_template",
                            "manifest_sha256": freeze.sha256(manifest_bytes), "rules_sha256": self.rules_sha, "model": models[i],
                            "records": [{"recording_id": row["recording_id"], "wav_sha256": row["wav_sha256"], "status": "not_run",
                                         "raw_text": None, "completeness": "unknown", "quality_flags": []}]}
                (work / name).write_bytes(freeze.json_bytes(envelope))
            command = [sys.executable, "-B", str(SCORER), "--manifest", str(work / "manifest.json"),
                       "--manifest-sha256", freeze.sha256(manifest_bytes), "--rules", str(FROZEN / "rules.json"),
                       "--rules-sha256", self.rules_sha, "--characters", str(FROZEN / "characters.json"), "--phrases", str(FROZEN / "phrases.json"),
                       "--model-a", str(work / "model-a.json"), "--model-b", str(work / "model-b.json"), "--output", str(work / "readout.json")]
            result = subprocess.run(command, text=True, capture_output=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads(result.stdout)
            readout = json.loads((work / "readout.json").read_bytes())
            self.assertEqual(receipt["rules_sha256"], self.rules_sha)
            self.assertEqual(readout["execution_kind"], "not_run_template")
            self.assertFalse(readout["inference_performed_by_this_tool"])
            self.assertEqual(readout["human_label_counts"], {"positive": 0, "negative": 0, "unknown": 2})


if __name__ == "__main__":
    unittest.main(verbosity=2)
