"""B17 source leakage and fixed-roster checks; no waveform generation or model calls."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b17_natural_sources as natural
import english_v6_b17_source as synthetic
import english_v6_b17_source_ancestry as lineage


class SourceContracts(unittest.TestCase):
    def test_all_actual_legacy_readers_excluded(self):
        assessment, ancestry = lineage.read_plan()
        actual = set(ancestry["legacy_natural"]["all_exposed_speakers"])
        self.assertEqual(len(actual), 40)
        self.assertIn("1462", actual)
        self.assertLessEqual(actual, set(ancestry["natural_excluded_speakers"]))
        self.assertEqual(ancestry["metadata_census"]["remaining_unexposed_counts"],
                         {"dev-clean": 0, "dev-other": 9, "test-clean": 16})
        for panel, readers in (("primary", 12), ("replication", 9)):
            identities = assessment[panel]["natural"]["speakers"]
            self.assertEqual(len(identities), readers)
            self.assertFalse(set(identities) & actual)
        self.assertEqual(natural.CORPUS_SPLIT, "test-clean")

    def test_old_replication_and_all_completed_sources_are_required(self):
        archives = synthetic.required_archives()
        self.assertEqual(len(archives), len(set(archives)))
        self.assertTrue({"14883618584", "14878153098", "14789951699", "14788972705",
                         "14225893574", "14878351529"} <= set(archives))
        self.assertEqual(set(lineage.required_archives()), {"14225893574", "14788972705", "14878351529"})

    def test_new_synthetic_ranges_and_unchanged_minima(self):
        primary, replica = (synthetic.source_plan(panel) for panel in ("primary", "replication"))
        self.assertEqual({r["speaker_index"] for r in primary["rows"]}, set(range(718, 742)))
        self.assertEqual({r["speaker_index"] for r in replica["rows"]}, set(range(742, 766)))
        self.assertFalse({r["source_speaker_id"] for r in primary["rows"]} &
                         {r["source_speaker_id"] for r in replica["rows"]})
        self.assertEqual(primary["frozen_source_gate"],
                         {"positive_originals_min": 12, "ordinary_originals_min": 24, "paired_source_groups_min": 8})
        self.assertEqual(len(primary["rows"]), 96)
        self.assertFalse(primary["promotion_allowed"])
        self.assertTrue(all(r["observed_class"] is None and not r["training_allowed"] for r in primary["rows"]))

    def test_malformed_legacy_archive_stops_before_opening_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = lineage.read_plan()[1]["legacy_natural"]
            (Path(tmp) / (str(lock["artifact_id"]) + ".zip")).write_bytes(b"not-the-original")
            with self.assertRaisesRegex(ValueError, "historical archive identity"):
                lineage.verify_history(Path(tmp))

    def fixture(self, omitted=None, legacy_speaker=False):
        design = dict(natural.plan())
        metadata = "".join(f"{sp} | M | test-clean | 1.0 | Fixture\n" for sp in design["speakers"]).encode()
        design["metadata_speakers_sha256"] = hashlib.sha256(metadata).hexdigest()
        if legacy_speaker:
            design["excluded_speakers"] = [*design["excluded_speakers"], design["speakers"][0]]
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w") as archive:
            def add(name, raw):
                item = tarfile.TarInfo(name); item.size = len(raw)
                archive.addfile(item, io.BytesIO(raw))
            add("LibriSpeech/SPEAKERS.TXT", metadata)
            for speaker in design["speakers"]:
                n = 1 if speaker == omitted else 3
                lines = [f"{speaker}-100-{i:04d} PLEASE READ THIS ORDINARY SENTENCE\n" for i in range(n)]
                add(f"LibriSpeech/test-clean/{speaker}/100/{speaker}-100.trans.txt", "".join(lines).encode())
        payload.seek(0)
        return payload, design

    def test_exact_reader_selection_is_deterministic(self):
        raw, design = self.fixture()
        with tarfile.open(fileobj=raw) as archive:
            picked = natural.pick(archive, design)
        self.assertEqual(len(picked["primary"]), 24)
        self.assertEqual([r["speaker"] for r in picked["primary"]], [sp for sp in design["speakers"] for _ in range(2)])
        raw.seek(0)
        with tarfile.open(fileobj=raw) as archive:
            self.assertEqual(natural.roster_bytes(picked), natural.roster_bytes(natural.pick(archive, design)))

    def test_frozen_reader_shortfall_never_replaced(self):
        raw, design = self.fixture(omitted=natural.plan()["speakers"][0])
        with tarfile.open(fileobj=raw) as archive, self.assertRaisesRegex(ValueError, "STOP without replacement"):
            natural.pick(archive, design)

    def test_historical_speaker_reintroduction_rejected(self):
        raw, design = self.fixture(legacy_speaker=True)
        with tarfile.open(fileobj=raw) as archive, self.assertRaisesRegex(ValueError, "no substitutions"):
            natural.pick(archive, design)

    def test_census_mismatch_rejected_before_transcript_selection(self):
        raw, design = self.fixture()
        design["metadata_speakers_sha256"] = "0" * 64
        with tarfile.open(fileobj=raw) as archive, self.assertRaisesRegex(ValueError, "speaker census"):
            natural.pick(archive, design)

    def source_fixture(self, directory, change):
        design = natural.plan()
        roster = [dict(speaker=sp, source_id=f"{sp}-100-{i:04d}", chapter="100",
                       transcript="PLEASE READ THIS ORDINARY SENTENCE", transcript_file_sha256="0" * 64)
                  for sp in design["speakers"] for i in range(2)]
        rows = [dict(row, id="b17-natural-" + row["source_id"], source_group_id="librispeech-test-clean-" + row["speaker"],
                     role="development", training_allowed=False, human_gold=False) for row in roster]
        change(roster, rows)
        raw = natural.roster_bytes({"primary": roster})
        (directory / "ROSTERS.json").write_bytes(raw)
        meta = dict(schema=natural.SCHEMA, panel="primary", corpus_split=natural.CORPUS_SPLIT,
                    corpus_sha256=design["upstream"]["sha256"], design_sha256=natural.ASSESSMENT_SHA,
                    source_ancestry_sha256=lineage.ANCESTRY_SHA, rosters_sha256=hashlib.sha256(raw).hexdigest(),
                    selected_original_slots=24, selected_speakers=12, kws_forwards=0, training_allowed=False,
                    human_gold=False, all_prior_replication_sealed=True, source_rejections=[], rows=rows)
        (directory / "MANIFEST.json").write_text(json.dumps(meta))

    def test_legacy_reader_cannot_hide_behind_new_group_prefix(self):
        def tamper(roster, rows):
            for item in (roster[0], rows[0]):
                item["speaker"] = "1462"
                item["source_id"] = "1462-100-0000"
            rows[0]["source_group_id"] = "librispeech-test-clean-1462"
        with tempfile.TemporaryDirectory() as tmp:
            self.source_fixture(Path(tmp), tamper)
            with patch.object(natural.qa.pcm_tools, "inspect_wav", side_effect=AssertionError("must reject before PCM read")):
                with self.assertRaisesRegex(ValueError, "preselected speaker order"):
                    natural.read_source(Path(tmp))

    def test_duplicate_selected_original_rejected_before_pcm_read(self):
        def tamper(roster, rows):
            roster[1] = dict(roster[0])
            rows[1] = dict(rows[0])
        with tempfile.TemporaryDirectory() as tmp:
            self.source_fixture(Path(tmp), tamper)
            with patch.object(natural.qa.pcm_tools, "inspect_wav", side_effect=AssertionError("must reject before PCM read")):
                with self.assertRaisesRegex(ValueError, "preselected speaker order"):
                    natural.read_source(Path(tmp))

    def test_selected_source_disappearance_needs_explicit_quality_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.source_fixture(Path(tmp), lambda roster, rows: rows.pop())
            with patch.object(natural.qa.pcm_tools, "inspect_wav", side_effect=AssertionError("must reject before PCM read")):
                with self.assertRaisesRegex(ValueError, "never substitute or fabricate"):
                    natural.read_source(Path(tmp))


if __name__ == "__main__":
    unittest.main()
