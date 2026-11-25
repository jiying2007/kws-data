"""Invented packaging fixtures only: no real PCM, decoder, model or network."""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import shutil
import stat
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent
definition = importlib.util.spec_from_file_location("b24r1_capacity_fixture", HERE / "english_v6_b24r1_replication_capacity.py")
cap = importlib.util.module_from_spec(definition)
definition.loader.exec_module(cap)


def packed(name="x", payload=b"invented fixture", compression=zipfile.ZIP_STORED):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=compression) as archive:
        archive.writestr(name, payload)
    return output.getvalue()


def primary(directory):
    directory.mkdir()
    for name in cap.PRIMARY_FILES:
        path = directory / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(packed() if name.endswith(".zip") else b"{}")


def corpus_fixture(path):
    roster = []
    with tarfile.open(path, "w:gz") as archive:
        for number in range(24):
            source_id = f"1-2-{number:04d}"
            row = dict(source_id=source_id, speaker="1", chapter="2")
            roster.append(row)
            samples = 300001 if number == 0 else 1
            fields = (16000 << 44) | (15 << 36) | samples
            header = b"fLaC" + b"\x80\0\0\x22" + b"\0" * 10 + fields.to_bytes(8, "big") + b"x" * 16
            member = tarfile.TarInfo(f"LibriSpeech/train-clean-100/1/2/{source_id}.flac")
            member.size = len(header)
            archive.addfile(member, io.BytesIO(header))
    return dict(roster=roster, upstream=dict(bytes=path.stat().st_size, sha256=cap.sha(path.read_bytes())))


def source_fixture(root):
    directory = root / "prior"
    directory.mkdir()
    primary(directory / "PRIMARY_QUALITY")
    for name in cap.KNOWN_FILES:
        (directory / name).write_bytes(packed() if name.endswith(".zip") else b"{}")
    (directory / "TRANSPORT").mkdir()
    (directory / "TRANSPORT/RESULT.json").write_bytes(b"{}")
    corpus = root / "corpus.tar.gz"
    design = corpus_fixture(corpus)
    budget = cap.source_budget(directory, corpus, design)
    source = root / "source"
    shutil.copytree(directory, source)
    (source / "CAPACITY.json").write_text(json.dumps(budget))
    for name in ("wav", "flac", "native"):
        (source / name).mkdir()
    for number in range(96):
        ident = f"b24r1-replication-invented-{number}"
        (source / "wav" / (ident + ".wav")).write_bytes(b"\0" * 46)
        native = source / "native" / ident
        native.mkdir()
        (native / "000.s16le").write_bytes(b"\0\0")
    for row in budget["natural"]["members"].values():
        stem = "b24r1-replication-natural-" + row["source_id"]
        (source / "wav" / (stem + ".wav")).write_bytes(b"\0" * row["wav_bytes"])
        (source / "flac" / (stem + ".flac")).write_bytes(b"\0" * row["flac_bytes"])
    return directory, source, corpus, design, budget


class CapacityContracts(unittest.TestCase):
    def test_nested_original_primary_allows_only_its_r1_source_composite(self):
        files = {name: dict(bytes=100, sha256='a' * 64) for name in cap.PRIMARY_FILES}
        files['inputs/source.zip']['bytes'] = 400
        value = dict(files=files, directories=[])
        with patch.object(cap, 'inventory', return_value=value), \
             patch.object(cap, 'CORE_LIMIT', 256), patch.object(cap, 'PRIMARY_SOURCE_LIMIT', 1024):
            self.assertEqual(cap._primary(Path('/invented-proof')), value)
            for name in cap.PRIMARY_FILES - {'inputs/source.zip'}:
                files[name]['bytes'] = 400
                with self.assertRaisesRegex(ValueError, 'member byte boundary'):
                    cap._primary(Path('/invented-proof'))
                files[name]['bytes'] = 100
            files['inputs/source.zip']['bytes'] = 1025
            with self.assertRaises(ValueError):
                cap._primary(Path('/invented-proof'))

    def test_only_two_explicit_composite_roles_have_larger_limit(self):
        self.assertEqual(cap.archive_limit("core"), 128_000_000)
        self.assertEqual(cap.archive_limit("replication_source"), 768_000_000)
        self.assertEqual(cap.archive_limit("replication_producer"), 512_000_000)
        self.assertEqual(cap.PRIMARY_SOURCE_LIMIT, 512_000_000)
        for role in (None, "primary", "source", "producer", "B0", "replication", "replication_result"):
            with self.assertRaises(ValueError):
                cap.archive_limit(role)

    def test_actual_compressed_and_expanded_boundaries_and_role_misuse(self):
        raw = packed(payload=b"z" * 400, compression=zipfile.ZIP_DEFLATED)
        with patch.object(cap, "CORE_LIMIT", 256), patch.object(cap, "COMPOSITE_LIMIT", 1024), \
             patch.object(cap, "SOURCE_LIMIT", 1024):
            with self.assertRaises(ValueError):
                cap.check_archive(raw, "core")
            for role in ("replication_source", "replication_producer"):
                with cap.check_archive(raw, role) as archive:
                    self.assertEqual(archive.read("x"), b"z" * 400)
        raw = packed(payload=b"z")
        with patch.object(cap, "CORE_LIMIT", len(raw)):
            with cap.check_archive(raw, "core"):
                pass
        with patch.object(cap, "CORE_LIMIT", len(raw) - 1):
            with self.assertRaises(ValueError):
                cap.check_archive(raw, "core")
        raw = packed(payload=b"z" * 2000, compression=zipfile.ZIP_DEFLATED)
        with patch.object(cap, "SOURCE_LIMIT", 1999):
            with self.assertRaises(ValueError):
                cap.check_archive(raw, "replication_source")

    def test_crc_paths_symlinks_and_member_ceiling_are_never_skipped(self):
        raw = packed(payload=b"unique-fixture-data")
        altered = raw.replace(b"unique-fixture-data", b"Unique-fixture-data", 1)
        with self.assertRaises((ValueError, zipfile.BadZipFile)):
            cap.check_archive(altered, "replication_source")
        for name in ("../x", "/x", "x/../y", "x\\y", "x" * 1025):
            with self.assertRaises(ValueError):
                cap.check_archive(packed(name), "replication_source")
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            entry = zipfile.ZipInfo("link")
            entry.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(entry, b"target")
        with self.assertRaises(ValueError):
            cap.check_archive(output.getvalue(), "replication_producer")
        with patch.object(cap, "MAX_MEMBERS", 0):
            with self.assertRaises(ValueError):
                cap.check_archive(packed(), "replication_source")

    def test_exact_tts_math_and_conservative_zip_envelope(self):
        self.assertEqual(cap.TTS_NATIVE_SAMPLES, 330750)
        self.assertEqual(cap.TTS_AUDIO_BOUND, 109588224)
        for size in (0, 1, 127, 65536):
            raw = packed(payload=b"x" * size, compression=zipfile.ZIP_DEFLATED)
            self.assertGreaterEqual(cap.zip_bound(size, 1, 1), len(raw))
        self.assertGreater(cap.zip_bound(100, 2, 2), cap.zip_bound(100, 1, 1))

    def test_retained_stop_layout_requires_prospective_replica_source_capacity(self):
        # Measured original B24 TRAIN/LIVE bytes plus a modest complete PRIMARY
        # proof and all fixed future source reserves. This is a packaging-only
        # example; it grants no source or quality permission.
        expanded = 240_000_000 + 50_759_903 + cap.TTS_AUDIO_BOUND + 10_000_000 + cap.NEW_METADATA_LIMIT
        total = cap.zip_bound(expanded, cap.MAX_MEMBERS, cap.MAX_MEMBERS * cap.MAX_PATH_BYTES)
        self.assertEqual(total, 524_411_765)
        self.assertGreater(total, 512_000_000)
        self.assertLess(total, cap.SOURCE_LIMIT)

    def test_whole_natural_rejects_and_all_original_proof_bytes_are_budgeted(self):
        with tempfile.TemporaryDirectory() as temporary:
            prior, source, corpus, design, budget = source_fixture(Path(temporary))
            first = budget["natural"]["members"]["LibriSpeech/train-clean-100/1/2/1-2-0000.flac"]
            self.assertEqual(first["samples"], 300001)
            self.assertEqual(first["wav_bytes"], 600046)
            self.assertTrue(budget["natural"]["duration_rejects_retained"])
            self.assertEqual(len(budget["primary_files"]), 18)
            self.assertEqual(budget["original_train_artifacts"], 4)
            report = cap.source_members(source)
            self.assertEqual(report["original_wavs"], 120)
            self.assertFalse(report["original_quality_authorized"])
            self.assertFalse(budget["original_quality_authorized"])
            self.assertLessEqual(report["maximum_zip_bytes"], budget["maximum_zip_bytes"])
            self.assertEqual(sum(budget[name] for name in ("new_model_calls", "new_source_calls", "new_observer_calls")), 0)

    def test_capacity_failure_occurs_before_consumption_or_source_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            prior, source, corpus, design, budget = source_fixture(Path(temporary))
            with patch.object(cap, "SOURCE_LIMIT", 1000):
                with self.assertRaisesRegex(ValueError, "before source consumption"):
                    cap.source_budget(prior, corpus, design)
            self.assertFalse((prior / "CONSUMPTION.json").exists())
            (prior / "CONSUMPTION.json").write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "precedes"):
                cap.source_budget(prior, corpus, design)

    def test_missing_changed_or_symlinked_original_proof_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            prior, source, corpus, design, budget = source_fixture(Path(temporary))
            target = source / "PRIMARY_QUALITY/inputs/B0.zip"
            target.write_bytes(b"changed original bytes")
            with self.assertRaisesRegex(ValueError, "every original PRIMARY"):
                cap.source_members(source)
            target.unlink()
            with self.assertRaises(ValueError):
                cap.source_members(source)
            target.symlink_to(prior / "PRIMARY_QUALITY/inputs/B0.zip")
            with self.assertRaisesRegex(ValueError, "symlinks"):
                cap.source_members(source)

    def test_whole_audio_and_metadata_envelopes_do_not_allow_silent_drops(self):
        with tempfile.TemporaryDirectory() as temporary:
            prior, source, corpus, design, budget = source_fixture(Path(temporary))
            original = source / "wav/b24r1-replication-natural-1-2-0000.wav"
            original.write_bytes(b"truncated")
            with self.assertRaisesRegex(ValueError, "whole natural"):
                cap.source_members(source)
        with tempfile.TemporaryDirectory() as temporary:
            prior, source, corpus, design, budget = source_fixture(Path(temporary))
            (source / "unexpected.bin").write_bytes(b"must not be omitted from budget")
            with self.assertRaisesRegex(ValueError, "complete new metadata"):
                cap.source_members(source)
            (source / "unexpected.bin").unlink()
            (source / "native/b24r1-replication-invented-0/000.s16le").write_bytes(b"x" * (2 * cap.TTS_NATIVE_SAMPLES + 1))
            with self.assertRaisesRegex(ValueError, "native PCM"):
                cap.source_members(source)

    def test_saved_budget_cannot_claim_a_larger_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            prior, source, corpus, design, budget = source_fixture(Path(temporary))
            budget["limit_bytes"] += 1
            (source / "CAPACITY.json").write_text(json.dumps(budget))
            with self.assertRaisesRegex(ValueError, "same fixed"):
                cap.source_members(source)

    def test_complete_producer_has_exact_roots_and_future_marker_reserve(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary(root / "primary_quality")
            for name in ("replication_producer", "source_producer"):
                (root / name).mkdir()
                (root / name / "original.py").write_bytes(b"# original fixture code\n")
            (root / "FETCH.json").write_bytes(b"{}")
            before = cap.producer_members(root)
            self.assertTrue(before["future_consumption_reserved"])
            self.assertFalse(before["original_quality_authorized"])
            (root / "CONSUMPTION.json").write_bytes(b"{}")
            after = cap.producer_members(root)
            self.assertEqual(before["actual_bytes"] - after["actual_bytes"], cap.CONSUMPTION_RESERVE - 2)
            with patch.object(cap, "COMPOSITE_LIMIT", after["maximum_zip_bytes"] - 1):
                with self.assertRaisesRegex(ValueError, "before scoring consumption"):
                    cap.producer_members(root)
            (root / "primary_quality/inputs/B0.zip").unlink()
            with self.assertRaises(ValueError):
                cap.producer_members(root)


if __name__ == "__main__":
    unittest.main()
