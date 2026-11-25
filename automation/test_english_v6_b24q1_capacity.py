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
definition = importlib.util.spec_from_file_location("b24q1_capacity_fixture", HERE / "english_v6_b24q1_capacity.py")
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
        ident = f"b24q1-replication-invented-{number}"
        (source / "wav" / (ident + ".wav")).write_bytes(b"\0" * 46)
        native = source / "native" / ident
        native.mkdir()
        (native / "000.s16le").write_bytes(b"\0\0")
    for row in budget["natural"]["members"].values():
        stem = "b24q1-replication-natural-" + row["source_id"]
        (source / "wav" / (stem + ".wav")).write_bytes(b"\0" * row["wav_bytes"])
        (source / "flac" / (stem + ".flac")).write_bytes(b"\0" * row["flac_bytes"])
    return directory, source, corpus, design, budget


class CapacityContracts(unittest.TestCase):
    def test_primary_producer_retains128_limit_and_reserves_consumed_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('primary_producer', 'source_producer'):
                (root / name).mkdir(); (root / name / 'original.py').write_bytes(b'# invented original\n')
            (root / 'FETCH.json').write_bytes(b'{}')
            report = cap.primary_producer_members(root)
            self.assertEqual(report['limit_bytes'], 128_000_000)
            self.assertTrue(report['future_consumption_reserved'])
            self.assertFalse(report['original_quality_authorized'])
            with patch.object(cap, 'CORE_LIMIT', 100):
                with self.assertRaises(ValueError): cap.primary_producer_members(root)
            self.assertFalse((root / 'CONSUMPTION.json').exists())

    def test_actual_retained_r1_bytes_require_new_primary_envelope(self):
        # Exact original retained bytes; future audio remains a frozen bound.
        stop = 308_497_677 + 1_697_604
        expanded = stop + 50_759_903 + cap.TTS_AUDIO_BOUND + cap.NEW_METADATA_LIMIT
        packed_bytes = cap.zip_bound(expanded, cap.MAX_MEMBERS, cap.MAX_MEMBERS * cap.MAX_PATH_BYTES)
        self.assertEqual(expanded, 502_543_408)
        self.assertEqual(packed_bytes, 593_072_007)
        self.assertGreater(packed_bytes, 512_000_000)
        self.assertLess(packed_bytes, cap.PRIMARY_SOURCE_LIMIT)

    def test_full_primary_r1_stop_budget_and_saved_member_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); prior = root / 'prior'; prior.mkdir()
            for name in cap.R1_STOP_FILES:
                target = prior / 'ORIGINAL_B24R1_STOP' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(packed() if name.endswith('.zip') else b'{}')
            for name in cap.PRIMARY_KNOWN_FILES:
                (prior / name).write_bytes(packed() if name.endswith('.zip') else b'{}')
            (prior / 'TRANSPORT').mkdir(); (prior / 'TRANSPORT/RESULT.json').write_bytes(b'{}')
            corpus = root / 'corpus.tar.gz'; design = corpus_fixture(corpus)
            budget = cap.primary_source_budget(prior, corpus, design)
            self.assertEqual(len(budget['r1_stop_files']), 21)
            self.assertEqual(budget['nested_original_R1_source_limit_bytes'], 512_000_000)
            self.assertEqual(budget['limit_bytes'], 768_000_000)
            self.assertFalse(budget['original_quality_authorized'])
            (prior / 'CAPACITY.json').write_text(json.dumps(budget))
            (prior / 'CONSUMPTION.json').write_bytes(b'{}')
            self.assertEqual(cap.primary_generation_capacity(prior, corpus, design), budget)
            changed = dict(budget, maximum_zip_bytes=budget['maximum_zip_bytes'] - 1)
            (prior / 'CAPACITY.json').write_text(json.dumps(changed))
            with self.assertRaises(ValueError): cap.primary_generation_capacity(prior, corpus, design)
            (prior / 'CAPACITY.json').unlink(); (prior / 'CONSUMPTION.json').unlink()
            source = root / 'source'; shutil.copytree(prior, source)
            (source / 'CAPACITY.json').write_text(json.dumps(budget))
            for name in ('wav', 'flac', 'native'): (source / name).mkdir()
            for number in range(96):
                ident = 'b24q1-primary-invented-' + str(number)
                (source / 'wav' / (ident + '.wav')).write_bytes(b'\0' * 46)
                folder = source / 'native' / ident; folder.mkdir()
                (folder / '000.s16le').write_bytes(b'\0\0')
            for row in budget['natural']['members'].values():
                stem = 'b24q1-primary-natural-' + row['source_id']
                (source / 'wav' / (stem + '.wav')).write_bytes(b'\0' * row['wav_bytes'])
                (source / 'flac' / (stem + '.flac')).write_bytes(b'\0' * row['flac_bytes'])
            checked = cap.primary_source_members(source)
            self.assertEqual(checked['original_wavs'], 120)
            self.assertEqual(checked['r1_stop_files'], 21)
            target = source / 'ORIGINAL_B24R1_STOP/source.zip'; original = target.read_bytes()
            target.write_bytes(b'changed')
            with self.assertRaises(ValueError): cap.primary_source_members(source)
            target.write_bytes(original)
            budget['nested_original_R1_source_limit_bytes'] += 1
            (source / 'CAPACITY.json').write_text(json.dumps(budget))
            with self.assertRaises(ValueError): cap.primary_source_members(source)
            with patch.object(cap, 'PRIMARY_SOURCE_LIMIT', 1000):
                with self.assertRaises(ValueError): cap.primary_source_budget(prior, corpus, design)
            self.assertFalse((prior / 'CONSUMPTION.json').exists())

    def test_nested_r1_source_keeps_its_old512_limit(self):
        files = {name: dict(bytes=1, sha256='a' * 64) for name in cap.R1_STOP_FILES}
        files['source.zip']['bytes'] = 512_000_000
        with patch.object(cap, 'inventory', return_value=dict(files=files, directories=[])):
            cap._r1_stop(Path('/invented'))
            files['source.zip']['bytes'] += 1
            with self.assertRaises(ValueError): cap._r1_stop(Path('/invented'))
            files['source.zip']['bytes'] = 1
            files['native_lv60.zip']['bytes'] = 128_000_001
            with self.assertRaises(ValueError): cap._r1_stop(Path('/invented'))

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
        self.assertEqual(cap.archive_limit("replication_source"), 1_280_000_000)
        self.assertEqual(cap.archive_limit("replication_producer"), 1_024_000_000)
        self.assertEqual(cap.PRIMARY_SOURCE_LIMIT, 768_000_000)
        self.assertEqual(cap.R1_SOURCE_LIMIT, 512_000_000)
        self.assertEqual(cap.archive_limit("primary_source"), 768_000_000)
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
            (prior / "CAPACITY.json").write_text(json.dumps(budget))
            (prior / "CONSUMPTION.json").write_bytes(b"{}")
            self.assertEqual(cap.replica_generation_capacity(prior, corpus, design), budget)
            changed = dict(budget, maximum_zip_bytes=budget["maximum_zip_bytes"] - 1)
            (prior / "CAPACITY.json").write_text(json.dumps(changed))
            with self.assertRaises(ValueError): cap.replica_generation_capacity(prior, corpus, design)
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
            original = source / "wav/b24q1-replication-natural-1-2-0000.wav"
            original.write_bytes(b"truncated")
            with self.assertRaisesRegex(ValueError, "whole natural"):
                cap.source_members(source)
        with tempfile.TemporaryDirectory() as temporary:
            prior, source, corpus, design, budget = source_fixture(Path(temporary))
            (source / "unexpected.bin").write_bytes(b"must not be omitted from budget")
            with self.assertRaisesRegex(ValueError, "complete new metadata"):
                cap.source_members(source)
            (source / "unexpected.bin").unlink()
            (source / "native/b24q1-replication-invented-0/000.s16le").write_bytes(b"x" * (2 * cap.TTS_NATIVE_SAMPLES + 1))
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
