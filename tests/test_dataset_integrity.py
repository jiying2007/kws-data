"""检查错误数据不能因更新 catalog 哈希而绕过语义校验。"""
import json
import pathlib
import shutil
import tempfile
import unittest
from tools.codex_assets import __main__ as archive


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.original = archive.ROOT
        self.temp = tempfile.TemporaryDirectory(prefix="kws-data-integrity-")
        self.root = pathlib.Path(self.temp.name) / "repo"
        shutil.copytree(self.original, self.root,
                        ignore=shutil.ignore_patterns(".git", "__pycache__"))
        archive.ROOT = self.root

    def tearDown(self):
        archive.ROOT = self.original
        self.temp.cleanup()

    def rebind(self, index, filename, key):
        catalog = json.loads((self.root / "catalog.json").read_text())
        entry = catalog["datasets"][index]
        entry[key] = archive.sha(self.root / entry["path"] / filename)
        archive.write_json(self.root / "catalog.json", catalog)

    def test_clean(self):
        result = archive.verify()
        self.assertEqual(result["recordings"], 42)

    def test_corrupted_audio(self):
        audio = next((self.root / "datasets/qwen3-reviewed-v1/audio").glob("*.wav"))
        payload = bytearray(audio.read_bytes())
        payload[-1] ^= 1
        audio.write_bytes(payload)
        with self.assertRaisesRegex(ValueError, "file SHA mismatch"):
            archive.verify()

    def test_relabel_with_rebound_manifest(self):
        manifest = self.root / "datasets/qwen3-reviewed-v1/manifest.json"
        data = json.loads(manifest.read_text())
        data["recordings"][0]["intended_text"] = "你好小温"
        archive.write_json(manifest, data)
        self.rebind(0, "manifest.json", "manifest_sha256")
        with self.assertRaisesRegex(ValueError, "positive keyword/text mismatch|review label mismatch"):
            archive.verify()

    def test_speaker_split_reassignment(self):
        path = self.root / "datasets/qwen3-reviewed-v1/splits.json"
        data = json.loads(path.read_text())
        data["recordings"][0]["split"] = "development_a"
        archive.write_json(path, data)
        self.rebind(0, "splits.json", "splits_sha256")
        with self.assertRaisesRegex(ValueError, "speaker split mismatch"):
            archive.verify()

    def test_asr_identity_reassignment(self):
        path = self.root / "datasets/qwen3-train-asr-v1/asr-review.jsonl"
        rows = archive.read_rows(path)
        rows[0]["asr_model_sha256"] = "0" * 64
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                        encoding="utf-8")
        self.rebind(1, "asr-review.jsonl", "review_sha256")
        with self.assertRaisesRegex(ValueError, "ASR asset identity mismatch"):
            archive.verify()

    def mutate_json(self, relative, update, index=None, rebind_key=None):
        path = self.root / relative
        data = json.loads(path.read_text())
        update(data)
        archive.write_json(path, data)
        if index is not None:
            self.rebind(index, path.name, rebind_key)

    def test_unknown_source(self):
        self.mutate_json('catalog.json', lambda c: c['datasets'][0].update(source_ref='absent'))
        with self.assertRaisesRegex(ValueError, 'unknown source_ref'):
            archive.verify()

    def test_dataset_identity_mismatch(self):
        self.mutate_json('catalog.json', lambda c: c['datasets'][0].update(dataset_id='other'))
        with self.assertRaisesRegex(ValueError, 'dataset ID mismatch'):
            archive.verify()

    def test_unknown_review_policy(self):
        self.mutate_json('catalog.json', lambda c: c['datasets'][0].update(review_policy_ref='absent'))
        with self.assertRaisesRegex(ValueError, 'unknown review_policy_ref'):
            archive.verify()

    def test_unknown_review_type(self):
        self.mutate_json('catalog.json', lambda c: c['datasets'][0].update(review_type='magic'))
        with self.assertRaisesRegex(ValueError, 'review type mismatch'):
            archive.verify()

    def test_source_model_mismatch(self):
        self.mutate_json('datasets/qwen3-reviewed-v1/manifest.json',
                         lambda m: m['source_model'].update(model_sha256='0' * 64), 0, 'manifest_sha256')
        with self.assertRaisesRegex(ValueError, 'source model identity mismatch'):
            archive.verify()

    def test_observed_flag_cannot_be_cleared(self):
        self.mutate_json('datasets/qwen3-reviewed-v1/splits.json',
                         lambda s: s['recordings'][0].update(observed_development=False), 0, 'splits_sha256')
        with self.assertRaisesRegex(ValueError, 'observed development flag required'):
            archive.verify()

    def test_split_qualification_cannot_be_enabled(self):
        self.mutate_json('datasets/qwen3-reviewed-v1/splits.json',
                         lambda s: s.update(formal_qualification_allowed=True), 0, 'splits_sha256')
        with self.assertRaisesRegex(ValueError, 'split qualification forbidden'):
            archive.verify()

    def test_source_scope_cannot_be_promoted(self):
        self.mutate_json('catalog.json', lambda c: next(iter(c['sources'].values()))['rights'].update(
            commercial_output_license='approved'))
        with self.assertRaisesRegex(ValueError, 'unsupported rights scope'):
            archive.verify()

    def test_unknown_kind(self):
        self.mutate_json('datasets/qwen3-reviewed-v1/manifest.json',
                         lambda m: m['recordings'][0].update(kind='anything'), 0, 'manifest_sha256')
        with self.assertRaisesRegex(ValueError, 'unsupported recording kind'):
            archive.verify()

    def test_bool_frame_count(self):
        self.mutate_json('datasets/qwen3-reviewed-v1/manifest.json',
                         lambda m: m['recordings'][0].update(frames=True), 0, 'manifest_sha256')
        with self.assertRaisesRegex(ValueError, 'invalid frame count'):
            archive.verify()

    def test_wrong_declared_sample_rate(self):
        self.mutate_json('datasets/qwen3-reviewed-v1/manifest.json',
                         lambda m: m['recordings'][0].update(sample_rate_hz=8000), 0, 'manifest_sha256')
        with self.assertRaisesRegex(ValueError, 'invalid declared sample rate'):
            archive.verify()

    def test_review_path_escape(self):
        self.mutate_json('catalog.json', lambda c: c['datasets'][0].update(review_file='../../README.md'))
        with self.assertRaisesRegex(ValueError, 'unsafe asset path'):
            archive.verify()

    def test_audio_symlink_forbidden(self):
        path = next((self.root / 'datasets/qwen3-reviewed-v1/audio').glob('*.wav'))
        saved = self.root / 'same-bytes.wav'
        path.rename(saved)
        path.symlink_to(saved)
        with self.assertRaisesRegex(ValueError, 'symlink asset path forbidden|unsafe asset path'):
            archive.verify()

    def test_bad_catalog_hash(self):
        self.mutate_json('catalog.json', lambda c: c['datasets'][0].update(manifest_sha256='not-a-hash'))
        with self.assertRaisesRegex(ValueError, 'invalid catalog SHA'):
            archive.verify()

    def test_content_identity_independent_of_catalog_format(self):
        before = archive.verify()['content_ids']
        catalog = self.root / 'catalog.json'
        catalog.write_text(json.dumps(json.loads(catalog.read_text()), ensure_ascii=False))
        self.assertEqual(before, archive.verify()['content_ids'])

    def test_export_native_roles_without_invented_alignment(self):
        from unittest.mock import patch
        commit = 'a' * 40
        selected = 'qwen3-xiaowo-reviewed-development-20260928'
        with patch.object(archive, 'git_identity', return_value=(commit, '')), \
                patch.object(archive, 'git_tracked_paths', return_value={
                    p.relative_to(self.root).as_posix() for p in self.root.rglob('*') if p.is_file()}):
            receipt = archive.export_receipt([selected], commit, archive.sha(self.root / 'catalog.json'))
        self.assertEqual(len(receipt['datasets'][0]['recordings']), 20)
        self.assertFalse(receipt['qualification_allowed'])
        rows = receipt['datasets'][0]['recordings']
        self.assertEqual({r['split'] for r in rows}, {'train', 'development_a', 'development_b'})
        self.assertTrue(all(r['review_method'] == 'human' for r in rows))
        self.assertFalse(any(set(r) & {'tokens', 'start_s', 'end_s', 'speaker_age'} for r in rows))

    def test_export_requires_pins_and_clean_checkout(self):
        from unittest.mock import patch
        commit = 'a' * 40
        selected = ['qwen3-xiaowo-reviewed-development-20260928']
        checksum = archive.sha(self.root / 'catalog.json')
        with patch.object(archive, 'git_identity', return_value=(commit, ' M catalog.json')):
            with self.assertRaisesRegex(ValueError, 'clean data checkout'):
                archive.export_receipt(selected, commit, checksum)
        with patch.object(archive, 'git_identity', return_value=(commit, '')), \
                patch.object(archive, 'git_tracked_paths', return_value={
                    p.relative_to(self.root).as_posix() for p in self.root.rglob('*') if p.is_file()}):
            with self.assertRaisesRegex(ValueError, 'data commit mismatch'):
                archive.export_receipt(selected, 'b' * 40, checksum)
            with self.assertRaisesRegex(ValueError, 'pinned catalog SHA mismatch'):
                archive.export_receipt(selected, commit, '0' * 64)
            with self.assertRaisesRegex(ValueError, 'unknown or duplicate dataset'):
                archive.export_receipt(['unknown'], commit, checksum)

    def test_current_catalog_rejects_legacy_schema(self):
        self.mutate_json('catalog.json', lambda c: c.update(schema_version=1))
        with self.assertRaisesRegex(ValueError, 'expected catalog schema 2'):
            archive.verify()

    def mutate_review(self, update):
        relative = 'datasets/qwen3-reviewed-v1/audio-review.jsonl'
        path = self.root / relative
        rows = archive.read_rows(path)
        update(rows)
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        self.rebind(0, path.name, 'review_sha256')

    def test_receipt_keyword_bool_rejected(self):
        self.mutate_review(lambda rows: rows[0].update(keyword_id=True))
        with self.assertRaisesRegex(ValueError, 'invalid review keyword ID'):
            archive.verify()

    def test_receipt_keyword_float_rejected(self):
        self.mutate_review(lambda rows: rows[0].update(keyword_id=1.0))
        with self.assertRaisesRegex(ValueError, 'invalid review keyword ID'):
            archive.verify()

    def test_receipt_schema_rejected(self):
        self.mutate_review(lambda rows: rows[0].update(schema_version=True))
        with self.assertRaisesRegex(ValueError, 'invalid review schema'):
            archive.verify()

    def test_receipt_missing_reviewer_rejected(self):
        self.mutate_review(lambda rows: rows[0].pop('reviewer_id'))
        with self.assertRaisesRegex(ValueError, 'missing human reviewer identity'):
            archive.verify()

    def test_manifest_schema_rejected(self):
        self.mutate_json('datasets/qwen3-reviewed-v1/manifest.json',
                         lambda m: m.update(schema_version=True), 0, 'manifest_sha256')
        with self.assertRaisesRegex(ValueError, 'invalid manifest schema'):
            archive.verify()

    def test_duplicate_recording_ids_rejected(self):
        def same_id(data):
            data['recordings'][1]['recording'] = data['recordings'][0]['recording']
        self.mutate_json('datasets/qwen3-reviewed-v1/manifest.json', same_id, 0, 'manifest_sha256')
        with self.assertRaisesRegex(ValueError, 'duplicate recording ID'):
            archive.verify()

    def test_export_rejects_asset_absent_from_pinned_tree(self):
        from unittest.mock import patch
        commit = 'a' * 40
        tracked = {p.relative_to(self.root).as_posix() for p in self.root.rglob('*') if p.is_file()}
        # Research WAVs outside the catalog are not consumer assets.
        manifest_path = self.root / 'datasets/qwen3-reviewed-v1/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        audio = manifest_path.parent / manifest['recordings'][0]['path']
        tracked.remove(audio.relative_to(self.root).as_posix())
        with patch.object(archive, 'git_identity', return_value=(commit, '')), \
                patch.object(archive, 'git_tracked_paths', return_value=tracked):
            with self.assertRaisesRegex(ValueError, 'absent from pinned Git tree'):
                archive.export_receipt(['qwen3-xiaowo-reviewed-development-20260928'],
                                       commit, archive.sha(self.root / 'catalog.json'))

    def test_real_git_export_rejects_ignored_audio(self):
        import subprocess
        def git(*args):
            return subprocess.check_output(['git', '-C', str(self.root),
                '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid'] + list(args),
                text=True, stderr=subprocess.DEVNULL).strip()
        git('init')
        git('add', '.')
        git('commit', '-m', 'fixture')
        selected = ['qwen3-xiaowo-reviewed-development-20260928']
        result = archive.export_receipt(selected, git('rev-parse', 'HEAD'),
                                        archive.sha(self.root / 'catalog.json'))
        self.assertEqual(len(result['datasets'][0]['recordings']), 20)
        manifest_path = self.root / 'datasets/qwen3-reviewed-v1/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        row = manifest['recordings'][0]
        source = manifest_path.parent / row['path']
        ignored = manifest_path.parent / 'incoming/ignored.wav'
        ignored.parent.mkdir()
        source.rename(ignored)
        row['path'] = 'incoming/ignored.wav'
        archive.write_json(manifest_path, manifest)
        self.rebind(0, 'manifest.json', 'manifest_sha256')
        git('add', '-A')
        git('commit', '-m', 'ignored asset fixture')
        self.assertEqual(git('status', '--porcelain'), '')
        with self.assertRaisesRegex(ValueError, 'absent from pinned Git tree'):
            archive.export_receipt(selected, git('rev-parse', 'HEAD'),
                                   archive.sha(self.root / 'catalog.json'))

    def test_human_archive_identity_mismatch(self):
        self.mutate_review(lambda rows: rows[0].update(review_archive_sha256='0' * 64))
        with self.assertRaisesRegex(ValueError, 'human review archive identity mismatch'):
            archive.verify()

    def test_stale_human_manifest_receipt_hash(self):
        self.mutate_review(lambda rows: rows[0].update(review_basis='new explanation'))
        with self.assertRaisesRegex(ValueError, 'human manifest receipt SHA mismatch'):
            archive.verify()

    def test_wrong_manifest_evidence_class(self):
        self.mutate_json('datasets/qwen3-reviewed-v1/manifest.json',
                         lambda m: m.update(evidence_class='asr-selected-synthetic-kws-dataset-v1'),
                         0, 'manifest_sha256')
        with self.assertRaisesRegex(ValueError, 'manifest evidence class mismatch'):
            archive.verify()


if __name__ == "__main__":
    unittest.main()
