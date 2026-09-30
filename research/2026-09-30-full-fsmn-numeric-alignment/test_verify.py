"""Mutation coverage for storage integrity and independent scalar semantics."""
import sys
sys.dont_write_bytecode = True
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

BASE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location('archive_verify', BASE / 'verify.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def reseal(root):
    rows = []
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name != 'archive-manifest.json':
            data = path.read_bytes()
            rows.append({'path': path.relative_to(root).as_posix(), 'bytes': len(data), 'sha256': v.digest(data)})
    write_json(root / 'archive-manifest.json', {'schema_version': 1,
               'purpose': 'historical-research-not-data-catalog', 'source_snapshot': v.SOURCE_SNAPSHOT, 'files': rows})


def rebind_logical(root, name, raw=None, packed=None):
    """Recompute *all* ordinary hashes, as an accidental/adversarial rebound would."""
    inventory = json.loads((root / 'logical-files.json').read_text())
    row = next(r for r in inventory['files'] if r['path'] == name)
    if raw is not None:
        row['original_bytes'], row['original_sha256'] = len(raw), v.digest(raw)
    if row['encoding'] == 'gzip-chunks':
        packed = gzip.compress(raw, compresslevel=9, mtime=0) if packed is None else packed
        for old in row['parts']:
            (root / old['path']).unlink()
        row['gzip_bytes'], row['gzip_sha256'], row['parts'] = len(packed), v.digest(packed), []
        for i, start in enumerate(range(0, len(packed), v.CHUNK_BYTES)):
            data = packed[start:start + v.CHUNK_BYTES]
            path = name + '.gz.parts/' + f'{i:04d}.bin'
            (root / path).write_bytes(data)
            row['parts'].append({'path': path, 'bytes': len(data), 'sha256': v.digest(data)})
    else:
        (root / name).write_bytes(raw)
    write_json(root / 'logical-files.json', inventory)
    reseal(root)


class ArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = v.read_archive(BASE)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.parent = Path(self.temp.name)
        self.root = self.parent / 'archive'
        shutil.copytree(BASE, self.root)

    def tearDown(self):
        self.temp.cleanup()

    def edit_index(self, edit):
        p = self.root / 'logical-files.json'
        data = json.loads(p.read_text())
        edit(data)
        write_json(p, data)
        reseal(self.root)

    def semantic_mutation(self, name, edit, error):
        payloads = dict(self.original)
        data = json.loads(payloads[name])
        edit(data)
        payloads[name] = json.dumps(data).encode()
        with self.assertRaisesRegex(ValueError, error):
            v.verify_semantics(payloads)

    def test_clean_archive(self):
        self.assertTrue(v.verify(self.root)['verified'])

    def test_source_snapshot_cannot_float(self):
        path = self.root / 'archive-manifest.json'
        data = json.loads(path.read_text())
        data['source_snapshot']['commit'] = 'main'
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, 'fixed source snapshot metadata'):
            v.verify(self.root)

    def test_required_license_inventory(self):
        (self.root / 'LICENSE.wekws').unlink()
        reseal(self.root)
        with self.assertRaisesRegex(ValueError, 'required physical inventory'):
            v.verify(self.root)

    def test_license_content_hash_rebound(self):
        (self.root / 'LICENSE.wekws').write_text('License omitted\n')
        reseal(self.root)
        with self.assertRaisesRegex(ValueError, 'historical Apache license identity'):
            v.verify(self.root)

    def test_attribution_notice_hash_rebound(self):
        (self.root / 'NOTICE.md').write_text('No attribution\n')
        reseal(self.root)
        with self.assertRaisesRegex(ValueError, 'historical attribution notice'):
            v.verify(self.root)

    def test_original_logical_identities(self):
        for name, sha in v.PINNED.items():
            self.assertEqual(v.digest(self.original[name]), sha)

    def test_corrupted_chunk(self):
        p = next(self.root.rglob('*.bin'))
        data = p.read_bytes()
        p.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        with self.assertRaisesRegex(ValueError, 'chunk identity'):
            v.verify(self.root)

    def test_missing_required_logical_file(self):
        self.edit_index(lambda x: x['files'].pop())
        with self.assertRaisesRegex(ValueError, 'required logical inventory'):
            v.verify(self.root)

    def test_duplicate_logical_inventory(self):
        self.edit_index(lambda x: x['files'].__setitem__(-1, x['files'][0]))
        with self.assertRaisesRegex(ValueError, 'required logical inventory'):
            v.verify(self.root)

    def test_piece_path_traversal_rebound(self):
        self.edit_index(lambda x: next(r for r in x['files'] if 'parts' in r)['parts'][0].update(path='../outside.bin'))
        with self.assertRaisesRegex(ValueError, 'unsafe path'):
            v.verify(self.root)

    def test_absolute_and_dot_paths(self):
        for name in ('/tmp/a', '../a', 'x/../a', 'x//a', 'x/./a', 'x\\a', ''):
            with self.subTest(name=name), self.assertRaises(ValueError):
                v.safe_read(self.root, name)

    def test_piece_order_rebound(self):
        self.edit_index(lambda x: next(r for r in x['files'] if r['path'] == 'pcm-native-result.json')['parts'].reverse())
        with self.assertRaisesRegex(ValueError, 'chunk path/order'):
            v.verify(self.root)

    def test_chunk_size_cap(self):
        p = next(self.root.rglob('*.bin'))
        p.write_bytes(b'x' * (v.CHUNK_BYTES + 1))
        with self.assertRaisesRegex(ValueError, 'physical member size cap'):
            v.verify(self.root)

    def test_logical_expansion_cap_rebound(self):
        self.edit_index(lambda x: next(r for r in x['files'] if 'parts' in r).update(original_bytes=v.MAX_LOGICAL_BYTES + 1))
        with self.assertRaisesRegex(ValueError, 'expanded size budget'):
            v.verify(self.root)

    def test_bounded_decompression_bomb(self):
        packed = gzip.compress(b'x' * (v.MAX_LOGICAL_BYTES + 1), mtime=0)
        with self.assertRaisesRegex(ValueError, 'expanded size budget'):
            v.decompress_bounded(packed, 100)

    def test_concatenated_gzip_rebound(self):
        name = 'full-oracle-result.json'
        packed = gzip.compress(self.original[name], mtime=0) + gzip.compress(b'{}', mtime=0)
        rebind_logical(self.root, name, packed=packed)
        with self.assertRaisesRegex(ValueError, 'concatenated/trailing gzip'):
            v.verify(self.root)

    def test_timestamped_gzip_rebound(self):
        name = 'full-oracle-result.json'
        rebind_logical(self.root, name, packed=gzip.compress(self.original[name], mtime=1))
        with self.assertRaisesRegex(ValueError, 'non-deterministic gzip header'):
            v.verify(self.root)

    def test_unexpected_binary_rebound(self):
        (self.root / 'model.bin').write_bytes(b'\xff\x00')
        reseal(self.root)
        with self.assertRaisesRegex(ValueError, 'required physical inventory'):
            v.verify(self.root)

    def test_unexpected_empty_directory(self):
        (self.root / 'unlisted').mkdir()
        with self.assertRaisesRegex(ValueError, 'unexpected directory'):
            v.verify(self.root)

    def test_binary_text_rebound(self):
        rebind_logical(self.root, 'pcm-native-run.log', b'\xff\x00')
        with self.assertRaises(UnicodeDecodeError):
            v.verify(self.root)

    def test_symlink_leaf(self):
        name = 'pcm-native-run.log'
        (self.root / name).unlink()
        (self.root / name).symlink_to(BASE / name)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            v.verify(self.root)

    def test_symlink_parent_directory(self):
        path = self.root / 'prior-stages'
        target = self.parent / 'moved'
        path.rename(target)
        path.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            v.verify(self.root)

    def test_symlink_archive_root(self):
        link = self.parent / 'alias'
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            v.verify(link)

    def test_symlink_ancestor_of_archive(self):
        link = self.parent / 'ancestor-alias'
        link.symlink_to(self.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            v.verify(link / 'archive')

    def test_approved_original_hash_rebound_rejected(self):
        name = 'resource-profile/result.json'
        data = json.loads(self.original[name])
        data['pipeline_cpu_RTF'] = 0.001
        rebind_logical(self.root, name, json.dumps(data).encode())
        with self.assertRaisesRegex(ValueError, 'approved original identity'):
            v.verify(self.root)

    def test_unpinned_historical_false_pass_hash_rebound(self):
        name = 'prior-stages/network-result-v1.json'
        data = json.loads(self.original[name])
        data['passed'] = True
        rebind_logical(self.root, name, json.dumps(data).encode())
        with self.assertRaisesRegex(ValueError, 'historical B failure'):
            v.verify(self.root)

    def test_local_affine_historical_source_hash_rebound(self):
        name = 'prior-stages/fsmn-before-full-oracle.c'
        rebind_logical(self.root, name, self.original[name] + b'\n/* changed */\n')
        with self.assertRaisesRegex(ValueError, 'local affine historical source binding'):
            v.verify(self.root)

    def test_cmvn_false_claim_hash_rebound(self):
        name = 'prior-stages/guard-cmvn-result.json'
        data = json.loads(self.original[name])
        data['cmvn_checks'][0]['max_abs'] = 0.5
        rebind_logical(self.root, name, json.dumps(data).encode())
        with self.assertRaisesRegex(ValueError, 'CMVN historical result'):
            v.verify(self.root)

    def test_generated_summary_hash_rebound(self):
        path = self.root / 'summary.json'
        data = json.loads(path.read_text())
        data['profile']['numerically_qualified'] = True
        write_json(path, data)
        reseal(self.root)
        with self.assertRaisesRegex(ValueError, 'recomputed summary mismatch'):
            v.verify(self.root)

    # Exercise semantic validators directly as well as end-to-end frozen hashes.
    # This proves the claims do not depend only on checksum failure.
    def test_semantic_pcm_full_field_false_pass(self):
        self.semantic_mutation('pcm-native-result.json', lambda x: x.update(full_fields_events_exact=True), 'PCM qualification flags')

    def test_semantic_pcm_non_score_event_change(self):
        self.semantic_mutation('pcm-native-result.json',
            lambda x: x['recordings'][0]['events'][0].update(available_audio_samples=1), 'non-score event equality')

    def test_semantic_pcm_score_summary(self):
        self.semantic_mutation('pcm-native-result.json',
            lambda x: x['recordings'][0].update(score_max_difference=0), 'event score summary')

    def test_semantic_pcm_probability_denominator(self):
        self.semantic_mutation('pcm-native-result.json',
            lambda x: x['recordings'][0]['calls'][0]['metrics']['probabilities'].update(elements=1), 'full2599 element coverage')

    def test_semantic_pcm_probability_finite(self):
        self.semantic_mutation('pcm-native-result.json',
            lambda x: x['recordings'][0]['calls'][0]['metrics']['probabilities'].update(max_abs=float('nan')), 'nonfinite JSON')

    def test_semantic_pcm_fbank_failure_retained(self):
        self.semantic_mutation('pcm-native-result.json',
            lambda x: next(c['metrics']['fbank'] for r in x['recordings'] for c in r['calls']
                           if c['metrics'] and c['metrics']['fbank']['fail_elements']).update(fail_elements=0), 'PCM retained scalar aggregate: fbank')

    def test_semantic_profile_audio_denominator(self):
        self.semantic_mutation('resource-profile/result.json', lambda x: x.update(audio_seconds=78.56), 'profile RTF numerator/denominator')

    def test_semantic_profile_recording_denominator(self):
        self.semantic_mutation('resource-profile/result.json',
            lambda x: x['passes'][0]['recordings'][0].update(audio_samples=1), 'profile saved native parity/denominator')

    def test_semantic_profile_disjoint_scope(self):
        self.semantic_mutation('resource-profile/result.json',
            lambda x: x['measurement_notes'].__setitem__(0, 'add callback twice'), 'profile nested callback scope')

    def test_semantic_profile_timing_component(self):
        self.semantic_mutation('resource-profile/result.json',
            lambda x: x['passes'][0]['recordings'][0]['components']['standalone_cmvn_micro'][0].update(cpu_ns=999), 'profile component timing totals')

    def test_semantic_profile_ssc_claim(self):
        self.semantic_mutation('resource-profile/result.json', lambda x: x.update(scope='SSC305 qualified'), 'profile qualification scope')

    def test_semantic_oracle_failure_not_overwritten(self):
        self.semantic_mutation('full-oracle-result.json', lambda x: x.update(old_B_still_failed=False), 'oracle does not overwrite B')

    def test_semantic_oracle_uninformative_bound(self):
        self.semantic_mutation('full-oracle-result.json',
            lambda x: [r.update(envelope_max=0.001) for r in x['records']], 'oracle envelope/final-logit outcome')

    def test_strict_json_duplicate_and_overflow(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                v.strict_json(raw)


if __name__ == '__main__':
    unittest.main(verbosity=2)
