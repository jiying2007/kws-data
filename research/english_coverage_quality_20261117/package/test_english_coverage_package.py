"""Meaningful byte/gate/receipt tests. No numerical model library is imported."""
from __future__ import annotations
import copy
import io
import math
from pathlib import Path
import struct
import sys
import unittest
import warnings
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_coverage_package as package


def npy(shape, value=0.0, descr='<f4', fortran=False):
    header = repr({'descr': descr, 'fortran_order': fortran, 'shape': shape}).encode()
    header += b' ' * ((64 - (10 + len(header) + 1) % 64) % 64) + b'\n'
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + struct.pack('<f', value) * math.prod(shape)


def npz(overrides=None):
    output = io.BytesIO()
    overrides = overrides or {}
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_STORED) as archive:
        for key, shape in package.SHAPES.items():
            raw = npy(shape, 1.0 if key == 'scale96' else 0.0)
            archive.writestr(key + '.npy', overrides.get(key, raw))
    return output.getvalue()


def retained_documents(npz_raw, binary):
    flags = dict.fromkeys(package.FALSE_FLAGS, False)
    pins = {'run_id': '123', 'head_sha': 'a' * 40, 'auditor_sha256': 'b' * 64,
            'quality_artifact': {'bytes': 12, 'sha256': 'c' * 64}}
    result = {'status': 'COMPLETED_WEAK_COVERAGE_QUALITY_RESEARCH',
        'research_retention': {'passed': True, 'criteria': {'new8_all_27_planned_development_cases_eligible': True}},
        'planned_new_development_conditions': 27, 'eligible_new8_conditions': 27,
        'independent_source_count_increment': 0, 'new8_new_source_holdout': False,
        'coverage_candidate': package.identity(npz_raw),
        'coverage_candidate_binary': package.identity(binary), 'run_id': '123', 'head_sha': 'a' * 40, 'run_attempt': '1',
        **flags}
    audit = {'schema': 'english-coverage-independent-audit-v1', 'status': 'VERIFIED',
        'expected_external_binding': True, 'research_retention': {'passed': True},
        'quality_artifact': pins['quality_artifact'], 'expected_run': '123', 'expected_head': 'a' * 40,
        'candidate_head': package.identity(npz_raw), 'candidate_binary': package.identity(binary),
        'auditor_sha256': 'b' * 64, 'parity': {'host': {'passed': True}, 'arm': {'passed': True}}, **flags}
    return result, audit, pins


def parity_fixture(npz_raw, binary):
    rows = [{'sample': 16000 + i * 1280, 'local_sample': 16000 + i * 1280,
             'case_index': 0, 'condition': 'clean', 'id': 'synthetic-source-only', 'ready': i >= 15, 'keyword': 0,
             'logits': [0.0, 0.0, 0.0], 'probabilities': [1.0, 0.0, 0.0] if i >= 15 else [0.0] * 3}
            for i in range(16)]
    stream = b'EHSTR01\0' + struct.pack('<I', len(rows)) + b''.join(
        struct.pack('<QI', row['sample'], int(i == 0)) + b'\0' * 384 for i, row in enumerate(rows))
    stdout = ''.join(f"{row['sample']} {int(row['ready'])} 0 0 0 0 {row['probabilities'][0]} 0 0\n"
                     for row in rows).encode()
    blobs = {'runtime/head-stream.bin': stream,
             'runtime/reference-frames.jsonl': b''.join(package.canonical(row) for row in rows),
             'runtime/CASE_MAP.json': package.canonical([{'condition': 'clean', 'id': 'synthetic-source-only',
                 'records': 16, 'first_record': 0, 'offset': 0, 'observed_samples': 35200, 'head_outputs': 1}]),
             'evidence/head-native.stdout': stdout}
    executable = bytearray(64); executable[:6] = b'\x7fELF\x02\x01'; struct.pack_into('<H', executable, 18, 62)
    blobs['runtime/head-native'] = bytes(executable)
    compiler = ['cc', '-std=c99', '-O2', '-ffp-contract=off', '-fno-fast-math', '-Wall', '-Wextra', '-Werror',
                'english_causal_head.c', 'head_replay.c', '-lm', '-o', '/fixture/head-native']
    runner = ['/fixture/head-native', '/fixture/models/coverage-head.bin', '/fixture/runtime/head-stream.bin']
    for name, argv in [('build-head-native', compiler), ('head-native', runner)]:
        blobs['evidence/' + name + '.exit.json'] = package.canonical({'argv': argv, 'returncode': 0,
            'error': None, 'process_group_killed': False, 'process_group_terminated': False})
    versions = {}
    for name, argv in [('host-compiler-version', ['cc', '--version']),
                       ('arm-compiler-version', ['arm-linux-gnueabihf-gcc', '--version']),
                       ('qemu-version', ['qemu-arm', '--version'])]:
        raw_version = ('synthetic-source-only-' + name).encode()
        blobs['evidence/' + name + '.stdout'] = raw_version
        blobs['evidence/' + name + '.exit.json'] = package.canonical({'argv': argv, 'returncode': 0,
            'error': None, 'process_group_killed': False, 'process_group_terminated': False})
        versions[name] = package.identity(raw_version)
    report = {'schema': 'english-coverage-host-c-head-parity-v1', 'passed': True,
        'scope': 'CANDIDATE_CACHED_EMBEDDING_HEAD_ONLY', 'candidate': package.identity(npz_raw),
        'candidate_binary': package.identity(binary), 'source_hashes': dict(package.C_SOURCE_SHA),
        'exact_samples': True, 'exact_ready': True, 'exact_events': True,
        'input_embeddings': 16, 'head_outputs': 1, 'cases': 1,
        'max_abs_logit_error': 0.0, 'max_abs_probability_error': 0.0,
        'threshold': 0.8, 'confirm_embeddings': 3, 'cooldown_embeddings': 13,
        'c_frontend': 'NOT_RUN', 'ssc305': 'NOT_RUN',
        'tool_versions': versions,
        'compiler_argv': compiler, 'runner_argv': runner, 'executable': package.identity(bytes(executable))}
    for field, path in {'stream_binary': 'runtime/head-stream.bin',
        'reference_frames': 'runtime/reference-frames.jsonl', 'case_map': 'runtime/CASE_MAP.json',
        'raw_stdout': 'evidence/head-native.stdout'}.items():
        report[field] = package.identity(blobs[path])
    return report, blobs


class CoveragePackageTests(unittest.TestCase):
    def test_byte_export_keeps_exact_existing_abi(self):
        raw = npz()
        actual = package.export_head_bytes(raw)
        expected = b'EHKWS01\0' + b'\0' * (96 * 4) + struct.pack('<f', 1.0) * 96
        expected += b'\0' * ((49475 - 192) * 4)
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 197908)
        self.assertEqual(raw, raw[:])

    def test_npy_refuses_wrong_abi_nonfinite_and_nonpositive_scale(self):
        cases = [{'w1': npy((1536, 32))}, {'mean96': npy((96,), descr='>f4')},
                 {'w1': npy((32, 1536), fortran=True)}, {'b2': npy((3,), float('nan'))},
                 {'scale96': npy((96,), 0.0)}]
        for changes in cases:
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                package.export_head_bytes(npz(changes))

    def test_unsafe_zip_and_duplicate_json_fail_closed(self):
        for names in [['../escape'], ['/absolute'], ['a\\b'], ['same', 'same']]:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                output = io.BytesIO()
                with zipfile.ZipFile(output, 'w') as archive:
                    for name in names:
                        archive.writestr(name, b'x')
            with self.subTest(names=names), self.assertRaises(ValueError):
                package.open_zip(output.getvalue())
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w') as archive:
            link = zipfile.ZipInfo('symlink'); link.create_system = 3; link.external_attr = 0o120777 << 16
            archive.writestr(link, b'outside')
        with self.assertRaises(ValueError):
            package.open_zip(output.getvalue())
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_JSON'):
            package.read_json(b'{"passed":false,"passed":true}')

    def test_audit_success_cannot_override_failed_quality_or_missing_arm(self):
        raw = npz(); binary = package.export_head_bytes(raw)
        result, audit, pins = retained_documents(raw, binary)
        package.validate_retention(result, audit, pins, raw, binary)
        failed = copy.deepcopy(result); failed['research_retention']['passed'] = False
        with self.assertRaisesRegex(ValueError, 'NOT_RETAINED'):
            package.validate_retention(failed, audit, pins, raw, binary)
        for field in ('expected_external_binding',):
            unbound = copy.deepcopy(audit); unbound[field] = False
            with self.assertRaises(ValueError):
                package.validate_retention(result, unbound, pins, raw, binary)
        old_arm = copy.deepcopy(audit); old_arm['parity']['arm']['passed'] = False
        with self.assertRaisesRegex(ValueError, 'HOST_AND_ARM'):
            package.validate_retention(result, old_arm, pins, raw, binary)

    def test_old_head_or_formal_claim_cannot_be_rebound(self):
        raw = npz(); binary = package.export_head_bytes(raw)
        result, audit, pins = retained_documents(raw, binary)
        old = copy.deepcopy(audit); old['candidate_head']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'ARTIFACT_AND_HEAD_BINDING'):
            package.validate_retention(result, old, pins, raw, binary)
        formal = copy.deepcopy(result); formal['qualification_allowed'] = True
        with self.assertRaisesRegex(ValueError, 'RESEARCH_FLAGS'):
            package.validate_retention(formal, audit, pins, raw, binary)

    def test_partial_new8_cannot_pass_via_a_true_summary(self):
        raw = npz(); binary = package.export_head_bytes(raw)
        result, audit, pins = retained_documents(raw, binary)
        result['eligible_new8_conditions'] = 26
        with self.assertRaisesRegex(ValueError, 'COMPLETE_NEW8'):
            package.validate_retention(result, audit, pins, raw, binary)

    def test_saved_c_receipt_is_compared_not_just_trusted(self):
        raw = npz(); binary = package.export_head_bytes(raw)
        report, blobs = parity_fixture(raw, binary)
        self.assertTrue(package.validate_parity('host', report, blobs, raw, binary)['passed'])
        changed = dict(blobs)
        changed['evidence/head-native.stdout'] = changed['evidence/head-native.stdout'].replace(
            b'35200 1 0', b'35200 1 2')
        rebound = copy.deepcopy(report)
        rebound['raw_stdout'] = package.identity(changed['evidence/head-native.stdout'])
        with self.assertRaisesRegex(ValueError, 'CLOCK_READY_EVENT'):
            package.validate_parity('host', rebound, changed, raw, binary)

    def test_runtime_contract_and_default_have_no_numerical_import(self):
        import contextlib
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(package.main([]), 0)
        self.assertIn('SOURCE_CONTRACT_ONLY', output.getvalue())
        self.assertEqual(package.POLICY['classes'], 3)
        self.assertEqual(package.POLICY['eof_padding_samples'], 0)
        self.assertFalse(set(('numpy', 'torch', 'onnx', 'onnxruntime')) & set(sys.modules))


if __name__ == '__main__':
    unittest.main()
