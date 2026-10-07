#!/usr/bin/env python3
"""Verify, restore and recount this saved research archive without model calls."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def decode(data):
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def safe_name(name):
    require(type(name) is str and bool(name) and '\\' not in name, 'Invalid member name')
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and str(p) == name, 'Unsafe member path')
    return p


def verify_identity(data, row):
    require(len(data) == row['bytes'] and digest(data) == row['sha256'], 'Byte/hash mismatch')
    if 'git_blob_sha1' in row:
        blob = b'blob ' + str(len(data)).encode() + b'\0' + data
        require(hashlib.sha1(blob).hexdigest() == row['git_blob_sha1'], 'Git blob mismatch')


def unpack(data, destination, expected_members, expanded_cap):
    require(not destination.exists(), 'Refusing existing restore path')
    destination.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        infos = z.infolist()
        require(len(infos) == len({x.filename for x in infos}), 'Duplicate ZIP member')
        files = [x for x in infos if not x.is_dir()]
        require(len(files) == expected_members, 'Wrong ZIP member count')
        require(sum(x.file_size for x in files) <= expanded_cap, 'Expanded ZIP bound')
        for info in infos:
            name = info.filename.rstrip('/') if info.is_dir() else info.filename
            safe_name(name)
            require(not stat.S_ISLNK(info.external_attr >> 16), 'ZIP symlink')
            require(not (info.flag_bits & 1), 'Encrypted ZIP member')
        payloads = {}
        for info in files:
            require(info.file_size <= 8 * 1024**2, 'Per-file cap')
            raw = z.read(info.filename)
            require(len(raw) == info.file_size, 'Expanded size mismatch')
            path = destination / info.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            payloads[info.filename] = raw
    if 'artifact-freeze.json' in payloads:
        frozen = decode(payloads['artifact-freeze.json'])['files']
        require(set(frozen) == set(payloads) - {'artifact-freeze.json'}, 'Freeze membership mismatch')
        for name, expected in frozen.items():
            require(digest(payloads[name]) == expected, 'Frozen member mismatch')
    return payloads


def verify(root, destination):
    require(not any(p.is_symlink() for p in root.rglob('*')), 'Archive symlink')
    manifest = decode((root / 'archive-manifest.json').read_bytes())
    records = manifest['files']
    require(len(records) == len({r['path'] for r in records}), 'Duplicate archive file record')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual == {r['path'] for r in records} | {'archive-manifest.json'}, 'Archive membership mismatch')
    for row in records:
        safe_name(row['path'])
        require(row['mode'] == '100644', 'Unexpected published mode')
        verify_identity((root / row['path']).read_bytes(), row)
    restored = {}
    total_members = 0
    require(len(manifest['raw_archives']) == 5 and {x['label'] for x in manifest['raw_archives']} ==
            {'tts', 'blind', 'asr', 'failure1', 'failure2'}, 'Raw archive scope')
    for item in manifest['raw_archives']:
        data = b''.join((root / row['path']).read_bytes() for row in item['parts'])
        verify_identity(data, item)
        for part in item['parts']:
            verify_identity((root / part['path']).read_bytes(), part)
        safe_name(item['restored_name'])
        archive_path = destination / 'original-zips' / item['restored_name']
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(data)
        view = destination / item['label']
        restored[item['label']] = (view, unpack(data, view, item['members'], item['expanded_cap_bytes']))
        total_members += item['members']
    tts, _ = restored['tts']
    asr, _ = restored['asr']
    blind, blind_data = restored['blind']
    resources = decode((tts / 'resources.json').read_bytes())
    require(digest(blind_data['blind-inputs.zip']) == resources['blind']['blind_archive_sha256'], 'Blind archive identity')
    require(digest(blind_data['blind-input-freeze.json']) == resources['blind']['blind_freeze_sha256'], 'Blind freeze identity')
    frozen = decode(blind_data['blind-input-freeze.json'])
    inner = unpack(blind_data['blind-inputs.zip'], destination / 'blind-inner', 7, 4 * 1024**2)
    require(set(inner) == {row['path'] for row in frozen['files']}, 'Blind inner membership')
    for row in frozen['files']:
        verify_identity(inner[row['path']], row)
    require(digest(inner['job.json']) == frozen['job_sha256'] == resources['blind']['blind_job_sha256'], 'Blind job identity')
    method = root / 'method'
    sys.path.insert(0, str(method))
    try:
        spec = importlib.util.spec_from_file_location('archive_compare_saved', method / 'compare_saved.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        comparison = module.compare_saved(root / 'plan.json', tts / 'generation', asr,
                                          resources['candidate_sha256'], resources['asr_candidate_sha256'])
    finally:
        sys.path.pop(0)
    projection = decode((root / 'result-projection.json').read_bytes())
    outcomes = {(x['audio_id'], x['model']): x for x in comparison['outcomes']}
    require(len(comparison['clips']) == len(projection['rows']) == 6, 'Comparison row count')
    for index, (actual, expected) in enumerate(zip(comparison['clips'], projection['rows']), 1):
        require(expected['source_id'] == f'qwen6-{index:03d}', 'Source order')
        for public, private in [('blind_audio_id', 'audio_id'), ('preset', 'preset'), ('keyword', 'keyword'),
                                ('intended_text', 'intended_text'), ('machine_state', 'machine_state'),
                                ('handling', 'handling'), ('quarantine_reasons', 'quarantine_reasons'),
                                ('tail_completeness', 'tail_completeness_truth'),
                                ('human_truth_established', 'human_truth_established'),
                                ('actual_positive_established', 'actual_positive_established'),
                                ('training_admitted', 'training_admitted')]:
            require(actual[private] == expected[public], 'Projected comparison mismatch: ' + public)
        for key, model in [('sensevoice_text', module.MODELS[0]), ('qwen_asr_text', module.MODELS[1])]:
            outcome = outcomes[(actual['audio_id'], model)]
            require(outcome['status'] == 'complete' and outcome['normalized_text'] == expected[key], 'ASR projection mismatch')
        require(actual['signal_measurements']['flags'] == expected['signal_flags'], 'Signal flags mismatch')
        require(actual['signal_measurements']['generation_flags'] == expected['generation_flags'], 'Generation flags mismatch')
    counts = Counter(x['machine_state'] for x in comparison['clips'])
    handling = Counter(x['handling'] for x in comparison['clips'])
    for key in ('machine_agreement_match', 'machine_agreement_mismatch', 'machine_dispute'):
        require(counts[key] == projection['counts'][key], 'Aggregate machine count mismatch')
    require(handling['weak_machine_supported_candidate'] == projection['counts']['weak_machine_supported'], 'Support count')
    require(handling['quarantine'] == projection['counts']['quarantined'], 'Quarantine count')
    require(not any(x['human_truth_established'] or x['actual_positive_established'] or x['training_admitted'] for x in comparison['clips']), 'Evidence scope')
    comparison_bytes = module._audio.encode_json(comparison)
    require(digest(comparison_bytes) == projection['source_hashes']['fixed_comparison_sha256'], 'Frozen comparison reproduction')
    comparison_path = destination / 'recomputed-comparison.json'
    comparison_path.write_bytes(comparison_bytes)
    adapter_path = root / 'panel-integration/integrate_saved_screen.py'
    spec = importlib.util.spec_from_file_location('archive_panel_adapter', adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    observations, gate, binding = adapter.integrate(root / 'panel-integration/panel', comparison_path, tts / 'generation')
    for name, value in [('OBSERVATIONS.json', observations), ('GATE-REPORT.json', gate), ('BINDING-REPORT.json', binding)]:
        encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        require(encoded == (root / 'panel-integration/result' / name).read_bytes(), 'Panel integration reproduction: ' + name)
    require(gate['status'] == 'REJECTED' and binding['gate_exit_code'] == 1, 'Expected panel decision')
    require(binding['training_rows'] == binding['sohee_screen_attempts'] == 0, 'Panel execution/admission scope')
    return {'status': 'PASS', 'physical_files': len(records) + 1, 'raw_archives': len(restored),
            'restored_zip_members': total_members, 'verified_blind_inner_members': 7,
            'compared_rows': 6, 'machine_states': dict(counts), 'handling': dict(handling),
            'model_calls': 0, 'network_calls': 0, 'human_truth': 'UNKNOWN',
            'panel_gate_status': gate['status'], 'panel_gate_return_code': binding['gate_exit_code'],
            'training_rows': 0, 'sohee_screen_attempts': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore', type=Path, help='Keep exact original ZIPs and extracted evidence in a new directory')
    args = parser.parse_args()
    if args.restore:
        destination = args.restore.resolve()
        require(not destination.exists(), 'Restore destination already exists')
        require(ROOT != destination and ROOT not in destination.parents, 'Restore outside the archive')
        require(not any(p.is_symlink() for p in (args.restore, *args.restore.parents)), 'Restore symlink path')
        destination.mkdir(parents=True)
        result = verify(ROOT, destination)
    else:
        with tempfile.TemporaryDirectory(prefix='qwen6-archive-') as temporary:
            result = verify(ROOT, Path(temporary))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
