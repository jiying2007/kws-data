"""Verify an assembled research SDK using saved bytes only; never run inference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def need(ok, message):
    if not ok:
        raise ValueError(message)


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def verify(root):
    root = Path(root).absolute()
    manifest = json.loads((root / 'PACKAGE_MANIFEST.json').read_bytes())
    need(manifest.get('schema') == 'english-research-sdk-delivery-v1', 'package schema')
    expected = manifest['files']
    files = set()
    for path in root.rglob('*'):
        need(not path.is_symlink(), 'package symlink')
        if path.is_file(): files.add(path.relative_to(root).as_posix())
    need(files == set(expected) | {'PACKAGE_MANIFEST.json', 'SHA256SUMS'}, 'package exact inventory')
    for name, pin in expected.items():
        need(not Path(name).is_absolute() and '..' not in Path(name).parts and '\\' not in name,
            'unsafe package member')
        need(identity((root / name).read_bytes()) == pin, 'package identity:' + name)
    checksum_names = sorted(set(expected) | {'PACKAGE_MANIFEST.json'})
    checksum_text = ''.join(hashlib.sha256((root / name).read_bytes()).hexdigest() + '  ' + name + '\n'
        for name in checksum_names)
    need((root / 'SHA256SUMS').read_text() == checksum_text, 'checksums exact inventory')
    h = module(root / 'verification/english_coverage_fullpcm_once.py', 'delivery_frozen_pcm_parser')
    inc = module(root / 'verification/english_hardneg_increment_once.py', 'delivery_frozen_head_parser')
    card = json.loads((root / 'MODEL_CARD.json').read_bytes())
    head = (root / 'models/selected-head.bin').read_bytes()
    npz = (root / 'models/selected-head.npz').read_bytes()
    need(identity(head) == card['head'] and identity(npz) == card['npz'], 'selected model identity')
    need(manifest.get('selected_model') == card['head'], 'manifest selected model differs from model card')
    inc.verify_saved_npz_bin(npz, head)
    need(card['model_kind'] == 'frozen_acoustic_embedding_plus_1536_to_32_to_3_research_head'
        and card['strong_english_fsmn_ctc'] is False and card['ssc305_qualified'] is False
        and card['independent_far'] == card['independent_frr'] == card['word_end_latency'] == 'NOT_MEASURED',
        'research completion boundary')
    result = json.loads((root / 'evidence/fullpcm/RESULT.json').read_bytes())
    receipt = json.loads((root / 'evidence/fullpcm/SUPERVISOR_EXIT.json').read_bytes())
    need(result['passed'] is True and result['candidate_head'] == card['head']
        and receipt['passed'] is True and receipt['returncode'] == 0
        and receipt['timed_out'] is False and receipt['error'] is None
        and not receipt['cleanup']['descendants_found'], 'same-model successful fullPCM evidence')
    need(str(result['run_id']) == str(card['fullpcm_source']['run_id']) and result['head_sha'] == card['fullpcm_source']['head_sha']
        and str(receipt['run_id']) == str(result['run_id']) and receipt['head_sha'] == result['head_sha'], 'fullPCM execution identity')
    if card['selection'] == 'teacher-retention-research':
        need(result['status'] == 'VERIFIED_TEACHER_RETENTION_HEAD_FULLPCM_NATIVE_ARM', 'new candidate lacks own fullPCM result')
        training = json.loads((root / 'evidence/training/RESULT.json').read_bytes())
        training_exit = json.loads((root / 'evidence/training/SUPERVISOR_EXIT.json').read_bytes())
        source = card['training_source']
        gates = {'all_291_case_regression', 'natural32_zero_events', 'source_balanced_reviewed_ce_decreased',
            'actual_weights_changed', 'normalization_byte_identical', 'native_saved_npz_bin_parity'}
        need(training.get('schema') == 'english-train-teacher-retention-once-v1'
            and training.get('status') == 'RETAINED_RESEARCH_TEACHER_RETENTION_HEAD'
            and training.get('passed') is True and training.get('research_retention_allowed') is True
            and training.get('numerical_execution_completed') is True
            and set(training.get('gates', {})) == gates and all(v is True for v in training['gates'].values()),
            'selected teacher candidate did not pass every fixed training gate')
        need(str(training.get('run_id')) == str(source['run_id']) and training.get('head_sha') == source['head_sha']
            and training.get('run_attempt') == 1 and training.get('candidate', {}).get('bin') == card['head']
            and training.get('candidate', {}).get('npz') == card['npz'], 'selected teacher training/weight identity')
        need(training.get('teacher_train_bag_forwards') == 225 and training.get('teacher_train_head_windows') == 8370
            and training.get('teacher_development_forwards') == training.get('teacher_reviewed_forwards') == 0,
            'teacher used unexpected data or compute')
        need(training_exit.get('passed') is True and training_exit.get('returncode') == 0
            and training_exit.get('error') is None and training_exit.get('timed_out') is False
            and not training_exit.get('cleanup', {}).get('descendants_found')
            and str(training_exit.get('run_id')) == str(source['run_id']) and training_exit.get('head_sha') == source['head_sha'],
            'selected teacher training supervisor did not pass')
    else:
        need(card['selection'] == 'coverage-retained-research' and result['status'] == 'VERIFIED_NEW_COVERAGE_INSTALLED_C_FULLPCM'
            and card['head'] == h.HEAD and card['npz'] == h.NPZ, 'unknown or changed coverage selection')
    observed = []
    for backend in ('native', 'arm'):
        need((root / ('sdk-' + backend) / 'models/causal-head.bin').read_bytes() == head, 'SDK stale weight')
        need(identity((root / ('sdk-' + backend) / 'lib/libekws_research_stream.a').read_bytes()) == h.SDK_LIBRARIES[backend], 'SDK library identity')
        need(hashlib.sha256((root / ('sdk-' + backend) / 'include/ekws_research_stream.h').read_bytes()).hexdigest() == h.PUBLIC_HEADER_SHA, 'SDK public header')
        for index, case in enumerate(card['cases']):
            rid = case['id']; reference = json.loads((root / 'tests' / (rid + '.reference.json')).read_bytes())
            need(reference['case'] == case and identity((root / 'tests' / (rid + '.pcm')).read_bytes()) ==
                {'bytes': 2*case['samples'], 'sha256': case['pcm_sha256']}, 'case PCM/reference identity')
            trace = (root / 'evidence/fullpcm/traces' / (backend + '-' + rid + '.trace')).read_bytes()
            parsed = h.parse_trace(trace, case)
            metadata = json.loads((root / 'evidence/fullpcm' / (backend + '-case' + str(index) + '.stdout')).read_bytes())
            item = {'case': case, 'reference': reference['scores'], 'embeddings': reference['embeddings']}
            report = h.compare_trace(item, parsed, metadata); report['backend'] = backend
            need(report['passed'], 'saved fullPCM mismatch:' + backend + ':' + rid)
            cached = json.loads((root / 'evidence/fullpcm' / (backend + '-case' + str(index) + '.comparison.json')).read_bytes())
            need(report == cached, 'saved comparison differs from actual trace')
            observed.append(report)
    need(len(observed) == 10 and observed == result['comparisons'], 'fullPCM complete comparisons')
    return {'status': 'VERIFIED_RESEARCH_SDK_PACKAGE_SAVED_BYTES_ONLY', 'files': len(files),
        'selected_model': card['head'], 'fullpcm_comparisons_rechecked': len(observed),
        'new_model_calls': 0, 'new_audio_decodes': 0, 'ssc305_qualified': False,
        'independent_far': 'NOT_MEASURED', 'independent_frr': 'NOT_MEASURED'}


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    print(json.dumps(verify(Path(__file__).resolve().parent), ensure_ascii=False, indent=2))
