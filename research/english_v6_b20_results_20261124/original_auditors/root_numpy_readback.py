"""Independent readback of original B20 outputs; never runs an acoustic model."""
from pathlib import Path
from collections import Counter
import hashlib
import importlib.util
import io
import json
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'b20_actual_primary_readback'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_bytes())


def decode(array):
    labels = np.argmax(array, axis=1).tolist()
    runs = [(i, token) for i, token in enumerate(labels)
            if token != 0 and (i == 0 or labels[i - 1] != token)]
    tokens = [token for _, token in runs]
    frames = [runs[i + 2][0] for i in range(len(runs) - 2)
              if tokens[i:i + 3] == [1, 2, 3]]
    return dict(tokens=tokens, event_frames=frames,
                event_samples=[512 + 160 * i for i in frames], events=len(frames))


def main():
    raw = (OUT / 'artifacts/result-original.zip').read_bytes()
    need(len(raw) == 5916607 and sha(raw) == '9972b33b74facbf1e9c055eec3820b981b8d0f9c829791e52c17468bb4891eb5',
         'original result ZIP identity')
    z = zipfile.ZipFile(io.BytesIO(raw))
    need(z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == 2020,
         'complete original ZIP CRC and member uniqueness')
    result_raw = z.read('RESULT.json')
    need(sha(result_raw) == '86d4cc6a84494f18b67479db9d986f722103566b81ba7c1f6772a5c56d719ff7', 'RESULT identity')
    result = json.loads(result_raw)
    original_sources = []
    for manifest, screen, cohort in (
        ('b20_actual_source_readback/source/MANIFEST.json', 'b20_actual_source_readback/final/SCREEN.json', 'synthetic'),
        ('b17_primary_source_readback/sources/natural/MANIFEST.json', 'b17_primary_source_readback/screens/natural/SCREEN.json', 'natural'),
    ):
        originals = read(manifest)['rows']
        labels = read(screen)['rows' if cohort == 'synthetic' else 'decisions']
        need(len(originals) == len(labels), 'original source and screen denominators')
        need(len({r['id'] for r in originals}) == len(originals), 'unique original IDs')
        for row, label in zip(originals, labels, strict=True):
            need(row['id'] == label['id'] and row['pcm_sha256'] == label['pcm_sha256'], 'original ID and PCM join')
            if cohort == 'synthetic':
                need(all(label[k] == v for k, v in row.items() if k != 'observed_class'), 'synthetic SCREEN source identity')
                if label['observed_class'] is None:
                    need(label['tier'] == 'unknown', 'unknown synthetic remains excluded')
                    continue
                need(label['observed_class'] == row['requested_class'] in (0, 2), 'canonical observed class')
                need(label['tier'] in ('silver', 'weak'), 'qualified source tier')
                cell, positive = row['cell_id'], label['observed_class'] == 2
            else:
                need(label['source_id'] == row['source_id'] and label['group'] == row['source_group_id'], 'natural utterance and reader')
                if label['tier'] == 'unknown':
                    continue
                need(label['tier'] in ('silver', 'weak') and label['full_wake_veto'] is False
                     and type(label['exact_votes']) is int and label['exact_votes'] >= 2, 'qualified natural source')
                cell, positive = 'read-speech', False
            identity = dict(id=row['id'], group=row['source_group_id'], cohort=cohort, cell=cell,
                            positive=positive, tier=label['tier'], pcm_sha256=row['pcm_sha256'])
            original_sources.append((identity, row['samples']))
    need(len(original_sources) == 84, '63 synthetic plus 21 natural source authority')
    need(json.loads(z.read('ADMISSION.json'))['admitted_rows'] == [s[0] for s in original_sources], 'source-derived ADMISSION equality')
    need(len(result['records']) == len(original_sources), 'complete result source population')
    arrays_raw = z.read('ALL_FRAME_LOGITS.npz')
    need(sha(arrays_raw) == result['logits_sha256'] == '272e69d0f9bfd72aa8a9da606d67db3677991a214dd950df5cd04aa5bcd315cb', 'complete NPZ bytes')
    arrays = np.load(io.BytesIO(arrays_raw), allow_pickle=False)
    verified, count, recomputed = set(), Counter(), []
    for record, (identity, samples) in zip(result['records'], original_sources, strict=True):
        need({k: record[k] for k in identity} == identity, 'RESULT complete source identity')
        output = dict(identity, gains={})
        need(set(record['gains']) == {'1.0', '0.75', '0.5'}, 'exact registered gains')
        for gain in ('1.0', '0.75', '0.5'):
            output['gains'][gain] = {}
            need(set(record['gains'][gain]) == {'B0', 'B9', 'control', 'challenger'}, 'exact registered models')
            for arm in ('B0', 'B9', 'control', 'challenger'):
                saved = record['gains'][gain][arm]
                key = identity['id'] + '__' + gain + '__' + arm
                need(saved['array'] == key and key not in verified, 'unique actual model/source/gain identity')
                array = arrays[key]
                need(array.dtype == np.dtype('<f4') and array.shape == ((samples - 512) // 160 + 1, 4)
                     and array.flags.c_contiguous and np.isfinite(array).all(), 'exact finite float32 source frame grid')
                need(sha(array.tobytes()) == saved['sha256'] and array.shape[0] == saved['frames'], 'saved array content')
                original = np.load(io.BytesIO(z.read('raw_outputs/' + key + '.npy')), allow_pickle=False)
                need(original.dtype == array.dtype and original.shape == (1,) + array.shape
                     and original.tobytes() == array.tobytes(), 'untouched raw model output equals saved array')
                attempt = json.loads(z.read('attempts/' + key + '.json'))
                need(attempt == dict(gain=gain, key=key, role=arm, source_id=identity['id'], status='consumed_unknown'),
                     'original write-before-forward attempt journal')
                decoded = decode(array)
                need(all(saved[k] == v for k, v in decoded.items()), 'actual greedy tokens/events/sample availability clocks')
                output['gains'][gain][arm] = decoded
                verified.add(key)
                count[arm] += 1
        recomputed.append(output)
    need(set(arrays.files) == verified, 'all and only original saved arrays')
    expected_members = {'RESULT.json', 'ADMISSION.json', 'STARTED.json', 'ALL_FRAME_LOGITS.npz'}
    expected_members |= {'raw_outputs/' + key + '.npy' for key in verified}
    expected_members |= {'attempts/' + key + '.json' for key in verified}
    need(set(z.namelist()) == expected_members, 'no unbound or missing archive members')
    need(result['actual_forwards'] == result['raw_output_count'] == result['saved_array_count'] == len(verified) == 1008,
         'actual attempted, returned and saved forward counts')
    need(result['model_forward_calls'] == dict(count) == dict.fromkeys(('B0', 'B9', 'control', 'challenger'), 252)
         and result['consumed_unknown'] == [], 'all four models complete, no unresolved consumption')
    decision_path = ROOT / 'kws-data/automation/english_v6_b14_decision.py'
    need(sha(decision_path.read_bytes()) == '6aa5f999598e47b41eb0a946acd82fdc5b47d30aea8990871c1c1a01065e4023', 'unchanged original quality function')
    need(sha((decision_path.parent / 'english_v6_b14_panels/assessment.json').read_bytes()) ==
         '944f5df3c7bc3244977a8a33f7e1c393bffff9f7f5c32d2d8cc79e2628e265d9', 'unchanged original quality assessment')
    spec = importlib.util.spec_from_file_location('b20_root_original_b14', decision_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    decision = module.decide(recomputed)
    need(all(result[k] == v for k, v in decision.items()), 'entire B14 outcome recomputed from original frame arrays')
    audit = dict(schema='b20-independent-root-numpy-output-readback-v1',
                 status='COMPLETE_OUTPUT_READBACK_MATCHES_NO_PROMOTION',
                 result_zip_sha256=sha(raw), result_json_sha256=sha(result_raw), all_frame_logits_sha256=sha(arrays_raw),
                 original_source_denominator=120, admitted_originals=len(original_sources),
                 source_identity_authority='Original manifests joined in frozen order to original source SCREEN files; not RESULT-selected rows',
                 source_native_QA_recomputed_by_this_script=False,
                 original_producer_archive_completeness_claimed=False,
                 whole_provenance_audit='Separate independent producer/input/source audit required',
                 exact_raw_and_saved_arrays=len(verified), model_forward_calls=dict(count),
                 original_attempt_journals_verified=len(verified), unresolved_consumed_unknown=[],
                 numpy_version=np.__version__, decision=decision,
                 new_model_forwards=0, new_TTS_ASR_calls=0, optimizer_updates=0,
                 authorization_granted=False, replication_open_allowed=False,
                 candidate_engineering_validation_allowed=False, promotion_allowed=False,
                 installed_software_baseline='B0', ssc305_physical='NOT_RUN')
    (OUT / 'ROOT_NUMPY_OUTPUT_READBACK.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: audit[k] for k in ('status', 'admitted_originals', 'exact_raw_and_saved_arrays', 'model_forward_calls', 'new_model_forwards', 'promotion_allowed')}))


if __name__ == '__main__':
    main()
