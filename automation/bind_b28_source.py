"""Bind complete actual B28 TRAIN/PRIMARY originals into a fresh offline review.

No target is changed in place. All future identities remain null until original
bytes and independently recomputed reports exist. This binder grants no model,
source, network or remote-write operation.
"""
from __future__ import annotations

import argparse
import base64
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b28_profile as profile
import english_v6_b28_training_readback as reader
import english_v6_b28_train as train
import english_v6_b28_source_metadata as metadata


def original_path(path):
    path = Path(path)
    profile.need(path.is_absolute() and path == path.resolve() and path.is_dir() and not path.is_symlink(),
        'explicit regular resolved complete original directory')
    return path


def transport(directory, files, artifact_ids):
    """Preserve original whole API bytes and only separately authenticated IDs."""
    directory = original_path(directory)
    profile.need(set(artifact_ids) == {name for name in files if name.endswith('.zip')},
        'every and only complete original ZIP has an authenticated specific artifact ID')
    result = {}
    for name, pin in files.items():
        raw = (directory / name).read_bytes()
        profile.need(len(raw) == pin['bytes'] and profile.sha(raw) == pin['sha256'],
            'same full original transport byte identity ' + name)
        result[name] = (dict(original_api_base64=base64.b64encode(raw).decode('ascii'))
            if name.endswith('.json') else dict(artifact_id=artifact_ids[name]))
    profile.validate_transport(result, files)
    return result


def training_transport(directory, files, lock):
    """Join the existing old41+18+catalog7 maps; never nest or repackage them."""
    directory = original_path(directory)
    training = reader.validate_lock(lock)
    previous = train.profile.binding('primary')
    opening, catalog = train.opening(), metadata.catalog_registration()
    locations = {'inputs/proof/training-origin/' + name: value for name, value in previous['transport'].items()}
    locations.update({'inputs/proof/primary-quality/' + name: value for name, value in opening['original_primary_transport'].items()})
    locations.update({'inputs/metadata/train360/' + name: value for name, value in catalog['transport'].items()})
    locations.update({'trained/TRAIN_' + role + '.zip': dict(artifact_id=pin['id'])
        for role, pin in training['artifacts'].items()})
    locations['trained/TRAIN_LIVE.json'] = dict(original_api_base64=base64.b64encode(
        (directory / 'trained/TRAIN_LIVE.json').read_bytes()).decode('ascii'))
    profile.need(set(locations) == set(files), 'all71 original transport roles without substitution or extras')
    profile.validate_transport(locations, files)
    for name, location in locations.items():
        if name.endswith('.json'):
            profile.need(base64.b64decode(location['original_api_base64'], validate=True) == (directory / name).read_bytes(),
                'entire retained original API body, never a summary or reserialized replacement ' + name)
    return locations


def quality_transport(directory, files, lock):
    import english_v6_b28_quality_readback as quality
    ids = {'originals/' + role + '.zip': lock[role]['id'] for role in ('result', 'producer', 'consumed')}
    with quality.original_archive(Path(directory) / 'originals/producer.zip', lock['producer'], lock['panel'], 'producer') as archive:
        name = lock['panel'] + '_producer/automation/english_v6_b28_' + lock['panel'] + '_inputs.json'
        raw = archive.read(name)
        profile.need(profile.sha(raw) == lock['inputs_sha256'], 'whole original same-B28 quality input registration')
        spec = profile.document(raw)
    pins = {**spec['inputs'], **{'native_' + role: pin for role, pin in spec['native_inputs'].items()},
        **{role + '_producer': pin for role, pin in spec['upstream_producers'].items()}}
    profile.need(set(pins) == profile.QUALITY_INPUTS, 'all twelve exact original B28 quality inputs')
    ids.update({'inputs/' + role + '.zip': pin['id'] for role, pin in pins.items()})
    return transport(directory, files, ids)


def verify_training(proof_root, lock):
    """Pre-binding full readback: no invented future bound report digest."""
    proof_root = original_path(proof_root)
    train.design(); train.opening(); train.assessment_binding()
    training = reader.validate_lock(lock)
    origin = proof_root / 'training-origin'
    files = profile.inventory(origin)
    profile.proof_file_pins(files, profile.LIMITS['primary_external_proof'])
    expected = profile.expected_training_inputs()
    profile.need(set(files) == {'trained/' + name for name in profile.TRAIN_FILES} | set(expected)
        and len(files) == 71 and all(files[name] == pin for name, pin in expected.items()),
        'all71 whole originals, including complete catalog Git tree, before independent TRAIN readback')
    for role, pin in training['artifacts'].items():
        profile.need(files['trained/TRAIN_' + role + '.zip'] == dict(bytes=pin['bytes'], sha256=pin['sha256']),
            'same actual original new B28 TRAIN ' + role)
    path = profile.HERE / 'english_v6_b28_training_readback.py'
    profile.need(Path(reader.__file__).resolve() == path and path.is_file() and not path.is_symlink(),
        'one complete local independent TRAIN reader')
    reader_sha = profile.sha(path.read_bytes())
    raw = profile._training_readback(proof_root, dict(files=files, training_lock=lock), reader)
    report = reader.validate_report(profile.document(raw), lock)
    result = report['result']
    profile.need(result['schema'] == train.SCHEMA and result['status'] == train.RESULT_STATUS
        and result['assessment_sha256'] == profile.profile()['assessment_sha256']
        and result['model_pt_sha256'] == training['model_pt_sha256']
        and result['model_pt_sha256']['control'] == profile.CONTROL_PT
        and result['model_pt_sha256']['challenger'] not in (profile.CONTROL_PT, profile.INITIAL_PT, profile.B27_PT)
        and report['complete_original_B27_NO_PROMOTION_verified'] is True
        and report['natural_no_event_terms_recomputed'] is True
        and report['saved_training_observations']['natural_objective_returns_verified'] == 384,
        'complete actual new no-event candidate after the independently reauthenticated B27 terminal')
    profile.need(profile.inventory(origin) == files and profile.sha(path.read_bytes()) == reader_sha,
        'all71 originals and independent reader unchanged through complete readback')
    return files, report, reader_sha


def bind(panel, proof_root, training_lock, output, primary_lock=None):
    profile.need(panel in ('primary', 'replication'), 'explicit independent B28 panel')
    proof_root, output = original_path(proof_root), Path(output)
    profile.need(output.is_absolute() and output == output.resolve() and output.parent.is_dir() and not output.exists(),
        'fresh review output; no existing binding can be retargeted')
    old = profile.binding(panel, False)
    profile.need(old == profile.inert_binding(panel), 'only an exact inert SOURCE binding may be bound once')
    expected_roots = {'training-origin'} if panel == 'primary' else {'training-origin', 'primary-quality'}
    profile.need({p.name for p in proof_root.iterdir()} == expected_roots,
        'exact independent proof layout; PRIMARY never borrows an earlier PASS')
    profile.need(primary_lock is None if panel == 'primary' else type(primary_lock) is dict and primary_lock.get('panel') == 'primary',
        'only a complete original same-B28 PRIMARY may open replica')
    files, report, reader_sha = verify_training(proof_root, training_lock)
    value = dict(old, status='BOUND_COMPLETE_ORIGINAL_TRAIN', assessment_sha256=report['result']['assessment_sha256'],
        training_lock=training_lock, training_reader_sha256=reader_sha,
        training_report_sha256=profile.sha(profile.canonical(report)), files=files,
        transport=training_transport(proof_root / 'training-origin', files, training_lock))
    if panel == 'replication':
        import english_v6_b28_quality_readback as quality
        primary = quality.verify_quality_original(proof_root / 'primary-quality', primary_lock, proof_root)
        quality.validate_report(primary, primary_lock)
        profile.need(primary['panel_gate'] == 'PASS' and primary['panel'] == 'primary'
            and primary['replication_open_allowed'] is True
            and primary['model_pt_sha256'] == report['result']['model_pt_sha256']
            and primary['assessment_sha256'] == report['result']['assessment_sha256']
            and primary['original_training_report_sha256'] == value['training_report_sha256'],
            'actual same-candidate new PRIMARY PASS; earlier scores or a different candidate cannot qualify')
        primary_files = profile.inventory(proof_root / 'primary-quality')
        value.update(primary_quality_lock=primary_lock,
            primary_quality_reader_sha256=profile.sha(Path(quality.__file__).read_bytes()),
            primary_quality_report_sha256=profile.sha(profile.canonical(primary)), primary_quality_files=primary_files,
            primary_quality_transport=quality_transport(proof_root / 'primary-quality', primary_files, primary_lock))
    profile.validate_binding(value, panel)
    profile.need(profile.inventory(proof_root / 'training-origin') == files,
        'unchanged complete original TRAIN after all evidence joins')
    if panel == 'replication':
        profile.need(profile.inventory(proof_root / 'primary-quality') == value['primary_quality_files'],
            'unchanged complete original PRIMARY after all evidence joins')
    profile.need(not {'torch', 'numpy', 'onnxruntime', 'soundfile', 'piper'} & set(sys.modules),
        'no scientific runtime in an offline original-proof binder')
    output.mkdir(); (output / 'automation').mkdir()
    target = output / 'automation' / ('english_v6_b28_' + panel + '_proof_binding.json')
    target.write_bytes(profile.canonical(value))
    (output / 'TRAINING_READBACK.json').write_bytes(profile.canonical(report))
    if panel == 'replication':
        (output / 'PRIMARY_READBACK.json').write_bytes(profile.canonical(primary))
    audit = dict(schema='english-v6-b28-source-proof-binding-audit-v1', panel=panel,
        status='BOUND_COMPLETE_ORIGINALS_FOR_REVIEW', binding_sha256=profile.sha(target.read_bytes()),
        original_training_files=len(files), original_training_bytes=sum(pin['bytes'] for pin in files.values()),
        original_training_report_sha256=value['training_report_sha256'],
        original_B27_failure_certification=train.failure_certification(),
        original_primary_pass_required=panel == 'replication', natural_objective_returns_recomputed=384,
        metadata_original_files=len(metadata.CATALOG_FILES), new_model_calls=0, new_source_calls=0,
        new_observer_calls=0, remote_mutations=0, scientific_execution_authorized=False)
    (output / 'BINDING_AUDIT.json').write_bytes(profile.canonical(audit))
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', choices=('primary', 'replication'))
    parser.add_argument('proof_root', type=Path)
    parser.add_argument('training_lock', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--primary-lock', type=Path)
    args = parser.parse_args()
    result = bind(args.panel, args.proof_root, profile.document(args.training_lock.read_bytes()), args.output,
        profile.document(args.primary_lock.read_bytes()) if args.primary_lock else None)
    print(profile.canonical(result).decode())
