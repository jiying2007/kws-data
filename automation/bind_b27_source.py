"""Offline B27 SOURCE proof binder; no model, source or remote write.

The first invocation follows complete actual B27 TRAIN. Replica binding also
requires the complete independently verified original B27 PRIMARY PASS. Output
is a fresh review directory and cannot itself execute any scientific budget.
"""
from __future__ import annotations
import argparse
import base64
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b27_profile as profile


def transport(directory, files, artifact_ids):
    result = {}
    for name in files:
        if name.endswith('.json'):
            result[name] = dict(original_api_base64=base64.b64encode((directory / name).read_bytes()).decode('ascii'))
        else:
            profile.need(name in artifact_ids, 'every complete original ZIP has an authenticated specific artifact ID')
            result[name] = dict(artifact_id=artifact_ids[name])
    profile.validate_transport(result, files)
    return result


def training_transport(directory, files, lock):
    import english_v6_b27_training_readback as reader
    train = reader.train
    ids = {'trained/TRAIN_' + role + '.zip': pin['id'] for role, pin in lock['training']['artifacts'].items()}
    old_lock = train.opening()['quality_lock']
    profile.need(profile.sha(profile.canonical(old_lock)) == profile.Q1_LOCK_SHA, 'unchanged full original Q1 failure lock')
    ids.update({'qualification/no-promotion/originals/' + role + '.zip': old_lock[role]['id']
                for role in ('result', 'producer', 'consumed')})
    with train.failure.original(directory / 'qualification/no-promotion/originals/producer.zip',
                                old_lock['producer'], panel='primary', role='producer') as archive:
        raw = archive.read('primary_producer/automation/english_v6_b24q1_primary_inputs.json')
        profile.need(profile.sha(raw) == old_lock['inputs_sha256'], 'original input map before using artifact IDs')
        original = profile.document(raw)
    pins = {**original['inputs'], **{'native_' + key: pin for key, pin in original['native_inputs'].items()},
            **{key + '_producer': pin for key, pin in original['upstream_producers'].items()}}
    ids.update({'qualification/no-promotion/inputs/' + role + '.zip': pin['id'] for role, pin in pins.items()})
    ids.update({'qualification/training/' + name: pin['artifact_id'] for name, pin in train.design()['input_archives'].items()})
    return transport(directory, files, ids)


def quality_transport(directory, files, lock):
    import english_v6_b27_quality_readback as quality
    ids = {'originals/' + role + '.zip': lock[role]['id'] for role in ('result', 'producer', 'consumed')}
    with quality.original_archive(directory / 'originals/producer.zip', lock['producer'], lock['panel'], 'producer') as archive:
        raw = archive.read(lock['panel'] + '_producer/automation/english_v6_b27_' + lock['panel'] + '_inputs.json')
        profile.need(profile.sha(raw) == lock['inputs_sha256'], 'actual original B27 quality input map')
        spec = profile.document(raw)
    pins = {**spec['inputs'], **{'native_' + role: pin for role, pin in spec['native_inputs'].items()},
            **{role + '_producer': pin for role, pin in spec['upstream_producers'].items()}}
    ids.update({'inputs/' + role + '.zip': pin['id'] for role, pin in pins.items()})
    return transport(directory, files, ids)


def bind(panel, proof_root, training_lock, output, primary_lock=None):
    proof_root, output = Path(proof_root), Path(output)
    profile.need(proof_root.is_absolute() and proof_root == proof_root.resolve()
                 and output.is_absolute() and output.parent.is_dir() and not output.exists(), 'regular original proof and fresh review output')
    old = profile.binding(panel, False)
    profile.need(old['status'] == 'INERT_UNTIL_COMPLETE_ORIGINAL_B27_TRAIN', 'once-only inert SOURCE binding; no retargeting')
    from english_v6_b27_training_readback import verify_training_original
    origin = proof_root / 'training-origin'
    files = profile.inventory(origin)
    profile.file_pins(files, profile.LIMITS['primary_external_proof'])
    report = verify_training_original(origin / 'trained', training_lock, origin / 'qualification')
    profile.need(report['complete_original_Q1_NO_PROMOTION_verified'] is True
                 and report['result']['assessment_sha256'] == profile.profile()['assessment_sha256']
                 and report['result']['model_pt_sha256']['control'] == profile.CONTROL_PT,
                 'complete original failed Q1 qualifies a separately frozen actual B27 TRAIN only')
    value = dict(old, status='BOUND_COMPLETE_ORIGINAL_TRAIN', training_lock=training_lock,
        training_reader_sha256=profile.sha((profile.HERE / 'english_v6_b27_training_readback.py').read_bytes()),
        training_report_sha256=profile.sha(profile.canonical(report)), files=files,
        transport=training_transport(origin, files, training_lock))
    if panel == 'replication':
        import english_v6_b27_quality_readback as quality
        profile.need(type(primary_lock) is dict and primary_lock['panel'] == 'primary', 'only new B27 PRIMARY can open replica')
        primary = quality.verify_quality_original(proof_root / 'primary-quality', primary_lock, proof_root)
        profile.need(primary['panel_gate'] == 'PASS' and primary['replication_open_allowed'] is True
                     and primary['model_pt_sha256'] == report['result']['model_pt_sha256']
                     and primary['assessment_sha256'] == report['result']['assessment_sha256'],
                     'complete same-candidate original new PRIMARY PASS only')
        prior_files = profile.inventory(proof_root / 'primary-quality')
        value.update(primary_quality_lock=primary_lock,
            primary_quality_reader_sha256=profile.sha(Path(quality.__file__).read_bytes()),
            primary_quality_report_sha256=profile.sha(profile.canonical(primary)), primary_quality_files=prior_files,
            primary_quality_transport=quality_transport(proof_root / 'primary-quality', prior_files, primary_lock))
    else:
        profile.need(primary_lock is None, 'primary cannot borrow any old PASS')
    profile.validate_binding(value, panel)
    profile.need(profile.inventory(origin) == files, 'all complete original bytes unchanged after independent readback')
    output.mkdir(); (output / 'automation').mkdir()
    target = output / 'automation' / ('english_v6_b27_' + panel + '_proof_binding.json')
    target.write_bytes(profile.canonical(value))
    (output / 'TRAINING_READBACK.json').write_bytes(profile.canonical(report))
    if panel == 'replication':
        (output / 'PRIMARY_READBACK.json').write_bytes(profile.canonical(primary))
    audit = dict(schema='english-v6-b27-source-proof-binding-audit-v1', panel=panel,
        status='BOUND_COMPLETE_ORIGINALS_FOR_REVIEW', binding_sha256=profile.sha(target.read_bytes()),
        original_training_files=len(files), original_training_bytes=sum(pin['bytes'] for pin in files.values()),
        original_training_report_sha256=value['training_report_sha256'], original_B24Q1_failure_report_sha256=profile.Q1_REPORT_SHA,
        original_primary_pass_required=panel == 'replication', new_model_calls=0, new_source_calls=0,
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
    report = bind(args.panel, args.proof_root, profile.document(args.training_lock.read_bytes()), args.output,
                  profile.document(args.primary_lock.read_bytes()) if args.primary_lock else None)
    print(profile.canonical(report).decode())
