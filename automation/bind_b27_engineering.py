"""Produce reviewable B27 engineering bindings from complete original evidence.

This command never edits an execution registration, changes selection, invokes
a scientific runtime, compiles or packages. Both modes write only a new review
directory after reopening originals through the exact independent readers.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b27_engineering as eng
import english_v6_b27_engineering_readback as readback


def fresh(path):
    path = Path(path)
    eng.need(path.is_absolute() and path == path.resolve() and path.parent.is_dir()
             and not path.exists() and all(not item.is_symlink() for item in path.parents),
             'fresh absolute review directory; existing originals and registrations remain unchanged')
    return path


def save(path, value):
    with path.open('xb') as handle:
        handle.write(readback.canonical(value))


def prepare_inputs(primary, primary_lock, primary_proof, replication, replication_lock, replication_proof, output):
    output = fresh(output)
    doc = readback.document(eng.SPEC.read_bytes())
    eng.need(doc['enabled'] is False and doc['candidate_files'] is None
             and doc['inputs']['pair'] is None and all(value is None for value in doc['quality_locks'].values()),
             'reviewed engineering registration is still unbound; no replacement candidate or replay')
    quality = eng.module(eng.QUALITY_READER)
    eng.need(eng.sha(Path(quality.__file__).read_bytes()) == readback.QUALITY_READER_SHA,
             'same exact reviewed independent B27 quality reader')
    proofs = {}
    for panel, directory, lock_path, proof_root in (
        ('primary', primary, primary_lock, primary_proof),
        ('replication', replication, replication_lock, replication_proof)):
        lock = readback.document(readback.read(Path(lock_path), readback.REPORT_LIMIT))
        eng.need(lock['panel'] == panel, 'explicit original panel identity')
        proofs[panel] = quality.verify_quality_original(Path(directory), lock, Path(proof_root))
        eng.need(proofs[panel]['panel_gate'] == 'PASS', 'actual complete original ' + panel + ' PASS required')
        doc['quality_locks'][panel] = lock
    doc.update(enabled=True, quality_reader_sha256=readback.QUALITY_READER_SHA,
        model_pt_sha256=proofs['primary']['model_pt_sha256'],
        original_training_report_sha256=proofs['primary']['original_training_report_sha256'])
    doc['inputs']['pair'] = proofs['primary']['pair_pin']
    with eng.original_zip(Path(primary) / 'inputs/pair.zip', doc['inputs']['pair']) as archive:
        receipt = readback.document(archive.read('RESULT.json'))
        doc['candidate_files'] = receipt['challenger']['files']
    eng.protocol(doc)
    eng.validate_dual_quality(proofs, doc)
    eng.original_pair(Path(primary) / 'inputs', doc, Path(primary_proof))
    output.mkdir()
    for name, value in [('engineering_inputs.json', doc), ('PRIMARY_READBACK.json', proofs['primary']),
                         ('REPLICATION_READBACK.json', proofs['replication'])]:
        save(output / name, value)
    result = dict(schema='english-v6-b27-reviewed-engineering-binding-v1', status='READY_FOR_REVIEW_ONLY',
        files={path.name: dict(bytes=path.stat().st_size, sha256=eng.sha(path.read_bytes())) for path in sorted(output.iterdir())},
        actual_registration_changed=False, engineering_executed=False, selection_changed=False,
        new_model_calls=0, new_source_calls=0, compiler_calls=0, packs_created=0)
    save(output / 'BINDING.json', result)
    return result


def original_engineering_lock(directory):
    """Derive pins from four actual ZIPs and eight original complete API bodies."""
    directory = Path(directory)
    api = {role: readback.read(directory / 'api' / (role + '.json'), readback.REPORT_LIMIT)
           for role in readback.API_ROLES}
    run = readback.document(api['run'])
    inventory = readback.document(api['artifacts'])
    eng.need(run['status'] == 'completed' and run['conclusion'] == 'success' and run['run_attempt'] == 1
             and run['path'] == eng.WORKFLOW and run['head_branch'] == eng.BRANCH
             and inventory['total_count'] == len(inventory['artifacts']) == 4,
             'only one complete actual successful B27 engineering run')
    names = dict(result='english-v6-b27-software-sdk-evidence-' + str(run['id']),
        producer='english-v6-b27-engineering-producer-' + str(run['id']), consumed=eng.MARKER,
        input_proof='english-v6-b27-engineering-input-proof-' + str(run['id']))
    artifacts = {}
    for role, name in names.items():
        matches = [value for value in inventory['artifacts'] if value['name'] == name]
        eng.need(len(matches) == 1, 'one exact original engineering artifact ' + role)
        item = matches[0]
        raw = readback.read(directory / 'originals' / (role + '.zip'))
        pin = dict(id=item['id'], run_id=run['id'], run_head=run['head_sha'], bytes=len(raw), sha256=eng.sha(raw))
        readback.validate_artifact(pin, run, item, name)
        artifacts[role] = pin
    with readback.original(directory / 'originals/producer.zip', artifacts['producer']) as archive:
        producer_sha = eng.sha(archive.read('PRODUCER.json'))
        prefix = 'repository/automation/'
        evaluator_sha = eng.sha(archive.read(prefix + 'english_v6_b27_engineering.py'))
        quality_sha = eng.sha(archive.read(prefix + 'english_v6_b27_quality_readback.py'))
        inputs_sha = eng.sha(archive.read(prefix + 'english_v6_b27_engineering_inputs.json'))
    with readback.original(directory / 'originals/result.zip', artifacts['result']) as archive:
        result_sha = eng.sha(archive.read('RESULT.json'))
        manifest_sha = eng.sha(archive.read('MANIFEST.json'))
    lock = dict(schema=readback.LOCK_SCHEMA, repository=eng.REPO, artifacts=artifacts,
        api={role: dict(bytes=len(raw), sha256=eng.sha(raw)) for role, raw in api.items()},
        evaluator_sha256=evaluator_sha, quality_reader_sha256=quality_sha, inputs_sha256=inputs_sha,
        producer_manifest_sha256=producer_sha, original_result_sha256=result_sha, result_manifest_sha256=manifest_sha)
    readback.validate_lock(lock)
    return lock


def bind_result(directory, output):
    output = fresh(output)
    lock = original_engineering_lock(directory)
    output.mkdir(); save(output / 'ORIGINAL_LOCK.json', lock)
    report = readback.verify(Path(directory), output / 'ORIGINAL_LOCK.json', output / 'readback')
    result = dict(schema='english-v6-b27-reviewed-engineering-result-binding-v1',
        status='COMPLETE_ORIGINAL_READBACK_READY_FOR_REVIEW', original_lock_sha256=eng.sha(readback.canonical(lock)),
        readback={name: dict(bytes=(output / 'readback' / name).stat().st_size,
                            sha256=eng.sha((output / 'readback' / name).read_bytes()))
                  for name in ('completion.json', 'readback.py')},
        original_training_report_sha256=report['original_training_report_sha256'],
        candidate_selection_changed=False, new_model_calls=0, compiler_calls=0, packs_created=0)
    save(output / 'BINDING.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    item = modes.add_parser('inputs')
    for name in ('primary', 'primary-lock', 'primary-proof', 'replication', 'replication-lock', 'replication-proof', 'output'):
        item.add_argument('--' + name, required=True, type=Path)
    item = modes.add_parser('result')
    item.add_argument('directory', type=Path); item.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.mode == 'inputs':
        value = prepare_inputs(args.primary, args.primary_lock, args.primary_proof,
                               args.replication, args.replication_lock, args.replication_proof, args.output)
    else:
        value = bind_result(args.directory, args.output)
    print(json.dumps(value, sort_keys=True))


if __name__ == '__main__':
    main()
