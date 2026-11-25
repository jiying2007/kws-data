"""Construct reviewable B27 openings and completed-TRAIN pins from real originals.

Both commands are readback/binding only. They never push, activate a workflow,
consume a scientific marker, select a new assessment or call a model.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b27_natural_weight as train
import english_v6_b27_training_readback as reader


def bind_opening(directory, lock, output):
    train.design(); train.assessment_binding()
    train.failure.validate_lock(lock)
    output = Path(output)
    train.need(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'fresh reviewable opening file only')
    report = train.failure.verify_failure_original(Path(directory), lock)
    train.need(report['panel_gate'] == 'NO_PROMOTION' and report['panel'] == 'primary'
         and report['model_pt_sha256']['challenger'] == train.CONTROL_PT
         and all(report[key] is False for key in train.failure.CLOSED), 'complete original Q1 failure has no downstream authority')
    value = dict(schema='english-v6-b27-after-original-b24q1-no-promotion-v1', condition=train.CONDITION,
        enabled=True, quality_lock=lock, baseline_challenger_pt_sha256=train.CONTROL_PT,
        failure_adapter=dict(path='automation/english_v6_b24q1_primary_failure.py',
            sha256=train.design()['failure_reader']['sha256'], base_reader_path='automation/english_v6_b24q1_quality_readback.py',
            base_reader_sha256=train.failure.BASE_READER_SHA),
        actual_primary_evaluator_sha256=lock['evaluator_sha256'],
        actual_primary_producer_manifest_sha256=lock['producer_manifest_sha256'],
        original_failure_report_sha256=train.sha(train.failure.canonical(report)))
    train.write(output, value)
    return value


def derive_lock(directory):
    """Derive actual artifact and result pins; this alone is not verification."""
    directory = Path(directory)
    live = reader.document((directory / 'TRAIN_LIVE.json').read_bytes())
    run, commit, inventory = live['run'], live['commit'], live['artifacts']
    names = reader.artifact_names(run['id'])
    train.need(inventory['total_count'] == len(inventory['artifacts']) == 4, 'four complete actual TRAIN artifacts')
    artifacts = {}
    for role, name in names.items():
        chosen = [item for item in inventory['artifacts'] if item['name'] == name]
        train.need(len(chosen) == 1, 'one original artifact for role ' + role)
        item = chosen[0]
        raw = (directory / ('TRAIN_' + role + '.zip')).read_bytes()
        train.need(len(raw) == item['size_in_bytes'] and item['digest'] == 'sha256:' + train.sha(raw),
                   'whole actual original artifact identity ' + role)
        artifacts[role] = dict(id=item['id'], bytes=len(raw), sha256=train.sha(raw), name=name)
    with train.failure.opened((directory / 'TRAIN_models.zip').read_bytes()) as archive:
        raw = archive.read('RESULT.json')
        result = reader.document(raw)
        result_sha = train.sha(raw)
    with train.failure.opened((directory / 'TRAIN_producer.zip').read_bytes()) as archive:
        producer_raw = archive.read('PRODUCER.json')
    train.need(len(commit['parents']) == 1, 'one prepared predecessor')
    value = dict(schema=reader.LOCK_SCHEMA, training=dict(run_id=run['id'], run_head=run['head_sha'],
        prepared_parent=commit['parents'][0]['sha'], run_branch=run['head_branch'], runtime_workflow=run['path'],
        artifacts=artifacts, result_sha256=result_sha, model_pt_sha256=result['model_pt_sha256']),
        training_producer_manifest_sha256=train.sha(producer_raw))
    reader.validate_live(live, value)
    return value, producer_raw


def bind_training(directory, inputs, output):
    output = Path(output)
    train.need(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'fresh reviewable binding output directory')
    lock, producer_raw = derive_lock(directory)
    report = reader.verify_training_original(directory, lock, inputs)
    output.mkdir()
    (output / 'TRAINING_LOCK.json').write_bytes(reader.canonical(lock))
    (output / 'TRAINING_READBACK.json').write_bytes(reader.canonical(report))
    (output / 'TRAINING_PRODUCER.json').write_bytes(producer_raw)
    value = dict(schema='english-v6-b27-completed-training-binding-candidate-v1',
        assessment_sha256=report['result']['assessment_sha256'], design_sha256=train.DESIGN_SHA,
        training=lock['training'], training_producer_manifest_sha256=lock['training_producer_manifest_sha256'],
        original_training_lock_sha256=train.sha(reader.canonical(lock)),
        independent_training_readback_sha256=train.sha(reader.canonical(report)),
        external_original_files=report['training_source']['complete_external_inputs']['files'],
        artifact_is_execution_authority=False, new_model_calls=0, new_source_calls=0, promotion_allowed=False)
    train.write(output / 'BINDING.json', value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ('opening', 'training'):
        item = sub.add_parser(mode)
        item.add_argument('--directory', type=Path, required=True)
        item.add_argument('--output', type=Path, required=True)
        if mode == 'opening': item.add_argument('--lock', type=Path, required=True)
        else: item.add_argument('--inputs', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'opening':
        value = bind_opening(args.directory, reader.document(args.lock.read_bytes()), args.output)
    else:
        value = bind_training(args.directory, args.inputs, args.output)
    print(json.dumps(value, sort_keys=True))
