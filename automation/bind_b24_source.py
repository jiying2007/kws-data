"""Bind only an actually completed original B24 TRAIN to the frozen new sources.

Input is a local directory with TRAIN_{models,producer,receipts,consumed}.zip
and TRAIN_LIVE.json containing original run/artifacts/commit/recursive-tree API
responses. No network or scientific work occurs. All bytes and dependency gates
are checked before either source binding file is written.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b24_source_ancestry as lineage
from english_v6_b24_b23_failure import opened


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def prepare(directory):
    directory = Path(directory).resolve()
    live = json.loads((directory / 'TRAIN_LIVE.json').read_bytes())
    run, artifacts = live['run'], live['artifacts']
    lineage.need(run['status'] == 'completed' and run['conclusion'] == 'success'
        and run['run_attempt'] == 1 and run['head_branch'] == lineage.TRAIN_BRANCH
        and run['path'] == lineage.TRAIN_WORKFLOW and run['event'] == 'push'
        and all(run[k]['full_name'] == lineage.REPO and run[k]['private'] is False for k in ('repository', 'head_repository')),
        'actual completed first public B24 TRAIN only')
    names = dict(models='english-v6-b24-b23-warm-start-models-' + str(run['id']),
        producer='english-v6-b24-train-producer-' + str(run['id']),
        receipts='english-v6-b24-original-input-receipts-' + str(run['id']),
        consumed='english-v6-b24-b23-warm-start-train-consumed')
    lineage.need(artifacts['total_count'] == len(artifacts['artifacts']) == 4, 'all four original TRAIN artifacts')
    pins = {}
    for role, name in names.items():
        matched = [item for item in artifacts['artifacts'] if item['name'] == name]
        lineage.need(len(matched) == 1, 'one original named TRAIN artifact ' + role)
        item = matched[0]
        raw = (directory / ('TRAIN_' + role + '.zip')).read_bytes()
        pin = dict(id=item['id'], name=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        lineage.validate_artifact(item, pin, run['id'], run['head_sha'])
        pins[role] = pin
    with opened((directory / 'TRAIN_models.zip').read_bytes()) as archive:
        result_raw = archive.read('RESULT.json')
        result = json.loads(result_raw)
    with opened((directory / 'TRAIN_producer.zip').read_bytes()) as archive:
        producer_raw = archive.read('PRODUCER.json')
    value = dict(schema='english-v6-b24-completed-train-source-binding-v1',
        assessment_sha256=lineage.ASSESSMENT_SHA, bound=True,
        training_producer_manifest_sha256=hashlib.sha256(producer_raw).hexdigest(),
        training=dict(run_id=run['id'], run_head=run['head_sha'],
            prepared_parent=result['identity']['prepared_parent'], run_branch=run['head_branch'],
            runtime_workflow=run['path'], artifacts=pins, result_sha256=hashlib.sha256(result_raw).hexdigest(),
            model_pt_sha256=result['model_pt_sha256']))
    with tempfile.TemporaryDirectory(prefix='b24-source-binding-review-') as temporary:
        staged = Path(temporary)
        inputs, producer = staged / lineage.INPUTS.name, staged / lineage.TRAIN_PRODUCER.name
        inputs.write_bytes(encoded(value))
        producer.write_bytes(producer_raw)
        with patch.object(lineage, 'INPUTS', inputs), patch.object(lineage, 'TRAIN_PRODUCER', producer):
            lineage.training_binding()
            verified = lineage.verify_training(directory)
    lineage.need(verified == result, 'entire completed original TRAIN and failure dependency verified before binding')
    return value, producer_raw


def bind(directory):
    value, producer_raw = prepare(directory)
    # Inputs stay closed if interruption occurs between these two local writes.
    current = json.loads(lineage.INPUTS.read_bytes())
    lineage.need(current['bound'] is False or lineage.INPUTS.read_bytes() == encoded(value),
        'never replace a different already bound original TRAIN')
    lineage.TRAIN_PRODUCER.write_bytes(producer_raw)
    lineage.INPUTS.write_bytes(encoded(value))
    return dict(status='BOUND_ORIGINAL_TRAIN_NOT_EXECUTED', training_run_id=value['training']['run_id'],
        inputs_sha256=hashlib.sha256(lineage.INPUTS.read_bytes()).hexdigest(),
        producer_manifest_sha256=hashlib.sha256(producer_raw).hexdigest(),
        new_source_calls=0, new_model_forwards=0, promotion_allowed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--write', action='store_true', help='write the two bindings only after complete successful readback')
    args = parser.parse_args()
    if args.write:
        result = bind(args.directory)
    else:
        value, producer = prepare(args.directory)
        result = dict(status='VERIFIED_ORIGINAL_TRAIN_NOT_BOUND', training_run_id=value['training']['run_id'],
            inputs_sha256=hashlib.sha256(encoded(value)).hexdigest(), producer_manifest_sha256=hashlib.sha256(producer).hexdigest(),
            new_source_calls=0, new_model_forwards=0, promotion_allowed=False)
    print(json.dumps(result, sort_keys=True))
