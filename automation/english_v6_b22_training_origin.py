"""Authenticate B22's preserved pre-fit STOP before one first-fit continuation."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / 'automation/english_v6_b22_training_origin.json'
EVIDENCE = ROOT / 'research/english_v6_b22_training_20261124/initial_failure'
REPO = 'jiying2007/kws-data'


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def gh(endpoint):
    return subprocess.run(['gh', 'api', 'repos/' + REPO + '/' + endpoint, '--allow-escape-sequences'],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout


def local():
    proof = json.loads(SPEC.read_bytes())
    need(proof['run_id'] == 45397676968 and proof['run_head'] == 'e5cd348d87297f4cfa7d3a5c4e68d9633eb3bfba'
         and proof['new_acoustic_model_forwards'] == proof['optimizer_updates'] == 0,
         'exact preserved pre-fit failure identity')
    for name, digest in proof['evidence_files'].items():
        need(sha((ROOT / name).read_bytes()) == digest, 'original failure evidence ' + name)
    producer = json.loads((EVIDENCE / 'PRODUCER.json').read_bytes())
    need(sha((EVIDENCE / 'PRODUCER.json').read_bytes()) == proof['original_producer_sha256']
         and len(producer['files']) == 243, 'entire actual first producer')
    for name, pin in producer['files'].items():
        raw = (ROOT / name).read_bytes()
        if name == producer['runtime_workflow']:
            token = b'__B22_REVIEWED_TRAIN_PARENT__'
            need(raw.count(token) == 2, 'original workflow stays inert')
            raw = raw.replace(token, proof['prepared_parent'].encode())
        need(len(raw) == pin['bytes'] and sha(raw) == pin['sha256'], 'unchanged first producer ' + name)
    log = json.loads((EVIDENCE / 'LOG.json').read_bytes())['content']
    need(sha(log.encode()) == proof['log_sha256'] and proof['failure'] in log
         and "paths, old_audit = prepare_union_paths" in log
         and "need(audit == json.loads(archive.read('SOURCE.json'))['positive_retention']" in log,
         'authentic exception before the first fit call')
    need(sorted(p.name for p in (EVIDENCE / 'partial').iterdir()) == proof['partial_members']
         == ['ASSESSMENT_BINDING.json', 'COMMON_DRAW.json', 'DESIGN.json', 'SOURCE.json', 'STARTED.json'],
         'complete partial output has no acoustic return, weight or optimizer trace')
    consumed = json.loads((EVIDENCE / 'consumed/CONSUMPTION.json').read_bytes())
    need((EVIDENCE / 'consumed/CONSUMPTION.json').read_bytes()
         == (EVIDENCE / 'receipts/CONSUMPTION.json').read_bytes()
         and consumed['identity'] == producer['identity']
         == json.loads((EVIDENCE / 'partial/STARTED.json').read_bytes())['identity'],
         'original consumed budget and STOP are preserved, never reset')
    return proof


def verify_live():
    proof = local()
    run = json.loads(gh('actions/runs/' + str(proof['run_id'])))
    need(run['id'] == proof['run_id'] and run['head_sha'] == proof['run_head']
         and run['run_attempt'] == 1 and run['status'] == 'completed' and run['conclusion'] == 'failure'
         and run['repository']['full_name'] == REPO and run['repository']['private'] is False,
         'original STOP must remain final and unrerun')
    jobs = json.loads(gh('actions/runs/' + str(proof['run_id']) + '/jobs?filter=all&per_page=100'))
    need(jobs['total_count'] == len(jobs['jobs']) == 1 and jobs['jobs'][0]['id'] == proof['job_id'],
         'only the original stopped training job')
    failed = [step['name'] for step in jobs['jobs'][0]['steps'] if step['conclusion'] == 'failure']
    need(failed == ['Fit one192-step challenger and retain every existing training output and gradient statistic'],
         'exact original failing step')
    need(sha(gh('actions/jobs/' + str(proof['job_id']) + '/logs')) == proof['log_sha256'],
         'original complete live job log')
    listing = json.loads(gh('actions/runs/' + str(proof['run_id']) + '/artifacts?per_page=100'))
    need(listing['total_count'] == len(listing['artifacts']) == len(proof['artifacts']) == 4,
         'complete original four-artifact inventory')
    for artifact in listing['artifacts']:
        pin = proof['artifacts'].get(artifact['name'])
        need(pin and artifact['id'] == pin['id'] and artifact['size_in_bytes'] == pin['bytes']
             and artifact['digest'] == 'sha256:' + pin['sha256'] and artifact['expired'] is False
             and artifact['workflow_run']['id'] == proof['run_id']
             and artifact['workflow_run']['head_sha'] == proof['run_head'], 'immutable original artifact')
        raw = gh('actions/artifacts/' + str(pin['id']) + '/zip')
        need(len(raw) == pin['bytes'] and sha(raw) == pin['sha256'], 'original archive bytes')
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            need(archive.testzip() is None, 'original archive CRC')
            if 'models-' in artifact['name']:
                need(sorted(archive.namelist()) == proof['partial_members'], 'complete original pre-fit output')
                for name in proof['partial_members']:
                    need(archive.read(name) == (EVIDENCE / 'partial' / name).read_bytes(), 'original partial member')
            elif 'train-producer-' in artifact['name']:
                need(archive.read('PRODUCER.json') == (EVIDENCE / 'PRODUCER.json').read_bytes(),
                     'complete actual original code identity')
    return dict(status='VERIFIED_ORIGINAL_ZERO_ACOUSTIC_CALL_STOP', original_run=proof['run_id'],
                original_head=proof['run_head'], new_acoustic_model_forwards=0, optimizer_updates=0,
                original_failure_preserved=True, original_consumption_preserved=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify_live() if args.live else local(), sort_keys=True))
