"""Bind one actual recommended-pronunciation corpus to the disabled trainer.

This offline helper reads original GitHub REST metadata and an exact artifact
ZIP. It authenticates every member, checks saved frozen sources against the
local published checkout, recomputes the pure policy, and validates typed
intake before writing enabled copies. No network, inference or training runs.
The source corpus is a formal training campaign, never a re-labelled probe.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile

sys.dont_write_bytecode = True
REPO = 'jiying2007/kws-data'
WORKER_PATH = 'automation/english_mero_meh_train_once.py'


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def regular(path, limit=4*1024**2):
    path = Path(path).absolute()
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
         'REGULAR_INPUT_REQUIRED')
    need(path.stat().st_size <= limit, 'INPUT_BOUND')
    return path.read_bytes()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_metadata(run, artifact, job, branch, zip_pin, source_design):
    need(type(run.get('id')) is int and run['id'] > 0
        and re.fullmatch('[0-9a-f]{40}', run.get('head_sha') or '') is not None,
        'ACTUAL_SOURCE_RUN_IDENTITY')
    need(run.get('repository', {}).get('full_name') == REPO
        and run.get('repository', {}).get('private') is False
        and run.get('event') == 'push' and run.get('head_branch') == 'main'
        and run.get('status') == 'completed' and run.get('conclusion') in ('success', 'failure')
        and type(run.get('run_attempt')) is int and run['run_attempt'] == 1
        and run.get('path') == source_design['workflow_path'],
        'COMPLETED_FIRST_PUBLIC_SOURCE_RUN')
    need(type(artifact.get('id')) is int and artifact['id'] > 0
        and artifact.get('name') == source_design['artifact_prefix']+str(run['id'])
        and artifact.get('expired') is False
        and artifact.get('workflow_run', {}).get('id') == run['id']
        and artifact.get('workflow_run', {}).get('head_sha') == run['head_sha']
        and type(artifact.get('size_in_bytes')) is int
        and artifact['size_in_bytes'] == zip_pin['bytes']
        and artifact.get('digest') == 'sha256:'+zip_pin['sha256'],
        'EXACT_CORPUS_ARTIFACT_AND_ZIP_BYTES')
    need(type(job.get('id')) is int and job['id'] > 0 and job.get('name') == 'combine'
        and job.get('run_id') == run['id'] and job.get('head_sha') == run['head_sha']
        and job.get('status') == 'completed' and job.get('conclusion') == 'success',
        'ACTUAL_SUCCESSFUL_COMBINE_JOB')
    need(branch.get('name') == 'main' and type(branch.get('commit')) is dict
        and re.fullmatch('[0-9a-f]{40}', branch['commit'].get('sha') or '') is not None,
        'SAVED_CURRENT_MAIN_BRANCH_RESPONSE')
    before = branch['commit']['sha']
    need(branch['commit'].get('url') == 'https://api.github.com/repos/'+REPO+'/commits/'+before,
        'MAIN_BRANCH_REPOSITORY_IDENTITY')
    return dict(zip_pin, run_id=run['id'], artifact_id=artifact['id'],
        name=artifact['name'], head_sha=run['head_sha']), before


def intake_for_packet(root, source, worker):
    """Construct a new binding only from the already authenticated full packet."""
    read = lambda name: worker.loads(regular(root/name, limit=worker.MAX_MEMBER))
    corpus, admission = read('CORPUS_MANIFEST.json'), read('ADMISSION.json')
    evidence, contract = read('MACHINE_EVIDENCE.json'), read('SOURCE_IDENTITY_CONTRACT.json')
    exclusion = read('EXCLUSION_EVIDENCE.json')
    digest = canonical_digest
    return {'schema': 'english-typed-training-intake-binding-v1',
        'campaign_id': worker.FRESH_SOURCE['campaign_id'],
        'producer_run': {'repository': REPO, 'run_id': source['run_id'],
            'commit': source['head_sha'], 'run_attempt': 1, 'event': 'push'},
        'source_artifact': {'id': source['artifact_id'],
            **{k: source[k] for k in ('name', 'bytes', 'sha256')}},
        'planned_row_ids': [r['id'] for r in corpus['rows']],
        'source_identity_contract_sha256': digest(contract),
        'corpus_sha256': digest(corpus), 'admission_sha256': digest(admission),
        'machine_evidence_sha256': digest(evidence),
        'admission_policy_sha256': worker.sha(regular(root/worker.FRESH_SOURCE['policy_path'])),
        'legacy_libri_ids': worker.LEGACY_LIBRI_IDS,
        'exclusion_evidence_sha256': digest(exclusion),
        'excluded_identities': exclusion['excluded_identities'],
        'parent_context_weak_policy': False}


def enabled_core(raw, binding_sha256):
    need(re.fullmatch('[0-9a-f]{64}', binding_sha256) is not None, 'FROZEN_BINDING_DIGEST')
    old = b'SOURCE_IDENTITY_CANDIDATE_ENABLED = False\nBOUND_INTAKE_BINDING_SHA256 = None'
    need(raw.count(old) == 1, 'ONLY_DISABLED_NEW_CORE_CAN_BE_BOUND')
    new = ("SOURCE_IDENTITY_CANDIDATE_ENABLED = True\nBOUND_INTAKE_BINDING_SHA256 = '"
           + binding_sha256 + "'").encode()
    return raw.replace(old, new)


def enabled_workflow(raw, before):
    need(re.fullmatch('[0-9a-f]{40}', before) is not None, 'EXACT_MAIN_PREDECESSOR')
    old = b'if: ${{ false }}'
    need(raw.count(old) == 1, 'UNIQUE_DISABLED_NUMERIC_GUARD')
    new = ("if: ${{ github.event_name == 'push' && github.ref == 'refs/heads/main' && "
           "github.run_attempt == 1 && github.event.before == '"+before+"' }}").encode()
    return raw.replace(old, new)


def bind(args):
    trainer = args.trainer_root.absolute()
    w = load_module(trainer/WORKER_PATH, 'mero_meh_offline_worker')
    disabled = w.loads(regular(trainer/w.PLAN_PATH))
    w.source_preflight(disabled)
    need(disabled['enabled'] is False, 'ONLY_UNCONSUMED_DISABLED_TRAINER')
    paths = {'run': args.run_metadata, 'artifact': args.artifact_metadata,
        'combine_job': args.combine_job_metadata, 'main': args.main_metadata}
    raw_metadata = {name: regular(path) for name, path in paths.items()}
    metadata = {name: w.loads(raw) for name, raw in raw_metadata.items()}
    zip_pin = w.file_identity(args.corpus_zip, limit=w.MAX_ZIP)
    source, before = validate_metadata(metadata['run'], metadata['artifact'],
        metadata['combine_job'], metadata['main'], zip_pin, w.FRESH_SOURCE)
    output = args.output.absolute()
    need(output.parent.is_dir() and not output.exists()
        and not any(p.is_symlink() for p in (output, *output.parents)), 'NEW_OUTPUT_DIRECTORY_ONLY')
    with tempfile.TemporaryDirectory(prefix='mero-meh-bind-', dir=output.parent) as temp:
        temp = Path(temp)
        corpus_root = temp/'corpus'
        all_pins = w.extract_archive(args.corpus_zip, corpus_root, source)
        fresh_pins = {name: pin for name, pin in all_pins.items() if name.startswith('frozen_source/')}
        need(w.FRESH_SOURCE['policy_path'] in fresh_pins, 'ACTUAL_SOURCE_POLICY_REQUIRED')
        # The actual source run must contain the published files the caller inspected.
        for name, pin in fresh_pins.items():
            relative = name.removeprefix('frozen_source/')
            need(w.identity(regular(trainer/relative, limit=w.MAX_MEMBER)) == pin,
                 'PUBLISHED_SOURCE_MATCHES_ACTUAL_CORPUS:'+relative)
        intake = intake_for_packet(corpus_root, source, w)
        digest = canonical_digest(intake)
        workflow = enabled_workflow(regular(trainer/w.WORKFLOW_PATH), before)
        core = enabled_core(regular(trainer/w.CORE_PATH), digest)
        pins = dict(disabled['source_pins'])
        pins[w.WORKFLOW_PATH], pins[w.CORE_PATH] = w.identity(workflow), w.identity(core)
        plan = w.expected_plan(disabled['worker_sha256'], pins, main_before=before, enabled=True,
            corpus_source=source, fresh_execution_plan=w.file_identity(corpus_root/'EXECUTION_PLAN.json'),
            corpus_run_conclusion=metadata['run']['conclusion'],
            corpus_combine_job={'job_id': metadata['combine_job']['id'], 'name': 'combine'},
            intake_binding=intake, fresh_source_pins=fresh_pins)
        # The expected digest here belongs to this authenticated offline binder,
        # not to the incoming packet. The future worker uses the core constant.
        packet = w.source_packet(corpus_root, plan, expected_binding_sha256=digest)
        w.verify_combine_job(metadata['combine_job'], plan)
        stage = temp/'trainer'
        changed = {w.WORKFLOW_PATH: workflow, w.CORE_PATH: core}
        for name in set(disabled['source_pins']) | {w.WORKER_PATH}:
            w.put(stage/name, changed.get(name, regular(trainer/name)))
        plan_raw = json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode()+b'\n'
        w.put(stage/w.PLAN_PATH, plan_raw)
        staged = load_module(stage/WORKER_PATH, 'mero_meh_bound_preflight')
        staged.source_preflight(plan)
        receipt = {'schema': 'english-mero-meh-training-binding-v1',
            'status': 'ACTUAL_CORPUS_VALIDATED_ENABLED_FILES_PREPARED',
            'repository': REPO, 'corpus_source': source, 'main_before': before,
            'intake_binding_sha256': digest, 'planned_rows': len(intake['planned_row_ids']),
            'byte_readback': packet['byte_readback'],
            'tier_counts': packet['admission'].get('tier_counts'),
            'balanced_groups_by_role': packet['admission'].get('balanced_groups_by_role'),
            'metadata_files': {name: w.identity(raw) for name, raw in raw_metadata.items()},
            'source_preflight': 'PASSED_ON_EXACT_ENABLED_STAGING_COPY',
            'new_model_calls': 0, 'new_network_calls': 0, 'remote_writes': 0,
            'training_executed': False, 'model_quality_result': 'NOT_RUN'}
        output.mkdir(mode=0o700)
        for name, raw in {**changed, w.PLAN_PATH: plan_raw}.items():
            w.put(output/name, raw)
        w.once(output/'BINDING_RECEIPT.json', receipt)
        for name, raw in raw_metadata.items():
            w.put(output/'metadata'/(name+'.json'), raw)
        w.once(output/'UPLOAD_MANIFEST.json', {'repository': REPO,
            'files': [{'path': name, 'local_path': str(output/name), **w.identity(raw),
                'git_blob_sha1': hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()}
                for name, raw in sorted({**changed, w.PLAN_PATH: plan_raw}.items())]})
    print(w.canonical(receipt).decode())
    return 0


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('trainer-root', 'run-metadata', 'artifact-metadata', 'combine-job-metadata',
                 'main-metadata', 'corpus-zip', 'output'):
        cli.add_argument('--'+name, type=Path, required=True)
    return bind(cli.parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
