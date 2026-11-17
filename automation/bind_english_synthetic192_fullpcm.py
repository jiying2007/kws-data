"""Bind a new full-PCM plan to actual successful saved synthetic-training metadata.

Offline only: never starts a workflow, subprocess, model, decoder or compiler.
Use API metadata saved by the caller; byte hashes are checked against both ZIPs.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import re


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('previous', 'candidate', 'run-metadata', 'artifact-metadata', 'output-plan'):
        cli.add_argument('--' + name, required=True, type=Path)
    cli.add_argument('--main-before', required=True)
    args = cli.parse_args(argv)
    path = Path(__file__).with_name('english_synthetic192_fullpcm_once.py')
    spec = importlib.util.spec_from_file_location('bound_teacher_fullpcm', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    h, inc = m.helpers()
    run, artifact = h.loads(h.regular(args.run_metadata)), h.loads(h.regular(args.artifact_metadata))
    m.need(run.get('repository', {}).get('full_name') == m.REPO and run.get('repository', {}).get('private') is False
        and run.get('event') == 'push' and run.get('head_branch') == 'main'
        and run.get('status') == 'completed' and run.get('conclusion') == 'success' and run.get('run_attempt') == 1,
        'ACTUAL_SUCCESSFUL_PUBLIC_SYNTHETIC_TRAINING_RUN_REQUIRED')
    m.need(artifact.get('workflow_run', {}).get('id') == run['id'] and artifact.get('workflow_run', {}).get('head_sha') == run['head_sha']
        and artifact.get('expired') is False and artifact.get('name') == 'english-synthetic192-training-' + str(run['id']),
        'MATCHING_SYNTHETIC_TRAINING_ARTIFACT_REQUIRED')
    data_pin = m.stream_identity(args.candidate)
    m.need(data_pin['bytes'] == artifact['size_in_bytes'] and artifact['digest'] == 'sha256:' + data_pin['sha256'], 'ACTUAL_CANDIDATE_ZIP_PIN')
    m.need(re.fullmatch('[0-9a-f]{40}', args.main_before) is not None, 'EXACT_NEW_MAIN_PREDECESSOR')
    source = dict(data_pin, run_id=run['id'], head_sha=run['head_sha'], artifact_id=artifact['id'], name=artifact['name'])
    archive = m.CandidateArchive(args.candidate, source, h, inc)
    try:
        archive.manifest('MANIFEST.json', exact=True)
        result = archive.json('evidence/RESULT.json')
        plan = m.fixed_plan(h.sha(path.read_bytes()), source, args.main_before,
            result.get('candidate', {}).get('bin'), result.get('candidate', {}).get('npz'))
        m.source_preflight(plan, h)
        m.candidate_gate(result, plan)
    finally:
        archive.z.close()
    prepared = m.prepare_inputs(args.previous, args.candidate, plan, h, inc)
    try:
        m.need(not args.output_plan.exists(), 'OUTPUT_PLAN_MUST_BE_NEW')
        h.put(args.output_plan, h.canonical(plan))
        print(h.canonical(prepared['summary']).decode(), end='')
    finally:
        prepared['previous'].z.close(); prepared['candidate'].z.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
