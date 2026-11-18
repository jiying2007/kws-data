"""Offline bind actual retained Mero MEH training bytes to the C/ARM replay.

Reads saved API metadata and both ZIPs. Runs byte parsers only; no network,
subprocess, compilation, optimization, or model execution is performed.
"""
from pathlib import Path
import argparse
import importlib.util
import re


def load_adapter():
    path = Path(__file__).with_name('english_mero_meh_fullpcm_once.py')
    spec = importlib.util.spec_from_file_location('bound_mero_meh_fullpcm', path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('previous', 'candidate', 'run-metadata', 'artifact-metadata', 'output-plan'):
        cli.add_argument('--' + name, required=True, type=Path)
    cli.add_argument('--main-before', required=True)
    args = cli.parse_args(argv)
    adapter = load_adapter()
    m = adapter.driver()
    h, inc = m.helpers()
    run = h.loads(h.regular(args.run_metadata))
    artifact = h.loads(h.regular(args.artifact_metadata))
    adapter.validate_training_run(run, m.need)
    m.need(artifact.get('workflow_run', {}).get('id') == run['id']
        and artifact.get('workflow_run', {}).get('head_sha') == run['head_sha']
        and artifact.get('expired') is False
        and artifact.get('name') == adapter.ARTIFACT_PREFIX + str(run['id']),
        'MATCHING_MERO_MEH_TRAINING_ARTIFACT')
    data_pin = m.stream_identity(args.candidate)
    m.need(data_pin['bytes'] == artifact['size_in_bytes']
        and artifact['digest'] == 'sha256:' + data_pin['sha256'], 'ACTUAL_CANDIDATE_ZIP_PIN')
    m.need(re.fullmatch('[0-9a-f]{40}', args.main_before) is not None, 'EXACT_NEW_MAIN_PREDECESSOR')
    source = dict(data_pin, run_id=run['id'], head_sha=run['head_sha'],
        artifact_id=artifact['id'], name=artifact['name'])
    archive = m.CandidateArchive(args.candidate, source, h, inc)
    try:
        archive.manifest('MANIFEST.json', exact=True)
        result = archive.json('evidence/RESULT.json')
        plan = m.fixed_plan(h.sha(h.regular(adapter.ROOT / adapter.WORKER)), source, args.main_before,
            result.get('candidate', {}).get('bin'), result.get('candidate', {}).get('npz'))
        m.source_preflight(plan, h)
        m.candidate_gate(result, plan)
    finally:
        archive.z.close()
    prepared = m.prepare_inputs(args.previous, args.candidate, plan, h, inc)
    try:
        m.need(args.output_plan.parent.is_dir() and not args.output_plan.exists()
            and not any(p.is_symlink() for p in (args.output_plan, *args.output_plan.parents)),
            'NEW_OUTPUT_PLAN_ONLY')
        with args.output_plan.open('xb') as stream:
            stream.write(h.canonical(plan))
        print(h.canonical(prepared['summary']).decode(), end='')
    finally:
        prepared['previous'].z.close()
        prepared['candidate'].z.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
