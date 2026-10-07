#!/usr/bin/env python3
"""Reuse the reviewed no-model CPU/import probe with local isolated paths."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'asr-runtime-identity-fix/research/asr_runtime'
sys.path.insert(0, str(SOURCE))
import qualify as q
q.require(q.sha256_file(SOURCE / 'container_stage.py') == 'dc10e8b07d35d17bccbfb0d95641bb43cdf20d7d75e57f5031ee7d76cd58f242',
          'reviewed probe source identity changed')
import container_stage

shutil.copyfile(SOURCE / 'locks/build-plan.json', ROOT / 'build-plan.json')
probe = container_stage.PROBE.replace('/inputs/', str(ROOT) + '/')
probe = probe.replace('/work/venv', str(ROOT / 'venv'))
probe = probe.replace('/output/', str(ROOT) + '/')
compile(probe, '<reviewed-cloud-cpu-probe>', 'exec')
(ROOT / 'probe.py').write_text(probe)
(ROOT / 'probe-source.receipt.json').write_bytes(q.canonical({
    'source_path': 'research/asr_runtime/container_stage.py',
    'reference_qualified_head': '1535031d186ee32190f40cd5263459c21b63b753',
    'reference_qualified_run': 'https://github.com/jiying2007/kws-pipeline/actions/runs/36937690401',
    'local_materialization_git_head': subprocess.check_output(
        ['git', '-C', str(SOURCE.parents[1]), 'rev-parse', 'HEAD'], text=True).strip(),
    'local_materialization_dirty_paths': subprocess.check_output(
        ['git', '-C', str(SOURCE.parents[1]), 'diff', '--name-only'], text=True).splitlines(),
    'source_identity_basis': 'Exact file SHA256 matches qualified run recipe; local git baseline is a connector-materialized snapshot, not the remote commit object',
    'source_sha256': q.sha256_file(SOURCE / 'container_stage.py'),
    'local_probe_sha256': q.sha256_file(ROOT / 'probe.py'),
    'adaptation': 'Only /inputs, /output and isolated venv filesystem locations replaced',
    'target_model_load_calls': 0, 'target_model_forward_calls': 0, 'user_audio_inputs': 0,
    'changed_dependency_versions': []}))
print(json.dumps({'probe': str(ROOT / 'probe.py'), 'sha256': q.sha256_file(ROOT / 'probe.py')}))
