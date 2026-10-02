#!/usr/bin/env python3
"""Build the five previously reviewed sdists twice, offline and serially."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'asr-runtime-identity-fix/research/asr_runtime'
sys.path.insert(0, str(SOURCE))
import qualify as q
import controller as c
import container_stage as stage

inventory_path = SOURCE / 'locks/inventory.json'
plan_path = SOURCE / 'locks/build-plan.json'
q.require(q.sha256_file(inventory_path) == '626b8c827ee63f95f07e828ae8351ef582197ab253cc68d80b29288679c796e1',
          'reviewed inventory identity changed')
q.require(q.sha256_file(plan_path) == 'fd20f0ddf3e3651880abc05fc851ad244fa839d34066e6b300e6c5c183d19425',
          'reviewed build plan identity changed')
files = q.validate_inventory(q.load_json(inventory_path), 143, 3060297109)
plan = q.load_json(plan_path)
limits = q.load_json(SOURCE / 'locks/admission.json')['limits']
artifacts = ROOT / 'artifacts'
by_name = {x['name']: x for x in files}
for name in plan['build_tool_names'] + [x['name'] for x in plan['sources']]:
    item = by_name[name]
    path = artifacts / item['filename']
    q.require(path.is_file() and path.stat().st_size == item['bytes'] and q.sha256_file(path) == item['sha256'],
              'build input size/hash mismatch: ' + name)
build_requirements = ROOT / 'build-requirements.txt'
build_requirements.write_text(c.requirements([by_name[n] for n in plan['build_tool_names']]))
work = ROOT / 'source-build-work'
build_env = dict(os.environ)
# This relocatable cloud Python was compiled with clang, which is absent here.
# Use the already-installed native GCC toolchain; upstream source stays unchanged.
compiler = Path('/usr/bin/gcc').resolve()
q.require(compiler.is_file(), 'existing native compiler is unavailable')
build_env.update(CC='/usr/bin/gcc -pthread', CXX='/usr/bin/g++ -pthread',
                 LDSHARED='/usr/bin/gcc -pthread -shared ' + (sysconfig.get_config_var('LDFLAGS') or ''))
(ROOT / 'source-toolchain.json').write_bytes(q.canonical({
    'compiler': str(compiler), 'compiler_sha256': q.sha256_file(compiler),
    'compiler_version': subprocess.check_output(['/usr/bin/gcc', '--version'], text=True).splitlines()[0],
    'CC': build_env['CC'], 'CXX': build_env['CXX'], 'LDSHARED': build_env['LDSHARED'],
    'python_include': sysconfig.get_path('include'), 'original_sources_modified': False}))
results = []
for label in ('a', 'b'):
    q.require(not work.is_symlink(), 'linked build work refused')
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    output = ROOT / ('built-' + label)
    q.require(not output.exists(), 'build output must be fresh')
    output.mkdir()
    subprocess.run([sys.executable, '-I', '-m', 'venv', str(work / 'venv')], check=True)
    python = str(work / 'venv/bin/python')
    subprocess.run([python, '-I', '-m', 'pip', '--isolated', 'install', '--no-index', '--no-cache-dir',
                    '--require-hashes', '--only-binary=:all:', '--find-links=' + str(artifacts),
                    '-r', str(build_requirements)], check=True)
    for project in plan['sources']:
        source = stage.extract_source(artifacts / project['filename'], project, work / 'sources')
        subprocess.run([python, '-I', '-m', 'pip', '--isolated', 'wheel', '--no-index', '--no-deps',
                        '--no-build-isolation', '--no-cache-dir', '--wheel-dir=' + str(output), str(source)],
                       check=True, env=build_env)
    # --no-deps is exclusively for individual source-wheel construction.
    result = c.validate_built_wheels(output, plan, limits)
    (ROOT / ('build-' + label + '-wheels.json')).write_bytes(q.canonical(result))
    results.append(result)
q.require(results[0] == results[1], 'independent source wheels differ')
for item in results[0]:
    shutil.copyfile(ROOT / 'built-a' / item['filename'], artifacts / item['filename'])
    by_name[item['name']] = item
runtime = [dict(by_name[name]) for name in plan['runtime_names']]
for item in runtime:
    item['install_extras'] = ['knf'] if item['name'] == 'funasr' else []
q.require(all(x['filename'].endswith('.whl') for x in runtime), 'runtime still contains source archives')
(ROOT / 'runtime-requirements.txt').write_text(c.requirements(runtime))
(ROOT / 'runtime-versions.json').write_text(json.dumps({x['name']: x['version'] for x in runtime}, sort_keys=True))
(ROOT / 'local-runtime-lock.json').write_bytes(q.canonical({'schema': 'kws.cloud-asr-runtime-lock.v1',
    'official_input_inventory_sha256': q.sha256_file(SOURCE / 'locks/inventory.json'),
    'variant': 'reviewed_pypi_torch_2.12.1_cuda_capable_cpu_execution_only',
    'independent_source_builds_byte_identical': True, 'runtime_files': runtime}))
print(json.dumps({'source_wheels': len(results[0]), 'byte_identical': True, 'runtime_packages': len(runtime)}))
