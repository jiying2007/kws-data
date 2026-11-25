"""Capture complete Git-bound B27 SOURCE code before source consumption."""
from __future__ import annotations
import argparse
import ast
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b27_profile as profile
import english_v6_b27_capacity as capacity
import english_v6_b21_producer_snapshot as common
import english_v6_b22_source_dynamic_preflight as dynamic

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'english-v6-b27-complete-source-producer-v1'
PRIMARY_DORMANT_QUALITY = frozenset(('english_v6_b27_quality_readback',
    'english_v6_b27_primary_eval', 'english_v6_b27_replication_eval'))
TRANSPORT_PLAN = 'automation/english_v6_b21_fixed_mirror_transport.json'
TRANSPORT_PLAN_SHA256 = '82f915a9ea4355c545dc3c8aabeeca25460896021a4303dfc1f3280597cd3850'
TRANSPORT_PROBE = 'automation/english_v6_b21_continuation_probe'
TRANSPORT_FAILURES = (
    ('automation/english_v6_b21_continuation_failure', 1693,
     '8e4745c640a6729361d70f73f20b5ad18c0972ebcd108fb1d5ca258f22479aeb'),
    ('automation/english_v6_b21_fixed_mirror_failure', 1595,
     'a1a0d2acf630fd8b23e773e1e7906401787a8a75cbca81fca2de08dd1d54c4ca'),
)
PRECONSUMPTION_DIRECTORY = 'automation/english_v6_b27_source_preconsumption_failure'
REVIEWED_METADATA_ADAPTERS = {
    'automation/english_v6_b27_capacity.py':
        '507e4c9c1f7bef3494bb56ff43cb2f7d74768f1db265abbd0e0858a8c383bb0f',
}


def transport_resource_paths(root=ROOT):
    """Complete original resource closure of the unchanged transport plan.

    Python import traversal cannot discover files read through Path.with_name.
    The exact original plan and two original proof manifests declare these
    twenty-five resources. Nothing is inferred from a directory wildcard.
    Original code/probe archives are retained in full; no TRAIN, SOURCE, model
    or PCM artifact is nested here.
    """
    root = Path(root).resolve()

    def read(name, expected=None):
        path = root / name
        profile.need(not Path(name).is_absolute() and '..' not in Path(name).parts
                     and path.is_file() and path.resolve().is_relative_to(root)
                     and not any(item.is_symlink() for item in (path, *path.parents)),
                     'complete regular original transport resource ' + name)
        raw = path.read_bytes()
        if expected is not None:
            profile.need(len(raw) == expected['bytes'] and profile.sha(raw) == expected['sha256'],
                         'unchanged original transport resource bytes ' + name)
        return raw

    plan_raw = read(TRANSPORT_PLAN, dict(bytes=1577, sha256=TRANSPORT_PLAN_SHA256))
    plan = profile.document(plan_raw)
    resources = {TRANSPORT_PLAN}
    probe_files = plan['probe']['files']
    profile.need(set(probe_files) == {'original.zip', 'api/run.json', 'api/artifact.json', 'original/RESULT.json'},
                 'all four original transport probe resources')
    for name, pin in probe_files.items():
        full = TRANSPORT_PROBE + '/' + name
        read(full, pin); resources.add(full)
    for directory, size, digest in TRANSPORT_FAILURES:
        manifest_name = directory + '/MANIFEST.json'
        manifest = profile.document(read(manifest_name, dict(bytes=size, sha256=digest)))
        resources.add(manifest_name)
        profile.need(len(manifest['files']) == 9, 'complete nine-file original transport failure proof')
        for name, pin in manifest['files'].items():
            full = directory + '/' + name
            read(full, pin); resources.add(full)
        found = {path.relative_to(root).as_posix() for path in (root / directory).rglob('*') if path.is_file()}
        expected = {manifest_name} | {directory + '/' + name for name in manifest['files']}
        profile.need(found == expected, 'all and only declared original transport failure resources')
    profile.need(len(resources) == 25, 'exact original transport resource closure')
    return resources


def preconsumption_resource_paths(root=ROOT):
    """Keep the complete immutable failed B27 event, including both originals."""
    import english_v6_b27_source_preconsumption_stop as stop
    root = Path(root).resolve()
    directory = root / PRECONSUMPTION_DIRECTORY
    profile.need(len(stop.FILES) == 9 and not directory.is_symlink(),
                 'exact nine-file B27 zero-consumption proof')
    resources = {PRECONSUMPTION_DIRECTORY + '/' + name for name in stop.FILES}
    for name in resources:
        path = root / name
        profile.need(path.is_file() and path.resolve().is_relative_to(root)
                     and not any(item.is_symlink() for item in (path, *path.parents)),
                     'complete regular original B27 preconsumption resource ' + name)
    found = {path.relative_to(root).as_posix() for path in directory.rglob('*') if path.is_file()}
    profile.need(found == resources, 'all and only nine original B27 preconsumption files')
    return resources


def resource_preflight():
    """Run the real original plan in a fresh captured-only read-only process.

    This entry is intentionally not usable inside the caller's process: its
    permanent audit hook forbids network, children, writes and scientific
    runtime imports. The actual plan verifies both complete original failures
    and the original probe; no function or successful result is mocked.
    """
    profile.need(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode,
                 'resource preflight requires a fresh -I -S -B process')
    resources = transport_resource_paths(ROOT) | preconsumption_resource_paths(ROOT)
    forbidden = dynamic.FORBIDDEN | {'tensorflow', 'jax', 'ctypes'}
    profile.need(not set(sys.modules) & forbidden, 'no scientific runtime before resource preflight')
    stdlib = Path(os.__file__).resolve().parent
    opened = set()

    def audit(event, args):
        if event == 'import':
            profile.need(args[0].split('.')[0] not in forbidden,
                         'no scientific or foreign runtime in original resource preflight')
        if event.startswith(('socket.', 'subprocess.', 'ctypes.')) or event in (
                'os.system', 'os.posix_spawn', 'os.fork', 'os.exec', 'os.remove',
                'os.rename', 'os.mkdir', 'os.rmdir', 'os.chmod', 'os.utime', 'os.link', 'os.symlink'):
            raise ValueError('read-only original resource preflight forbids ' + event)
        if event == 'open':
            path, mode, flags = args
            profile.need(isinstance(path, (str, bytes, os.PathLike)), 'no untracked descriptor reads in resource preflight')
            profile.need(not (flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)),
                         'no writes in original resource preflight')
            path = Path(os.fsdecode(path)).absolute()
            resolved = path.resolve()
            if resolved.is_relative_to(ROOT):
                name = resolved.relative_to(ROOT).as_posix()
                profile.need(not any(item.is_symlink() for item in (path, *path.parents))
                             and (name in resources or name.endswith(('.py', '.json', '.yml', '.yaml', '.md', '.txt'))
                                  or (name.endswith('.pyc') and not resolved.exists())),
                             'only captured code and declared original transport resources')
                opened.add(name)
            else:
                profile.need(resolved.is_relative_to(stdlib)
                             and 'site-packages' not in resolved.parts
                             and 'dist-packages' not in resolved.parts,
                             'no checkout fallback or external asset read in resource preflight')

    sys.addaudithook(audit)
    import english_v6_b21_fixed_mirror_transport as transport
    plan = transport.plan()
    import english_v6_b27_source_preconsumption_stop as stop
    stopped = stop.verify()
    profile.need(transport.PLAN.resolve() == ROOT / TRANSPORT_PLAN
                 and resources <= opened,
                 'real original plan reopens every captured transport resource')
    code = {}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not isinstance(filename, str):
            continue
        path = Path(filename).resolve()
        if path.is_relative_to(ROOT) and path.is_file():
            code[path.relative_to(ROOT).as_posix()] = profile.sha(path.read_bytes())
        if name.startswith(('english_', 'multilingual_campaign', 'kws_data_contract')):
            profile.need(path.is_relative_to(ROOT), 'all project modules are inside the captured root')
    profile.need(not set(sys.modules) & forbidden, 'no scientific runtime imported by original plan')
    with zipfile.ZipFile(ROOT / PRECONSUMPTION_DIRECTORY / 'producer.zip') as archive:
        original = profile.document(archive.read('PRODUCER.json'))['files']
        added_code = {'automation/english_v6_b27_source_producer.py',
                      'automation/english_v6_b27_source_preconsumption_stop.py'}
        adapters = {name: digest for name, digest in code.items() if name in REVIEWED_METADATA_ADAPTERS}
        profile.need(adapters == REVIEWED_METADATA_ADAPTERS,
                     'exact reviewed archive-accounting adapter; original numerical capacity limits unchanged')
        helpers = {name: digest for name, digest in code.items()
                   if name not in added_code and name not in REVIEWED_METADATA_ADAPTERS}
        profile.need(helpers and all(name in original and digest == original[name]['sha256']
                                    for name, digest in helpers.items()),
                     'all actually loaded original helpers match the authenticated failed producer')
    return dict(schema='english-v6-b27-captured-source-resource-preflight-v1',
        status='PASS_ORIGINAL_TRANSPORT_AND_PROOFS',
        original_plan_sha256=TRANSPORT_PLAN_SHA256,
        resource_files={name: dict(bytes=(ROOT / name).stat().st_size, sha256=profile.sha((ROOT / name).read_bytes()))
                        for name in sorted(resources)},
        loaded_repository_modules=dict(sorted(code.items())),
        authenticated_original_helper_files=len(helpers),
        reviewed_metadata_adapters=adapters,
        corpus_bytes=plan['bytes'], corpus_sha256=plan['sha256'],
        original_preconsumption_report_sha256=profile.sha(profile.canonical(stopped)),
        original_preconsumption_run_id=stop.RUN,
        original_plan_executions=1, new_source_calls=0, new_model_calls=0,
        new_network_calls=0, new_audio_reads=0, optimizer_updates=0,
        original_helpers_unchanged=True, acoustic_evidence=False)


def runtime_closure(directory, roots, blocked=frozenset()):
    """Original local static traversal, stopping before unreachable panel code.

    Blocking the three frozen replica-only imports before parsing their files
    ensures future quality dependencies cannot affect PRIMARY SOURCE capture.
    The original closure implementation and every historical module stay intact.
    """
    directory = Path(directory)
    repository = directory.parent
    def local(name):
        if name in blocked or not re.fullmatch(r'[a-zA-Z_][a-zA-Z_0-9]*', name):
            return []
        for parent in (directory, repository):
            module = parent / (name + '.py')
            if module.is_file():
                return [module]
            package = parent / name
            if (package / '__init__.py').is_file():
                return sorted(p for p in package.rglob('*.py') if p.is_file())
        return []
    pending, seen = [], set()
    for name in roots:
        path = directory / (name + '.py')
        profile.need(name not in blocked and path.is_file(), 'complete reachable source dependency ' + name)
        pending.append(path)
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        profile.need(path.resolve().is_relative_to(repository.resolve()), 'source code stays in captured repository')
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_bytes())):
            names = []
            if isinstance(node, ast.Import):
                names = [value.name.split('.')[0] for value in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split('.')[0]]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                literal = re.fullmatch(r'([a-zA-Z_][a-zA-Z_0-9]*)(?:\.py)?', node.value)
                if literal:
                    names = [literal.group(1)]
            pending.extend(p for name in names for p in local(name) if p not in seen)
    return {str(path.relative_to(repository)) for path in seen}


def configuration(panel):
    profile.need(panel in ('primary', 'replication'), 'explicit B27 SOURCE panel')
    name = 'english_v6_b27_source_ancestry' if panel == 'primary' else 'english_v6_b27_replication_source_ancestry'
    lineage = importlib.import_module(name)
    template = 'automation/english_v6_b27_' + ('' if panel == 'primary' else 'replication_') + 'source_workflow_template.yml'
    token = ('__B27_REVIEWED_' + panel.upper() + '_SOURCE_PARENT__').encode()
    return lineage, template, token


def runtime_paths(root=ROOT, panel='primary'):
    """Explicit reachable runtime closure; PRIMARY has no quality-reader authority.

    The shared profile's replica-only import is dormant while the immutable
    panel is primary and every primary_quality field is null. Its eventual
    implementation is not an input to PRIMARY SOURCE. Replica captures it and
    the already passed PRIMARY code/registration before any replica call.
    """
    root = Path(root)
    profile.need(panel in ('primary', 'replication'), 'registered source runtime profile')
    roots = ('english_v6_b27_source' if panel == 'primary' else 'english_v6_b27_replication_source',
             'english_v6_b27_source_producer', 'english_v6_b27_original_proof')
    if panel == 'replication':
        roots += ('english_v6_b27_quality_readback', 'english_v6_b27_primary_eval')
    blocked = PRIMARY_DORMANT_QUALITY if panel == 'primary' else frozenset()
    paths = runtime_closure(root / 'automation', roots, blocked)
    bases = tuple(name.removesuffix('.py') for name in profile.profile(False)['fixed_base_sha256'])
    paths |= runtime_closure(root / 'automation', bases, blocked)
    profile.need(not any(name.endswith(('_primary_eval.py', '_replication_eval.py')) and 'b27_' in name
                        for name in paths) if panel == 'primary' else True,
                 'new PRIMARY source does not depend on any later quality evaluator')
    return paths


def select_paths(tracked, root=ROOT, panel='primary'):
    root = Path(root)
    lineage, template, _ = configuration(panel)
    import english_v6_b27_producer as training
    paths = training.paths() | runtime_paths(root, panel) | set(dynamic.required_paths(root))
    paths |= transport_resource_paths(root)
    paths |= preconsumption_resource_paths(root)
    paths |= set(lineage.metadata.validate()['hashes'])
    paths |= {lineage.WORKFLOW, template, 'automation/english_v6_b27_execution_profile.json',
        'automation/english_v6_b27_' + panel + '_proof_binding.json'}
    if panel == 'replication':
        paths |= {'automation/english_v6_b27_primary_proof_binding.json',
                  'automation/english_v6_b27_primary_inputs.json', 'automation/english_v6_b27_primary_producer.json'}
    forbidden = {'automation/bind_b27_quality.py',
        'automation/english_v6_b27_' + panel + '_inputs.json',
        'automation/english_v6_b27_' + panel + '_producer.json',
        'automation/english_v6_b27_' + panel + '_workflow_template.yml',
        '.github/workflows/english-v6-b27-' + panel + '-model-eval-once.yml'}
    if panel == 'primary':
        forbidden |= {'automation/english_v6_b27_quality_readback.py',
            'automation/english_v6_b27_primary_eval.py', 'automation/english_v6_b27_replication_eval.py',
            'automation/english_v6_b27_replication_proof_binding.json',
            'automation/english_v6_b27_replication_inputs.json', 'automation/english_v6_b27_replication_producer.json'}
    profile.need(paths <= set(tracked) and not paths & forbidden,
                 'all actual source dependencies captured; future quality registration and code excluded; '
                 + 'missing=' + repr(sorted(paths - set(tracked)))
                 + '; forbidden=' + repr(sorted(paths & forbidden)))
    return paths


def envelope(files):
    total = sum(pin['bytes'] for pin in files.values())
    reserve = 4_000_000
    bound = capacity.base.zip_bound(total + reserve, len(files) * 2 + 64,
        sum(len(name.encode()) for name in files) * 2 + 65_536)
    profile.need(total + reserve <= capacity.CORE_LIMIT and bound <= capacity.CORE_LIMIT,
                 'whole original code/metadata producer fits128MB before any source calls')
    transport_archives = sorted(name for name in files if name.endswith('.zip') and (
        name.startswith(TRANSPORT_PROBE + '/')
        or any(name.startswith(directory + '/') for directory, _, _ in TRANSPORT_FAILURES)))
    preconsumption_artifacts = sorted(name for name in files
        if name.startswith(PRECONSUMPTION_DIRECTORY + '/') and name.endswith('.zip'))
    profile.need(len(transport_archives) == 5 and len(preconsumption_artifacts) == 2,
                 'all seven original code, probe and preconsumption archives retained without model or PCM inputs')
    return dict(expanded_bytes=total, metadata_reserve=reserve, maximum_zip_bytes=bound,
                limit_bytes=capacity.CORE_LIMIT, training_and_quality_input_archives_nested=0,
                retained_source_preconsumption_artifacts=len(preconsumption_artifacts),
                retained_source_preconsumption_artifact_paths=preconsumption_artifacts,
                retained_transport_original_archives=len(transport_archives),
                retained_transport_original_archive_paths=transport_archives)


def capture(output, prepared, panel='primary', root=ROOT):
    root, output = Path(root).resolve(), Path(output)
    lineage, template_path, token = configuration(panel)
    identity = lineage.execution_guard(prepared)
    profile.binding(panel)
    changed = common.git(root, 'diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD').decode().splitlines()
    parents = common.git(root, 'rev-list', '--parents', '-n', '1', 'HEAD').decode().split()[1:]
    profile.need(changed == [lineage.WORKFLOW] and parents == [prepared], 'one reviewed workflow-only source activation')
    template = (root / template_path).read_bytes()
    profile.need(template.count(token) == 2 and (root / lineage.WORKFLOW).read_bytes() == template.replace(token, prepared.encode()),
                 'only two frozen parent substitutions activate SOURCE')
    tracked = common.tree_entries(common.git(root, 'ls-tree', '-rz', '--full-tree', 'HEAD'))
    paths = select_paths(tracked, root, panel)
    payloads, files = {}, {}
    for name in sorted(paths):
        path = root / name
        profile.need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), 'regular tracked source member')
        raw = path.read_bytes()
        profile.need(common.blob(raw) == tracked[name]['git_blob_sha1'], 'every actual source producer Git byte')
        payloads[name] = raw
        files[name] = dict(tracked[name], bytes=len(raw), sha256=profile.sha(raw))
    measured = envelope(files)
    profile.need(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'fresh captured producer root')
    output.mkdir()
    for name, raw in payloads.items():
        target = output / 'repository' / name
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
    receipt = json.loads(subprocess.check_output([sys.executable, '-I', '-S', '-B',
        str(output / 'repository/automation/english_v6_b22_source_dynamic_preflight.py')]))
    resource_receipt = json.loads(subprocess.check_output([sys.executable, '-I', '-S', '-B',
        str(output / 'repository/automation/english_v6_b27_source_producer.py'), '--resource-preflight']))
    manifest = dict(schema=SCHEMA, panel=panel, **identity, runtime_workflow=lineage.WORKFLOW,
        git_tree=common.git(root, 'rev-parse', 'HEAD^{tree}').decode().strip(), files=files,
        python_files=sorted(name for name in paths if name.endswith('.py')),
        resource_files=sorted(name for name in paths if not name.endswith('.py')),
        external_proof_binding_sha256=profile.sha(lineage.INPUTS.read_bytes()),
        execution_profile_sha256=profile.sha(profile.PROFILE.read_bytes()), capacity=measured,
        captured_dynamic_import_preflight=receipt, source_code_reads_only=True,
        captured_resource_preflight=resource_receipt,
        runtime_dependency_policy='complete_original_train_plus_explicit_reachable_source_profile',
        primary_source_depends_on_future_quality=False,
        include_hidden_files=True, runtime_workflow_preserved_hidden=True,
        new_source_calls=0, new_model_forwards=0, optimizer_updates=0, promotion_allowed=False)
    (output / 'PRODUCER.json').write_bytes(profile.canonical(manifest))
    return manifest


if __name__ == '__main__':
    if sys.argv[1:] == ['--resource-preflight']:
        print(json.dumps(resource_preflight(), sort_keys=True, separators=(',', ':')))
        raise SystemExit(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--panel', choices=('primary', 'replication'), required=True)
    parser.add_argument('--prepared-parent', required=True)
    args = parser.parse_args()
    result = capture(args.output, args.prepared_parent, args.panel)
    print(json.dumps(dict(files=len(result['files']), manifest_sha256=profile.sha(profile.canonical(result))), sort_keys=True))
