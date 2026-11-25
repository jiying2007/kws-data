"""Capture complete Git-bound B27 SOURCE code before source consumption."""
from __future__ import annotations
import argparse
import ast
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b27_profile as profile
import english_v6_b27_capacity as capacity
import english_v6_b21_producer_snapshot as common
import english_v6_b22_source_dynamic_preflight as dynamic

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'english-v6-b27-complete-source-producer-v1'
PRIMARY_DORMANT_QUALITY = frozenset(('english_v6_b27_quality_readback',
    'english_v6_b27_primary_eval', 'english_v6_b27_replication_eval'))


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
                 'all actual source dependencies captured; future quality registration and code excluded')
    return paths


def envelope(files):
    total = sum(pin['bytes'] for pin in files.values())
    reserve = 4_000_000
    bound = capacity.base.zip_bound(total + reserve, len(files) * 2 + 64,
        sum(len(name.encode()) for name in files) * 2 + 65_536)
    profile.need(total + reserve <= capacity.CORE_LIMIT and bound <= capacity.CORE_LIMIT,
                 'whole original code/metadata producer fits128MB before any source calls')
    return dict(expanded_bytes=total, metadata_reserve=reserve, maximum_zip_bytes=bound,
                limit_bytes=capacity.CORE_LIMIT, original_input_archives_nested=0)


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
    manifest = dict(schema=SCHEMA, panel=panel, **identity, runtime_workflow=lineage.WORKFLOW,
        git_tree=common.git(root, 'rev-parse', 'HEAD^{tree}').decode().strip(), files=files,
        python_files=sorted(name for name in paths if name.endswith('.py')),
        resource_files=sorted(name for name in paths if not name.endswith('.py')),
        external_proof_binding_sha256=profile.sha(lineage.INPUTS.read_bytes()),
        execution_profile_sha256=profile.sha(profile.PROFILE.read_bytes()), capacity=measured,
        captured_dynamic_import_preflight=receipt, source_code_reads_only=True,
        runtime_dependency_policy='complete_original_train_plus_explicit_reachable_source_profile',
        primary_source_depends_on_future_quality=False,
        include_hidden_files=True, runtime_workflow_preserved_hidden=True,
        new_source_calls=0, new_model_forwards=0, optimizer_updates=0, promotion_allowed=False)
    (output / 'PRODUCER.json').write_bytes(profile.canonical(manifest))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--panel', choices=('primary', 'replication'), required=True)
    parser.add_argument('--prepared-parent', required=True)
    args = parser.parse_args()
    result = capture(args.output, args.prepared_parent, args.panel)
    print(json.dumps(dict(files=len(result['files']), manifest_sha256=profile.sha(profile.canonical(result))), sort_keys=True))
