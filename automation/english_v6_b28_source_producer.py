"""Capture complete B28 SOURCE code and small metadata; all big originals stay external."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b28_profile as profile
import english_v6_b28_producer as training
import english_v6_b28_source_metadata as metadata
import english_v6_b28_capacity as capacity
import english_v6_b21_producer_snapshot as common
import english_v6_b22_source_dynamic_preflight as dynamic

original = profile.preserved('english_v6_b27_source_producer')
ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'english-v6-b28-complete-source-producer-v1'
DORMANT = frozenset(('english_v6_b28_quality_readback', 'english_v6_b28_primary_eval', 'english_v6_b28_replication_eval'))


def configuration(panel):
    profile.need(panel in ('primary', 'replication'), 'explicit B28 source panel')
    name = 'english_v6_b28_source_ancestry' if panel == 'primary' else 'english_v6_b28_replication_source_ancestry'
    lineage = importlib.import_module(name)
    template = 'automation/english_v6_b28_' + ('' if panel == 'primary' else 'replication_') + 'source_workflow_template.yml'
    return lineage, template, ('__B28_REVIEWED_' + panel.upper() + '_SOURCE_PARENT__').encode()


def runtime_paths(root=ROOT, panel='primary'):
    profile.need(panel in ('primary', 'replication'), 'registered B28 runtime')
    roots = ('english_v6_b28_source' if panel == 'primary' else 'english_v6_b28_replication_source',
        'english_v6_b28_source_producer', 'english_v6_b28_original_proof')
    if panel == 'replication':
        roots += ('english_v6_b28_quality_readback', 'english_v6_b28_primary_eval')
    blocked = DORMANT if panel == 'primary' else frozenset()
    paths = original.runtime_closure(Path(root) / 'automation', roots, blocked)
    bases = tuple(name.removesuffix('.py') for name in profile.profile(False)['fixed_base_sha256'])
    paths |= original.runtime_closure(Path(root) / 'automation', bases, blocked)
    profile.need(not any(Path(name).stem in DORMANT for name in paths) if panel == 'primary' else True,
        'PRIMARY SOURCE never depends on a future B28 quality evaluator')
    return paths


def select_paths(tracked, root=ROOT, panel='primary'):
    root = Path(root)
    lineage, template, _ = configuration(panel)
    paths = training.paths() | runtime_paths(root, panel) | set(dynamic.required_paths(root))
    paths |= set(metadata.validate_registration()['hashes'])
    paths |= {lineage.WORKFLOW, template, 'automation/english_v6_b28_execution_profile.json',
        'automation/english_v6_b28_' + panel + '_proof_binding.json',
        'automation/english_v6_b28_multicorpus_source.py', 'automation/test_english_v6_b28_multicorpus_source.py',
        'automation/test_english_v6_b28_sources.py'}
    if panel == 'replication':
        paths |= {'automation/english_v6_b28_primary_proof_binding.json', 'automation/english_v6_b28_primary_inputs.json',
            'automation/english_v6_b28_primary_producer.json'}
    forbidden = {'automation/bind_b28_quality.py', 'automation/english_v6_b28_' + panel + '_inputs.json',
        'automation/english_v6_b28_' + panel + '_producer.json',
        'automation/english_v6_b28_' + panel + '_workflow_template.yml',
        '.github/workflows/english-v6-b28-' + panel + '-model-eval-once.yml'}
    if panel == 'primary':
        forbidden |= {'automation/' + name + '.py' for name in DORMANT}
        forbidden |= {'automation/english_v6_b28_replication_proof_binding.json',
            'automation/english_v6_b28_replication_inputs.json', 'automation/english_v6_b28_replication_producer.json'}
    profile.need(paths <= set(tracked) and not paths & forbidden,
        'complete source closure, without future quality or external catalog originals; missing=' + repr(sorted(paths - set(tracked)))
        + '; forbidden=' + repr(sorted(paths & forbidden)))
    training.original_archive_resources(paths, training.original_primary_manifest())
    return paths


def resource_preflight(panel='primary'):
    """Fresh captured-only stdlib process; all actual resource reads are local."""
    profile.need(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode, 'fresh -I -S -B source preflight')
    stdlib = Path(os.__file__).resolve().parent
    blocked = dynamic.FORBIDDEN | {'tensorflow', 'jax', 'ctypes'}
    profile.need(not set(sys.modules) & blocked, 'no scientific runtime before resource preflight')
    opened = set()
    def audit(event, args):
        if event == 'import':
            profile.need(args[0].split('.')[0] not in blocked, 'no scientific import in captured source preflight')
        if event.startswith(('socket.', 'subprocess.', 'ctypes.')) or event in (
            'os.system', 'os.posix_spawn', 'os.fork', 'os.exec', 'os.remove', 'os.rename', 'os.mkdir',
            'os.rmdir', 'os.chmod', 'os.utime', 'os.link', 'os.symlink'):
            raise ValueError('read-only source preflight forbids ' + event)
        if event == 'open':
            path, _, flags = args
            profile.need(isinstance(path, (str, bytes, os.PathLike))
                and not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND), 'only tracked read-only files')
            path = Path(os.fsdecode(path)).absolute()
            resolved = path.resolve()
            if resolved.is_relative_to(ROOT):
                profile.need(not any(p.is_symlink() for p in (path, *path.parents)), 'captured regular resource only')
                opened.add(resolved.relative_to(ROOT).as_posix())
            else:
                profile.need(resolved.is_relative_to(stdlib) and 'site-packages' not in resolved.parts
                    and 'dist-packages' not in resolved.parts, 'no checkout fallback or external input')
    sys.addaudithook(audit)
    checked = metadata.validate_registration()
    collector = metadata._collector()
    collector.registration()
    owner = importlib.import_module('english_v6_b28_source' if panel == 'primary' else 'english_v6_b28_replication_source')
    owner.lineage.read_plan()
    owner.source_plan()
    contract = owner.code_contract()
    owner.validate_config(owner.PIPER_CONFIG)
    owner.natural.plan()
    profile.binding(panel)
    profile.need(set(checked['hashes']) <= opened and set(collector.FROZEN) <= opened,
        'every exact frozen metadata and publisher resource reopened inside the captured producer')
    return dict(schema='english-v6-b28-captured-source-resource-preflight-v1', status='PASS_FROZEN_REGISTRATION_ONLY',
        panel=panel, metadata_hashes=checked['hashes'], collector_sha256=metadata.COLLECTOR_SHA,
        source_policy=owner.source_policy(), scientific_contract_sha256=profile.sha(profile.canonical(contract)),
        opened_resources=sorted(opened), complete_external_originals_still_required=True,
        new_model_calls=0, new_source_calls=0, new_network_calls=0, new_audio_reads=0, optimizer_updates=0)


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
        'only two exact parent substitutions activate new SOURCE')
    tracked = common.tree_entries(common.git(root, 'ls-tree', '-rz', '--full-tree', 'HEAD'))
    paths = select_paths(tracked, root, panel)
    payloads, files = {}, {}
    for name in sorted(paths):
        path = root / name
        profile.need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), 'ordinary tracked source file')
        raw = path.read_bytes()
        profile.need(common.blob(raw) == tracked[name]['git_blob_sha1'], 'every captured source file has its exact Git identity')
        payloads[name] = raw
        files[name] = dict(tracked[name], bytes=len(raw), sha256=profile.sha(raw))
    envelope = original.envelope(files)
    profile.need(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'fresh complete B28 source producer')
    output.mkdir()
    for name, raw in payloads.items():
        target = output / 'repository' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(raw)
    dynamic_receipt = json.loads(subprocess.check_output([sys.executable, '-I', '-S', '-B',
        str(output / 'repository/automation/english_v6_b22_source_dynamic_preflight.py')]))
    resource_receipt = json.loads(subprocess.check_output([sys.executable, '-I', '-S', '-B',
        str(output / 'repository/automation/english_v6_b28_source_producer.py'), '--resource-preflight', panel]))
    manifest = dict(schema=SCHEMA, panel=panel, **identity, runtime_workflow=lineage.WORKFLOW,
        git_tree=common.git(root, 'rev-parse', 'HEAD^{tree}').decode().strip(), files=files,
        python_files=sorted(n for n in paths if n.endswith('.py')), resource_files=sorted(n for n in paths if not n.endswith('.py')),
        external_proof_binding_sha256=profile.sha(lineage.INPUTS.read_bytes()),
        execution_profile_sha256=profile.sha(profile.PROFILE.read_bytes()), capacity=envelope,
        captured_dynamic_import_preflight=dynamic_receipt, captured_resource_preflight=resource_receipt,
        source_code_reads_only=True, runtime_dependency_policy='complete_original_train_and_external_catalog_plus_explicit_source',
        primary_source_depends_on_future_quality=False, complete_catalog_artifacts_nested=0,
        include_hidden_files=True, runtime_workflow_preserved_hidden=True,
        new_source_calls=0, new_model_forwards=0, optimizer_updates=0, promotion_allowed=False)
    (output / 'PRODUCER.json').write_bytes(profile.canonical(manifest))
    return manifest


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--resource-preflight' and sys.argv[2] in ('primary', 'replication'):
        print(json.dumps(resource_preflight(sys.argv[2]), sort_keys=True))
        raise SystemExit(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--panel', choices=('primary', 'replication'), required=True)
    parser.add_argument('--prepared-parent', required=True)
    args = parser.parse_args()
    value = capture(args.output, args.prepared_parent, args.panel)
    print(json.dumps(dict(files=len(value['files']), manifest_sha256=profile.sha(profile.canonical(value))), sort_keys=True))
