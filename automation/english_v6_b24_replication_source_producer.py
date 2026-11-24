"""Capture the complete source execution root, including dynamic old dependencies.

The capture includes the repository's tracked automation library and historical
workflow/resource bytes. Prospective B24 scoring bindings are explicitly outside
the source producer because binding them occurs only after this source result.
Execution and all subsequent observer jobs run from this captured root.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b21_producer_snapshot as common
import english_v6_b24_replication_source_ancestry as lineage
import english_v6_b22_source_dynamic_preflight as dynamic

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = 'automation/english_v6_b24_replication_source_workflow_template.yml'
TOKEN = b'__B24_REVIEWED_REPLICATION_SOURCE_PARENT__'
EXTRA = {'research/english_hardneg_increment_20261117/OWNER_REVIEW.json'}
FUTURE_PREFIXES = (
    'automation/english_v6_b24_replication_eval', 'automation/english_v6_b24_replication_inputs',
    'automation/english_v6_b24_replication_producer.json','automation/english_v6_b24_replication_workflow_template',
    'automation/test_english_v6_b24_replication_eval','automation/english_v6_b24_replication_bind',
    'automation/english_v6_b24_engineering','automation/test_english_v6_b24_engineering',
    '.github/workflows/english-v6-b24-replication-model-', '.github/workflows/english-v6-b24-engineering-',
)


def select_paths(tracked, root=ROOT):
    original = json.loads((Path(root) / 'automation/english_v6_b23_source_training_producer.json').read_bytes())
    common.need(original['schema'] == 'english-v6-b23-complete-git-producer-v1'
                and len(original['files']) == 306, 'complete actual historical B23 TRAIN closure')
    paths = {name for name in tracked if (name.startswith('automation/')
             or name.startswith('.github/workflows/')
             or name.startswith('research/english_v6_b22_training_20261124/initial_failure/')
             or name.startswith('research/english_v6_b22_source_20261124/initial_failure/') or name in EXTRA)
             and not name.startswith(FUTURE_PREFIXES)}
    paths |= set(original['files'])
    # These are read dynamically by raw_asr._load, outside automation/. Both
    # runner and its campaign import are also used by Qwen asset verification.
    paths |= set(dynamic.required_paths(root))
    binding = json.loads((Path(root) / 'automation/english_v6_b24_source_inputs.json').read_bytes())
    if binding['bound']:
        raw = (Path(root) / 'automation/english_v6_b24_source_training_producer.json').read_bytes()
        common.need(common.sha(raw) == binding['training_producer_manifest_sha256'], 'exact successful complete TRAIN producer')
        paths |= set(json.loads(raw)['files'])
    required = EXTRA | {'automation/english_v6_b24_replication_opening.json', 'automation/english_v6_b22_replication_reuse.py', lineage.WORKFLOW, TEMPLATE, 'automation/english_v6_b24_replication_source.py',
        'automation/english_v6_b24_replication_source_ancestry.py', 'automation/english_v6_b24_replication_source_observe.py',
        'automation/english_v6_b24_replication_transport.py', 'automation/english_v6_b24_replication_natural_sources.py',
        'automation/english_v6_b24_replication_source_producer.py', 'automation/english_v6_b24_source_inputs.json',
        'automation/english_v6_b24_source_training_producer.json',
        'automation/english_v6_b24_source_training_proof.py',
        'automation/english_v6_b23_source_training_producer.json',
        'automation/english_v6_b22_source_training_producer.json',
        'automation/english_v6_b24_replication_capacity.py',
        'automation/test_english_v6_b24_replication_capacity.py',
        'automation/english_v6_b22_source_dynamic_preflight.py',
        'automation/test_english_v6_b24_replication_sources.py'}
    common.need(required <= paths, 'all declared source, observer and producer entrypoints captured')
    common.need(paths <= set(tracked), 'all original dynamic and external dependencies remain tracked')
    return paths


def capture(output, prepared, root=ROOT):
    root, output = Path(root).resolve(), Path(output)
    identity = lineage.execution_guard(prepared)
    lineage.training_binding()
    head = common.git(root, 'rev-parse', 'HEAD').decode().strip()
    parents = common.git(root, 'rev-list', '--parents', '-n', '1', 'HEAD').decode().split()[1:]
    changed = common.git(root, 'diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD').decode().splitlines()
    common.need(identity['run_head'] == head and parents == [prepared] and changed == [lineage.WORKFLOW],
                'single workflow-only child of reviewed preparation')
    template = (root / TEMPLATE).read_bytes()
    common.need(template.count(TOKEN) == 2 and (root / lineage.WORKFLOW).read_bytes() == template.replace(TOKEN, prepared.encode()),
                'only two exact parent substitutions activate the reviewed workflow')
    tracked = common.tree_entries(common.git(root, 'ls-tree', '-rz', '--full-tree', 'HEAD'))
    paths = select_paths(tracked, root)
    common.need(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'new immutable source producer root')
    files, payloads = {}, {}
    for name in sorted(paths):
        path = root / name
        common.need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
                    'tracked regular source file only: ' + name)
        raw = path.read_bytes()
        common.need(common.blob(raw) == tracked[name]['git_blob_sha1'], 'original Git bytes: ' + name)
        files[name] = dict(tracked[name], bytes=len(raw), sha256=common.sha(raw))
        payloads[name] = raw
    output.mkdir()
    for name, raw in payloads.items():
        target = output / 'repository' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    process = subprocess.run([sys.executable, '-I', '-S', '-B',
        str(output / 'repository/automation/english_v6_b22_source_dynamic_preflight.py')],
        cwd=output / 'repository', check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    runtime_check = json.loads(process.stdout)
    common.need(runtime_check['helper_pins'] == dynamic.required_paths(root)
                and runtime_check['code_reads_only'] is runtime_check['no_model'] is runtime_check['no_network'] is True,
                'actual isolated captured-root observer import, checksum and runner preflight')
    manifest = dict(schema='english-v6-b24-complete-source-producer-v1', role='source',
        **dict(identity, run_branch=identity['run_branch'].removeprefix('refs/heads/')),
        runtime_workflow=lineage.WORKFLOW, git_tree=common.git(root, 'rev-parse', 'HEAD^{tree}').decode().strip(),
        files=files, python_files=sorted(n for n in paths if n.endswith('.py')),
        resource_files=sorted(n for n in paths if not n.endswith('.py')),
        runtime_dependency_preflight=runtime_check,
        closure_rule='complete_original_TRAIN_and_frozen_PRIMARY_plus_replica_source_excluding_future_replica_scoring',
        excluded_future_prefixes=list(FUTURE_PREFIXES), source_code_reads_only=True,
        runtime_workflow_preserved_hidden=True, include_hidden_files=True, network_calls=0, new_source_calls=0,
        new_model_forwards=0, optimizer_updates=0, promotion_allowed=False)
    raw = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    (output / 'PRODUCER.json').write_bytes(raw)
    print(json.dumps(dict(files=len(files), bytes=sum(p['bytes'] for p in files.values()),
        manifest_sha256=common.sha(raw), run_head=head, prepared_parent=prepared)))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--prepared-parent', required=True)
    args = parser.parse_args()
    capture(args.output, args.prepared_parent)
