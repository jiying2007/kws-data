"""Capture B24Q1 source execution with full original B24 TRAIN, immutable R1 STOP and approved Q1 policy.

Every job executes from this captured root. Future B24Q1 scoring bindings are
excluded because they are frozen only after the new source result exists.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import english_v6_b21_producer_snapshot as common
import english_v6_b22_source_dynamic_preflight as dynamic
import english_v6_b24q1_source_ancestry as lineage

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = 'automation/english_v6_b24q1_source_workflow_template.yml'
TOKEN = b'__B24Q1_REVIEWED_SOURCE_PARENT__'
EXTRA = {'research/english_hardneg_increment_20261117/OWNER_REVIEW.json',
    'research/english_v6_b24q1_preparation_20261125/APPROVAL.json',
    'research/english_v6_b24q1_preparation_20261125/APPROVED_PROPOSAL_SOURCE.json',
    'research/english_v6_b24r1_source_stop_20261125/READBACK.json',
    'research/english_v6_b24r1_source_stop_20261125/ORIGINAL_LOCK.json'}
FUTURE_PREFIXES = (
    'automation/english_v6_b24q1_primary_', 'automation/test_english_v6_b24q1_primary_',
    'automation/bind_b24q1_primary', 'automation/test_bind_b24q1_primary',
    'automation/english_v6_b24q1_quality_', 'automation/test_english_v6_b24q1_quality_',
    'automation/english_v6_b24q1_replication_', 'automation/test_english_v6_b24q1_replication_',
    'automation/english_v6_b24q1_engineering', 'automation/test_english_v6_b24q1_engineering',
    '.github/workflows/english-v6-b24q1-primary-model-',
    '.github/workflows/english-v6-b24q1-primary-eval-',
    '.github/workflows/english-v6-b24q1-replication-', '.github/workflows/english-v6-b24q1-engineering-',
)


def select_paths(tracked, root=ROOT):
    root = Path(root)
    paths = {name for name in tracked if (name.startswith('automation/')
        or name.startswith('.github/workflows/')
        or name.startswith('research/english_v6_b22_training_20261124/initial_failure/')
        or name.startswith('research/english_v6_b22_source_20261124/initial_failure/') or name in EXTRA)
        and not name.startswith(FUTURE_PREFIXES)}
    # The original B23 TRAIN closure includes resources outside automation.
    prior = json.loads((root / 'automation/english_v6_b23_source_training_producer.json').read_bytes())
    paths |= set(prior['files']) | set(dynamic.required_paths(root))
    # Include every exact metadata input, even when outside automation.
    paths |= set(lineage.metadata.validate()['hashes'])
    binding = json.loads((root / 'automation/english_v6_b24_source_inputs.json').read_bytes())
    if binding['bound']:
        raw = (root / 'automation/english_v6_b24_source_training_producer.json').read_bytes()
        common.need(common.sha(raw) == binding['training_producer_manifest_sha256'], 'exact completed original B24 TRAIN producer')
        paths |= set(json.loads(raw)['files'])
    required = EXTRA | {lineage.WORKFLOW, TEMPLATE, 'automation/english_v6_b22_source_dynamic_preflight.py',
        'automation/english_v6_b24q1_source.py', 'automation/english_v6_b24q1_source_ancestry.py',
        'automation/english_v6_b24q1_source_observe.py', 'automation/english_v6_b24q1_natural_sources.py',
        'automation/english_v6_b24q1_source_transport.py', 'automation/english_v6_b24q1_source_producer.py',
        'automation/english_v6_b24_source_training_proof.py', 'automation/english_v6_b24_source_inputs.json',
        'automation/english_v6_b24_source_training_producer.json', 'automation/test_english_v6_b24q1_source_entry.py',
        'automation/test_english_v6_b24q1_quarantine.py',
        'automation/test_english_v6_b24q1_source_contract.py',
        'automation/english_v6_b24q1_quarantine.py',
        'automation/english_v6_b24q1_capacity.py',
        'automation/english_v6_b24q1_evidence_scope.py',
        'automation/test_english_v6_b24q1_evidence_scope.py',
        'automation/english_v6_b24q1_orchestration_contract.json',
        'automation/english_v6_b24q1_r1_source_failure.py',
        'automation/english_v6_b24q1_r1_git_transport.py',
        'automation/english_v6_b24q1_r1_git_transport.json',
        'automation/test_english_v6_b24q1_r1_git_transport.py',
        'automation/english_v6_b24r1_source_stop_readback.py',
        'automation/test_english_v6_b24q1_r1_source_failure.py',
        'automation/english_v6_b24q1_source_failure_inputs.json',
        'automation/english_v6_b24q1_evaluation_binding.json',
        'automation/english_v6_b24q1_evaluation_design.json'}
    common.need(required <= paths and paths <= set(tracked), 'all source and original dynamic dependencies stay tracked')
    common.need(not any(name.startswith(FUTURE_PREFIXES) for name in paths),
                'future B24Q1 PRIMARY/replica/engineering bindings cannot change the source capture')
    return paths


def capture(output, prepared, root=ROOT):
    root, output = Path(root).resolve(), Path(output)
    identity = lineage.execution_guard(prepared)
    lineage.training_binding()
    lineage.failure.read_binding()
    head = common.git(root, 'rev-parse', 'HEAD').decode().strip()
    parents = common.git(root, 'rev-list', '--parents', '-n', '1', 'HEAD').decode().split()[1:]
    changed = common.git(root, 'diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD').decode().splitlines()
    common.need(identity['run_head'] == head and parents == [prepared] and changed == [lineage.WORKFLOW],
                'one workflow-only child of the reviewed source preparation')
    template = (root / TEMPLATE).read_bytes()
    common.need(template.count(TOKEN) == 2 and (root / lineage.WORKFLOW).read_bytes() == template.replace(TOKEN, prepared.encode()),
                'only two exact reviewed parent substitutions activate this new source experiment')
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
    receipt = json.loads(subprocess.check_output([sys.executable, '-SIB',
        str(output / 'repository/automation/english_v6_b22_source_dynamic_preflight.py')], text=True))
    manifest = dict(schema='english-v6-b24q1-complete-source-producer-v1', role='source',
        **dict(identity, run_branch=identity['run_branch'].removeprefix('refs/heads/')),
        runtime_workflow=lineage.WORKFLOW, git_tree=common.git(root, 'rev-parse', 'HEAD^{tree}').decode().strip(),
        files=files, python_files=sorted(name for name in paths if name.endswith('.py')),
        resource_files=sorted(name for name in paths if not name.endswith('.py')),
        closure_rule='all_tracked_automation_workflows_exact_approved_Q1_metadata_complete_original_B24_TRAIN_and_all19_dynamic_helpers_excluding_future_B24Q1_scoring',
        captured_dynamic_import_preflight=receipt, excluded_future_prefixes=list(FUTURE_PREFIXES),
        source_code_reads_only=True, runtime_workflow_preserved_hidden=True, include_hidden_files=True,
        network_calls=0, new_source_calls=0, new_model_forwards=0, optimizer_updates=0, promotion_allowed=False)
    raw = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    (output / 'PRODUCER.json').write_bytes(raw)
    print(json.dumps(dict(files=len(files), bytes=sum(pin['bytes'] for pin in files.values()),
        manifest_sha256=common.sha(raw), run_head=head, prepared_parent=prepared)))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--prepared-parent', required=True)
    args = parser.parse_args()
    capture(args.output, args.prepared_parent)
