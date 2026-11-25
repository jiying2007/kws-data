"""Capture the complete source execution root, including dynamic old dependencies.

The capture includes the repository's tracked automation library and historical
workflow/resource bytes. Prospective B24Q1 scoring bindings are explicitly outside
the source producer because binding them occurs only after this source result.
Execution and all subsequent observer jobs run from this captured root.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b21_producer_snapshot as common
import english_v6_b24q1_replication_source_ancestry as lineage
import english_v6_b22_source_dynamic_preflight as dynamic

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = 'automation/english_v6_b24q1_replication_source_workflow_template.yml'
TOKEN = b'__B24Q1_REVIEWED_REPLICATION_SOURCE_PARENT__'
EXTRA = {'research/english_hardneg_increment_20261117/OWNER_REVIEW.json',
    'research/english_v6_b24q1_preparation_20261125/APPROVAL.json',
    'research/english_v6_b24q1_preparation_20261125/APPROVED_PROPOSAL_SOURCE.json',
    'research/english_v6_b24r1_source_stop_20261125/READBACK.json',
    'research/english_v6_b24r1_source_stop_20261125/ORIGINAL_LOCK.json'}
FUTURE_PREFIXES = (
    'automation/english_v6_b24q1_replication_eval', 'automation/english_v6_b24q1_replication_inputs',
    'automation/english_v6_b24q1_replication_producer.json','automation/english_v6_b24q1_replication_workflow_template',
    'automation/test_english_v6_b24q1_replication_eval','automation/english_v6_b24q1_replication_bind',
    'automation/english_v6_b24q1_engineering','automation/test_english_v6_b24q1_engineering',
    '.github/workflows/english-v6-b24q1-replication-model-', '.github/workflows/english-v6-b24q1-engineering-',
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
    # The full approved assessment and exact input resources are part of the
    # source producer, including resources outside the automation directory.
    paths |= set(lineage.metadata.validate()['hashes'])
    # These are read dynamically by raw_asr._load, outside automation/. Both
    # runner and its campaign import are also used by Qwen asset verification.
    paths |= set(dynamic.required_paths(root))
    binding = json.loads((Path(root) / 'automation/english_v6_b24_source_inputs.json').read_bytes())
    if binding['bound']:
        raw = (Path(root) / 'automation/english_v6_b24_source_training_producer.json').read_bytes()
        common.need(common.sha(raw) == binding['training_producer_manifest_sha256'], 'exact successful complete TRAIN producer')
        paths |= set(json.loads(raw)['files'])
    required = EXTRA | {'automation/english_v6_b24q1_replication_opening.json', 'automation/english_v6_b22_replication_reuse.py', lineage.WORKFLOW, TEMPLATE, 'automation/english_v6_b24q1_replication_source.py',
        'automation/english_v6_b24q1_replication_source_ancestry.py', 'automation/english_v6_b24q1_replication_source_observe.py',
        'automation/english_v6_b24q1_replication_transport.py', 'automation/english_v6_b24q1_replication_natural_sources.py',
        'automation/english_v6_b24q1_replication_source_producer.py', 'automation/english_v6_b24_source_inputs.json',
        'automation/english_v6_b24_source_training_producer.json',
        'automation/english_v6_b24_source_training_proof.py',
        'automation/english_v6_b23_source_training_producer.json',
        'automation/english_v6_b22_source_training_producer.json',
        'automation/english_v6_b24q1_capacity.py',
        'automation/english_v6_b24q1_evidence_scope.py',
        'automation/test_english_v6_b24q1_evidence_scope.py',
        'automation/test_english_v6_b24q1_capacity.py',
        'automation/english_v6_b24q1_primary_eval.py',
        'automation/english_v6_b24q1_primary_inputs.json',
        'automation/english_v6_b24q1_primary_producer.json',
        'automation/english_v6_b24q1_primary_workflow_template.yml',
        '.github/workflows/english-v6-b24q1-primary-model-eval-once.yml',
        '.github/workflows/english-v6-b24q1-primary-sources-once.yml',
        'automation/english_v6_b24q1_quality_readback.py',
        'automation/english_v6_b24q1_r1_source_failure.py',
        'automation/english_v6_b24q1_source_failure_inputs.json',
        'automation/english_v6_b24q1_r1_git_transport.py',
        'automation/english_v6_b24q1_r1_git_transport.json',
        'automation/test_english_v6_b24q1_r1_git_transport.py',
        'automation/english_v6_b24r1_source_stop_readback.py',
        'automation/english_v6_b24q1_quarantine.py',
        'automation/test_english_v6_b24q1_quarantine.py',
        'automation/english_v6_b24q1_orchestration_contract.json',
        'automation/english_v6_b24q1_source_metadata.py',
        'automation/english_v6_b24q1_evaluation_design.json',
        'automation/english_v6_b24q1_evaluation_binding.json',
        'automation/english_v6_b22_source_dynamic_preflight.py',
        'automation/test_english_v6_b24q1_replication_sources.py'}
    common.need(required <= paths, 'all declared source, observer and producer entrypoints captured')
    common.need(paths <= set(tracked), 'all original dynamic and external dependencies remain tracked')
    common.need(not any(name.startswith(FUTURE_PREFIXES) for name in paths),
                'future replica scoring and engineering cannot change the replica source capture')
    return paths


def captured_preflight(repository, root=ROOT):
    """Actually import/check only the captured stdlib source; no opening or calls."""
    repository = Path(repository)
    process = subprocess.run([sys.executable, '-I', '-S', '-B',
        str(repository / 'automation/english_v6_b22_source_dynamic_preflight.py')],
        cwd=repository, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    dynamic_check = json.loads(process.stdout)
    common.need(dynamic_check['helper_pins'] == dynamic.required_paths(root)
        and dynamic_check['code_reads_only'] is dynamic_check['no_model'] is dynamic_check['no_network'] is True,
        'actual isolated original19 helper dependency preflight')
    source = subprocess.run([sys.executable, '-I', '-S', '-B',
        str(repository / 'automation/english_v6_b24q1_replication_source.py'), 'check'],
        cwd=repository, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    common.need(source.stdout == b'PASS: sealed B24Q1 replica metadata; zero new audio, observations or KWS forwards\n'
        and source.stderr == b'', 'captured complete Q1 source policy, metadata and resources import in isolation')
    return dict(schema='english-v6-b24q1-replication-source-dependency-preflight-v1',
        original_dynamic_helpers=dynamic_check, source_check_stdout_sha256=common.sha(source.stdout),
        code_reads_only=True, no_model=True, no_network=True,
        expected_original_B24R1_STOP_location='PRIMARY_QUALITY/inputs/source.zip')


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
    runtime_check = captured_preflight(output / 'repository', root)
    manifest = dict(schema='english-v6-b24q1-complete-source-producer-v1', role='source',
        **dict(identity, run_branch=identity['run_branch'].removeprefix('refs/heads/')),
        runtime_workflow=lineage.WORKFLOW, git_tree=common.git(root, 'rev-parse', 'HEAD^{tree}').decode().strip(),
        files=files, python_files=sorted(n for n in paths if n.endswith('.py')),
        resource_files=sorted(n for n in paths if not n.endswith('.py')),
        runtime_dependency_preflight=runtime_check,
        closure_rule='complete_original_TRAIN_and_frozen_PRIMARY_plus_replica_source_excluding_future_replica_scoring',
        approved_proposal_sha256=lineage.metadata.PROPOSAL_SHA,
        approval_sha256=lineage.metadata.APPROVAL_SHA,
        orchestration_contract_sha256=lineage.metadata.ORCHESTRATION_SHA,
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
