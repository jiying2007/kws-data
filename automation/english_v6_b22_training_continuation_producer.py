"""Capture B22 from the complete actual B21 manifest plus declared new resources."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b22_union_retention_continuation as train
import english_v6_b22_training_origin as origin
import english_v6_b21_producer_snapshot as common

NEW_FILES = {
    'automation/english_v6_b22_union_retention.py',
    'automation/english_v6_b22_union_retention_design.json',
    'automation/english_v6_b22_assessment_binding.json',
    'automation/english_v6_b22_b21_producer_manifest.json',
    'automation/english_v6_b22_producer.py',
    'automation/test_english_v6_b22_union_retention.py',
    'automation/english_v6_b22_training_continuation_workflow_template.yml',
}


def paths(require_frozen=True):
    train.assessment_binding(require_frozen)
    origin.local()
    first = json.loads((origin.EVIDENCE / 'PRODUCER.json').read_bytes())
    proof = json.loads(origin.SPEC.read_bytes())
    extra = {
        'automation/english_v6_b22_union_retention_continuation.py',
        'automation/english_v6_b22_training_continuation_producer.py',
        'automation/english_v6_b22_training_continuation_workflow_template.yml',
        'automation/english_v6_b22_training_origin.py',
        'automation/english_v6_b22_training_origin.json',
        'automation/test_english_v6_b22_training_continuation.py',
    }
    names = set(first['files']) | set(proof['evidence_files']) | extra | {train.WORKFLOW}
    train.need(all((train.ROOT / name).is_file() and not (train.ROOT / name).is_symlink() for name in names),
               'complete original243 plus explicit first-fit continuation dependency inventory')
    return names


def capture(output, parent):
    identity = train.guard(parent)
    root = train.ROOT
    names = paths()
    changed = common.git(root, 'diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD').decode().splitlines()
    parents = common.git(root, 'rev-list', '--parents', '-n', '1', 'HEAD').decode().split()[1:]
    train.need(changed == [train.WORKFLOW] and parents == [parent], 'one workflow-only child of reviewed preparation')
    template = (root / 'automation/english_v6_b22_training_continuation_workflow_template.yml').read_bytes()
    token = b'__B22_REVIEWED_TRAIN_CONTINUATION_PARENT__'
    train.need(template.count(token) == 2 and (root / train.WORKFLOW).read_bytes() == template.replace(token, parent.encode()),
               'exact reviewed B22 workflow with two parent substitutions only')
    tracked = common.tree_entries(common.git(root, 'ls-tree', '-rz', '--full-tree', 'HEAD'), names)
    output = Path(output)
    train.need(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'fresh immutable producer snapshot')
    payloads, files = {}, {}
    for name in sorted(names):
        raw = (root / name).read_bytes()
        train.need(name in tracked and common.blob(raw) == tracked[name]['git_blob_sha1'], 'exact Git-bound producer member ' + name)
        files[name] = dict(tracked[name], bytes=len(raw), sha256=train.sha(raw))
        payloads[name] = raw
    output.mkdir()
    for name, raw in payloads.items():
        target = output / 'repository' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    manifest = dict(schema='english-v6-b22-complete-git-producer-v1', identity=identity,
        git_tree=common.git(root, 'rev-parse', 'HEAD^{tree}').decode().strip(), files=files,
        original_b21_dependency_files=221, original_b22_dependency_files=243, dependency_source='all_original243_plus_zero_call_origin_and_first_fit_continuation',
        source_code_reads_only=True, runtime_workflow=train.WORKFLOW, include_hidden_files=True,
        recipe_sha256=train.DESIGN_SHA, assessment_binding_sha256=train.sha(train.BINDING.read_bytes()),
        new_model_forwards=0, optimizer_updates=0, new_source_calls=0)
    train.write(output / 'PRODUCER.json', manifest)
    print(json.dumps(dict(files=len(files), original_dependencies=221, manifest_sha256=train.sha((output / 'PRODUCER.json').read_bytes()))))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--prepared-parent', required=True)
    args = parser.parse_args()
    capture(args.output, args.prepared_parent)
