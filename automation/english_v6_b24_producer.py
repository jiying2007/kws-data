"""Capture B24 with the full original B23 TRAIN producer and explicit new roots."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b24_b23_warm_start as train
import english_v6_b21_producer_snapshot as common
import english_v6_b23_primary_eval as closure

TEMPLATE = 'automation/english_v6_b24_training_workflow_template.yml'
TOKEN = b'__B24_REVIEWED_B23_WARM_START_PARENT__'


def paths(require_frozen=True):
    recipe = train.design()
    binding = train.assessment_binding(require_frozen)
    raw = (train.ROOT / recipe['original_b23_producer_manifest_path']).read_bytes()
    train.need(train.sha(raw) == recipe['original_b23_producer_manifest_sha256'], 'actual complete original B23 producer manifest')
    original = json.loads(raw)
    train.need(len(original['files']) == 306, 'all306 inherited B23 TRAIN dependencies')
    for name, pin in original['files'].items():
        current = (train.ROOT / name).read_bytes()
        if name == original['runtime_workflow']:
            parent = original.get('identity', original)['prepared_parent'].encode()
            token = b'__B23_REVIEWED_PREFIX_PEAK_PARENT__'
            train.need(current.count(token) == 2, 'original B23 workflow remains inert')
            current = current.replace(token, parent)
        train.need(train.sha(current) == pin['sha256'] and len(current) == pin['bytes'], 'unchanged original B23 dependency ' + name)
    names = set(original['files']) | set(binding['files'])
    names |= closure.static_closure(train.HERE, ('english_v6_b24_b23_warm_start', 'english_v6_b24_b23_failure',
        'english_v6_b24_producer', 'english_v6_b24_source_metadata', 'test_english_v6_b24_b23_warm_start',
        'test_english_v6_b24_source_metadata'))
    names |= {recipe['original_b23_producer_manifest_path'], TEMPLATE, train.WORKFLOW,
        train.DESIGN.relative_to(train.ROOT).as_posix(), train.BINDING.relative_to(train.ROOT).as_posix(),
        train.OPENING.relative_to(train.ROOT).as_posix(), recipe['train_only_mechanism_evidence']['path'], recipe['train_only_mechanism_memo']['path'],
        'automation/english_v6_b23_primary_eval.py'}
    train.need(all((train.ROOT / name).is_file() and not (train.ROOT / name).is_symlink() for name in names),
               'all original and explicit new scientific dependencies exist')
    return names


def capture(output, parent):
    identity = train.guard(parent)
    names = paths()
    root = train.ROOT
    changed = common.git(root, 'diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD').decode().splitlines()
    parents = common.git(root, 'rev-list', '--parents', '-n', '1', 'HEAD').decode().split()[1:]
    train.need(changed == [train.WORKFLOW] and parents == [parent], 'one workflow-only child of reviewed frozen preparation')
    template = (root / TEMPLATE).read_bytes()
    train.need(template.count(TOKEN) == 2 and (root / train.WORKFLOW).read_bytes() == template.replace(TOKEN, parent.encode()),
               'only two parent pin substitutions activate original B24 workflow')
    tracked = common.tree_entries(common.git(root, 'ls-tree', '-rz', '--full-tree', 'HEAD'), names)
    output = Path(output)
    train.need(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'new immutable B24 producer root')
    payloads, files = {}, {}
    for name in sorted(names):
        raw = (root / name).read_bytes()
        train.need(name in tracked and common.blob(raw) == tracked[name]['git_blob_sha1'], 'exact original Git bytes ' + name)
        files[name] = dict(tracked[name], bytes=len(raw), sha256=train.sha(raw))
        payloads[name] = raw
    output.mkdir()
    for name, raw in payloads.items():
        destination = output / 'repository' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    manifest = dict(schema='english-v6-b24-complete-git-producer-v1', identity=identity,
        git_tree=common.git(root, 'rev-parse', 'HEAD^{tree}').decode().strip(), files=files,
        original_b23_dependency_files=306, dependency_source='all_original306_plus_explicit_B24_and_fresh_metadata',
        source_code_reads_only=True, runtime_workflow=train.WORKFLOW, include_hidden_files=True,
        recipe_sha256=train.DESIGN_SHA, assessment_binding_sha256=train.sha(train.BINDING.read_bytes()),
        opening_sha256=train.sha(train.OPENING.read_bytes()), new_model_forwards=0, optimizer_updates=0, new_source_calls=0)
    train.write(output / 'PRODUCER.json', manifest)
    print(json.dumps(dict(files=len(files), bytes=sum(p['bytes'] for p in files.values()),
        manifest_sha256=train.sha((output / 'PRODUCER.json').read_bytes()))))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--prepared-parent', required=True)
    args = parser.parse_args()
    capture(args.output, args.prepared_parent)
