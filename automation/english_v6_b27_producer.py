"""Capture complete B27 code and frozen metadata before its single TRAIN budget."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b27_natural_weight as train
import english_v6_b21_producer_snapshot as common
import english_v6_b23_primary_eval as closure

TEMPLATE = 'automation/english_v6_b27_training_workflow_template.yml'
TOKEN = b'__B27_REVIEWED_NATURAL_WEIGHT_PARENT__'
SCHEMA = 'english-v6-b27-complete-git-producer-v1'


def paths(require_frozen=True):
    recipe = train.design()
    binding = train.assessment_binding(require_frozen)
    original = train.failure.document((train.ROOT / recipe['original_training_producer']['path']).read_bytes())
    train.need(len(original['files']) == recipe['original_training_dependency_files'] == 338,
               'all338 complete original B24 TRAIN dependencies')
    for name, pin in original['files'].items():
        raw = (train.ROOT / name).read_bytes()
        if name == original['runtime_workflow']:
            token = b'__B24_REVIEWED_B23_WARM_START_PARENT__'
            train.need(raw.count(token) == 2, 'original B24 workflow remains inert')
            raw = raw.replace(token, original['identity']['prepared_parent'].encode())
        train.need(train.sha(raw) == pin['sha256'] and len(raw) == pin['bytes'], 'unchanged historical dependency ' + name)
    names = set(original['files']) | set(binding['files'])
    names |= closure.static_closure(train.HERE, ('english_v6_b27_natural_weight', 'english_v6_b27_producer', 'bind_b27_training',
        'english_v6_b27_training_readback', 'test_english_v6_b27_natural_weight', 'test_english_v6_b27_training_readback'))
    names |= {TEMPLATE, train.WORKFLOW, train.DESIGN.relative_to(train.ROOT).as_posix(),
        train.BINDING.relative_to(train.ROOT).as_posix(), train.OPENING.relative_to(train.ROOT).as_posix()}
    for key in ('baseline_recipe', 'baseline_fit', 'baseline_wrapper', 'proposal', 'evidence', 'bounded_proposal',
                'original_training_binding', 'original_training_producer', 'failure_reader', 'base_quality_reader',
                'original_B24_quality_function'):
        names.add(recipe[key]['path'])
    names |= {'research/english_v6_b27_preparation_20261125/' + name for name in ('README.md', 'ID_AVAILABILITY.json')}
    train.need(all((train.ROOT / name).is_file() and not (train.ROOT / name).is_symlink() for name in names),
               'all original and explicit new scientific dependencies exist')
    return names


def workflow(parent):
    template = (train.ROOT / TEMPLATE).read_bytes()
    train.need(template.count(TOKEN) == 2 and train.failure.valid_hash(parent, 40),
               'only two reviewed predecessor substitutions activate B27')
    return template.replace(TOKEN, parent.encode())


def capacity_receipt(files):
    total = sum(pin['bytes'] for pin in files.values())
    # PRODUCER.json, all possible directory entries and their names fit this
    # prospective reserve; neither raw source nor old ZIPs enter the producer.
    reserve = 4_000_000
    bound = train.capacity.zip_bound(total + reserve, len(files) * 2 + 64,
        sum(len(name.encode('utf-8')) for name in files) * 2 + 64 * 1024)
    train.need(total + reserve <= train.capacity.CORE_LIMIT and bound <= train.capacity.CORE_LIMIT,
               'complete code/metadata producer fits unchanged128MB limit before inputs or budget')
    return dict(expanded_bytes=total, manifest_and_directory_reserve=reserve,
                zip_bound=bound, limit=train.capacity.CORE_LIMIT, original_input_archives_embedded=0)


def capture(output, parent):
    identity = train.guard(parent)
    names, root = paths(), train.ROOT
    changed = common.git(root, 'diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD').decode().splitlines()
    parents = common.git(root, 'rev-list', '--parents', '-n', '1', 'HEAD').decode().split()[1:]
    train.need(changed == [train.WORKFLOW] and parents == [parent], 'one workflow-only child of reviewed frozen preparation')
    train.need((root / train.WORKFLOW).read_bytes() == workflow(parent), 'exact activated workflow from reviewed template')
    tracked = common.tree_entries(common.git(root, 'ls-tree', '-rz', '--full-tree', 'HEAD'), names)
    output = Path(output)
    train.need(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'fresh immutable B27 producer root')
    payloads, files = {}, {}
    for name in sorted(names):
        raw = (root / name).read_bytes()
        train.need(name in tracked and common.blob(raw) == tracked[name]['git_blob_sha1'], 'exact original Git bytes ' + name)
        files[name] = dict(tracked[name], bytes=len(raw), sha256=train.sha(raw))
        payloads[name] = raw
    envelope = capacity_receipt(files)
    output.mkdir()
    for name, raw in payloads.items():
        target = output / 'repository' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    manifest = dict(schema=SCHEMA, identity=identity, files=files,
        git_tree=common.git(root, 'rev-parse', 'HEAD^{tree}').decode().strip(),
        original_b24_dependency_files=338, dependency_source='all_original338_plus_explicit_B27_and_fresh_metadata',
        source_code_reads_only=True, runtime_workflow=train.WORKFLOW, include_hidden_files=True,
        recipe_sha256=train.DESIGN_SHA, assessment_binding_sha256=train.sha(train.BINDING.read_bytes()),
        opening_sha256=train.sha(train.OPENING.read_bytes()), capacity=envelope,
        new_model_forwards=0, optimizer_updates=0, new_source_calls=0)
    train.write(output / 'PRODUCER.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--prepared-parent', required=True)
    args = parser.parse_args()
    result = capture(args.output, args.prepared_parent)
    print(json.dumps(dict(files=len(result['files']), manifest_sha256=train.sha(train.encoded(result))), sort_keys=True))
