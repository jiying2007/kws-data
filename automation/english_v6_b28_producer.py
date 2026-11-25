"""Capture all original and new B28 dependencies before its once-only budget."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b28_train as train
import english_v6_b28_source_metadata as metadata
import english_v6_b27_producer as previous
import english_v6_b21_producer_snapshot as common
import english_v6_b23_primary_eval as closure

TEMPLATE = 'automation/english_v6_b28_training_workflow_template.yml'
TOKEN = b'__B28_REVIEWED_NATURAL_EVENT_PARENT__'
SCHEMA = 'english-v6-b28-complete-git-producer-v1'
ORIGINAL_PRIMARY_MANIFEST = 'automation/english_v6_b27_primary_producer.json'
ORIGINAL_PRIMARY_MANIFEST_SHA = '1a174d05f22254009f1d8023cf5e2366edb44941e6825d7bf66c7a1a1b678ebe'
NEW_MODULES = ('english_v6_b28_train', 'english_v6_b28_producer',
    'english_v6_b27_primary_no_promotion_readback', 'test_english_v6_b27_primary_no_promotion_readback',
    'english_v6_b28_natural_event_fit', 'english_v6_b28_no_event_objective',
    'english_v6_b28_no_event_reference', 'english_v6_b28_objective_readback',
    'english_v6_b28_training_readback', 'english_v6_b28_source_metadata',
    'test_english_v6_b28_train', 'test_english_v6_b28_natural_event_fit',
    'test_english_v6_b28_objective_readback', 'test_english_v6_b28_training_readback',
    'test_english_v6_b28_source_metadata')


def original_primary_manifest():
    raw = (train.ROOT / ORIGINAL_PRIMARY_MANIFEST).read_bytes()
    train.need(train.sha(raw) == ORIGINAL_PRIMARY_MANIFEST_SHA,
               'exact complete original B27 PRIMARY/source/TRAIN producer registration bytes')
    manifest = train.failure.document(raw)
    train.need(manifest['frozen'] is True and manifest['schema'] == 'english-v6-b27-primary-producer-v1',
               'complete original B27 PRIMARY/source/TRAIN producer registration')
    return manifest


def original_archive_resources(names, manifest):
    # Seven small historical failed-transport/source receipts are part of the
    # already authenticated original code closure. They are not the 59 dynamic
    # TRAIN/PRIMARY originals, which must remain outside this new producer.
    resources = {name: digest for name, digest in manifest['resources'].items()
                 if Path(name).suffix.lower() == '.zip'}
    upstream = manifest['upstream_manifests']['source']['files']
    train.need(len(resources) == 7 and {name for name in names if Path(name).suffix.lower() == '.zip'} == set(resources),
               'only seven exact manifest-bound historical ZIP resources; dynamic original proof stays external')
    pins = {}
    for name, digest in sorted(resources.items()):
        train.capacity.safe_name(name)
        pin = upstream[name]
        path = train.ROOT / name
        train.need(pin['sha256'] == digest and path.is_file() and not path.is_symlink(),
                   'original inherited ZIP resource is explicitly bound ' + name)
        raw = path.read_bytes()
        train.need(len(raw) == pin['bytes'] and train.sha(raw) == digest
                   and common.blob(raw) == pin['git_blob_sha1'] and pin['mode'] == '100644',
                   'all original inherited ZIP resource bytes and Git identity ' + name)
        pins[name] = dict(bytes=pin['bytes'], sha256=digest)
    return pins


def paths(require_frozen=True):
    recipe = train.design(require_frozen)
    binding = train.assessment_binding(require_frozen)
    names = set(previous.paths())
    # PRIMARY readback dynamically restores historical producers and resources.
    # Preserve its full existing closures, not only static Python imports.
    manifest = original_primary_manifest()
    names.add(ORIGINAL_PRIMARY_MANIFEST)
    names |= set(manifest['resources'])
    for role in ('primary', 'source'):
        names |= set(manifest[role]['files']) | set(manifest[role].get('resources', {}))
    for upstream in manifest['upstream_manifests'].values():
        names |= set(upstream['files'])
    # Both validators are loaded by explicit strings, so a static import walk
    # cannot establish their complete non-Python resource closure. The original
    # B27 assessment binding is already verified and retained by previous.paths.
    collector = metadata._collector()
    collector.registration(train.ROOT)
    metadata._previous()
    names |= set(collector.paths()) | {metadata.PREVIOUS}
    names |= closure.static_closure(train.HERE, (Path(metadata.COLLECTOR).stem, Path(metadata.PREVIOUS).stem))
    names |= closure.static_closure(train.HERE, NEW_MODULES)
    names |= {'automation/' + name + '.py' for name in NEW_MODULES}
    names |= {TEMPLATE, train.WORKFLOW, train.DESIGN.relative_to(train.ROOT).as_posix(),
        train.BINDING.relative_to(train.ROOT).as_posix(), train.OPENING.relative_to(train.ROOT).as_posix(),
        'automation/english_v6_b27_execution_profile.json', 'automation/english_v6_b27_primary_proof_binding.json'}
    names |= set(binding['files'])
    names |= {pin['path'] for pin in recipe['frozen_dependencies']}
    train.need(all((train.ROOT / name).is_file() and not (train.ROOT / name).is_symlink() for name in names),
               'all338 historical files, complete B27 producer resources and all explicit new dependencies exist')
    original_archive_resources(names, manifest)
    return names


def workflow(parent):
    template = (train.ROOT / TEMPLATE).read_bytes()
    train.need(template.count(TOKEN) == 2 and train.failure.valid_hash(parent, 40),
               'only two exact reviewed-parent substitutions activate the new B28 workflow')
    return template.replace(TOKEN, parent.encode())


def capacity_receipt(files):
    return previous.capacity_receipt(files)


def capture(output, parent):
    identity = train.guard(parent)
    names, root = paths(), train.ROOT
    changed = common.git(root, 'diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD').decode().splitlines()
    parents = common.git(root, 'rev-list', '--parents', '-n', '1', 'HEAD').decode().split()[1:]
    train.need(changed == [train.WORKFLOW] and parents == [parent], 'one workflow-only child of reviewed preparation')
    train.need((root / train.WORKFLOW).read_bytes() == workflow(parent), 'exact original activated workflow bytes')
    tracked = common.tree_entries(common.git(root, 'ls-tree', '-rz', '--full-tree', 'HEAD'), names)
    output = Path(output)
    train.need(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'fresh immutable B28 producer directory')
    payloads, files = {}, {}
    for name in sorted(names):
        raw = (root / name).read_bytes()
        train.need(name in tracked and common.blob(raw) == tracked[name]['git_blob_sha1'], 'exact committed original Git file ' + name)
        files[name] = dict(tracked[name], bytes=len(raw), sha256=train.sha(raw))
        payloads[name] = raw
    envelope = capacity_receipt(files)
    output.mkdir()
    for name, raw in payloads.items():
        target = output / 'repository' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as handle:
            handle.write(raw)
    manifest = dict(schema=SCHEMA, identity=identity, files=files,
        git_tree=common.git(root, 'rev-parse', 'HEAD^{tree}').decode().strip(),
        original_b24_dependency_files=338, original_b27_dependency_files=len(previous.paths()),
        dependency_source='all_original338_plus_complete_B27_primary_source_TRAIN_and_explicit_B28',
        source_code_reads_only=True, runtime_workflow=train.WORKFLOW, include_hidden_files=True,
        recipe_sha256=train.DESIGN_SHA, assessment_binding_sha256=train.sha(train.BINDING.read_bytes()),
        opening_sha256=train.sha(train.OPENING.read_bytes()), capacity=envelope, **train.objective_fit.contract(),
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
