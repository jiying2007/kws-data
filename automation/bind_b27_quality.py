"""Bind all actual original B27 quality inputs offline into a fresh review root."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import importlib
from pathlib import Path
import sys
import tempfile
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b27_profile as profile
import english_v6_b27_quality_readback as quality

PROVIDERS = ('qwen', 'whisper', 'parakeet', 'xlsr', 'lv60')
ROLES = ('B0', 'B9', 'pair', 'source', 'screen', *('native_' + p for p in PROVIDERS), 'train_producer', 'source_producer')
GROUPS = ('B0', 'B9', 'train', 'source')


def artifact_name(role, run, panel):
    if role == 'pair': return 'english-v6-b27-natural-weight-models-' + str(run)
    if role == 'train_producer': return 'english-v6-b27-train-producer-' + str(run)
    if role.startswith('native_'): return 'b27-' + panel + '-dev-observer-' + role.removeprefix('native_')
    suffix = {'source': 'source', 'screen': 'frozen-screen', 'source_producer': 'source-producer'}[role]
    return 'english-v6-b27-' + panel + '-' + suffix + '-' + str(run)


def metadata(bundle, base, evaluator, panel):
    profile.need(bundle['schema'] == 'english-v6-b27-' + panel + '-original-quality-binding-inputs-v1'
                 and set(bundle['groups']) == set(GROUPS) and set(bundle['archives']) == set(ROLES),
                 'four complete actual API groups and twelve original archives')
    groups = {name: {key: profile.document((base / path).read_bytes()) for key, path in group.items()}
              for name, group in bundle['groups'].items()}
    pins, evidence = {}, {}
    for role in ROLES:
        key = 'train' if role in ('pair', 'train_producer') else role if role in ('B0', 'B9') else 'source'
        group = groups[key]
        profile.need({'run', 'commit', 'artifacts'} <= set(group), 'entire final original API group')
        run, commit = group['run'], group['commit']
        inventory = group['artifacts']
        profile.need(inventory['total_count'] == len(inventory['artifacts']), 'untruncated whole original artifact inventory')
        if role in evaluator.BASELINES:
            selected = [item for item in inventory['artifacts'] if item['id'] == evaluator.BASELINES[role]['id']]
        else:
            selected = [item for item in inventory['artifacts'] if item['name'] == artifact_name(role, run['id'], panel)]
        profile.need(len(selected) == 1 and selected[0]['digest'].startswith('sha256:'), 'exact original named artifact ' + role)
        item = selected[0]
        pin = dict(id=item['id'], run_id=run['id'], run_head=run['head_sha'], bytes=item['size_in_bytes'],
                   sha256=item['digest'].removeprefix('sha256:'))
        evaluator.validate_provenance(pin, dict(run=run, commit=commit, artifact=item), role=role)
        if role in evaluator.BASELINES: profile.need(pin == evaluator.BASELINES[role], 'immutable original baseline')
        pins[role], evidence[role] = pin, dict(group, artifact=item)
    profile.need(len({pin['id'] for pin in pins.values()}) == 12
                 and groups['train']['run']['head_branch'] == evaluator.TRAIN_BRANCH
                 and groups['source']['run']['head_branch'] == evaluator.SOURCE_BRANCH,
                 'actual distinct new B27 training and source events')
    return groups, pins, evidence


def registration(evaluator, panel, pins, training):
    source = importlib.import_module(evaluator.SOURCE_MODULE)
    bound = profile.binding(panel)
    return dict(schema=evaluator.SCHEMA, panel=panel, scoring_allowed=False,
        prepared_parent=None, parent_authority='reviewed_workflow', source_module=evaluator.SOURCE_MODULE,
        source_schema=evaluator.SOURCE_SCHEMA, screen_schema=evaluator.SCREEN_SCHEMA,
        pair_schema=evaluator.PAIR_SCHEMA, pair_stems=evaluator.STEMS, parent_pt_sha256=profile.INITIAL_PT,
        model_pt_sha256=training['result']['model_pt_sha256'], model_comparison=evaluator.MODEL_COMPARISON,
        same_job_matched_control=False, historical_control_retrained=False,
        gains=list(evaluator.GAINS), source_gate=evaluator.GATE, maximum_model_forwards=1440,
        **dict.fromkeys(evaluator.FLAGS, False), replication_opened=panel == 'replication',
        original_primary_pass_required=panel == 'replication', primary_quality_lock=bound['primary_quality_lock'],
        inputs={role: pins[role] for role in evaluator.INPUT_KEYS},
        native_inputs={provider: pins['native_' + provider] for provider in PROVIDERS},
        upstream_producers={role: pins[role + '_producer'] for role in ('train', 'source')},
        assessment_sha256=evaluator.EVALUATION_ASSESSMENT, training_assessment_sha256=evaluator.TRAINING_ASSESSMENT,
        original_B24_training_assessment_sha256=evaluator.ORIGINAL_B24_TRAINING_ASSESSMENT,
        training_design_sha256=profile.DESIGN_SHA, execution_profile_sha256=profile.sha(profile.PROFILE.read_bytes()),
        external_proof_binding_sha256=evaluator.spec_binding_sha(), producer_manifest_sha256=None,
        complete_original_B24Q1_failure_required=True, all_old_stops_and_reserves_preserved=True,
        new_optimizer_updates=0, evaluation_only=True, source_policy=source.source_policy())


def producer_document(evaluator, panel, snapshots, spec, root):
    """Freeze the complete original upstream closure without loading any model."""
    root = Path(root)
    source_python = {name for name in snapshots['source']['files'] if name.endswith('.py')}
    source_resources = {name for name in snapshots['source']['files'] if not name.endswith('.py') and name != evaluator.SOURCE_WORKFLOW}
    code = evaluator.static_closure(root / 'automation', evaluator.PRODUCER_ROOTS)
    code |= {'automation/' + name for name in profile.profile(False)['fixed_base_sha256']}
    resources = set(source_resources)
    for role in ('train', 'source'):
        code |= {name for name in snapshots[role]['files'] if name.endswith('.py')}
        resources |= {name for name in snapshots[role]['files'] if not name.endswith('.py')
                      and name != snapshots[role]['runtime_workflow']}
    import english_v6_b27_source_producer as captured_source
    profile.need(captured_source.runtime_paths(root, panel) <= source_python,
                 'all actual SOURCE dynamic and static code already captured originally')
    resources -= {'automation/' + evaluator.SPEC.name, 'automation/' + evaluator.PRODUCER.name,
                  evaluator.WORKFLOW, evaluator.SOURCE_WORKFLOW}
    resources |= {'automation/english_v6_b14_panels/assessment.json', 'automation/english_v6_b27_panels/assessment.json',
                  'automation/english_v6_b27_natural_weight_design.json', 'automation/english_v6_b27_execution_profile.json',
                  'automation/english_v6_b27_' + panel + '_proof_binding.json'}
    if panel == 'replication':
        resources |= {'automation/english_v6_b27_primary_inputs.json', 'automation/english_v6_b27_primary_producer.json'}
        profile.need(profile.sha((root / 'automation/english_v6_b27_primary_inputs.json').read_bytes()) == spec['primary_quality_lock']['inputs_sha256']
                     and profile.sha((root / 'automation/english_v6_b27_primary_producer.json').read_bytes()) == spec['primary_quality_lock']['producer_manifest_sha256'],
                     'same complete originally passed PRIMARY registration')
    def hashes(names): return {name: profile.sha((root / name).read_bytes()) for name in sorted(names)}
    producer = dict(schema='english-v6-b27-' + panel + '-producer-v1', frozen=True,
        **{panel: dict(files=hashes(code), workflow_path=evaluator.WORKFLOW,
                       workflow_template_sha256=profile.sha(evaluator.WORKFLOW_TEMPLATE.read_bytes()))},
        source=dict(files=hashes(source_python), resources=hashes(source_resources), workflow_path=evaluator.SOURCE_WORKFLOW,
            workflow_sha256=snapshots['source']['files'][evaluator.SOURCE_WORKFLOW]['sha256']),
        resources=hashes(resources), upstream_manifests=snapshots)
    return producer, code, resources


def bind(panel, bundle_path, proof_root, output):
    root = Path(__file__).resolve().parents[1]
    bundle_path, proof_root, output = Path(bundle_path), Path(proof_root), Path(output)
    profile.need(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'fresh immutable review output')
    evaluator = importlib.import_module('english_v6_b27_' + panel + '_eval')
    profile.need(profile.document(evaluator.SPEC.read_bytes())['scoring_allowed'] is False,
                 'bind only the inert original panel; never retarget an existing scorer')
    with profile.proof_scope(proof_root, panel), quality.offline_only(proof_root, panel):
        training = profile.require_proof(panel)
        bundle = profile.document(bundle_path.read_bytes())
        groups, pins, meta = metadata(bundle, bundle_path.parent, evaluator, panel)
        spec = registration(evaluator, panel, pins, training)
        with ExitStack() as stack, tempfile.TemporaryDirectory(prefix='b27-quality-binding-') as temporary:
            archives = {role: stack.enter_context(evaluator.original_zip(bundle_path.parent / bundle['archives'][role], pin, role=role))
                        for role, pin in pins.items()}
            snapshots = {role: evaluator.snapshot_contract(archives[role + '_producer'], role, meta[role + '_producer'], root)
                         for role in ('train', 'source')}
            train = evaluator.validate_training_archive(archives['pair'], spec, meta['pair'])
            for role in ('control', 'challenger'):
                profile.need(profile.sha(archives['pair'].read(evaluator.STEMS[role] + '.pt')) == spec['model_pt_sha256'][role],
                             'actual full original trained checkpoint hash ' + role)
            directory, providers = Path(temporary) / 'source', Path(temporary) / 'providers'
            directory.mkdir(); archives['source'].extractall(directory)
            for provider in PROVIDERS:
                target = providers / provider; target.mkdir(parents=True)
                archives['native_' + provider].extractall(target)
            _, _, source, plan = evaluator.helpers(spec)
            owner = source.evaluation_preflight(directory)
            screen_raw = archives['screen'].read('SCREEN.json'); screen = profile.document(screen_raw)
            source.verify_screen_contract(directory, screen, providers)
            audit = evaluator.source_audit(screen, owner['rows'], owner['all_original_rows'], source.source_policy(), screen_raw)
            execution, captured = owner['source_execution'], snapshots['source']
            profile.need(str(execution['run_id']) == str(pins['source']['run_id'])
                         and execution['run_head'] == pins['source']['run_head']
                         and execution['prepared_parent'] == captured['prepared_parent'], 'same original first source identity')
            selected = evaluator.exact_admitted_join(owner['rows'], screen['rows'])
            reserve = set()
            if panel == 'primary':
                reserve = {evaluator.person(row) for row in source.source_plan('replication')['rows']}
                reserve.update(str(int(value)) for value in plan['replication']['natural']['speakers'])
            else:
                prior = training['primary_quality']
                profile.need(owner['primary_quality'] == prior and prior['panel_gate'] == 'PASS', 'same full original PRIMARY PASS')
                previous = prior['source_inventory']
                profile.need(not set(previous['pcm_sha256']) & {r['pcm_sha256'] for r in owner['all_original_rows']}
                             and not set(previous['canonical_people']) & {evaluator.person(r) for r in owner['all_original_rows']}
                             and not set(previous['source_group_id']) & {r['source_group_id'] for r in owner['all_original_rows']},
                             'all120 new replica originals independent of all120 primary originals')
            evaluator.reject_overlap(owner['all_original_rows'], train['base_train'] + train['natural_train'], owner['history'], reserve)
            counts = evaluator.source_counts(selected)
            profile.need(counts == screen['source_coverage'] and all(counts[key] >= value for key, value in evaluator.GATE.items())
                         and screen['source_gate'] == 'SOURCE_GATE_PASS' and screen['source_shortfall'] == {},
                         'unchanged whole source population meets every prospective minimum')
            envelope = evaluator.capacity.score_budget(selected)
            producer, code, resources = producer_document(evaluator, panel, snapshots, spec, root)
            producer_raw = profile.canonical(producer)
            spec.update(scoring_allowed=True, producer_manifest_sha256=profile.sha(producer_raw))
            spec_raw = profile.canonical(spec); spec_file = Path(temporary) / evaluator.SPEC.name; spec_file.write_bytes(spec_raw)
            with mock.patch.object(evaluator, 'SPEC', spec_file), mock.patch.object(evaluator, 'CANDIDATE_PT', spec['model_pt_sha256']['challenger']):
                evaluator.protocol()
            profile.need(not {'torch', 'numpy', 'onnxruntime'} & set(sys.modules), 'no scientific runtime during original binding')
            output.mkdir(); target = output / 'automation'; target.mkdir()
            (target / evaluator.SPEC.name).write_bytes(spec_raw); (target / evaluator.PRODUCER.name).write_bytes(producer_raw)
            report = dict(schema='english-v6-b27-original-quality-binding-audit-v1', panel=panel,
                status='BOUND_ORIGINAL_INPUTS_FOR_REVIEWED_ONCE_QUALITY', evaluator_sha256=profile.sha(Path(evaluator.__file__).read_bytes()),
                registration_sha256=profile.sha(spec_raw), producer_manifest_sha256=profile.sha(producer_raw),
                input_artifacts=pins, original_source_rows=120, observer_eligible_rows=len(owner['rows']), admitted_originals=len(selected),
                source_coverage=counts, source_audit=audit, output_capacity=envelope, planned_model_forwards=len(selected) * 12,
                full_original_training_report_sha256=profile.binding(panel)['training_report_sha256'],
                all_original_screens_recomputed=True, every_actual_upstream_producer_member_preserved=True,
                scientific_code_files=len(code), resource_files=len(resources), model_pt_sha256=spec['model_pt_sha256'],
                new_model_calls=0, new_source_calls=0, new_observer_calls=0, remote_mutations=0,
                execution_started=False, promotion_allowed=False, product_released=False)
            (output / 'BINDING_AUDIT.json').write_bytes(profile.canonical(report))
            return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', choices=('primary', 'replication'))
    parser.add_argument('bundle', type=Path)
    parser.add_argument('proof_root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(profile.canonical(bind(args.panel, args.bundle, args.proof_root, args.output)).decode())
