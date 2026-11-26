"""B28 once-only four-model quality adapter, inert until complete originals bind.

Unchanged B27/B24Q1 code objects implement source accounting, model loading,
the actual forward ledger, full raw FP32 capture, and the B14 decision. This
module supplies the new TRAIN identity and checks every complete upstream.
"""
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b28_profile as profile

original = profile.preserved('english_v6_b27_primary_eval')
profile.inherit(original, globals())
capacity = profile.preserved('english_v6_b27_capacity')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PANEL = 'primary'
SPEC = HERE / 'english_v6_b28_primary_inputs.json'
PRODUCER = HERE / 'english_v6_b28_primary_producer.json'
WORKFLOW_TEMPLATE = HERE / 'english_v6_b28_primary_workflow_template.yml'
TRAIN_DESIGN = HERE / 'english_v6_b28_natural_event_design.json'
ASSESSMENT = HERE / 'english_v6_b28_panels/assessment.json'
TRAINING_DESIGN = profile.DESIGN_SHA
TRAINING_ASSESSMENT = EVALUATION_ASSESSMENT = profile.profile(False)['assessment_sha256']
ORIGINAL_B24_TRAINING_ASSESSMENT = original.ORIGINAL_B24_TRAINING_ASSESSMENT
CONTROL_PT, INITIAL_PT, B27_PT = profile.CONTROL_PT, profile.INITIAL_PT, profile.B27_PT
MODEL_COMPARISON = 'B28_new_challenger_against_immutable_historical_B24_challenger'
PAIR_SCHEMA = 'english-v6-b28-train-natural-event-v1'
PAIR_STATUS = 'TRAINED_NATURAL_EVENT_CANDIDATE_REQUIRES_NEW_INDEPENDENT_PRIMARY'
STEMS = {'control': 'control/models/mero43-b24-challenger192',
         'challenger': 'challenger/models/mero43-b28-challenger192'}
SCHEMA = 'english-v6-b28-fresh-primary-four-frozen-models-v1'
SOURCE_MODULE = 'english_v6_b28_source'
SOURCE_SCHEMA = 'english-v6-b28-fresh-combined-source-v1'
SCREEN_SCHEMA = 'english-v6-b28-120-original-quarantine-source-screen-v1'
BRANCH = 'refs/heads/research/en-v6-b28-primary-scoring-once-20261125'
WORKFLOW = '.github/workflows/english-v6-b28-primary-model-eval-once.yml'
SOURCE_BRANCH = 'research/en-v6-b28-primary-sources-once-20261125'
SOURCE_WORKFLOW = '.github/workflows/english-v6-b28-primary-sources-once.yml'
SOURCE_TEMPLATE = 'automation/english_v6_b28_source_workflow_template.yml'
SOURCE_PARENT_TOKEN = '__B28_REVIEWED_PRIMARY_SOURCE_PARENT__'
TRAIN_BRANCH = 'research/en-v6-b28-natural-event-train-once-20261125'
TRAIN_WORKFLOW = '.github/workflows/english-v6-b28-natural-event-train-once.yml'
PARENT_TOKEN = '__B28_REVIEWED_PRIMARY_SCORING_PARENT__'
MARKER = 'english-v6-b28-primary-consumed'
FLAGS = original.FLAGS
PRODUCER_ROOTS = ('english_v6_b28_primary_eval', SOURCE_MODULE,
                  'english_v6_b28_quality_readback', 'english_v6_four_model_panel', 'english_v6_b14_decision')
CANDIDATE_PT = profile.document(SPEC.read_bytes()).get('model_pt_sha256', {}).get('challenger') if SPEC.is_file() else None
_original_write = write
_original_score = score


def objective_contract():
    return importlib.import_module('english_v6_b28_natural_event_fit').contract()


def spec_binding_sha():
    return sha(profile.proof_binding_path(PANEL).read_bytes())


def protocol():
    p = profile.profile()
    need(SPEC.is_file(), 'INERT: actual complete B28 TRAIN and independently fresh SOURCE are unbound')
    doc = profile.document(SPEC.read_bytes())
    need(doc.get('scoring_allowed') is True, 'INERT: B28 scoring has no original evidence registration')
    bound = profile.binding(PANEL)
    training = bound['training_lock']['training']
    need(doc['schema'] == SCHEMA and doc['panel'] == PANEL and doc['source_module'] == SOURCE_MODULE
         and doc['source_schema'] == SOURCE_SCHEMA and doc['screen_schema'] == SCREEN_SCHEMA
         and doc['prepared_parent'] is None and doc['parent_authority'] == 'reviewed_workflow'
         and doc['pair_schema'] == PAIR_SCHEMA and doc['pair_stems'] == STEMS
         and doc['parent_pt_sha256'] == INITIAL_PT and doc['model_pt_sha256'] == training['model_pt_sha256']
         and doc['model_pt_sha256']['B0'] == B0_PT and doc['model_pt_sha256']['B9'] == B9_PT
         and doc['model_pt_sha256']['control'] == CONTROL_PT and doc['model_pt_sha256']['challenger'] == CANDIDATE_PT
         and CANDIDATE_PT not in (CONTROL_PT, INITIAL_PT, B27_PT)
         and set(doc['model_pt_sha256']) == set(ARMS)
         and all(valid_hash(value) for value in doc['model_pt_sha256'].values())
         and len(set(doc['model_pt_sha256'].values())) == 4
         and doc['model_comparison'] == MODEL_COMPARISON
         and doc['same_job_matched_control'] is doc['historical_control_retrained'] is False
         and doc['gains'] == list(GAINS) and doc['source_gate'] == GATE
         and type(doc['maximum_model_forwards']) is int and doc['maximum_model_forwards'] == MAX_FORWARDS
         and all(doc[key] is False for key in FLAGS)
         and doc['replication_opened'] is (PANEL == 'replication')
         and doc['original_primary_pass_required'] is (PANEL == 'replication')
         and set(doc['inputs']) == INPUT_KEYS and set(doc['native_inputs']) == set(PROVIDERS)
         and set(doc['upstream_producers']) == {'train', 'source'},
         'actual B28 candidate, immutable B24 control, four fixed roles and unchanged B14/source gates')
    need(doc['assessment_sha256'] == doc['training_assessment_sha256'] == EVALUATION_ASSESSMENT
         == p['assessment_sha256'] == sha(ASSESSMENT.read_bytes())
         and doc['training_design_sha256'] == TRAINING_DESIGN == sha(TRAIN_DESIGN.read_bytes())
         and doc['original_B24_training_assessment_sha256'] == ORIGINAL_B24_TRAINING_ASSESSMENT
         and doc['execution_profile_sha256'] == sha(profile.PROFILE.read_bytes())
         and doc['external_proof_binding_sha256'] == spec_binding_sha()
         and doc['original_training_report_sha256'] == bound['training_report_sha256']
         and valid_hash(doc['producer_manifest_sha256'])
         and doc['complete_original_B27_failure_required'] is True
         and doc['all_old_stops_and_reserves_preserved'] is True
         and type(doc['new_optimizer_updates']) is int and doc['new_optimizer_updates'] == 0
         and doc['evaluation_only'] is True
         and all(doc.get(key) == value for key, value in objective_contract().items()),
         'preoptimizer B28 freeze, independently recomputed objective and all earlier terminal outcomes')
    validate_source_policy(doc['source_policy'])
    pins = archive_pins(doc)
    for role, pin in pins.items():
        valid_artifact(pin, role=role)
    need(all(doc['inputs'][role] == pin for role, pin in BASELINES.items()), 'unchanged original B0 and B9')
    for role, key in (('models', 'pair'), ('producer', 'train_producer')):
        pin = training['artifacts'][role]
        need(pins[key] == dict(id=pin['id'], bytes=pin['bytes'], sha256=pin['sha256'],
            run_id=training['run_id'], run_head=training['run_head']), 'actual original new B28 TRAIN artifact ' + role)
    source_pins = [pins[key] for key in ('source', 'screen', 'source_producer', *('native_' + p for p in PROVIDERS))]
    need(len({pin['id'] for pin in pins.values()}) == 12
         and all(all(pin[key] == source_pins[0][key] for key in ('run_id', 'run_head')) for pin in source_pins)
         and source_pins[0]['run_id'] != training['run_id'], 'twelve distinct complete, explicitly registered artifacts')
    if PANEL == 'primary':
        need(doc['primary_quality_lock'] is None, 'PRIMARY cannot borrow an earlier PASS')
    else:
        need(doc['primary_quality_lock'] == bound['primary_quality_lock'], 'exact new same-candidate original PRIMARY lock')
    return doc


def source_check():
    profile.profile(False)
    bound = profile.binding(PANEL, False)
    spec = profile.document(SPEC.read_bytes()) if SPEC.is_file() else {'scoring_allowed': False}
    return dict(schema=SCHEMA, status='BOUND_NOT_EXECUTED' if spec.get('scoring_allowed') is True else 'INERT_SCAFFOLD',
        original_proof_status=bound['status'], model_calls=0, source_calls=0, pcm_reads=0,
        scientific_execution_authorized=False, promotion_allowed=False, product_released=False)


def validate_source_policy(policy):
    source = importlib.import_module(SOURCE_MODULE)
    need(policy == source.source_policy() and policy['schema'] == 'english-v6-b28-source-policy-v1'
         and policy['experiment_id'] == 'B28' and policy['orchestration_contract_sha256'] == ORCHESTRATION_SHA
         and policy['b28_execution_profile_sha256'] == sha(profile.PROFILE.read_bytes()),
         'unchanged full-original native-empty quarantine and five-valid source admission')
    return policy


def validate_pair_receipt(receipt, spec):
    hashes = spec['model_pt_sha256']
    contract = objective_contract()
    need(set(hashes) == set(ARMS) and hashes['B0'] == B0_PT and hashes['B9'] == B9_PT
         and hashes['control'] == CONTROL_PT and hashes['challenger'] not in (CONTROL_PT, INITIAL_PT, B27_PT)
         and len(set(hashes.values())) == 4 and all(valid_hash(value) for value in hashes.values())
         and receipt.get('schema') == PAIR_SCHEMA and receipt.get('status') == PAIR_STATUS
         and type(receipt.get('actual_optimizer_updates')) is int and receipt['actual_optimizer_updates'] == 192
         and type(receipt.get('actual_new_model_forwards')) is int and receipt['actual_new_model_forwards'] == 1847
         and receipt.get('natural_weight') == 0.16 and receipt.get('model_pt_sha256') == hashes
         and receipt.get('assessment_sha256') == spec['training_assessment_sha256'] == TRAINING_ASSESSMENT
         and receipt.get('design_sha256') == receipt.get('recipe_sha256') == spec['training_design_sha256'] == TRAINING_DESIGN
         and receipt.get('phase_initialization_pt_sha256') == INITIAL_PT
         and receipt.get('optimizer_initialization') == 'fresh_AdamW'
         and receipt.get('reused_ordinary_teacher_arrays') == 264
         and receipt.get('original_retention_conditions') == 120 and receipt.get('union_retention_conditions') == 122
         and receipt.get('original_B27_NO_PROMOTION_reauthenticated') is True
         and receipt.get('natural_objective_returns') == 384 and receipt.get('prefix_blank_reduction') == 'amax'
         and receipt.get('natural_objective_schema') == contract['objective_schema']
         and all(receipt.get(key) == value for key, value in contract.items())
         and all(receipt.get(key) is False for key in ('restored_optimizer_state', 'exact_optimizer_continuation',
             'historical_numerical_replay_claimed', 'same_job_matched_control', 'promotion_allowed',
             'B27_metrics_used_to_choose_parameters', 'source_execution_allowed', 'new_primary_allowed',
             'replication_allowed', 'candidate_engineering_allowed', 'product_released'))
         and all(type(receipt.get(key)) is int and receipt[key] == 0 for key in ('old_B23_optimizer_updates_reexecuted',
             'old_B24_optimizer_updates_reexecuted', 'new_control_optimizer_updates', 'new_control_forwards',
             'new_teacher_forwards', 'dev_cal_locked_forwards', 'new_tts_asr_calls')),
         'complete actual B28 TRAIN receipt with384 no-event terms, old science unchanged and no quality permission')
    need(receipt['control']['output_stem'] == STEMS['control']
         and receipt['challenger']['stem'] == STEMS['challenger']
         and receipt['challenger']['parent_pt_sha256'] == INITIAL_PT
         and receipt['challenger']['historical_control_pt_sha256'] == CONTROL_PT
         and receipt['challenger']['optimization_forwards'] == 1844
         and receipt['challenger']['checkpoint_reload_forwards'] == 2
         and receipt['challenger']['onnx_export_forwards'] == 1, 'one candidate and the original immutable B24 control')
    for role in ('control', 'challenger'):
        stem = Path(STEMS[role]).name
        need(set(receipt[role]['files']) == {stem + '.' + extension for extension in ('pt', 'npz', 'onnx')},
             'all complete original model formats')
        for pin in receipt[role]['files'].values():
            need(type(pin['bytes']) is int and pin['bytes'] > 10_000 and valid_hash(pin['sha256']), 'full original model member')
        need(receipt[role]['files'][stem + '.pt']['sha256'] == hashes[role], 'exact original model role ' + role)


def producer_contract(spec, prepared, root=ROOT):
    root = Path(root)
    raw = (root / 'automation' / PRODUCER.name).read_bytes()
    need(sha(raw) == spec['producer_manifest_sha256'], 'exact complete B28 scoring producer')
    doc = profile.document(raw)
    need(doc['schema'] == 'english-v6-b28-' + PANEL + '-producer-v1' and doc['frozen'] is True
         and set(doc['upstream_manifests']) == {'train', 'source'}, 'both entire original upstream manifests')
    required = static_closure(root / 'automation', PRODUCER_ROOTS)
    required |= {'automation/' + name for name in profile.profile(False)['fixed_base_sha256']}
    for upstream in doc['upstream_manifests'].values():
        required |= {name for name in upstream['files'] if name.endswith('.py')}
    need(set(doc[PANEL]['files']) == required, 'complete explicit static and dynamic Python closure')
    source_files = {name: pin['sha256'] for name, pin in doc['upstream_manifests']['source']['files'].items() if name.endswith('.py')}
    source_resources = {name: pin['sha256'] for name, pin in doc['upstream_manifests']['source']['files'].items()
                        if not name.endswith('.py') and name != SOURCE_WORKFLOW}
    need(doc['source']['files'] == source_files and doc['source']['resources'] == source_resources,
         'every complete actual original SOURCE producer byte remains unchanged')
    for role in (PANEL, 'source'):
        for name, digest in doc[role]['files'].items():
            need(valid_hash(digest) and sha((root / name).read_bytes()) == digest, 'frozen complete code ' + name)
    resources = {'automation/english_v6_b28_execution_profile.json',
        'automation/english_v6_b28_' + PANEL + '_proof_binding.json',
        'automation/english_v6_b28_panels/assessment.json', 'automation/english_v6_b28_natural_event_design.json',
        'automation/english_v6_b28_opening.json', 'automation/english_v6_b28_assessment_binding.json',
        'automation/english_v6_b14_panels/assessment.json'}
    for upstream in doc['upstream_manifests'].values():
        resources |= {name for name in upstream['files'] if not name.endswith('.py') and name not in (TRAIN_WORKFLOW, SOURCE_WORKFLOW)}
    if PANEL == 'replication':
        resources |= {'automation/english_v6_b28_primary_inputs.json', 'automation/english_v6_b28_primary_producer.json',
                      'automation/english_v6_b28_primary_proof_binding.json'}
        need(sha((root / 'automation/english_v6_b28_primary_inputs.json').read_bytes()) == spec['primary_quality_lock']['inputs_sha256']
             and sha((root / 'automation/english_v6_b28_primary_producer.json').read_bytes()) == spec['primary_quality_lock']['producer_manifest_sha256'],
             'entire originally passed PRIMARY registration retained for the replica')
    need(resources <= set(doc['resources']), 'every original dynamic resource and separately frozen B28 registration')
    for name, digest in doc['resources'].items():
        path = PurePosixPath(name)
        need(path.parts and not path.is_absolute() and '..' not in path.parts and '\\' not in name
             and valid_hash(digest) and sha((root / name).read_bytes()) == digest, 'complete original resource ' + name)
    need(all(doc['resources'].get(name) == digest for name, digest in source_resources.items()), 'same entire SOURCE resources')
    template = (root / 'automation' / WORKFLOW_TEMPLATE.name).read_bytes()
    need(doc[PANEL]['workflow_path'] == WORKFLOW and doc['source']['workflow_path'] == SOURCE_WORKFLOW
         and sha(template) == doc[PANEL]['workflow_template_sha256'] and template.count(PARENT_TOKEN.encode()) == 2
         and b'include-hidden-files: true' in template and valid_hash(doc['source']['workflow_sha256'])
         and valid_hash(prepared, 40), 'exact complete once-only workflow and reviewed parent')
    workflow = template.replace(PARENT_TOKEN.encode(), prepared.encode())
    need((root / WORKFLOW).read_bytes() == workflow, 'only two reviewed parent substitutions')
    return doc, workflow


def source_resource_contract(receipt, policy, scientific, metadata_hashes, collector_sha, collector_paths, captured_paths):
    counters = ('new_model_calls', 'new_source_calls', 'new_network_calls', 'new_audio_reads', 'optimizer_updates')
    need(type(receipt) is dict and set(receipt) == {'schema', 'status', 'panel', 'metadata_hashes',
         'collector_sha256', 'source_policy', 'scientific_contract_sha256', 'opened_resources',
         'complete_external_originals_still_required', *counters}
         and receipt['schema'] == 'english-v6-b28-captured-source-resource-preflight-v1'
         and receipt['status'] == 'PASS_FROZEN_REGISTRATION_ONLY' and receipt['panel'] == PANEL
         and receipt['metadata_hashes'] == metadata_hashes and receipt['collector_sha256'] == collector_sha
         and receipt['source_policy'] == policy
         and receipt['scientific_contract_sha256'] == sha(profile.canonical(scientific))
         and receipt['complete_external_originals_still_required'] is True
         and all(type(receipt[key]) is int and receipt[key] == 0 for key in counters),
         'original captured-only source preflight reopens exact B28 metadata and unchanged source science without execution')
    opened = receipt['opened_resources']
    need(type(opened) is list and all(type(name) is str for name in opened)
         and opened == sorted(set(opened))
         and set(metadata_hashes) | set(collector_paths) <= set(opened) <= set(captured_paths),
         'every frozen metadata and original collector dependency was reopened within the complete captured closure')
    for name in opened:
        path = PurePosixPath(name)
        need(name and not path.is_absolute() and path.as_posix() == name and '\\' not in name and '\0' not in name
             and all(part not in ('', '.', '..') for part in name.split('/')), 'normalized captured resource path')
    return receipt


def snapshot_contract(archive, role, meta, repository=None):
    need(role in ('train', 'source'), 'explicit B28 upstream role')
    doc = profile.document(archive.read('PRODUCER.json'))
    execution = doc.get('identity', doc)
    run, commit, tree = meta['run'], meta['commit'], meta['tree']
    branch, workflow = (TRAIN_BRANCH, TRAIN_WORKFLOW) if role == 'train' else (SOURCE_BRANCH, SOURCE_WORKFLOW)
    need(str(execution['run_id']) == str(run['id']) and execution['run_head'] == run['head_sha'] == commit['sha']
         and execution['run_attempt'] == 1 and execution['run_branch'].removeprefix('refs/heads/') == branch
         and doc['runtime_workflow'] == workflow and doc['git_tree'] == tree['sha'] == commit['commit']['tree']['sha']
         and tree['truncated'] is False and [v['sha'] for v in commit['parents']] == [execution['prepared_parent']]
         and run['path'] == workflow and [v['filename'] for v in commit['files']] == [workflow]
         and doc['source_code_reads_only'] is True
         and all(type(doc[key]) is int and doc[key] == 0 for key in ('new_model_forwards', 'optimizer_updates', 'new_source_calls')),
         'actual one-time pre-execution upstream Git identity and all complete original bytes')
    need({v.filename for v in archive.infolist() if not v.is_dir()} == {'PRODUCER.json'} | {'repository/' + n for n in doc['files']},
         'entire original producer archive including every hidden workflow and dynamic dependency')
    for name, pin in doc['files'].items():
        path = PurePosixPath(name)
        need(path.parts and not path.is_absolute() and '..' not in path.parts and '\\' not in name, 'safe producer member')
        body = archive.read('repository/' + name)
        need(len(body) == pin['bytes'] and sha(body) == pin['sha256']
             and git_blob(body) == pin['git_blob_sha1'] == tree_member(tree, name)
             and pin['mode'] in ('100644', '100755'), 'every original Git-bound member ' + name)
        if repository is not None and name != workflow:
            need((Path(repository) / name).read_bytes() == body, 'same upstream bytes retained by scoring')
    if role == 'train':
        need(doc['schema'] == 'english-v6-b28-complete-git-producer-v1'
             and doc['recipe_sha256'] == TRAINING_DESIGN and doc['original_b24_dependency_files'] == 338
             and all(doc.get(key) == value for key, value in objective_contract().items()),
             'complete original B28 no-event TRAIN with unchanged338 B24 dependencies')
        template_path, token = 'automation/english_v6_b28_training_workflow_template.yml', '__B28_REVIEWED_NATURAL_EVENT_PARENT__'
    else:
        need(doc['schema'] == 'english-v6-b28-complete-source-producer-v1'
             and doc['panel'] == PANEL and doc['external_proof_binding_sha256'] == spec_binding_sha(),
             'complete same-panel SOURCE and same complete external original proof registration')
        source = importlib.import_module(SOURCE_MODULE)
        metadata = importlib.import_module('english_v6_b28_source_metadata')
        source_resource_contract(doc['captured_resource_preflight'], source.source_policy(), source.code_contract(),
            metadata.validate_registration()['hashes'], metadata.COLLECTOR_SHA, metadata._collector().FROZEN, doc['files'])
        template_path, token = SOURCE_TEMPLATE, SOURCE_PARENT_TOKEN
    template = archive.read('repository/' + template_path)
    need(template.count(token.encode()) == 2 and template.replace(token.encode(), execution['prepared_parent'].encode())
         == archive.read('repository/' + workflow), 'exact workflow-only activation from the reviewed original template')
    return doc


def preflight(inputs, prepared):
    spec = protocol()
    identity = guard(spec, prepared)
    producer, workflow = producer_contract(spec, identity['prepared_parent'])
    capacity.producer_members(inputs, PANEL)
    fetched = profile.document((inputs / 'FETCH.json').read_bytes())
    verify_fetched(inputs, spec, identity, producer, workflow, fetched)
    quality = importlib.import_module('english_v6_b28_quality_readback')
    proof = profile.require_proof(PANEL)
    value = quality.source_admission(sys.modules[__name__], inputs, spec, identity, workflow, fetched, PANEL,
                                    proof.get('primary_quality'))
    common, decision, _source, _plan = helpers(spec)
    return spec, identity, common, decision, [(row, row['cohort']) for row in value['selected_rows']], proof['training_source'], value['admission']


def write(path, value):
    # The exact inherited scoring loop writes these objects before the first
    # forward and in its failure-safe finally block. Bind the added ancestry
    # claims to the already completed full TRAIN read, not to a supplied summary.
    if Path(path).name in ('STARTED.json', 'RESULT.json') and value.get('schema') == SCHEMA:
        proof = profile.require_proof(PANEL)
        need(proof['complete_original_B27_NO_PROMOTION_verified'] is True
             and proof['natural_no_event_terms_recomputed'] is True,
             'complete original B27 terminal and independent B28 objective precede scored output')
        value.update(B27_failed_primary_preserved=True, complete_original_B27_failure_verified=True,
            original_training_report_sha256=profile.binding(PANEL)['training_report_sha256'],
            external_proof_binding_sha256=spec_binding_sha(),
            natural_objective_schema=objective_contract()['objective_schema'],
            natural_objective_returns_verified=384, derived_B24_originals=5,
            derived_B24_is_new_qualification_source=False)
    return _original_write(path, value)


if __name__ == '__main__':
    main()
