"""B27 four-frozen-model quality profile; inert until full originals are bound.

B14, source minima, gains, the forward ledger and full raw output capture retain
their original algorithms. This adapter binds actual B27 TRAIN and compares the
new candidate to immutable B24, B0 and B9. No old quality gate can open B27.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b27_profile as profile
original = profile.preserved('english_v6_b24q1_primary_eval')
profile.inherit(original, globals())
import english_v6_b27_capacity as capacity
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PANEL = 'primary'
SPEC = HERE / 'english_v6_b27_primary_inputs.json'
PRODUCER = HERE / 'english_v6_b27_primary_producer.json'
WORKFLOW_TEMPLATE = HERE / 'english_v6_b27_primary_workflow_template.yml'
TRAIN_DESIGN = HERE / 'english_v6_b27_natural_weight_design.json'
ASSESSMENT = HERE / 'english_v6_b27_panels/assessment.json'
TRAINING_DESIGN = profile.DESIGN_SHA
TRAINING_ASSESSMENT = EVALUATION_ASSESSMENT = profile.profile(False)['assessment_sha256']
ORIGINAL_B24_TRAINING_ASSESSMENT = original.TRAINING_ASSESSMENT
CONTROL_PT, INITIAL_PT = profile.CONTROL_PT, profile.INITIAL_PT
MODEL_COMPARISON = 'B27_new_challenger_against_immutable_historical_B24_challenger'
PAIR_SCHEMA = 'english-v6-b27-train-natural-weight-v1'
STEMS = {'control': 'control/models/mero43-b24-challenger192',
         'challenger': 'challenger/models/mero43-b27-challenger192'}
SCHEMA = 'english-v6-b27-fresh-primary-four-frozen-models-v1'
SOURCE_MODULE = 'english_v6_b27_source'
SOURCE_SCHEMA = 'english-v6-b27-fresh-combined-source-v1'
SCREEN_SCHEMA = 'english-v6-b27-120-original-quarantine-source-screen-v1'
BRANCH = 'refs/heads/research/en-v6-b27-primary-scoring-once-20261125'
WORKFLOW = '.github/workflows/english-v6-b27-primary-model-eval-once.yml'
SOURCE_BRANCH = 'research/en-v6-b27-primary-sources-once-20261125'
SOURCE_WORKFLOW = '.github/workflows/english-v6-b27-primary-sources-once.yml'
SOURCE_TEMPLATE = 'automation/english_v6_b27_source_workflow_template.yml'
SOURCE_PARENT_TOKEN = '__B27_REVIEWED_PRIMARY_SOURCE_PARENT__'
TRAIN_BRANCH = 'research/en-v6-b27-natural-weight-train-once-20261125'
TRAIN_WORKFLOW = '.github/workflows/english-v6-b27-natural-weight-train-once.yml'
PARENT_TOKEN = '__B27_REVIEWED_PRIMARY_SCORING_PARENT__'
MARKER = 'english-v6-b27-primary-consumed'
FLAGS = tuple(key for key in original.FLAGS if key != 'replication_opened')
PRODUCER_ROOTS = ('english_v6_b27_primary_eval', SOURCE_MODULE,
                  'english_v6_b27_quality_readback', 'english_v6_four_model_panel', 'english_v6_b14_decision')
# The candidate is never inferred from B24. Null remains inert before actual TRAIN.
CANDIDATE_PT = profile.document(SPEC.read_bytes()).get('model_pt_sha256', {}).get('challenger')


def archive_limit(role=None):
    need(role in {None, 'result', 'producer', 'consumed', *INPUT_KEYS,
         'train_producer', 'source_producer', *('native_' + p for p in PROVIDERS)}, 'explicit B27 original role')
    return profile.LIMITS['source'] if role == 'source' else profile.LIMITS['core']


def original_zip(path, pin, *, role=None):
    valid_artifact(pin, role=role)
    maximum = archive_limit(role)
    raw = path.read_bytes()
    need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"] and len(raw) <= maximum, "original complete input ZIP")
    z = zipfile.ZipFile(io.BytesIO(raw))
    try:
        infos = z.infolist()
        need(len(infos) <= 8192 and len(infos) == len({i.filename for i in infos})
             and sum(i.file_size for i in infos) <= maximum, "bounded unique archive members")
        for info in infos:
            name = PurePosixPath(info.filename)
            need(name.parts and not name.is_absolute() and ".." not in name.parts and "\\" not in info.filename
                 and not stat.S_ISLNK(info.external_attr >> 16), "safe original archive member")
        need(z.testzip() is None, "original archive CRC")
        if role == "source":
            manifest = json.loads(z.read("MANIFEST.json"))
            need(manifest["schema"] == SOURCE_SCHEMA and manifest["panel"] == PANEL
                 and manifest["assessment_sha256"] == EVALUATION_ASSESSMENT,
                 "composite envelope belongs only to the original registered B24Q1 SOURCE")
        return z
    except BaseException:
        z.close()
        raise

def protocol():
    p = profile.profile()
    doc = profile.document(SPEC.read_bytes())
    need(doc.get('scoring_allowed') is True, 'INERT: full original B27 TRAIN and independent fresh SOURCE required')
    binding = profile.binding(PANEL)
    training = binding['training_lock']['training']
    need(doc['schema'] == SCHEMA and doc['panel'] == PANEL and doc['source_module'] == SOURCE_MODULE
         and doc['source_schema'] == SOURCE_SCHEMA and doc['screen_schema'] == SCREEN_SCHEMA
         and doc['prepared_parent'] is None and doc['parent_authority'] == 'reviewed_workflow'
         and doc['pair_schema'] == PAIR_SCHEMA and doc['pair_stems'] == STEMS
         and doc['parent_pt_sha256'] == INITIAL_PT and doc['model_pt_sha256'] == training['model_pt_sha256']
         and doc['model_pt_sha256']['B0'] == B0_PT and doc['model_pt_sha256']['B9'] == B9_PT
         and doc['model_pt_sha256']['control'] == CONTROL_PT
         and doc['model_pt_sha256']['challenger'] == CANDIDATE_PT
         and CANDIDATE_PT not in (CONTROL_PT, INITIAL_PT)
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
         'new B27 candidate, original B24 comparison and exact unchanged source/B14 rules')
    need(doc['assessment_sha256'] == doc['training_assessment_sha256'] == EVALUATION_ASSESSMENT
         == p['assessment_sha256'] == sha(ASSESSMENT.read_bytes())
         and doc['training_design_sha256'] == TRAINING_DESIGN == sha(TRAIN_DESIGN.read_bytes())
         and doc['original_B24_training_assessment_sha256'] == ORIGINAL_B24_TRAINING_ASSESSMENT
         and doc['execution_profile_sha256'] == sha(profile.PROFILE.read_bytes())
         and doc['external_proof_binding_sha256'] == sha((HERE / ('english_v6_b27_' + PANEL + '_proof_binding.json')).read_bytes())
         and valid_hash(doc['producer_manifest_sha256'])
         and doc['complete_original_B24Q1_failure_required'] is True
         and doc['all_old_stops_and_reserves_preserved'] is True
         and doc['new_optimizer_updates'] == 0 and doc['evaluation_only'] is True,
         'preoptimizer B27 freeze remains distinct from immutable original B24 ancestor')
    validate_source_policy(doc['source_policy'])
    pins = archive_pins(doc)
    for role, pin in pins.items(): valid_artifact(pin, role=role)
    need(all(doc['inputs'][role] == pin for role, pin in BASELINES.items()), 'original B0/B9 unchanged')
    for role, key in (('models', 'pair'), ('producer', 'train_producer')):
        pin = training['artifacts'][role]
        need(pins[key] == dict(id=pin['id'], bytes=pin['bytes'], sha256=pin['sha256'],
                              run_id=training['run_id'], run_head=training['run_head']), 'actual original B27 TRAIN artifact ' + role)
    source_pins = [pins[key] for key in ('source', 'screen', 'source_producer', *('native_' + p for p in PROVIDERS))]
    need(len({pin['id'] for pin in pins.values()}) == 12
         and all(all(pin[k] == source_pins[0][k] for k in ('run_id', 'run_head')) for pin in source_pins)
         and source_pins[0]['run_id'] != training['run_id'], 'twelve distinct complete original artifacts')
    if PANEL == 'primary':
        need(doc['primary_quality_lock'] is None, 'PRIMARY cannot borrow any previous PASS')
    else:
        need(doc['primary_quality_lock'] == binding['primary_quality_lock'], 'new original same-candidate PRIMARY lock required')
    return doc


def source_check():
    profile.profile(); profile.binding(PANEL, False)
    doc = profile.document(SPEC.read_bytes())
    return dict(schema=SCHEMA, status='INERT_SCAFFOLD' if doc.get('scoring_allowed') is not True else 'BOUND_NOT_EXECUTED',
                model_calls=0, source_calls=0, pcm_reads=0, promotion_allowed=False)


def validate_source_policy(policy):
    source = importlib.import_module(SOURCE_MODULE)
    expected = source.source_policy()
    need(policy == expected and policy['schema'] == 'english-v6-b27-source-policy-v1'
         and policy['experiment_id'] == 'B27' and policy['orchestration_contract_sha256'] == ORCHESTRATION_SHA
         and policy['b27_execution_profile_sha256'] == sha(profile.PROFILE.read_bytes()),
         'same frozen native-empty quarantine, five-valid admission and explicit B27 profile')
    return policy


def validate_pair_receipt(receipt, spec):
    hashes = spec['model_pt_sha256']
    need(hashes['control'] == CONTROL_PT and hashes['challenger'] not in (CONTROL_PT, INITIAL_PT)
         and receipt.get('schema') == PAIR_SCHEMA
         and receipt.get('status') == 'TRAINED_NATURAL_WEIGHT_CANDIDATE_REQUIRES_NEW_INDEPENDENT_PRIMARY'
         and receipt.get('actual_optimizer_updates') == 192 and receipt.get('actual_new_model_forwards') == 1847
         and receipt.get('natural_weight') == 0.16 and receipt.get('model_pt_sha256') == hashes
         and receipt.get('assessment_sha256') == spec['training_assessment_sha256'] == TRAINING_ASSESSMENT
         and receipt.get('design_sha256') == receipt.get('recipe_sha256') == spec['training_design_sha256'] == TRAINING_DESIGN
         and receipt.get('phase_initialization_pt_sha256') == INITIAL_PT
         and receipt.get('optimizer_initialization') == 'fresh_AdamW'
         and receipt.get('reused_ordinary_teacher_arrays') == 264
         and receipt.get('original_retention_conditions') == 120 and receipt.get('union_retention_conditions') == 122
         and receipt.get('original_B24Q1_NO_PROMOTION_reauthenticated') is True
         and all(receipt.get(key) is False for key in ('restored_optimizer_state', 'exact_optimizer_continuation',
             'historical_numerical_replay_claimed', 'same_job_matched_control', 'promotion_allowed', 'Q1_metrics_used_to_choose_parameters'))
         and all(type(receipt.get(key)) is int and receipt[key] == 0 for key in ('old_B23_optimizer_updates_reexecuted',
             'old_B24_optimizer_updates_reexecuted', 'new_control_optimizer_updates', 'new_control_forwards',
             'new_teacher_forwards', 'dev_cal_locked_forwards', 'new_tts_asr_calls')),
         'complete actual B27 TRAIN receipt; B23 initialization is never the B24 quality control')
    need(receipt['control']['output_stem'] == STEMS['control']
         and receipt['challenger']['stem'] == STEMS['challenger']
         and receipt['challenger']['parent_pt_sha256'] == INITIAL_PT
         and receipt['challenger']['historical_control_pt_sha256'] == CONTROL_PT
         and receipt['challenger']['optimization_forwards'] == 1844
         and receipt['challenger']['checkpoint_reload_forwards'] == 2
         and receipt['challenger']['onnx_export_forwards'] == 1, 'same one candidate and exact original B24 control')
    for role in ('control', 'challenger'):
        stem = Path(STEMS[role]).name
        need(set(receipt[role]['files']) == {stem + '.' + extension for extension in ('pt', 'npz', 'onnx')},
             'all complete original model formats')
        for pin in receipt[role]['files'].values():
            need(type(pin['bytes']) is int and pin['bytes'] > 10_000 and valid_hash(pin['sha256']), 'full original model member')
        need(receipt[role]['files'][stem + '.pt']['sha256'] == hashes[role], 'exact original model role ' + role)


def validate_training_archive(pair, spec, meta):
    proof = profile.require_proof(PANEL)
    receipt = profile.document(pair.read('RESULT.json'))
    validate_pair_receipt(receipt, spec)
    need(receipt == proof['result'] and sha(pair.read('RESULT.json')) == proof['result_sha256']
         and sha(pair.read('SOURCE.json')) == proof['source_sha256'], 'same fully independently verified original TRAIN')
    train = profile.document(pair.read('SOURCE.json'))
    need(train == proof['training_source'], 'all original88 training identities and122 protected conditions')
    return train


def producer_contract(spec, prepared, root=ROOT):
    root = Path(root)
    raw = (root / 'automation' / PRODUCER.name).read_bytes()
    need(sha(raw) == spec['producer_manifest_sha256'], 'exact complete B27 scoring producer')
    doc = profile.document(raw)
    need(doc['schema'] == 'english-v6-b27-' + PANEL + '-producer-v1' and doc['frozen'] is True
         and set(doc['upstream_manifests']) == {'train', 'source'}, 'both full actual upstream manifests')
    required = static_closure(root / 'automation', PRODUCER_ROOTS)
    required |= {'automation/' + name for name in profile.profile(False)['fixed_base_sha256']}
    for upstream in doc['upstream_manifests'].values():
        required |= {name for name in upstream['files'] if name.endswith('.py')}
    need(set(doc[PANEL]['files']) == required, 'complete explicit static and dynamic original Python closure')
    source_files = {name: pin['sha256'] for name, pin in doc['upstream_manifests']['source']['files'].items() if name.endswith('.py')}
    source_resources = {name: pin['sha256'] for name, pin in doc['upstream_manifests']['source']['files'].items()
                        if not name.endswith('.py') and name != SOURCE_WORKFLOW}
    need(doc['source']['files'] == source_files and doc['source']['resources'] == source_resources,
         'complete actual original SOURCE producer unchanged')
    for role in (PANEL, 'source'):
        for name, digest in doc[role]['files'].items():
            need(valid_hash(digest) and sha((root / name).read_bytes()) == digest, 'frozen full producer code ' + name)
    expected_resources = {'automation/english_v6_b27_execution_profile.json',
        'automation/english_v6_b27_' + PANEL + '_proof_binding.json',
        'automation/english_v6_b27_panels/assessment.json', 'automation/english_v6_b27_natural_weight_design.json',
        'automation/english_v6_b14_panels/assessment.json'}
    for upstream in doc['upstream_manifests'].values():
        expected_resources |= {name for name in upstream['files'] if not name.endswith('.py') and name not in (TRAIN_WORKFLOW, SOURCE_WORKFLOW)}
    if PANEL == 'replication':
        expected_resources |= {'automation/english_v6_b27_primary_inputs.json', 'automation/english_v6_b27_primary_producer.json'}
        need(sha((root / 'automation/english_v6_b27_primary_inputs.json').read_bytes()) == spec['primary_quality_lock']['inputs_sha256']
             and sha((root / 'automation/english_v6_b27_primary_producer.json').read_bytes()) == spec['primary_quality_lock']['producer_manifest_sha256'],
             'complete originally passed PRIMARY registration needed by replica inheritance')
    need(expected_resources <= set(doc['resources']), 'every original dynamic resource and new frozen registration')
    for name, digest in doc['resources'].items():
        path = PurePosixPath(name)
        need(path.parts and not path.is_absolute() and '..' not in path.parts and '\\' not in name
             and valid_hash(digest) and sha((root / name).read_bytes()) == digest, 'complete original resource ' + name)
    need(all(doc['resources'].get(name) == digest for name, digest in source_resources.items()), 'same entire SOURCE resources')
    template = (root / 'automation' / WORKFLOW_TEMPLATE.name).read_bytes()
    need(doc[PANEL]['workflow_path'] == WORKFLOW and doc['source']['workflow_path'] == SOURCE_WORKFLOW
         and sha(template) == doc[PANEL]['workflow_template_sha256'] and template.count(PARENT_TOKEN.encode()) == 2
         and b'include-hidden-files: true' in template and valid_hash(doc['source']['workflow_sha256'])
         and valid_hash(prepared, 40), 'exact hidden-file-preserving once workflows')
    workflow = template.replace(PARENT_TOKEN.encode(), prepared.encode())
    need((root / WORKFLOW).read_bytes() == workflow, 'only two reviewed parent substitutions')
    return doc, workflow


def producer_files(spec, producer, workflow, role):
    if role == PANEL:
        extras = {**producer['resources'], WORKFLOW: sha(workflow),
            'automation/' + SPEC.name: sha(SPEC.read_bytes()), 'automation/' + PRODUCER.name: spec['producer_manifest_sha256'],
            'automation/' + WORKFLOW_TEMPLATE.name: producer[PANEL]['workflow_template_sha256']}
    else:
        need(role == 'source', 'explicit B27 producer role')
        extras = {**producer['source']['resources'], SOURCE_WORKFLOW: producer['source']['workflow_sha256']}
    return {**producer[role]['files'], **extras}


def snapshot_contract(archive, role, meta, repository=None):
    need(role in ('train', 'source'), 'explicit B27 upstream role')
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
         and all(doc[k] == 0 for k in ('new_model_forwards', 'optimizer_updates', 'new_source_calls')),
         'actual one-time pre-execution upstream producer')
    need({v.filename for v in archive.infolist() if not v.is_dir()} == {'PRODUCER.json'} | {'repository/' + n for n in doc['files']},
         'entire original producer archive, including hidden workflow and every dynamic dependency')
    for name, pin in doc['files'].items():
        path = PurePosixPath(name)
        need(path.parts and not path.is_absolute() and '..' not in path.parts and '\\' not in name, 'safe producer member')
        body = archive.read('repository/' + name)
        need(len(body) == pin['bytes'] and sha(body) == pin['sha256']
             and git_blob(body) == pin['git_blob_sha1'] == tree_member(tree, name)
             and pin['mode'] in ('100644', '100755'), 'every original Git-bound producer member ' + name)
        if repository is not None and name != workflow:
            need((Path(repository) / name).read_bytes() == body, 'same upstream bytes retained by scoring')
    if role == 'train':
        need(doc['schema'] == 'english-v6-b27-complete-git-producer-v1'
             and doc['recipe_sha256'] == TRAINING_DESIGN and doc['original_b24_dependency_files'] == 338,
             'complete B27 TRAIN with unchanged original338 B24 dependencies')
        template_path, token = 'automation/english_v6_b27_training_workflow_template.yml', '__B27_REVIEWED_NATURAL_WEIGHT_PARENT__'
    else:
        need(doc['schema'] == 'english-v6-b27-complete-source-producer-v1'
             and doc['panel'] == PANEL and doc['external_proof_binding_sha256'] == spec_binding_sha(),
             'complete same-panel B27 SOURCE producer and exact original-proof binding')
        template_path, token = SOURCE_TEMPLATE, SOURCE_PARENT_TOKEN
    template = archive.read('repository/' + template_path)
    need(template.count(token.encode()) == 2 and template.replace(token.encode(), execution['prepared_parent'].encode())
         == archive.read('repository/' + workflow), 'exact workflow-only activation from reviewed template')
    return doc


def spec_binding_sha():
    return sha((HERE / ('english_v6_b27_' + PANEL + '_proof_binding.json')).read_bytes())


def preflight(inputs, prepared):
    spec = protocol()
    identity = guard(spec, prepared)
    producer, workflow = producer_contract(spec, identity['prepared_parent'])
    capacity.producer_members(inputs, PANEL)
    fetched = profile.document((inputs / 'FETCH.json').read_bytes())
    verify_fetched(inputs, spec, identity, producer, workflow, fetched)
    import english_v6_b27_quality_readback as quality
    proof = profile.require_proof(PANEL)
    evidence = quality.source_admission(sys.modules[__name__], inputs, spec, identity, workflow, fetched, PANEL,
        proof.get('primary_quality'))
    train = proof['training_source']
    common, decision, source, plan = helpers(spec)
    chosen = [(row, row['cohort']) for row in evidence['selected_rows']]
    return spec, identity, common, decision, chosen, train, evidence['admission']


def verify_fetched(inputs, spec, identity, producer, workflow, fetched):
    need(fetched['schema'] == SCHEMA and fetched['identity'] == identity
         and fetched['inputs_sha256'] == sha(SPEC.read_bytes())
         and fetched['producer_manifest_sha256'] == spec['producer_manifest_sha256']
         and fetched['workflow_sha256'] == sha(workflow)
         and fetched['external_proof_binding_sha256'] == spec_binding_sha()
         and fetched['prior_consumed_markers']['total_count'] == 0, 'same original acquisition and unconsumed budget')
    validate_execution_metadata(identity, fetched[PANEL + '_run'], fetched[PANEL + '_commit'])
    runs = fetched['original_workflow_run_inventory']
    need(runs['total_count'] == len(runs['workflow_runs']) == 1 and runs['workflow_runs'][0]['id'] == int(identity['run_id']),
         'only the actual first scoring run')
    pins = archive_pins(spec)
    need(set(fetched['source_metadata']) == set(pins), 'every full original input API retained')
    for key, pin in pins.items(): validate_provenance(pin, fetched['source_metadata'][key], role=key)
    for role, commit in ((PANEL, fetched[PANEL + '_commit']), ('source', fetched['source_metadata']['source']['commit'])):
        tree = fetched['producer_trees'][role]
        need(tree['sha'] == commit['commit']['tree']['sha'] and tree['truncated'] is False, 'complete actual original Git tree')
        expected = producer_files(spec, producer, workflow, role)
        target = Path(inputs) / (role + '_producer')
        need(set(expected) == {p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()}, 'complete captured code/resource root')
        for name, digest in expected.items():
            body = (target / name).read_bytes()
            need(sha(body) == digest and git_blob(body) == tree_member(tree, name), 'every actual original Git blob ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'fetch', 'preflight', 'score'))
    parser.add_argument('--proof', type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prepared-parent')
    args = parser.parse_args()
    if args.mode == 'check':
        print(json.dumps(source_check(), sort_keys=True)); return
    protocol()
    need(args.proof is not None and args.inputs is not None, 'complete explicit originals before execution')
    with profile.proof_scope(args.proof, PANEL):
        if args.mode == 'fetch': fetch(args.inputs, args.prepared_parent)
        elif args.mode == 'preflight':
            checked = preflight(args.inputs, args.prepared_parent)
            write(args.inputs / 'CONSUMPTION.json', dict(identity=checked[1], marker_name=MARKER,
                  admission_sha256=sha(profile.canonical(checked[-1]))))
            print(json.dumps(checked[-1], sort_keys=True))
        else:
            need(args.output is not None, 'fresh full original output path')
            score(args.inputs, args.output, args.prepared_parent)


def fetch(inputs, prepared):
    spec = protocol()
    identity = guard(spec, prepared)
    producer, workflow = producer_contract(spec, identity["prepared_parent"])
    need(inputs.is_absolute() and inputs.parent.is_dir() and not inputs.exists(), "fresh immutable evidence directory")
    prior = json.loads(gh("actions/artifacts?name=" + MARKER + "&per_page=1"))
    need(prior["total_count"] == 0, "consumed B24Q1 primary cannot replay")
    run_inventory = unused_workflow_run(identity)
    run, commit = json.loads(gh("actions/runs/" + identity["run_id"])), json.loads(gh("commits/" + identity["run_head"]))
    validate_execution_metadata(identity, run, commit)
    inputs.mkdir()
    metadata = {}
    for key, pin in archive_pins(spec).items():
        metadata[key] = {"run": json.loads(gh("actions/runs/" + str(pin["run_id"]))),
                         "artifact": json.loads(gh("actions/artifacts/" + str(pin["id"]))),
                         "commit": json.loads(gh("commits/" + pin["run_head"]))}
        validate_provenance(pin, metadata[key], role=key)
        raw = gh("actions/artifacts/" + str(pin["id"]) + "/zip")
        need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "whole original input " + key)
        (inputs / (key + ".zip")).write_bytes(raw)
    trees = {PANEL: git_tree(commit), "source": git_tree(metadata["source"]["commit"]),
             "train": git_tree(metadata["pair"]["commit"])}
    with original_zip(inputs / "source_producer.zip", spec["upstream_producers"]["source"]) as source_archive:
        for role in (PANEL, "source"):
            for path, expected in producer_files(spec, producer, workflow, role).items():
                # The checkout and the original source archive already contain full bytes;
                # verify their actual Git blobs instead of thousands of network blob calls.
                raw = (ROOT / path).read_bytes() if role == PANEL else source_archive.read("repository/" + path)
                need(sha(raw) == expected and git_blob(raw) == tree_member(trees[role], path), "actual complete Git-bound producer member " + path)
                target = inputs / (role + "_producer") / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
    write(inputs / "FETCH.json", dict(schema=SCHEMA, identity=identity, **{PANEL + '_run': run, PANEL + '_commit': commit},
        inputs_sha256=sha(SPEC.read_bytes()), producer_manifest_sha256=spec["producer_manifest_sha256"], workflow_sha256=sha(workflow),
        external_proof_binding_sha256=spec_binding_sha(), source_metadata=metadata, producer_trees=trees, prior_consumed_markers=prior, original_workflow_run_inventory=run_inventory))
    capacity.producer_members(inputs, PANEL)

def source_audit(screen, rows, originals, policy, screen_raw):
    """Retain full120 exposure, fixed N and every planned slot before KWS.

    The caller must first replay the captured verify_screen_contract over every
    original raw payload. These checks independently forbid shrinking N, hiding
    quarantine or selecting a favorable subset of the qualified originals.
    """
    validate_source_policy(policy)
    count = len(rows)
    original_ids = {r["id"]: r for r in originals}
    need(len(originals) == len({r["id"] for r in originals}) == len({r["pcm_sha256"] for r in originals}) == 120
         and 96 <= count <= 120 and len({r["id"] for r in rows}) == count
         and all(r["id"] in original_ids and all(r[k] == original_ids[r["id"]][k]
                 for k in ("cohort", "pcm_sha256", "wav_sha256", "samples", "requested_class")) for r in rows)
         and screen["selected_original_slots"] == 120 and screen["source_rows"] == count
         and screen["original_quality_rejections"] == 120 - count and screen["source_policy"] == policy,
         "all120 originals and unchanged pre-observer N retained with explicit policy")
    ids = screen["quarantined_ids"]
    need(type(ids) is list and ids == sorted(set(ids)) and set(ids) <= {r["id"] for r in rows}
         and type(screen["quarantine"]) is dict and set(screen["quarantine"]) == set(ids)
         and all(type(v) is dict and bool(v) and set(v) <= set(PROVIDERS) for v in screen["quarantine"].values()),
         "one immutable eligible original per quarantine, no missing or duplicate ID")
    selected = exact_admitted_join(rows, screen["rows"])
    quarantined = [r for r in screen["rows"] if r["id"] in set(ids)]
    need(all(r["observed_class"] is None and r["tier"] == "unknown" and r["reason"] == QUARANTINE_REASON for r in quarantined)
         and {r["id"] for r in screen["rows"] if r.get("reason") == QUARANTINE_REASON} == set(ids)
         and not set(ids) & {r["id"] for r in selected}, "all five votes and every KWS call excluded for each quarantined original")
    partition = dict(pre_observer_rejected=120-count, quarantined_invalid_empty=len(ids),
        five_valid_unqualified=count-len(ids)-len(selected), qualified=len(selected))
    need(screen["partition"] == partition and sum(partition.values()) == 120
         and all(type(v) is int and v >= 0 for v in partition.values()), "complete original120 partition; no subset accuracy imputation")
    accounting = screen["native_slot_accounting"]
    fields = {"planned", "attempted", "returned", "valid", "invalid_empty", "fatal", "unknown"}
    need(set(accounting) == {"per_provider", "total", "maximum_per_panel"} and accounting["maximum_per_panel"] == 600
         and set(accounting["per_provider"]) == set(PROVIDERS), "all five original predeclared native slot populations")
    for key in ("coverage", "completed_coverage", "invalid_empty_coverage"):
        need(set(screen[key]) == set(PROVIDERS) and all(type(v) is int and v >= 0 for v in screen[key].values()),
             "complete integer original provider coverage " + key)
    for provider in PROVIDERS:
        value = accounting["per_provider"][provider]
        need(set(value) == fields and all(type(v) is int and v >= 0 for v in value.values())
             and value["planned"] == value["attempted"] == value["returned"] == screen["completed_coverage"][provider] == count
             and value["valid"] == screen["coverage"][provider] and value["invalid_empty"] == screen["invalid_empty_coverage"][provider]
             and value["invalid_empty"] == sum(provider in item for item in screen["quarantine"].values())
             and value["valid"] + value["invalid_empty"] == count and value["fatal"] == value["unknown"] == 0,
             "every planned slot returned exactly once; valid and proved-empty remain distinct")
    need(accounting["total"] == {k: sum(v[k] for v in accounting["per_provider"].values()) for k in fields}
         and accounting["total"]["planned"] == 5 * count <= 600,
         "exact5*N total; no extra calls to fill an unused600 ceiling")
    fields = ("source_policy", "coverage", "completed_coverage", "invalid_empty_coverage", "quarantined_ids",
              "quarantine", "native_slot_accounting", "partition", "attrition")
    return dict(schema="english-v6-b27-full-source-and-quarantine-audit-v1", screen_sha256=sha(screen_raw),
        original_source_rows=120, observer_eligible_rows=count, qualified_rows=len(selected),
        quarantined_kws_forwards=0, source_metrics_scope="all_original_exposure_and_fixed_observer_population",
        quality_metrics_scope="all_five_valid_source_qualified_originals_only",
        **{key: screen[key] for key in fields})

def snapshots(directory, spec, torch, np, common):
    """Authenticate original B0/B9 and B24Q1 challenger and historical-control checkpoints without any forward."""
    models = {}
    with original_zip(directory / "B0.zip", spec["inputs"]["B0"]) as archive, tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp)
        for name in ["RESULT.json"] + ["models/mero-fsmn-ctc-expanded256." + e for e in ("pt", "npz", "onnx")]:
            dest = target / name
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(archive.read(name))
        model, mel, mean, scale, hashes = common.b0.model(torch, np, target, "B0")
        common.need(hashes["pt"] == B0_PT, "installed B0 identity")
        models["B0"] = (model, mel, mean, scale, hashes["pt"])
    with original_zip(directory / "B9.zip", spec["inputs"]["B9"]):
        prior, _ = common.b9.verify_b9_archive(directory / "B9.zip", torch, np)
    models["B9"] = (*prior["B9_control"], common.b9.MODELS["B9_control"]["pt"])
    common.need(all(np.array_equal(a, b) for a, b in zip(models["B0"][1:4], models["B9"][1:4])),
                "B0/B9 use identical frozen acoustic frontend and normalization")
    with original_zip(directory / "pair.zip", spec["inputs"]["pair"]) as archive:
        receipt = json.loads(archive.read("RESULT.json"))
        validate_pair_receipt(receipt, spec)
        sources = json.loads(archive.read("SOURCE.json"))
        for arm in ("control", "challenger"):
            members = {}
            for ext in ("pt", "npz", "onnx"):
                filename = STEMS[arm] + "." + ext
                raw = archive.read(filename)
                expected = receipt[arm]["files"][Path(filename).name]
                common.need(common.sha(raw) == expected["sha256"] and len(raw) == expected["bytes"],
                            "complete trained B24Q1 artifact member " + filename)
                members[ext] = raw
            weights = torch.load(io.BytesIO(members["pt"]), map_location="cpu", weights_only=True)
            net = common.b9.v5.build_model(torch)
            net.load_state_dict(weights, strict=True)
            net.eval()
            common.need(sum(p.numel() for p in net.parameters()) == 43876 and
                        all(bool(torch.isfinite(p).all()) for p in net.parameters()), "finite actual B24Q1 weights")
            with np.load(io.BytesIO(members["npz"]), allow_pickle=False) as npz:
                common.need(set(npz.files) == set(weights) | {"mel_filters", "train_mean", "train_scale"} and
                            all(np.array_equal(npz[k], t.cpu().numpy()) for k, t in weights.items()),
                            "B24Q1 PT and NPZ learned tensor identity")
                frontend = tuple(npz[k].copy() for k in ("mel_filters", "train_mean", "train_scale"))
            common.need(all(np.array_equal(a, b) for a, b in zip(frontend, models["B0"][1:4])),
                        "candidate retains B0 frozen acoustic frontend")
            models[arm] = (net, *frontend, common.sha(members["pt"]))
    common.need(set(models) == set(common.NAMES), "four explicitly authenticated frozen roles")
    return models, sources

class Ledger:
    """Every actual forward entry consumes one immutable key, even if it fails."""
    def __init__(self, directory):
        self.directory = directory
        self.counts = dict.fromkeys(ARMS, 0)
        self.current = None
        self.consumed = {}

    def entered(self, role):
        need(self.current is not None and self.current["role"] == role, "one explicitly bound actual model call")
        key = self.current["key"]
        need(key not in self.consumed, "successful or consumed-unknown case must never be replayed")
        need(sum(self.counts.values()) < MAX_FORWARDS, "hard ceiling before an additional model forward entry")
        self.counts[role] += 1
        self.consumed[key] = "consumed_unknown"
        write(self.directory / (key + ".json"), dict(**self.current, status="consumed_unknown"))

    def saved(self, key):
        need(self.consumed.get(key) == "consumed_unknown", "one returned array per actual forward")
        self.consumed[key] = "saved_array"

def score(inputs, output, prepared):
    spec = protocol()  # Before paths, helper imports and model initialization.
    checked = preflight(inputs, prepared)
    spec, identity, common, decision, chosen, train, admission = checked
    permit = json.loads((inputs / "CONSUMPTION.json").read_bytes())
    need(permit["identity"] == identity and permit["marker_name"] == MARKER
         and permit["admission_sha256"] == sha(json.dumps(admission, sort_keys=True, separators=(",", ":")).encode()),
         "one uploaded consumed marker must bind this exact complete preflight")
    unused_workflow_run(identity)
    remote = json.loads(gh("actions/artifacts?name=" + MARKER + "&per_page=100"))
    need(remote["total_count"] == 1 and len(remote["artifacts"]) == 1
         and remote["artifacts"][0]["workflow_run"]["id"] == int(identity["run_id"])
         and remote["artifacts"][0]["workflow_run"]["head_sha"] == identity["run_head"],
         "exact current-run consumed marker; any older or ambiguous consumption forbids replay")
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(), "new result directory only")
    output.mkdir()
    attempts = output / "attempts"; attempts.mkdir()
    raw_outputs = output / "raw_outputs"; raw_outputs.mkdir()
    ledger = Ledger(attempts)
    report = dict(schema=SCHEMA, **identity, assessment_sha256=spec["assessment_sha256"],
        training_assessment_sha256=spec["training_assessment_sha256"], training_design_sha256=spec["training_design_sha256"],
        registered_inputs_sha256=sha(SPEC.read_bytes()), implementation_sha256=sha(Path(__file__).read_bytes()),
        producer_manifest_sha256=spec["producer_manifest_sha256"], workflow_sha256=admission["workflow_sha256"],
        source_gate="SOURCE_GATE_PASS", source_coverage=admission["source_coverage"], source_shortfall={},
        source_audit=admission["source_audit"],
        panel_gate="STARTED", records=[], actual_forwards=0, saved_array_count=0, raw_output_count=0,
        optimizer_updates=0, model_pt_sha256=spec["model_pt_sha256"], installed_software_baseline="B0",
        panel=PANEL, replication_opened=PANEL == 'replication', research_replication_passed=False,
        replication_open_allowed=False, B24Q1_failed_primary_preserved=True, all_old_stops_and_reserves_preserved=True, B20_failed_primary_preserved=True, B21_failed_primary_preserved=True, B22_failed_primary_preserved=True, B23_failed_primary_preserved=True, B24_original_source_stop_preserved=True,
        same_job_matched_control=False, historical_control_retrained=False, model_comparison=MODEL_COMPARISON,
        original_natural_reused_without_reobservation=False, fresh_source_originals=True, ssc305_physical=False,
        B24R1_original_source_stop_preserved=True, original_whole_panel_failure_contract_unchanged=False,
        natural_field_far_per_hour=None, field_3_5m_frr=None, **dict.fromkeys(FLAGS, False))
    if PANEL == 'replication':
        report.update(primary_result_sha256=admission['primary_result_sha256'], primary_quality_lock_sha256=admission['primary_quality_lock_sha256'])
    write(output / "ADMISSION.json", admission)
    write(output / "STARTED.json", report)
    hooks, arrays = [], None
    try:
        import numpy as np
        import torch
        need(np.__version__ == "1.26.4" and torch.__version__ == "2.9.1+cpu", "unchanged pinned CPU scoring runtime")
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True); torch.backends.mkldnn.enabled = False
        models, loaded_train = snapshots(inputs, spec, torch, np, common)
        need(loaded_train == train and {k: v[-1] for k, v in models.items()} == spec["model_pt_sha256"],
             "same preflight TRAIN and all four original models/frontends")
        for role, values in models.items():
            def entered(_module, _args, role=role):
                ledger.entered(role)
            hooks.append(values[0].register_forward_pre_hook(entered))
        arrays = zipfile.ZipFile(output / "ALL_FRAME_LOGITS.npz", "x", compression=zipfile.ZIP_DEFLATED)
        with ExitStack() as stack:
            archive = stack.enter_context(original_zip(inputs / "source.zip", spec["inputs"]["source"], role="source"))
            for row, cohort in chosen:
                record = score_identity(row, cohort); record["gains"] = {}; report["records"].append(record)
                raw = archive.read("wav/" + row["id"] + ".wav")
                with wave.open(io.BytesIO(raw)) as audio:
                    need(audio.getparams()[:3] == (1, 2, 16000) and audio.getnframes() == row["samples"], "whole admitted PCM")
                    pcm = audio.readframes(audio.getnframes())
                need(sha(raw) == row["wav_sha256"] and sha(pcm) == row["pcm_sha256"], "same preflight original PCM")
                for gain in GAINS:
                    signal = common.b9.ref.gain_pcm(pcm, float(gain), np)
                    record["gains"][gain] = {}
                    for role in ARMS:
                        net, mel, mean, scale, _ = models[role]
                        feature = common.b9.v5.features(signal, row, torch, np, mel)
                        x = torch.from_numpy(((feature - mean) / scale).astype(np.float32)).unsqueeze(0)
                        key = row["id"] + "__" + gain + "__" + role
                        need(re.fullmatch(r"[A-Za-z0-9_.-]+", key), "safe immutable array key")
                        ledger.current = dict(key=key, role=role, source_id=row["id"], gain=gain)
                        with torch.inference_mode():
                            y = net(x)
                        raw_y = y.detach().cpu().numpy()
                        with (raw_outputs / (key + ".npy")).open("xb") as handle:
                            np.save(handle, raw_y, allow_pickle=False)
                            handle.flush(); os.fsync(handle.fileno())
                        report["raw_output_count"] += 1
                        need(raw_y.shape == (1, len(feature), 4) and raw_y.dtype == np.dtype("float32") and np.isfinite(raw_y).all(),
                             "complete actual finite FP32 frame output")
                        matrix = np.ascontiguousarray(raw_y[0], dtype="<f4")
                        encoded = io.BytesIO(); np.save(encoded, matrix, allow_pickle=False)
                        arrays.writestr(key + ".npy", encoded.getvalue())
                        arrays.fp.flush(); os.fsync(arrays.fp.fileno())
                        report["saved_array_count"] += 1
                        ledger.saved(key)
                        labels = matrix.argmax(-1).tolist()
                        record["gains"][gain][role] = dict(array=key, sha256=sha(matrix.tobytes()), frames=len(matrix),
                            **ctc_events(labels))
        validate_scored_join(report["records"], admission["admitted_rows"])
        need(sum(ledger.counts.values()) == report["saved_array_count"] == report["raw_output_count"] == 12 * len(chosen) <= MAX_FORWARDS
             and all(v == 3 * len(chosen) for v in ledger.counts.values()), "all and only the exact admitted original/gain/model calls completed")
        decision_result = decision.decide(report["records"])
        report.update(decision_result,
            replication_open_allowed=PANEL == 'primary' and decision_result['panel_gate'] == 'PASS',
            candidate_engineering_validation_allowed=PANEL == 'replication' and decision_result['panel_gate'] == 'PASS',
            research_replication_passed=PANEL == 'replication' and decision_result['panel_gate'] == 'PASS')
    except BaseException as exc:
        report.update(panel_gate="STOP", error_type=type(exc).__name__, error=str(exc)[:2000], replication_open_allowed=False, candidate_engineering_validation_allowed=False, research_replication_passed=False)
        raise
    finally:
        for hook in hooks:
            hook.remove()
        if arrays is not None:
            arrays.close()
            report["logits_sha256"] = sha((output / "ALL_FRAME_LOGITS.npz").read_bytes())
        report.update(actual_forwards=sum(ledger.counts.values()), model_forward_calls=ledger.counts,
                      consumed_unknown=[key for key, value in ledger.consumed.items() if value != "saved_array"],
                      replay_allowed=False)
        write(output / "RESULT.json", report)

if __name__ == '__main__':
    main()
