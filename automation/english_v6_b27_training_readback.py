"""Independent original B27 TRAIN provenance and complete saved-array readback.

This reader uses stdlib bytes only. It does not load a checkpoint, run a model,
recompute a gradient or interpret TRAIN observations as a promotion decision.
The complete external Q1 and original TRAIN inputs remain mandatory evidence.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b27_natural_weight as train
import english_v6_b27_producer as producer

need, sha, document, canonical = train.need, train.sha, train.failure.document, train.failure.canonical
LOCK_SCHEMA = 'english-v6-b27-original-training-lock-v1'
REPORT_SCHEMA = 'english-v6-b27-original-training-readback-v1'
ARTIFACT_ROLES = ('models', 'producer', 'receipts', 'consumed')
OBSERVATIONS = ('TRAIN_FORWARD_INDEX.json', 'TRAIN_FORWARD_LOGITS.npz', 'OPTIMIZER_TRACE.json',
                'POSITIVE_RETENTION_TRACE.json', 'ORDINARY_MARGIN_TRACE.json', 'CHECKPOINT_RELOAD_LOGITS.npz')


def artifact_names(run):
    return dict(models='english-v6-b27-natural-weight-models-' + str(run),
        producer='english-v6-b27-train-producer-' + str(run),
        receipts='english-v6-b27-original-input-receipts-' + str(run), consumed=train.MARKER)


def validate_lock(lock):
    need(type(lock) is dict and set(lock) == {'schema', 'training', 'training_producer_manifest_sha256'}
         and lock['schema'] == LOCK_SCHEMA and train.failure.valid_hash(lock['training_producer_manifest_sha256']),
         'explicit complete actual B27 TRAIN lock')
    t = lock['training']
    fields = {'run_id', 'run_head', 'prepared_parent', 'run_branch', 'runtime_workflow',
              'artifacts', 'result_sha256', 'model_pt_sha256'}
    need(type(t) is dict and set(t) == fields and type(t['run_id']) is int and t['run_id'] > 0
         and all(train.failure.valid_hash(t[key], 40) for key in ('run_head', 'prepared_parent'))
         and t['run_head'] != t['prepared_parent'] and t['run_branch'] == train.BRANCH.removeprefix('refs/heads/')
         and t['runtime_workflow'] == train.WORKFLOW and set(t['artifacts']) == set(ARTIFACT_ROLES)
         and train.failure.valid_hash(t['result_sha256']), 'one actual workflow-only training child and four complete originals')
    names = artifact_names(t['run_id'])
    for role, pin in t['artifacts'].items():
        need(set(pin) == {'id', 'bytes', 'sha256', 'name'} and type(pin['id']) is int and pin['id'] > 0
             and type(pin['bytes']) is int and 0 < pin['bytes'] <= train.capacity.CORE_LIMIT
             and train.failure.valid_hash(pin['sha256']) and pin['name'] == names[role], 'exact named original TRAIN artifact ' + role)
    need(len({p['id'] for p in t['artifacts'].values()}) == 4, 'four distinct original TRAIN artifacts')
    hashes = t['model_pt_sha256']
    need(set(hashes) == {'B0', 'B9', 'control', 'challenger'} and all(train.failure.valid_hash(v) for v in hashes.values())
         and len(set(hashes.values())) == 4 and hashes['B0'] == train.b24.old.B0_PT
         and hashes['B9'] == train.b24.old.b10.b9.MODELS['B9_control']['pt'] and hashes['control'] == train.CONTROL_PT
         and hashes['challenger'] != train.INITIAL_PT, 'original B0/B9, immutable B24 control and actual distinct B27 challenger')
    return t


def validate_live(live, lock):
    t = validate_lock(lock)
    run, commit, tree, inventory = (live[key] for key in ('run', 'commit', 'tree', 'artifacts'))
    need(run['id'] == t['run_id'] and run['head_sha'] == t['run_head'] and run['run_attempt'] == 1
         and run['event'] == 'push' and run['head_branch'] == t['run_branch'] and run['path'] == train.WORKFLOW
         and run['status'] == 'completed' and run['conclusion'] == 'success'
         and all(run[key]['full_name'] == train.REPO and run[key]['private'] is False for key in ('repository', 'head_repository')),
         'actual completed first public TRAIN run')
    need(commit['sha'] == t['run_head'] and [v['sha'] for v in commit['parents']] == [t['prepared_parent']]
         and [v['filename'] for v in commit['files']] == [train.WORKFLOW]
         and tree['sha'] == commit['commit']['tree']['sha'] and tree['truncated'] is False,
         'original workflow-only activation and complete Git tree')
    need(inventory['total_count'] == len(inventory['artifacts']) == 4
         and {v['id'] for v in inventory['artifacts']} == {v['id'] for v in t['artifacts'].values()},
         'complete original four-artifact inventory')
    by_id = {v['id']: v for v in inventory['artifacts']}
    for pin in t['artifacts'].values():
        item = by_id[pin['id']]
        need(item['name'] == pin['name'] and item['expired'] is False and item['size_in_bytes'] == pin['bytes']
             and item['digest'] == 'sha256:' + pin['sha256'] and item['workflow_run']['id'] == t['run_id']
             and item['workflow_run']['head_sha'] == t['run_head'], 'original artifact API identity ' + pin['name'])
    return dict(run_id=str(t['run_id']), run_head=t['run_head'], prepared_parent=t['prepared_parent'],
                run_attempt=1, run_branch=train.BRANCH)


def _files(archive):
    return {item.filename for item in archive.infolist() if not item.is_dir()}


def sequence(raw):
    value = document(b'{"items":' + raw + b'}')['items']
    need(type(value) is list, 'complete original JSON array')
    return value


def verify_producer(archive, live, lock, identity):
    raw = archive.read('PRODUCER.json')
    need(sha(raw) == lock['training_producer_manifest_sha256'], 'exact original TRAIN producer manifest')
    manifest = document(raw)
    need(manifest['schema'] == producer.SCHEMA and manifest['identity'] == identity
         and manifest['git_tree'] == live['tree']['sha'] and manifest['runtime_workflow'] == train.WORKFLOW
         and manifest['original_b24_dependency_files'] == 338 and manifest['source_code_reads_only'] is True
         and manifest['include_hidden_files'] is True and manifest['recipe_sha256'] == train.DESIGN_SHA
         and manifest['assessment_binding_sha256'] == sha(train.BINDING.read_bytes())
         and manifest['opening_sha256'] == sha(train.OPENING.read_bytes())
         and all(manifest[key] == 0 for key in ('new_model_forwards', 'optimizer_updates', 'new_source_calls')),
         'complete producer captured before every scientific operation')
    names = producer.paths()
    need(set(manifest['files']) == names and _files(archive) == {'PRODUCER.json'} | {'repository/' + name for name in names},
         'all338 historical dependencies and all explicit new dependencies, with no supplement')
    tree = {item['path']: item for item in live['tree']['tree']}
    for name, pin in manifest['files'].items():
        body = archive.read('repository/' + name)
        expected = producer.workflow(identity['prepared_parent']) if name == train.WORKFLOW else (train.ROOT / name).read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()
        need(body == expected and len(body) == pin['bytes'] and sha(body) == pin['sha256']
             and blob == pin['git_blob_sha1'] == tree[name]['sha'] and tree[name]['type'] == 'blob'
             and tree[name]['mode'] == pin['mode'], 'every original producer byte and original Git blob ' + name)
    need(manifest['capacity'] == producer.capacity_receipt(manifest['files']), 'prospectively bounded complete producer')
    return manifest


def expected_forward_schedule(slots, source):
    need(len(slots) == 192 and len(source['positive_retention']) == 120 and len(source['retention_additions']) == 2,
         'exact original draw and all122 positive paths')
    groups = defaultdict(lambda: defaultdict(list))
    for row in source['positive_retention']:
        groups[row['group']][row['id']].append(dict(row, teacher='B9', newly_protected=False))
    for row in source['retention_additions']:
        groups[row['group']][row['id']].append(dict(row, newly_protected=True))
    need(len(groups) == 34 and sum(len(v) for v in groups.values()) == 43, 'all original positive groups and originals')
    samples = {}
    for slot in slots:
        for kind in ('ordinary', 'positive', 'natural'):
            row = slot[kind]
            need(type(row['samples']) is int and row['samples'] >= 512
                 and samples.get(row['id'], row['samples']) == row['samples'], 'same whole original length on every fixed draw')
            samples[row['id']] = row['samples']
    result = []
    for step, slot in enumerate(slots, 1):
        for kind in ('ordinary', 'positive', 'natural'):
            for gain in (1.0, 0.5):
                result.append(dict(step=step, kind='natural_ordinary' if kind == 'natural' else 'synthetic_' + kind,
                    id=slot[kind]['id'], gain=gain, prefix=slot['prefix'],
                    frames=(slot[kind]['samples'] + slot['prefix'] - 512) // 160 + 1))
        group = sorted(groups)[(step - 1) % len(groups)]
        for uid in sorted(groups[group]):
            for meta in groups[group][uid]:
                result.append(dict(step=step, kind='positive_retention', id=uid, group=group, gain=float(meta['gain']),
                    prefix=0, teacher=meta['teacher'], newly_protected=meta['newly_protected'],
                    frames=(samples[uid] - 512) // 160 + 1))
    need(len(result) == 1844, 'exact unchanged1844 optimization forward entries')
    return result


def saved_training_arrays(models, result, source):
    pins = result['saved_training_observations']
    need(set(pins) == set(OBSERVATIONS), 'all six original saved observation artifacts')
    data = {}
    for name, pin in pins.items():
        raw = models.read('challenger/' + name)
        need(set(pin) == {'bytes', 'sha256'} and len(raw) == pin['bytes'] and sha(raw) == pin['sha256'],
             'complete original saved training observation ' + name)
        data[name] = raw
    slots = sequence(models.read('COMMON_DRAW.json'))
    expected = expected_forward_schedule(slots, source)
    index = document(data['TRAIN_FORWARD_INDEX.json'])
    need(index['count'] == len(index['records']) == 1844 and index['pending_unreturned_forward'] is None
         and index['logits_are_raw'] is True and index['arrays_sha256'] == sha(data['TRAIN_FORWARD_LOGITS.npz']),
         'complete1844 original optimization returns; no unknown or missing entry')
    frames = 0
    with train.failure.opened(data['TRAIN_FORWARD_LOGITS.npz']) as arrays:
        need(set(arrays.namelist()) == {'forward_' + str(i).zfill(5) + '.npy' for i in range(1844)},
             'all and only the complete original optimization arrays')
        for index_number, (row, metadata) in enumerate(zip(index['records'], expected, strict=True)):
            key = 'forward_' + str(index_number).zfill(5)
            need(set(row) == set(metadata) | {'array', 'frames', 'sha256'}
                 and {name: row[name] for name in metadata} == metadata and row['array'] == key,
                 'same scheduled source, gain, prefix and positive path at every forward')
            body, _labels = train.failure.read_npy(arrays.read(key + '.npy'), (row['frames'], 4))
            need(sha(body) == row['sha256'], 'every finite original full-grid training return')
            frames += row['frames']
    optimizer = sequence(data['OPTIMIZER_TRACE.json'])
    positives = sequence(data['POSITIVE_RETENTION_TRACE.json'])
    ordinary = sequence(data['ORDINARY_MARGIN_TRACE.json'])
    need(len(optimizer) == 192 and len(positives) == 692 and len(ordinary) == 768, 'complete traces for fixed original schedule')
    for step, (row, slot) in enumerate(zip(optimizer, slots, strict=True), 1):
        need(row['step'] == step and row['prefix'] == slot['prefix'] and row['ordinary_id'] == slot['ordinary']['id']
             and row['positive_id'] == slot['positive']['id'] and row['natural_id'] == slot['natural']['id']
             and row['ceiling_coefficient'] == 0.08 and type(row['gradient_clipped']) is bool
             and row['gradient_clipped'] == (row['preclip_gradient_norm'] > 5)
             and all(type(row[key]) in (int, float) and math.isfinite(row[key]) for key in (
                 'loss', 'synthetic_ctc', 'gain_consistency', 'natural_ctc', 'positive_retention',
                 'ordinary_ceiling', 'preclip_gradient_norm')),
             'same source draw and finite complete loss/gradient trace for each original update')
        computed = sum(row[key] for key in ('synthetic_ctc', 'gain_consistency', 'natural_ctc', 'positive_retention'))
        computed += row['ceiling_coefficient'] * row['ordinary_ceiling']
        need(math.isclose(row['loss'], computed, rel_tol=2e-6, abs_tol=2e-6), 'saved scalar components match the original total')
    retained = [row for row in expected if row['kind'] == 'positive_retention']
    for row, expected_row in zip(positives, retained, strict=True):
        need(all(row[key] == expected_row[key] for key in ('step', 'id', 'group', 'teacher', 'newly_protected'))
             and float(row['gain']) == expected_row['gain'] and type(row['unweighted_loss']) in (int, float)
             and math.isfinite(row['unweighted_loss']) and row['unweighted_loss'] >= 0,
             'all692 original positive path losses with exact source identity')
    expected_ordinary = [row for row in expected if row['kind'] in ('synthetic_ordinary', 'natural_ordinary')]
    source_groups = {slot[kind]['id']: slot[kind]['source_group_id'] for slot in slots for kind in ('ordinary', 'natural')}
    for row, expected_row in zip(ordinary, expected_ordinary, strict=True):
        need(all(row[key] == expected_row[key] for key in ('step', 'id', 'gain', 'prefix', 'frames'))
             and row['group'] == source_groups[row['id']]
             and all(type(row[key]) in (int, float) and math.isfinite(row[key]) for key in ('mean', 'maximum', 'active_fraction'))
             and type(row['active_cells']) is int and type(row['cells']) is int
             and 0 <= row['active_cells'] <= row['cells'] and row['cells'] == row['frames'] * 3,
             'all768 complete original ordinary margin observations')
    with train.failure.opened(data['CHECKPOINT_RELOAD_LOGITS.npz']) as reload:
        need(set(reload.namelist()) == {'saved.npy', 'reloaded.npy'}, 'only the two already-budgeted checkpoint reload returns')
        shape = (index['records'][0]['frames'], 4)
        saved = train.failure.read_npy(reload.read('saved.npy'), shape)[0]
        rebuilt = train.failure.read_npy(reload.read('reloaded.npy'), shape)[0]
        need(saved == rebuilt, 'identical original saved/reloaded checkpoint outputs; no new forward')
    return dict(optimization_arrays=1844, checkpoint_reload_arrays=2, optimizer_trace_entries=192,
                positive_trace_entries=692, ordinary_trace_entries=768, verified_optimization_frames=frames,
                exported_onnx_forward_count_from_original_producer=1)


def _verify_worker(directory, lock, inputs, proof):
    directory, inputs = Path(directory), Path(inputs)
    train.design(); train.assessment_binding(); train.opening()
    live = document((directory / 'TRAIN_LIVE.json').read_bytes())
    identity = validate_live(live, lock)
    t = lock['training']
    with ExitStack() as stack, tempfile.TemporaryDirectory(prefix='b27-original-training-') as temporary:
        archives = {}
        for role, pin in t['artifacts'].items():
            raw = (directory / ('TRAIN_' + role + '.zip')).read_bytes()
            need(len(raw) == pin['bytes'] and sha(raw) == pin['sha256'], 'exact complete original B27 TRAIN artifact ' + role)
            archives[role] = stack.enter_context(train.failure.opened(raw))
        manifest = verify_producer(archives['producer'], live, lock, identity)
        models, receipts, consumed = archives['models'], archives['receipts'], archives['consumed']
        retained = Path(temporary)
        receipts.extractall(retained)
        expected_receipts = {'FETCH.json', 'SOURCE_PREFLIGHT.json', 'CONSUMPTION.json', 'EXTERNAL_INPUTS.json',
                            'original_b24/TRAIN_LIVE.json'}
        expected_receipts |= {'original_b24/TRAIN_' + role + '.zip' for role in ARTIFACT_ROLES}
        expected_receipts |= {'no-promotion/api/' + name + '.json' for name in ('run', 'commit', 'artifacts')}
        expected_receipts |= {'no-promotion/originals/' + name + '.zip' for name in ('result', 'producer', 'consumed')}
        need(_files(receipts) == expected_receipts, 'every retained original proof and external-input index; no missing or supplemented member')
        for name in expected_receipts:
            if name.startswith('no-promotion/'):
                need(receipts.read(name) == (inputs / name).read_bytes(), 'same complete original Q1 evidence inside and outside receipts')
        old_result = train.old_proof.verify(retained / 'original_b24')
        original_control = train.historical_control(retained / 'original_b24')
        external = train.external_inputs(inputs)
        need(receipts.read('EXTERNAL_INPUTS.json') == train.encoded(external), 'all complete Q1 and TRAIN external bytes remain bound')
        raw = models.read('RESULT.json')
        result, source = document(raw), document(models.read('SOURCE.json'))
        need(sha(raw) == t['result_sha256'] and result['schema'] == train.SCHEMA and result['status'] == train.RESULT_STATUS
             and result['identity'] == identity and result['recipe_sha256'] == result['design_sha256'] == train.DESIGN_SHA
             and result['model_pt_sha256'] == t['model_pt_sha256'] and result['natural_weight'] == 0.16
             and result['assessment_sha256'] == train.assessment_binding()['assessment']['sha256']
             and result['actual_optimizer_updates'] == 192 and result['actual_new_model_forwards'] == 1847
             and result['reused_ordinary_teacher_arrays'] == 264 and result['original_retention_conditions'] == 120
             and result['union_retention_conditions'] == 122 and result['prefix_blank_reduction'] == 'amax'
             and result['phase_initialization_pt_sha256'] == train.INITIAL_PT
             and result['optimizer_initialization'] == 'fresh_AdamW'
             and result['original_B24Q1_NO_PROMOTION_reauthenticated'] is True
             and all(result[key] is False for key in ('restored_optimizer_state', 'exact_optimizer_continuation',
                 'historical_numerical_replay_claimed', 'same_job_matched_control', 'promotion_allowed',
                 'physical_ssc305', 'Q1_metrics_used_to_choose_parameters'))
             and all(result[key] == 0 for key in ('old_B23_optimizer_updates_reexecuted', 'old_B24_optimizer_updates_reexecuted',
                 'new_control_optimizer_updates', 'new_control_forwards', 'new_teacher_forwards', 'dev_cal_locked_forwards', 'new_tts_asr_calls'))
             and result['field_far_per_hour'] is None and result['architecture'] == train.b24.baseline.architecture_contract(),
             'actual one bounded natural-weight phase with unchanged science and no quality/promotion claim')
        for field, name in (('source_sha256', 'SOURCE.json'), ('common_draw_sha256', 'COMMON_DRAW.json'),
                ('union_evidence_sha256', 'UNION_EVIDENCE.json'), ('old120_references_sha256', 'OLD120_POSITIVE_REFERENCES.json'),
                ('opening_sha256', 'OPENING.json'), ('assessment_binding_sha256', 'ASSESSMENT_BINDING.json')):
            need(sha(models.read(name)) == result[field], 'exact original result dependency ' + name)
        for name, path in (('DESIGN.json', train.DESIGN), ('OPENING.json', train.OPENING), ('ASSESSMENT_BINDING.json', train.BINDING)):
            need(models.read(name) == path.read_bytes(), 'unchanged before-optimizer ' + name)
        need(sha(models.read('COMMON_DRAW.json')) == train.design()['common_draw_sha256'], 'fixed original192 draws')
        need(_files(consumed) == {'CONSUMPTION.json'} and consumed.read('CONSUMPTION.json') == receipts.read('CONSUMPTION.json')
             and receipts.read('SOURCE_PREFLIGHT.json') == models.read('SOURCE.json'), 'one original durable budget and same complete before-fit source')
        marker, fetched = document(consumed.read('CONSUMPTION.json')), document(receipts.read('FETCH.json'))
        need(marker['schema'] == train.SCHEMA and marker['status'] == 'TRAINING_BUDGET_CONSUMED_UNKNOWN'
             and marker['identity'] == fetched['identity'] == identity and marker['budget'] == train.design()['budget']
             and marker['promotion_allowed'] is False and marker['source_preflight_sha256'] == result['source_sha256']
             and all(marker[key] == fetched[key] == result[key] for key in ('recipe_sha256', 'assessment_binding_sha256', 'opening_sha256'))
             and fetched['previous_failure'] == source['original_B24Q1_failed_primary'] == proof
             and fetched['external_inputs_sha256'] == sha(receipts.read('EXTERNAL_INPUTS.json'))
             and source['complete_external_inputs'] == external and source['original_B24_train'] == old_result
             and source['historical_control'] == original_control,
             'same complete actual qualification proof, original input bytes and durable one-shot consumption')
        need(source['receipt_capacity'] == fetched['receipt_capacity'] == train.receipt_capacity(retained)
             and sum(len(receipts.read(name)) for name in
                 ('FETCH.json', 'SOURCE_PREFLIGHT.json', 'CONSUMPTION.json', 'EXTERNAL_INPUTS.json')) <= train.RECEIPT_RESERVE,
             'complete retained originals and all new receipts fit their before-budget byte envelope')
        history = train.source_history(inputs)
        need(source['original_B24Q1_source_exclusion'] == fetched['original_B24Q1_source_exclusion'] == history
             and source['prospective_assessment_sha256'] == result['assessment_sha256']
             and source['schema'] == 'english-v6-b27-authenticated-train-source-v1'
             and source['initialization_pt_sha256'] == train.INITIAL_PT and source['natural_weight'] == 0.16
             and source['all_original_train_sources_and_teachers_unchanged'] is True
             and source['Q1_metrics_used_to_choose_parameters'] is False
             and all(source[key] == 0 for key in ('dev_rows_used_for_training', 'new_model_forwards', 'new_source_calls', 'new_dev_model_forwards')),
             'new assessment excludes full Q1 history and contributes no training row or model call')
        with train.original_models(retained / 'original_b24') as (old_models, _old):
            original_source = document(old_models.read('SOURCE.json'))
            for key in ('base_train', 'natural_train', 'train_originals', 'positive_retention', 'retention_additions',
                        'union', 'input_archives', 'registered_positive_retention_forwards', 'registered_base_ctc_forwards'):
                need(source[key] == original_source[key], 'unchanged entire TRAIN identity and protection set ' + key)
            need(models.read('OLD120_POSITIVE_REFERENCES.json') == old_models.read('OLD120_POSITIVE_REFERENCES.json')
                 and models.read('UNION_EVIDENCE.json') == old_models.read('UNION_EVIDENCE.json'),
                 'all original120 references and the same exact two additions')
        need(len(source['base_train']) == 79 and len(source['natural_train']) == 9 and len(source['train_originals']) == 88,
             'complete frozen TRAIN identity inventory available for all later overlap checks')
        expected_members = {'RESULT.json', 'STARTED.json', 'SOURCE.json', 'COMMON_DRAW.json', 'DESIGN.json',
                            'OPENING.json', 'ASSESSMENT_BINDING.json', 'UNION_EVIDENCE.json', 'OLD120_POSITIVE_REFERENCES.json'}
        expected_members |= {'challenger/' + name for name in OBSERVATIONS}
        for role, stem in (('control', 'mero43-b24-challenger192'), ('challenger', train.STEM)):
            need(set(result[role]['files']) == {stem + suffix for suffix in ('.pt', '.npz', '.onnx')}, 'all three exact model formats ' + role)
            for name, pin in result[role]['files'].items():
                member = role + '/models/' + name
                expected_members.add(member)
                body = models.read(member)
                need(len(body) == pin['bytes'] and sha(body) == pin['sha256'], 'complete original export ' + member)
            need(result[role]['files'][stem + '.pt']['sha256'] == t['model_pt_sha256'][role], 'exact fixed PT role ' + role)
        need(_files(models) == expected_members and result['control']['files'] == original_control['files']
             and result['control']['output_stem'] == 'control/models/mero43-b24-challenger192'
             and result['challenger']['stem'] == 'challenger/models/' + train.STEM
             and result['challenger']['parent_pt_sha256'] == train.INITIAL_PT
             and result['challenger']['historical_control_pt_sha256'] == train.CONTROL_PT
             and result['challenger']['natural_weight'] == 0.16
             and result['challenger']['optimization_forwards'] == 1844
             and result['challenger']['checkpoint_reload_forwards'] == 2
             and result['challenger']['onnx_export_forwards'] == 1,
             'only the complete final candidate and exact original B24 control; no partial checkpoint selection')
        started = document(models.read('STARTED.json'))
        need(started == dict(schema=train.SCHEMA, identity=identity, budget=train.design()['budget'], recipe_sha256=train.DESIGN_SHA),
             'honest before-fit STARTED receipt')
        observed = saved_training_arrays(models, result, source)
    return dict(schema=REPORT_SCHEMA, result=result, training_source=source,
        original_lock_sha256=sha(canonical(lock)), result_sha256=t['result_sha256'], source_sha256=result['source_sha256'],
        producer_manifest_sha256=lock['training_producer_manifest_sha256'], saved_training_observations=observed,
        all_original_producer_git_verified=True, all_original_inputs_verified=True, all_saved_training_arrays_verified=True,
        complete_original_Q1_NO_PROMOTION_verified=True, original_B24_control_unchanged=True,
        new_model_calls=0, new_source_calls=0, new_optimizer_updates=0, promotion_allowed=False, product_released=False)


def verify_training_original(directory, lock, inputs):
    validate_lock(lock)
    need(not {'numpy', 'torch', 'onnxruntime'} & set(sys.modules), 'independent TRAIN readback precedes scientific runtime imports')
    with tempfile.TemporaryDirectory(prefix='b27-train-lock-') as temporary:
        pin = Path(temporary) / 'lock.json'
        pin.write_bytes(canonical(lock))
        raw = subprocess.check_output([sys.executable, '-I', '-S', '-B', str(Path(__file__).resolve()), '--verify-worker',
            str(Path(directory).resolve()), str(pin), str(Path(inputs).resolve())])
    report = document(raw)
    need(report['schema'] == REPORT_SCHEMA and report['original_lock_sha256'] == sha(canonical(lock))
         and report['result_sha256'] == lock['training']['result_sha256']
         and all(report[key] is True for key in ('all_original_producer_git_verified', 'all_original_inputs_verified',
             'all_saved_training_arrays_verified', 'complete_original_Q1_NO_PROMOTION_verified', 'original_B24_control_unchanged'))
         and all(report[key] == 0 for key in ('new_model_calls', 'new_source_calls', 'new_optimizer_updates'))
         and report['promotion_allowed'] is report['product_released'] is False,
         'complete isolated original TRAIN evidence only; no independent quality claim')
    return report


def fetch_training_original(directory, lock, inputs):
    t = validate_lock(lock)
    directory = Path(directory)
    need(directory.is_absolute() and directory.parent.is_dir() and not directory.exists(), 'fresh immutable original TRAIN proof root')
    directory.mkdir()
    commit = document(train.gh('commits/' + t['run_head']))
    live = dict(run=document(train.gh('actions/runs/' + str(t['run_id']))), commit=commit,
        artifacts=document(train.gh('actions/runs/' + str(t['run_id']) + '/artifacts?per_page=100')),
        tree=document(train.gh('git/trees/' + commit['commit']['tree']['sha'] + '?recursive=1')))
    validate_live(live, lock)
    train.write(directory / 'TRAIN_LIVE.json', live)
    for role, pin in t['artifacts'].items():
        raw = train.gh('actions/artifacts/' + str(pin['id']) + '/zip')
        need(len(raw) == pin['bytes'] and sha(raw) == pin['sha256'], 'complete original fetched TRAIN artifact ' + role)
        (directory / ('TRAIN_' + role + '.zip')).write_bytes(raw)
    return verify_training_original(directory, lock, inputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-worker', action='store_true')
    parser.add_argument('directory', type=Path)
    parser.add_argument('lock', type=Path)
    parser.add_argument('inputs', type=Path)
    args = parser.parse_args()
    bound = document(args.lock.read_bytes())
    if args.verify_worker:
        # The only child is the exact frozen Q1 fail-only reader. Its original
        # worker enforces the recursive offline import and process boundaries.
        proof = train.previous_failure(args.inputs)
        with train.failure.offline_only():
            report = _verify_worker(args.directory, bound, args.inputs, proof)
    else:
        report = verify_training_original(args.directory, bound, args.inputs)
    sys.stdout.buffer.write(canonical(report))
