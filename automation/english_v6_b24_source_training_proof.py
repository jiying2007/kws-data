"""Pure local verification of the original TRAIN dependency of B24 sources.

The original B24 TRAIN entry already reauthenticated all twelve old B23 input
archives and source admission. This layer verifies that actual immutable run,
its gate code, original receipts and every old returned KWS array. It does not
reopen the old audio/provider ZIPs or claim a second old-source admission.
"""
from __future__ import annotations
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile

import english_v6_b24_b23_failure as failure
import english_v6_b24_source_ancestry as lineage

need, sha = lineage.need, lineage.digest
ROOT = Path(__file__).resolve().parents[1]
SCOPE = dict(original_B23_complete_source_admission_authority='actual_original_B24_TRAIN_entry',
    old_B23_input_archives_reopened_by_source=False,
    old_B23_original_result_arrays_reverified_by_source=True,
    actual_B24_TRAIN_producer_and_receipts_verified=True)


def verify_previous_failure(receipts, opening, source):
    """Verify original result/producer/marker, using TRAIN-verified source labels.

Whole original metadata is separately frozen in B24 ancestry; it supplies the
unchanged full frame lengths. Admission labels remain explicitly attributable
to the actual TRAIN gate, not to another run of the five native observers.
"""
    need(opening['schema'] == 'english-v6-b24-after-original-b23-no-promotion-v1'
         and opening['condition'] == 'only_after_actual_complete_B23_primary_NO_PROMOTION' and opening['enabled'] is True
         and opening['baseline_challenger_pt_sha256'] == lineage.metadata.CONTROL_PT,
         'actual original B23 failure opening; no synthetic enable switch')
    lock, proof = opening['quality_lock'], source['previous_failure']
    failure.validate_lock(lock)
    adapter = opening['failure_adapter']
    need(adapter == dict(path='automation/english_v6_b24_b23_failure.py', sha256=sha(Path(failure.__file__).read_bytes()),
             base_reader_path='automation/english_v6_b23_quality_readback.py', base_reader_sha256=failure.BASE_READER_SHA)
         and opening['actual_primary_evaluator_sha256'] == lock['evaluator_sha256']
         and opening['actual_primary_producer_manifest_sha256'] == lock['producer_manifest_sha256'],
         'same independently frozen reader and actual original PRIMARY provenance authority')
    need(proof['schema'] == 'english-v6-b24-original-b23-no-promotion-v1'
         and proof['panel'] == 'primary' and proof['panel_gate'] == 'NO_PROMOTION'
         and proof['model_pt_sha256']['challenger'] == opening['baseline_challenger_pt_sha256']
         and proof['original_lock_sha256'] == sha(failure.canonical(lock))
         and proof['original_result_sha256'] == lock['original_result_sha256']
         and proof['failure_reader_sha256'] == adapter['sha256']
         and proof['base_reader_sha256'] == failure.BASE_READER_SHA
         and proof['all_original_arrays_verified'] is proof['original_source_admission_recomputed']
             is proof['original_producer_git_verified'] is True
         and all(proof[key] is False for key in ('authorization_granted', 'replication_open_allowed',
             'candidate_engineering_validation_allowed', 'promotion_allowed', 'product_released'))
         and all(proof[key] == 0 for key in ('new_model_calls', 'new_source_calls', 'new_observer_calls')),
         'complete actual TRAIN-entry failure proof, with every downstream permission closed')
    run = json.loads(receipts.read('no-promotion/api/run.json'))
    commit = json.loads(receipts.read('no-promotion/api/commit.json'))
    inventory = json.loads(receipts.read('no-promotion/api/artifacts.json'))
    raw = {name: receipts.read('no-promotion/originals/' + name + '.zip')
           for name in ('result', 'producer', 'consumed')}
    for name, body in raw.items():
        need(len(body) == lock[name]['bytes'] and sha(body) == lock[name]['sha256'],
             'exact original saved B23 quality artifact ' + name)
    with ExitStack() as stack, tempfile.TemporaryDirectory(prefix='b24-train-gate-receipts-') as temporary:
        archives = {name: stack.enter_context(failure.opened(body)) for name, body in raw.items()}
        directory = Path(temporary)
        archives['producer'].extractall(directory)
        root = directory / 'primary_producer'
        driver = root / 'automation/english_v6_b23_primary_eval.py'
        spec_path = root / 'automation/english_v6_b23_primary_inputs.json'
        producer_path = root / 'automation/english_v6_b23_primary_producer.json'
        need(sha(driver.read_bytes()) == lock['evaluator_sha256']
             and sha(spec_path.read_bytes()) == lock['inputs_sha256']
             and sha(producer_path.read_bytes()) == lock['producer_manifest_sha256'],
             'actual original B23 evaluator, registration and producer identity')
        primary = failure.load(driver, 'b24_saved_b23_primary_eval')
        names = dict(result='english-v6-b23-historical-control-primary-logits-' + str(run['id']),
            producer='english-v6-b23-primary-producer-evidence-' + str(run['id']),
            consumed='english-v6-b23-primary-consumed')
        for name in raw:
            selected = [item for item in inventory['artifacts'] if item['id'] == lock[name]['id']]
            need(len(selected) == 1 and selected[0]['name'] == names[name], 'original named B23 artifact ' + name)
            primary.validate_provenance(lock[name], dict(run=run, commit=commit, artifact=selected[0]))
        spec = primary.protocol()
        fetched = json.loads(archives['producer'].read('FETCH.json'))
        identity = fetched['identity']
        failure.identity(identity)
        primary.validate_execution_metadata(identity, run, commit)
        need(identity == proof['original_identity'] and identity['run_id'] == str(lock['result']['run_id'])
             and identity['run_head'] == lock['result']['run_head'], 'same actual original failure execution')
        producer, workflow = primary.producer_contract(spec, identity['prepared_parent'], root=root)
        need(fetched['schema'] == primary.SCHEMA and fetched['inputs_sha256'] == lock['inputs_sha256']
             and fetched['producer_manifest_sha256'] == lock['producer_manifest_sha256']
             and fetched['workflow_sha256'] == sha(workflow) and fetched['prior_consumed_markers']['total_count'] == 0,
             'actual original acquisition and reviewed workflow before first failure scoring')
        original_commit = fetched['primary_commit']
        primary.validate_execution_metadata(identity, fetched['primary_run'], original_commit)
        primary.validate_execution_metadata(identity, run, original_commit)
        need(commit['sha'] == original_commit['sha']
             and [item['sha'] for item in commit['parents']] == [item['sha'] for item in original_commit['parents']]
             and commit['commit']['tree']['sha'] == original_commit['commit']['tree']['sha'],
             'independently saved original commit and captured acquisition agree')
        prior_runs = fetched['original_workflow_run_inventory']
        need(prior_runs['total_count'] == len(prior_runs['workflow_runs']) == 1
             and prior_runs['workflow_runs'][0]['id'] == int(identity['run_id']), 'one first original B23 scoring run')
        pins = primary.archive_pins(spec)
        need(set(fetched['source_metadata']) == set(pins), 'all twelve original input provenance receipts')
        for name, pin in pins.items():
            primary.validate_provenance(pin, fetched['source_metadata'][name])
        expected_files = {'FETCH.json', 'CONSUMPTION.json'}
        for role, source_commit in (('primary', original_commit), ('source', fetched['source_metadata']['source']['commit'])):
            tree = fetched['producer_trees'][role]
            need(tree['sha'] == source_commit['commit']['tree']['sha'] and tree['truncated'] is False,
                 'complete original captured Git tree ' + role)
            expected = primary.producer_files(spec, producer, workflow, role)
            target = directory / (role + '_producer')
            need(set(expected) == {path.relative_to(target).as_posix() for path in target.rglob('*') if path.is_file()},
                 'all and only complete original producer members ' + role)
            for name, digest in expected.items():
                body = (target / name).read_bytes()
                need(sha(body) == digest and primary.git_blob(body) == primary.tree_member(tree, name),
                     'every original producer byte matches its original Git blob ' + name)
                expected_files.add(role + '_producer/' + name)
        need({item.filename for item in archives['producer'].infolist() if not item.is_dir()} == expected_files,
             'original B23 producer ZIP has no missing or supplemented members')
        admission = proof['admission']
        consumed_raw = archives['producer'].read('CONSUMPTION.json')
        need({item.filename for item in archives['consumed'].infolist() if not item.is_dir()} == {'CONSUMPTION.json'}
             and archives['consumed'].read('CONSUMPTION.json') == consumed_raw
             and json.loads(consumed_raw) == dict(identity=identity, marker_name=primary.MARKER,
                 admission_sha256=sha(failure.canonical(admission))), 'original durable marker binds TRAIN-verified admission')
        members = {}
        pins = source['previous_source_metadata_members']
        need(set(pins) == {'ORIGINAL_B23_SOURCE_MANIFEST.json', 'ORIGINAL_B23_SOURCE_HISTORY.json'},
             'both complete original historical metadata members retained in TRAIN receipts')
        for member, expected_sha in lineage.metadata.SOURCE_MEMBERS.items():
            name = 'ORIGINAL_B23_SOURCE_' + member
            body = receipts.read(name)
            need(pins[name] == dict(bytes=len(body), sha256=sha(body), original_member=member)
                 and sha(body) == expected_sha, 'exact original old source metadata bytes ' + member)
            members[member] = body
        public = lineage.metadata.derive_original_metadata(members, fetched['source_metadata']['source'], spec['inputs']['source'])
        for name, value in public.items():
            need(lineage.metadata.encoded(value) == (lineage.metadata.PANELS / name).read_bytes(),
                 'public exclusion whitelist reexports every exact original PCM/person ' + name)
        reexport = source['prospective_history_reexport']
        need(reexport['schema'] == 'english-v6-b24-original-source-metadata-reexport-v1'
             and reexport['status'] == 'VERIFIED_ORIGINAL_METADATA_REEXPORT'
             and reexport['source_artifact'] == spec['inputs']['source'] == lineage.metadata.SOURCE_PIN
             and reexport['original_metadata_sha256'] == lineage.metadata.SOURCE_MEMBERS
             and reexport['original_manifest_sha256'] == sha(members['MANIFEST.json'])
             and reexport['original_history_sha256'] == sha(members['HISTORY.json'])
             and reexport['original_rows'] == 120 and reexport['prior_panel_people_excluded'] == 72
             and reexport['whole_zip_sha256_verified'] is reexport['original_source_identity_verified']
                 is reexport['metadata_member_crc_verified'] is True
             and reexport['scientific_execution_authorized'] is reexport['quality_results_read'] is False
             and all(reexport[key] == 0 for key in ('audio_members_interpreted', 'observer_receipts_read',
                 'model_calls', 'new_source_calls')),
             'the full ZIP check belongs to actual TRAIN; this layer rechecks the original metadata only')
        originals = json.loads(members['MANIFEST.json'])['rows']
        need(proof['source_inventory'] == dict(original_count=len(originals),
             pcm_sha256=sorted({row['pcm_sha256'] for row in originals}),
             source_group_id=sorted({row['source_group_id'] for row in originals}),
             canonical_people=sorted({primary.person(row) for row in originals})),
             'TRAIN-verified old source population equals all120 original frozen metadata identities')
        by_id = {row['id']: row for row in originals}
        selected = []
        for label in admission['admitted_rows']:
            original = by_id[label['id']]
            row = dict(original, cell_id=label['cell'], observed_class=2 if label['positive'] else 0, tier=label['tier'])
            need(primary.score_identity(row, row['cohort']) == label
                 and row['observed_class'] == row['requested_class'], 'unchanged full metadata and original TRAIN-verified label')
            selected.append(row)
        decision = failure.load(root / 'automation/english_v6_b14_decision.py', 'b24_saved_original_b14_decision')
        checked = failure.saved_failure_arrays(raw['result'], spec, 'primary', admission, selected, primary, decision)
        need(checked['original_result_sha256'] == proof['original_result_sha256']
             and checked['logits_sha256'] == proof['logits_sha256']
             and checked['result']['actual_forwards'] == proof['actual_forwards']
             and checked['result']['model_forward_calls'] == proof['model_forward_calls']
             and decision.decide(checked['result']['records']) == proof['decision']
             and proof['model_pt_sha256'] == spec['model_pt_sha256']
             and proof['pair_pin'] == spec['inputs']['pair'],
             'every original returned array and unchanged B14 independently reproduce the locked failure')
    return dict(old_B23_input_archives_reopened_by_source=False,
        old_B23_original_result_arrays_reverified_by_source=True,
        original_B23_result_sha256=proof['original_result_sha256'],
        actual_B24_TRAIN_verification_performed_by_this_helper=False)


def verify(directory):
    directory = Path(directory)
    live = json.loads((directory / 'TRAIN_LIVE.json').read_bytes())
    t = lineage.validate_training_live(live)
    identity = dict(run_id=str(t['run_id']), run_head=t['run_head'], run_attempt=1,
        prepared_parent=t['prepared_parent'], run_branch='refs/heads/' + t['run_branch'])
    with ExitStack() as stack:
        archives = {}
        for role, pin in t['artifacts'].items():
            raw = (directory / ('TRAIN_' + role + '.zip')).read_bytes()
            need(len(raw) == pin['bytes'] and sha(raw) == pin['sha256'], 'exact saved original B24 TRAIN ZIP ' + role)
            archives[role] = stack.enter_context(failure.opened(raw))
        producer = archives['producer']
        need(producer.read('PRODUCER.json') == lineage.TRAIN_PRODUCER.read_bytes(), 'entire actual TRAIN producer manifest')
        manifest = json.loads(producer.read('PRODUCER.json'))
        need(manifest['identity'] == identity and manifest['runtime_workflow'] == lineage.TRAIN_WORKFLOW
             and manifest['git_tree'] == live['tree']['sha'] and manifest['original_b23_dependency_files'] == 306
             and manifest['source_code_reads_only'] is True and manifest['include_hidden_files'] is True
             and all(manifest[key] == 0 for key in ('new_model_forwards', 'optimizer_updates', 'new_source_calls')),
             'actual complete original TRAIN producer captured before any scientific call')
        tree = {item['path']: item for item in live['tree']['tree']}
        expected = {'PRODUCER.json'} | {'repository/' + name for name in manifest['files']}
        need({item.filename for item in producer.infolist() if not item.is_dir()} == expected,
             'all and only original complete TRAIN producer members')
        for name, pin in manifest['files'].items():
            raw = producer.read('repository/' + name)
            need(len(raw) == pin['bytes'] and sha(raw) == pin['sha256']
                 and hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == pin['git_blob_sha1']
                 and tree[name]['type'] == 'blob' and tree[name]['sha'] == pin['git_blob_sha1']
                 and tree[name]['mode'] == pin['mode'], 'all original TRAIN bytes and actual Git tree blobs ' + name)
        for name in ('english_v6_b24_b23_warm_start.py', 'english_v6_b24_b23_failure.py',
                     'english_v6_b24_producer.py', 'english_v6_b24_assessment_binding.json',
                     'english_v6_b24_opening.json'):
            need(producer.read('repository/automation/' + name) == (ROOT / 'automation' / name).read_bytes(),
                 'same reviewed actual TRAIN-entry gate implementation and opening ' + name)
        for name, digest in lineage.metadata.validate()['hashes'].items():
            need(sha(producer.read('repository/' + name)) == digest,
                 'actual training captured every exact prospective B24 assessment dependency ' + name)
        template = producer.read('repository/automation/english_v6_b24_training_workflow_template.yml')
        token = b'__B24_REVIEWED_B23_WARM_START_PARENT__'
        need(template.count(token) == 2 and producer.read('repository/' + lineage.TRAIN_WORKFLOW)
             == template.replace(token, t['prepared_parent'].encode()), 'only reviewed two-parent substitutions activated TRAIN')
        models, receipts = archives['models'], archives['receipts']
        raw = models.read('RESULT.json')
        result = json.loads(raw)
        need(sha(raw) == t['result_sha256'] and result['schema'] == 'english-v6-b24-train-b23-warm-start-v1'
             and result['status'] == 'TRAINED_B23_WARM_START_CANDIDATE_REQUIRES_NEW_INDEPENDENT_PRIMARY'
             and result['identity'] == identity and result['recipe_sha256'] == result['design_sha256'] == lineage.metadata.DESIGN_SHA
             and result['assessment_sha256'] == lineage.ASSESSMENT_SHA and result['model_pt_sha256'] == t['model_pt_sha256']
             and result['actual_optimizer_updates'] == 192 and result['actual_new_model_forwards'] == 1847
             and result['original_retention_conditions'] == 120 and result['union_retention_conditions'] == 122
             and result['reused_ordinary_teacher_arrays'] == 264 and result['prefix_blank_reduction'] == 'amax'
             and result['original_B23_NO_PROMOTION_reauthenticated'] is True
             and result['phase_initialization_pt_sha256'] == lineage.metadata.CONTROL_PT
             and result['optimizer_initialization'] == 'fresh_AdamW'
             and result['restored_optimizer_state'] is result['exact_optimizer_continuation'] is False
             and result['old_B23_optimizer_updates_reexecuted'] == 0
             and result['challenger']['parent_pt_sha256'] == lineage.metadata.CONTROL_PT
             and result['challenger']['optimizer_initialization'] == 'fresh_AdamW'
             and result['challenger']['restored_optimizer_state'] is result['challenger']['exact_optimizer_continuation'] is False
             and result['challenger']['old_B23_optimizer_updates_reexecuted'] == 0
             and all(result[key] == 0 for key in ('new_control_optimizer_updates', 'new_control_forwards',
                 'new_teacher_forwards', 'dev_cal_locked_forwards', 'new_tts_asr_calls'))
             and result['promotion_allowed'] is result['historical_numerical_replay_claimed']
                 is result['same_job_matched_control'] is result['physical_ssc305'] is False
             and result['field_far_per_hour'] is None,
             'actual full new192-step B23-initialized fresh-AdamW TRAIN candidate; no release or same-job control claim')
        for key, name in (('source_sha256', 'SOURCE.json'), ('common_draw_sha256', 'COMMON_DRAW.json'),
                ('union_evidence_sha256', 'UNION_EVIDENCE.json'), ('old120_references_sha256', 'OLD120_POSITIVE_REFERENCES.json'),
                ('opening_sha256', 'OPENING.json'), ('assessment_binding_sha256', 'ASSESSMENT_BINDING.json')):
            need(sha(models.read(name)) == result[key], 'every original TRAIN result dependency ' + name)
        for name, path in (('OPENING.json', 'english_v6_b24_opening.json'),
                           ('ASSESSMENT_BINDING.json', 'english_v6_b24_assessment_binding.json'),
                           ('DESIGN.json', 'english_v6_b24_b23_warm_start_design.json')):
            need(models.read(name) == producer.read('repository/automation/' + path),
                 'original trained result retains the exact before-fit ' + name)
        binding = json.loads(models.read('ASSESSMENT_BINDING.json'))
        need(binding['assessment'] == dict(path='automation/english_v6_b24_panels/assessment.json', sha256=lineage.ASSESSMENT_SHA)
             and manifest['recipe_sha256'] == lineage.metadata.DESIGN_SHA
             and manifest['assessment_binding_sha256'] == result['assessment_binding_sha256']
             and manifest['opening_sha256'] == result['opening_sha256'], 'prospective assessment and real failure opening remain identical')
        consumed_raw = receipts.read('CONSUMPTION.json')
        need(archives['consumed'].read('CONSUMPTION.json') == consumed_raw
             and {item.filename for item in archives['consumed'].infolist() if not item.is_dir()} == {'CONSUMPTION.json'},
             'same durable separately uploaded before-fit consumed marker')
        consumed, source = json.loads(consumed_raw), json.loads(models.read('SOURCE.json'))
        fetched = json.loads(receipts.read('FETCH.json'))
        need(consumed['schema'] == result['schema'] and consumed['status'] == 'TRAINING_BUDGET_CONSUMED_UNKNOWN'
             and consumed['identity'] == fetched['identity'] == identity
             and consumed['recipe_sha256'] == fetched['recipe_sha256'] == result['recipe_sha256']
             and consumed['assessment_binding_sha256'] == fetched['assessment_binding_sha256'] == result['assessment_binding_sha256']
             and consumed['opening_sha256'] == fetched['opening_sha256'] == result['opening_sha256']
             and consumed['source_preflight_sha256'] == sha(receipts.read('SOURCE_PREFLIGHT.json')) == result['source_sha256']
             and receipts.read('SOURCE_PREFLIGHT.json') == models.read('SOURCE.json')
             and fetched['previous_failure'] == source['previous_failure']
             and fetched['prospective_history_reexport'] == source['prospective_history_reexport']
             and fetched['previous_source_metadata_members'] == source['previous_source_metadata_members'],
             'actual source authorization completed before the single original TRAIN budget was consumed')
        need(source['schema'] == 'english-v6-b24-authenticated-train-source-v1'
             and source['all_original_train_sources_and_teachers_unchanged'] is source['saved_B23_dev_readback_for_authorization'] is True
             and all(source[key] == 0 for key in ('new_model_forwards', 'new_source_calls', 'new_dev_model_forwards', 'dev_rows_used_for_training'))
             and source['prospective_assessment_sha256'] == lineage.ASSESSMENT_SHA
             and source['initialization'] == json.loads(models.read('DESIGN.json'))['initialization'],
             'original TRAIN sources alone; saved B23 readback is authorization evidence')
        verify_previous_failure(receipts, json.loads(models.read('OPENING.json')), source)
        stems = dict(control='mero43-b23-challenger192', challenger='mero43-b24-challenger192')
        for role, stem in stems.items():
            need(set(result[role]['files']) == {stem + extension for extension in ('.pt', '.npz', '.onnx')},
                 'all three complete original model formats ' + role)
            for name, pin in result[role]['files'].items():
                raw = models.read(role + '/models/' + name)
                need(len(raw) == pin['bytes'] and sha(raw) == pin['sha256'], 'unchanged original model bytes ' + name)
                if name.endswith('.pt'):
                    need(pin['sha256'] == t['model_pt_sha256'][role], 'exact original fixed PT role ' + role)
        need(result['challenger']['stem'] == 'challenger/models/' + stems['challenger']
             and result['control']['output_stem'] == 'control/models/' + stems['control']
             and result['control']['files'] == source['original_control']['files'], 'same immutable original B23 control exports')
    return result
