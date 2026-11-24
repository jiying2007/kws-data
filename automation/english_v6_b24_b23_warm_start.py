"""One conditional B24 phase starts from actual B23 weights with fresh AdamW.

No original B23 update is replayed. The unchanged B23 fit loop performs a new
192-update phase with the same draws, sources, teachers, amax loss, coefficients
and 1847-forward budget. Original B23 exports are the immutable control.
Complete independent B23 NO_PROMOTION and new metadata are required first.
"""
from __future__ import annotations
import argparse
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b23_prefix_peak as b23
import english_v6_b24_b23_failure as failure

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DESIGN = HERE / 'english_v6_b24_b23_warm_start_design.json'
DESIGN_SHA = '4fa88cf4050aa276323a7306f21d53163d47c27e3c011c2547d5eeebe6980309'
BINDING = HERE / 'english_v6_b24_assessment_binding.json'
OPENING = HERE / 'english_v6_b24_opening.json'
REPO = 'jiying2007/kws-data'
BRANCH = 'refs/heads/research/en-v6-b24-b23-warm-start-train-once-20261124'
WORKFLOW = '.github/workflows/english-v6-b24-b23-warm-start-train-once.yml'
MARKER = 'english-v6-b24-b23-warm-start-train-consumed'
SCHEMA = 'english-v6-b24-train-b23-warm-start-v1'
INITIAL_PT = '784def9df6c2a8bfbecdb7ba2d7b70630337a1cda72561c05392327aa980297f'
STEM = 'mero43-b24-challenger192'
baseline, old = b23.baseline, b23.old
need, sha, encoded, write, gh = b23.need, b23.sha, b23.encoded, b23.write, b23.gh
prefix_peak_suffix_loss = b23.prefix_peak_suffix_loss
Recorder, positive_loss, margin_statistics = b23.Recorder, b23.positive_loss, b23.margin_statistics


def design():
    need(sha(DESIGN.read_bytes()) == DESIGN_SHA, 'fixed pre-DEV warm-start-only phase design')
    value = json.loads(DESIGN.read_bytes())
    before = b23.design()
    descriptive = {'schema', 'contrast', 'scope', 'historical_control', 'parent_pt_sha256',
        'baseline_engine_path', 'baseline_engine_sha256', 'baseline_recipe_path', 'baseline_recipe_sha256',
        'opening_condition', 'train_only_mechanism_evidence', 'prefix_blank_loss'}
    retained = set(before) - descriptive
    need(set(before) <= set(value) and all(value[k] == before[k] for k in retained),
         'every original B23 numeric/data/draw/teacher/loss/optimizer field stays identical')
    prefix = dict(before['prefix_blank_loss'], original=before['prefix_blank_loss']['candidate'])
    need(value['prefix_blank_loss'] == prefix and prefix['candidate'] == 'softplus(blank_margin).amax()',
         'the same known-silence amax is retained, not a second loss intervention')
    need(sha((ROOT / value['baseline_engine_path']).read_bytes()) == value['baseline_engine_sha256']
         and b23.DESIGN_SHA == value['baseline_recipe_sha256']
         and value['parent_pt_sha256'] == value['historical_control']['pt_sha256'] == INITIAL_PT
         and value['opening_condition'] == 'only_after_actual_complete_B23_primary_NO_PROMOTION'
         and value['budget'] == before['budget'] and value['updates'] == 192,
         'exact actual B23 scientific parent, immutable control and unchanged new budget')
    initial = value['initialization']
    need(all(initial[k] == v for k, v in value['historical_control'].items())
         and initial['optimizer'] == 'fresh_AdamW' and initial['restore_optimizer_state'] is False
         and initial['exact_optimizer_continuation'] is False and initial['old_B23_steps_reexecuted'] is False,
         'one fresh optimizer phase from actual B23 weights, not reconstructed continuation')
    for key in ('train_only_mechanism_evidence', 'train_only_mechanism_memo'):
        pin = value[key]
        need(sha((ROOT / pin['path']).read_bytes()) == pin['sha256'], 'unchanged prior TRAIN-only mechanism record')
    baseline.architecture_contract()
    return value

def assessment_binding(require_frozen=True):
    value = json.loads(BINDING.read_bytes())
    need(value['schema'] == 'english-v6-b24-prospective-assessment-binding-v1'
         and value['recipe_path'] == DESIGN.relative_to(ROOT).as_posix()
         and value['recipe_sha256'] == DESIGN_SHA, 'prospective panels bind this exact warm-start phase')
    if not require_frozen and value['frozen'] is False:
        need(value['assessment'] is None and value['files'] == {} and value['optimizer_allowed'] is False,
             'inert incomplete metadata grants no execution budget')
        return value
    need(value['frozen'] is True and value['optimizer_allowed'] is False
         and type(value['files']) is dict and bool(value['files']),
         'complete metadata freeze is separate from failed-primary permission')
    for name, digest in value['files'].items():
        path = Path(name)
        need(not path.is_absolute() and '..' not in path.parts
             and re.fullmatch(r'[0-9a-f]{64}', digest or ''), 'safe complete metadata resource pin')
        target = ROOT / path
        need(target.is_file() and not target.is_symlink() and sha(target.read_bytes()) == digest,
             'unchanged prospective assessment member ' + name)
    pin = value['assessment']
    need(set(pin) == {'path', 'sha256'} and value['files'].get(pin['path']) == pin['sha256'],
         'assessment is covered by complete metadata inventory')
    report = json.loads((ROOT / pin['path']).read_bytes())
    need(report['freeze_before_optimizer_and_new_source'] is True
         and report['training']['design_path'] == DESIGN.relative_to(ROOT).as_posix()
         and report['training']['design_sha256'] == DESIGN_SHA,
         'fresh independent panels are prospectively frozen')
    import english_v6_b24_source_metadata as metadata
    need(Path(metadata.__file__).resolve() == HERE / 'english_v6_b24_source_metadata.py'
         and metadata.validate()['hashes'] == value['files'], 'complete deterministic fresh metadata validation')
    return value

def opening(require_enabled=True):
    value = json.loads(OPENING.read_bytes())
    expected = {'schema', 'condition', 'enabled', 'quality_lock', 'baseline_challenger_pt_sha256',
        'failure_adapter', 'actual_primary_evaluator_sha256', 'actual_primary_producer_manifest_sha256'}
    need(set(value) == expected and value['schema'] == 'english-v6-b24-after-original-b23-no-promotion-v1'
         and value['condition'] == 'only_after_actual_complete_B23_primary_NO_PROMOTION'
         and value['baseline_challenger_pt_sha256'] == design()['historical_control']['pt_sha256'],
         'specific complete original B23 failure dependency, never a generic permit')
    if not require_enabled and value['enabled'] is False:
        need(all(value[k] is None for k in ('quality_lock', 'failure_adapter', 'actual_primary_evaluator_sha256',
             'actual_primary_producer_manifest_sha256')), 'inert opening contains no fabricated future result')
        return value
    need(value['enabled'] is True, 'B24 is sealed until actual complete B23 NO_PROMOTION is bound')
    failure.validate_lock(value['quality_lock'])
    pin = value['failure_adapter']
    need(type(pin) is dict and set(pin) == {'path', 'sha256', 'base_reader_path', 'base_reader_sha256'}
         and pin['path'] == 'automation/english_v6_b24_b23_failure.py'
         and pin['base_reader_path'] == 'automation/english_v6_b23_quality_readback.py'
         and sha((ROOT / pin['path']).read_bytes()) == pin['sha256']
         and sha((ROOT / pin['base_reader_path']).read_bytes()) == pin['base_reader_sha256'] == failure.BASE_READER_SHA,
         'pre-result frozen fail-only algorithm; actual evaluator and producer remain original authority')
    need(value['actual_primary_evaluator_sha256'] == value['quality_lock']['evaluator_sha256'] == failure.PRIMARY_CODE_SHA
         and value['actual_primary_producer_manifest_sha256'] == value['quality_lock']['producer_manifest_sha256'],
         'opening names exact original PRIMARY evidence, not an inferred future code version')
    return value


def previous_failure(inputs):
    bound = opening()
    report = failure.verify_failure_original(Path(inputs) / 'no-promotion', bound['quality_lock'])
    need(report['panel'] == 'primary' and report['panel_gate'] == 'NO_PROMOTION'
         and report['model_pt_sha256']['challenger'] == bound['baseline_challenger_pt_sha256']
         and report['all_original_arrays_verified'] is report['original_source_admission_recomputed']
             is report['original_producer_git_verified'] is True
         and all(report[k] is False for k in ('authorization_granted', 'replication_open_allowed',
             'candidate_engineering_validation_allowed', 'promotion_allowed', 'product_released')),
         'actual complete B23 failure from all original source/tensor/Git evidence')
    return report


def historical_control(inputs):
    path = Path(inputs) / 'control/B23.zip'
    pin = design()['historical_control']
    need(path.is_file() and not path.is_symlink() and path.stat().st_size == pin['bytes']
         and sha(path.read_bytes()) == pin['sha256'], 'original B23 parent/control artifact bytes')
    with zipfile.ZipFile(path) as archive:
        need(archive.testzip() is None, 'complete original B23 parent/control archive')
        raw = archive.read('RESULT.json')
        receipt = json.loads(raw)
        need(sha(raw) == pin['result_sha256']
             and receipt['schema'] == 'english-v6-b23-train-prefix-peak-v1'
             and receipt['status'] == 'TRAINED_PREFIX_PEAK_CANDIDATE_REQUIRES_NEW_INDEPENDENT_PRIMARY'
             and receipt['identity']['run_id'] == str(pin['run_id']) and receipt['identity']['run_head'] == pin['run_head']
             and receipt['actual_optimizer_updates'] == 192 and receipt['actual_new_model_forwards'] == 1847
             and receipt['model_pt_sha256']['challenger'] == pin['pt_sha256']
             and receipt['challenger']['stem'] == pin['stem'] and receipt['prefix_blank_reduction'] == 'amax'
             and receipt['recipe_sha256'] == b23.DESIGN_SHA
             and sha(archive.read('SOURCE.json')) == receipt['source_sha256'] == pin['source_sha256']
             and sha(archive.read('COMMON_DRAW.json')) == receipt['common_draw_sha256'] == pin['common_draw_sha256'],
             'same actual B23 fit, source, draw and complete original model identity')
        expected = {'mero43-b23-challenger192' + ext for ext in ('.pt', '.npz', '.onnx')}
        need(set(receipt['challenger']['files']) == expected, 'all three actual B23 model formats')
        for name, item in receipt['challenger']['files'].items():
            raw = archive.read('challenger/models/' + name)
            need(len(raw) == item['bytes'] and sha(raw) == item['sha256'], 'unchanged actual B23 parent/control ' + name)
        need(receipt['challenger']['files']['mero43-b23-challenger192.pt']['sha256'] == INITIAL_PT,
             'actual checkpoint is the sole phase initialization')
    return receipt['challenger']


def source_preflight(inputs):
    inputs = Path(inputs)
    proof = previous_failure(inputs)
    control = historical_control(inputs)
    assessment = assessment_binding()
    history_reexport = prospective_history_proof(inputs)
    previous_members = previous_source_metadata(inputs, history_reexport)
    report, *values = baseline.source_preflight(inputs / 'training')
    need(sha(encoded(values[-1])) == design()['common_draw_sha256'], 'same original 192 COMMON_DRAW without rerun')
    value = dict(report, schema='english-v6-b24-authenticated-train-source-v1',
        original_b22_training_source_sha256=sha(encoded(report)), previous_failure=proof,
        original_control=control, initialization=design()['initialization'],
        prospective_history_reexport=history_reexport,
        previous_source_metadata_members=previous_members,
        all_original_train_sources_and_teachers_unchanged=True,
        original_retention_conditions=120, union_retention_conditions=122,
        new_model_forwards=0, new_source_calls=0, new_dev_model_forwards=0,
        saved_B23_dev_readback_for_authorization=True, dev_rows_used_for_training=0,
        prospective_assessment_sha256=assessment['assessment']['sha256'])
    value.pop('new_dev_reads')
    return value, *values


def prospective_history_proof(inputs):
    """Reexport the minimal public exposure list from the authenticated old source.

    previous_failure has already verified these exact original producer bytes.
    This extra metadata-only join binds every excluded PCM/person to that source;
    no audio is opened and no observer or model output is used for selection.
    """
    import english_v6_b24_source_metadata as metadata
    inputs = Path(inputs)
    bound = opening()['quality_lock']
    with zipfile.ZipFile(inputs / 'no-promotion/originals/producer.zip') as archive:
        raw = archive.read('primary_producer/automation/english_v6_b23_primary_inputs.json')
        need(sha(raw) == bound['inputs_sha256'], 'exact already-verified original B23 registration')
        registration = json.loads(raw)
        fetched = json.loads(archive.read('FETCH.json'))
    return metadata.verify_original_source(inputs / 'no-promotion/inputs/source.zip',
        fetched['source_metadata']['source'], registration['inputs']['source'])


def previous_source_metadata(inputs, history_reexport, create=False):
    """Retain two exact original private artifact members, never public API data."""
    inputs = Path(inputs)
    result = {}
    with zipfile.ZipFile(inputs / 'no-promotion/inputs/source.zip') as archive:
        for member, field in (('MANIFEST.json', 'original_manifest_sha256'),
                              ('HISTORY.json', 'original_history_sha256')):
            raw = archive.read(member)
            need(sha(raw) == history_reexport[field], 'metadata from the exact authenticated original B23 source')
            name = 'ORIGINAL_B23_SOURCE_' + member
            target = inputs / name
            if create:
                with target.open('xb') as handle:
                    handle.write(raw)
            need(target.is_file() and not target.is_symlink() and target.read_bytes() == raw,
                 'unaltered original source metadata retained in the TRAIN artifact only')
            result[name] = dict(bytes=len(raw), sha256=sha(raw), original_member=member)
    return result


def guard(parent, env=None, event=None, git_parent=None):
    design(); opening(); assessment_binding()
    e = os.environ if env is None else env
    need(re.fullmatch(r'[0-9a-f]{40}', parent or ''), 'exact reviewed B24 prepared predecessor')
    need(all(e.get(k) == v for k, v in dict(GITHUB_ACTIONS='true', GITHUB_REPOSITORY=REPO,
         GITHUB_EVENT_NAME='push', GITHUB_REF=BRANCH, GITHUB_RUN_ATTEMPT='1').items()),
         'only first public B24 execution on its prospectively named branch')
    head, run = e.get('GITHUB_SHA', ''), e.get('GITHUB_RUN_ID', '')
    need(re.fullmatch(r'[0-9a-f]{40}', head) and re.fullmatch(r'[1-9][0-9]*', run), 'actual public execution identity')
    if event is None:
        event = json.loads(Path(e['GITHUB_EVENT_PATH']).read_bytes())
    if git_parent is None:
        git_parent = subprocess.check_output(['git', '-C', e['GITHUB_WORKSPACE'], 'rev-parse', 'HEAD^1'], text=True).strip()
    need(event.get('before') == git_parent == parent and event.get('after') == head
         and event.get('ref') == BRANCH and event.get('forced') is event.get('deleted') is False
         and event.get('repository', {}).get('full_name') == REPO and event['repository'].get('private') is False,
         'single workflow-only reviewed public activation; no force or retry')
    return dict(run_id=run, run_head=head, run_attempt=1, prepared_parent=parent, run_branch=BRANCH)

def fetch(inputs, parent):
    identity = guard(parent)
    inputs = Path(inputs)
    need(inputs.is_absolute() and inputs.parent.is_dir() and not inputs.exists(), 'fresh immutable B24 input root')
    inputs.mkdir()
    proof = failure.fetch_failure_original(inputs / 'no-promotion', opening()['quality_lock'])
    need(proof['model_pt_sha256']['challenger'] == design()['historical_control']['pt_sha256'], 'failure belongs to original B23 control')
    history_reexport = prospective_history_proof(inputs)
    previous_members = previous_source_metadata(inputs, history_reexport, create=True)
    receipts = {}
    for subdir, pins in (('training', design()['input_archives']), ('control', {'B23.zip': design()['historical_control']})):
        target = inputs / subdir
        target.mkdir()
        for name, pin in pins.items():
            artifact = json.loads(gh('actions/artifacts/' + str(pin['artifact_id'])))
            need(artifact['id'] == pin['artifact_id'] and artifact['expired'] is False
                 and artifact['digest'] == 'sha256:' + pin['sha256'], 'live original input artifact ' + name)
            raw = gh('actions/artifacts/' + str(pin['artifact_id']) + '/zip')
            need(sha(raw) == pin['sha256'] and len(raw) == artifact['size_in_bytes'], 'exact original input bytes ' + name)
            (target / name).write_bytes(raw)
            receipts[subdir + '/' + name] = artifact
    historical_control(inputs)
    write(inputs / 'FETCH.json', dict(identity=identity, artifacts=receipts, recipe_sha256=DESIGN_SHA,
        assessment_binding_sha256=sha(BINDING.read_bytes()), opening_sha256=sha(OPENING.read_bytes()), previous_failure=proof,
        prospective_history_reexport=history_reexport, previous_source_metadata_members=previous_members,
        new_model_forwards=0))

def consume(inputs, parent):
    identity = guard(parent)
    inputs = Path(inputs)
    report, *_ = source_preflight(inputs)
    marker = inputs / 'CONSUMPTION.json'
    need(not marker.exists(), 'one B24 budget marker; original B23 markers remain untouched')
    prior = json.loads(gh('actions/artifacts?name=' + MARKER + '&per_page=100'))
    need(prior['total_count'] == 0 and prior['artifacts'] == [], 'B24 budget has never been consumed')
    write(inputs / 'SOURCE_PREFLIGHT.json', report)
    value = dict(schema=SCHEMA, status='TRAINING_BUDGET_CONSUMED_UNKNOWN', identity=identity,
        recipe_sha256=DESIGN_SHA, assessment_binding_sha256=sha(BINDING.read_bytes()),
        opening_sha256=sha(OPENING.read_bytes()), source_preflight_sha256=sha((inputs / 'SOURCE_PREFLIGHT.json').read_bytes()),
        budget=design()['budget'], promotion_allowed=False)
    write(marker, value)
    return value

def verify_consumed(inputs, parent):
    identity = guard(parent)
    local = json.loads((inputs / 'CONSUMPTION.json').read_bytes())
    need(local['identity'] == identity and local['budget'] == design()['budget']
         and local['recipe_sha256'] == DESIGN_SHA and local['assessment_binding_sha256'] == sha(BINDING.read_bytes())
         and local['opening_sha256'] == sha(OPENING.read_bytes())
         and local['source_preflight_sha256'] == sha((inputs / 'SOURCE_PREFLIGHT.json').read_bytes()),
         'same frozen input receipt and preoptimizer budget')
    runs = json.loads(gh('actions/workflows/' + Path(WORKFLOW).name + '/runs?branch=' + BRANCH.removeprefix('refs/heads/') + '&per_page=100'))
    need(runs['total_count'] == len(runs['workflow_runs']) == 1
         and str(runs['workflow_runs'][0]['id']) == identity['run_id']
         and runs['workflow_runs'][0]['head_sha'] == identity['run_head']
         and runs['workflow_runs'][0]['run_attempt'] == 1, 'exactly one first B24 training execution')
    items = json.loads(gh('actions/artifacts?name=' + MARKER + '&per_page=100'))
    need(items['total_count'] == len(items['artifacts']) == 1, 'one durably uploaded B24 budget')
    item = items['artifacts'][0]
    need(item['expired'] is False and item['workflow_run']['id'] == int(identity['run_id'])
         and item['workflow_run']['head_sha'] == identity['run_head'], 'same first execution owns the marker')
    raw = gh('actions/artifacts/' + str(item['id']) + '/zip')
    need(sha(raw) == item['digest'].removeprefix('sha256:') and len(raw) == item['size_in_bytes'], 'original consumed artifact bytes')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = [v.filename for v in archive.infolist() if not v.is_dir()]
        need(names == ['CONSUMPTION.json'] and json.loads(archive.read(names[0])) == local,
             'uploaded exact marker precedes first optimization')
    return identity

def fit(initial, slots, base_cache, natural_cache, positive_paths, teachers, frontend, output, torch, np, F):
    """Execute the exact frozen B23 fit function; change only exported role metadata."""
    # b23.fit builds a new AdamW after load_state_dict(initial). It neither restores
    # optimizer state nor replays an old update. Its complete scientific loop is
    # reused by identity rather than copied or monkey-patched.
    result = b23.fit(initial, slots, base_cache, natural_cache, positive_paths, teachers, frontend, output, torch, np, F)
    need(result['parent_pt_sha256'] == old.B0_PT and result['optimizer_updates'] == 192
         and result['actual_new_model_forwards'] == 1847, 'unchanged original loop metadata before role correction')
    files = {}
    for name, pin in result['files'].items():
        need(name.startswith('mero43-b23-challenger192.'), 'original fit export namespace')
        old_path = output / 'models' / name
        new_name = STEM + Path(name).suffix
        target = output / 'models' / new_name
        need(not target.exists() and old_path.stat().st_size == pin['bytes'] and sha(old_path.read_bytes()) == pin['sha256'],
             'only export names change after the already-budgeted fit')
        old_path.rename(target)
        files[new_name] = pin
    return dict(result, stem='challenger/models/' + STEM, files=files, parent_pt_sha256=INITIAL_PT,
        optimizer_initialization='fresh_AdamW', restored_optimizer_state=False, exact_optimizer_continuation=False,
        old_B23_optimizer_updates_reexecuted=0)


def load_initial_state(inputs, frontend, torch, np):
    """Read exact original B23 tensor bytes; there is no checkpoint forward."""
    historical_control(inputs)
    pin = design()['initialization']
    with zipfile.ZipFile(Path(inputs) / 'control/B23.zip') as archive:
        raw = archive.read(pin['stem'] + '.pt')
        need(sha(raw) == INITIAL_PT, 'B23 weights, never B0 initialization')
        initial = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        need(sum(v.numel() for v in initial.values()) == 43876
             and all(torch.isfinite(v).all() for v in initial.values()), 'finite exact fixed-architecture parent')
        with np.load(io.BytesIO(archive.read(pin['stem'] + '.npz')), allow_pickle=False) as bundle:
            need(set(bundle.files) == set(initial) | {'mel_filters', 'train_mean', 'train_scale'}
                 and all(np.array_equal(bundle[k], value.numpy()) for k, value in initial.items())
                 and all(np.array_equal(bundle[k], value) for k, value in
                    zip(('mel_filters', 'train_mean', 'train_scale'), frontend, strict=True)),
                 'exact parent PT/NPZ and unchanged original B0/B9 frontend')
    return {k: value.detach().clone() for k, value in initial.items()}


def execute(inputs, output, parent):
    inputs, output = Path(inputs), Path(output)
    identity = verify_consumed(inputs, parent)
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'one fresh output; no training retry')
    report, rows, waves, groups, extra, extra_waves, slots = source_preflight(inputs)
    need(sha(encoded(report)) == json.loads((inputs / 'CONSUMPTION.json').read_bytes())['source_preflight_sha256'],
         'same complete source proof as the durable consumption marker')
    output.mkdir()
    write(output / 'STARTED.json', dict(schema=SCHEMA, identity=identity, budget=design()['budget'], recipe_sha256=DESIGN_SHA))
    write(output / 'SOURCE.json', report)
    write(output / 'COMMON_DRAW.json', slots)
    (output / 'DESIGN.json').write_bytes(DESIGN.read_bytes())
    (output / 'ASSESSMENT_BINDING.json').write_bytes(BINDING.read_bytes())
    (output / 'OPENING.json').write_bytes(OPENING.read_bytes())
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    _unused_b0, _unused_b9, frontend = old.b17.load_parents(inputs / 'training', torch, np)
    base_cache = old.b13.paired.make_features(rows, waves, frontend, torch, np)
    natural_cache = old.b13.paired.make_features(extra, extra_waves, frontend, torch, np)
    paths, old_audit = baseline.prepare_union_paths(inputs / 'training', rows, waves, frontend, np, torch)
    teachers = baseline.saved_ordinary_teachers(inputs / 'training', torch, np)
    write(output / 'UNION_EVIDENCE.json', report['union'])
    write(output / 'OLD120_POSITIVE_REFERENCES.json', old_audit)
    control_dir = output / 'control/models'
    control_dir.mkdir(parents=True)
    control_files = {}
    with zipfile.ZipFile(inputs / 'control/B23.zip') as archive:
        receipt = json.loads(archive.read('RESULT.json'))['challenger']
        for name, pin in receipt['files'].items():
            raw = archive.read('challenger/models/' + name)
            need(len(raw) == pin['bytes'] and sha(raw) == pin['sha256'], 'unchanged historical control export')
            (control_dir / name).write_bytes(raw)
            control_files[name] = pin
    target = output / 'challenger'
    target.mkdir()
    candidate = fit(load_initial_state(inputs, frontend, torch, np), slots, base_cache, natural_cache,
                    paths, teachers, frontend, target, torch, np, F)
    result = dict(schema=SCHEMA, status='TRAINED_B23_WARM_START_CANDIDATE_REQUIRES_NEW_INDEPENDENT_PRIMARY', identity=identity,
        recipe_sha256=DESIGN_SHA, assessment_binding_sha256=sha(BINDING.read_bytes()),
        design_sha256=DESIGN_SHA, assessment_sha256=assessment_binding()['assessment']['sha256'],
        source_sha256=sha((output / 'SOURCE.json').read_bytes()),
        prefix_blank_reduction='amax', original_B23_NO_PROMOTION_reauthenticated=True,
        phase_initialization_pt_sha256=INITIAL_PT, optimizer_initialization="fresh_AdamW",
        restored_optimizer_state=False, exact_optimizer_continuation=False, old_B23_optimizer_updates_reexecuted=0,
        opening_sha256=sha(OPENING.read_bytes()),
        common_draw_sha256=sha((output / 'COMMON_DRAW.json').read_bytes()),
        union_evidence_sha256=sha((output / 'UNION_EVIDENCE.json').read_bytes()),
        old120_references_sha256=sha((output / 'OLD120_POSITIVE_REFERENCES.json').read_bytes()),
        model_pt_sha256=dict(B0=old.B0_PT, B9=old.b10.b9.MODELS['B9_control']['pt'],
                            control=design()['historical_control']['pt_sha256'],
                            challenger=candidate['files'][STEM + '.pt']['sha256']),
        control=dict(**design()['historical_control'], output_stem='control/models/mero43-b23-challenger192', files=control_files),
        challenger=candidate, actual_optimizer_updates=192, actual_new_model_forwards=1847,
        new_control_optimizer_updates=0, new_control_forwards=0, new_teacher_forwards=0,
        reused_ordinary_teacher_arrays=264, original_retention_conditions=120, union_retention_conditions=122,
        dev_cal_locked_forwards=0, new_tts_asr_calls=0, historical_numerical_replay_claimed=False,
        same_job_matched_control=False, architecture=baseline.architecture_contract(), promotion_allowed=False,
        physical_ssc305=False, field_far_per_hour=None)
    write(output / 'RESULT.json', result)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    sub.add_parser('check')
    for mode in ('fetch', 'consume', 'execute', 'preflight'):
        item = sub.add_parser(mode)
        item.add_argument('--inputs', type=Path, required=True)
        if mode != 'preflight': item.add_argument('--prepared-parent', required=True)
        if mode == 'execute': item.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'check':
        design(); assessment_binding(False); opening(False)
        print('PASS: B24 warm-start-only new phase; actual complete B23 failure and frozen new panels required')
    elif args.mode == 'fetch': fetch(args.inputs, args.prepared_parent)
    elif args.mode == 'consume': print(json.dumps(consume(args.inputs, args.prepared_parent), sort_keys=True))
    elif args.mode == 'preflight': print(json.dumps(source_preflight(args.inputs)[0], sort_keys=True))
    else: print(json.dumps(execute(args.inputs, args.output, args.prepared_parent), sort_keys=True))
