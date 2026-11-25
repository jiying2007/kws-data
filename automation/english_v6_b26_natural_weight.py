"""Inert B26 TRAIN hypothesis: natural-negative weight 0.08 to 0.16 only.

The exact original B23 fit code and B24 role wrapper are reused with isolated
globals. Neither historical module is mutated. The initializer remains the
saved B23 state; immutable B24 exports become the historical control. No
execution workflow or fresh assessment population is registered in this stage.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
import io
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b24_b23_warm_start as b24
import english_v6_b26_b24r1_failure as failure

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DESIGN = HERE / 'english_v6_b26_natural_weight_design.json'
DESIGN_SHA = 'da6853da24ce47ca685aa8deb2132fc03045bdab80a8f4a5693221f28c41c97b'
BINDING = HERE / 'english_v6_b26_assessment_binding.json'
OPENING = HERE / 'english_v6_b26_opening.json'
PROPOSAL_SHA = '4f2df013204098f1014e366def6e8b0b6b983fcaba73805db1204a415f116b8c'
INITIAL_PT = '784def9df6c2a8bfbecdb7ba2d7b70630337a1cda72561c05392327aa980297f'
CONTROL_PT = 'f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4'
NATURAL_WEIGHT = 0.16
STEM = 'mero43-b26-challenger192'
SCHEMA = 'english-v6-b26-train-natural-weight-v1'
CONDITION = 'only_after_actual_complete_B24R1_primary_NO_PROMOTION'
need, sha, encoded = b24.need, b24.sha, b24.encoded


def design():
    need(sha(DESIGN.read_bytes()) == DESIGN_SHA, 'exact pre-result single-variable B26 design')
    value = failure.document(DESIGN.read_bytes())
    for key in ('baseline_recipe', 'baseline_fit', 'baseline_wrapper', 'proposal', 'evidence'):
        pin = value[key]
        path = ROOT / pin['path']
        need(path.is_file() and not path.is_symlink() and sha(path.read_bytes()) == pin['sha256'],
             'unchanged frozen TRAIN-only scientific dependency ' + key)
    before = b24.design()
    proposal = failure.document((ROOT / value['proposal']['path']).read_bytes())
    need(value['proposal']['sha256'] == PROPOSAL_SHA
         and proposal['hypothesis']['scientific_variable'] == 'NATURAL_WEIGHT'
         and proposal['hypothesis']['old_value'] == before['natural_weight'] == b24.old.NATURAL_WEIGHT == 0.08
         and proposal['hypothesis']['new_value'] == NATURAL_WEIGHT == 0.16
         and value['scientific_change'] == dict(field='natural_weight', **{'global': 'NATURAL_WEIGHT'}, before=0.08, after=0.16),
         'one exact scalar contrast; no silent change to a historical module')
    need(value['initial_pt_sha256'] == before['parent_pt_sha256'] == INITIAL_PT
         and value['historical_control_pt_sha256'] == CONTROL_PT
         and value['opening_condition'] == CONDITION
         and value['optimizer_updates'] == before['updates'] == 192
         and value['maximum_new_model_forwards'] == before['budget']['total_new_model_forwards'] == 1847
         and before['union_retention_conditions'] == 122
         and before['gain_consistency_weight'] == 0.03
         and before['positive_retention_weight'] == before['ordinary_ceiling_weight'] == 0.08
         and value['new_training_audio'] == value['new_teacher_forwards'] == value['new_control_forwards'] == 0
         and all(value[key] is False for key in ('exact_optimizer_continuation', 'historical_numerical_replay_claimed',
             'old_B25_condition_changed', 'quality_thresholds_changed', 'scientific_execution_authorized')),
         'same original initializer, data, teachers, draws, budget and fixed quality rules')
    return value


def scientific_recipe():
    """Resolve the one changed numeric field over the immutable original recipe."""
    design()
    value = copy.deepcopy(b24.design())
    value['natural_weight'] = NATURAL_WEIGHT
    return value


def assessment_binding(require_frozen=True):
    """This preparation deliberately has no new people, PCM or executable freeze."""
    value = failure.document(BINDING.read_bytes())
    expected = dict(schema='english-v6-b26-prospective-assessment-binding-v1',
        recipe_path=DESIGN.relative_to(ROOT).as_posix(), recipe_sha256=DESIGN_SHA,
        frozen=False, optimizer_allowed=False, assessment=None, files={}, original_B24R1_source_exclusion=None)
    need(value == expected, 'B26 assessment stays inert until complete R1 source PCM/person exclusion is known')
    need(not require_frozen, 'fresh B26 assessment and original R1 exclusion verification are not implemented or registered')
    return value


def opening(require_enabled=True):
    value = failure.document(OPENING.read_bytes())
    fields = {'schema', 'condition', 'enabled', 'quality_lock', 'failure_adapter',
        'baseline_challenger_pt_sha256', 'actual_primary_evaluator_sha256', 'actual_primary_producer_manifest_sha256'}
    need(set(value) == fields and value['schema'] == 'english-v6-b26-after-original-b24r1-no-promotion-v1'
         and value['condition'] == CONDITION and value['baseline_challenger_pt_sha256'] == CONTROL_PT,
         'only complete original B24R1 primary failure, never B24 SOURCE STOP or old B25 permission')
    if not require_enabled and value['enabled'] is False:
        need(all(value[k] is None for k in ('quality_lock', 'failure_adapter',
             'actual_primary_evaluator_sha256', 'actual_primary_producer_manifest_sha256')),
             'unobserved future quality results have no invented pins')
        return value
    need(value['enabled'] is True, 'B26 is sealed until complete original B24R1 PRIMARY NO_PROMOTION')
    failure.validate_lock(value['quality_lock'])
    expected = dict(path='automation/english_v6_b26_b24r1_failure.py',
        sha256=sha(Path(failure.__file__).read_bytes()),
        base_reader_path='automation/english_v6_b24r1_quality_readback.py', base_reader_sha256=failure.BASE_READER_SHA)
    need(value['failure_adapter'] == expected
         and value['actual_primary_evaluator_sha256'] == value['quality_lock']['evaluator_sha256'] == failure.PRIMARY_CODE_SHA
         and value['actual_primary_producer_manifest_sha256'] == value['quality_lock']['producer_manifest_sha256'],
         'opening binds exact original evaluator and complete producer, not summary metrics')
    return value


def previous_failure(inputs):
    bound = opening()
    report = failure.verify_failure_original(Path(inputs) / 'no-promotion', bound['quality_lock'])
    need(report['schema'] == 'english-v6-b26-original-b24r1-no-promotion-v1'
         and report['panel'] == 'primary' and report['panel_gate'] == 'NO_PROMOTION'
         and report['model_pt_sha256']['challenger'] == CONTROL_PT
         and report['assessment_sha256'] == failure.ASSESSMENT_SHA
         and report['training_assessment_sha256'] == failure.TRAINING_ASSESSMENT_SHA
         and report['training_design_sha256'] == b24.DESIGN_SHA
         and report['all_original_arrays_verified'] is report['original_source_admission_recomputed']
             is report['original_producer_git_verified'] is True
         and all(report[key] is False for key in ('authorization_granted', 'replication_open_allowed',
             'candidate_engineering_validation_allowed', 'promotion_allowed', 'product_released')),
         'entire original R1 source, Git producer and every actual return independently reproduce NO_PROMOTION')
    return report


@contextmanager
def original_models(directory):
    """Read pinned old model bytes only; never deserialize or infer a model here."""
    pin = design()['original_B24_models']
    path = Path(directory) / 'TRAIN_models.zip'
    need(path.is_file() and not path.is_symlink() and path.stat().st_size == pin['bytes'], 'original B24 model ZIP')
    raw = path.read_bytes()
    need(sha(raw) == pin['sha256'], 'exact original B24 model ZIP bytes')
    with failure.opened(raw) as archive:
        result_raw = archive.read('RESULT.json')
        result = failure.document(result_raw)
        need(sha(result_raw) == pin['result_sha256']
             and result['schema'] == 'english-v6-b24-train-b23-warm-start-v1'
             and result['identity']['run_id'] == str(pin['run_id']) and result['identity']['run_head'] == pin['run_head']
             and result['recipe_sha256'] == b24.DESIGN_SHA
             and result['phase_initialization_pt_sha256'] == result['model_pt_sha256']['control'] == INITIAL_PT
             and result['model_pt_sha256']['challenger'] == CONTROL_PT
             and result['actual_optimizer_updates'] == 192 and result['actual_new_model_forwards'] == 1847
             and archive.read('DESIGN.json') == b24.DESIGN.read_bytes()
             and sha(archive.read('COMMON_DRAW.json')) == b24.design()['common_draw_sha256'],
             'exact already-trained B24 run, unchanged B23 initializer and original COMMON_DRAW')
        for role, prefix, wanted, stem in (
                ('control', 'control/models/', INITIAL_PT, 'mero43-b23-challenger192'),
                ('challenger', 'challenger/models/', CONTROL_PT, 'mero43-b24-challenger192')):
            files = result[role]['files']
            need(set(files) == {stem + ext for ext in ('.pt', '.npz', '.onnx')}, 'three complete immutable exports ' + role)
            for name, item in files.items():
                body = archive.read(prefix + name)
                need(len(body) == item['bytes'] and sha(body) == item['sha256'], 'exact saved export ' + name)
            need(files[stem + '.pt']['sha256'] == wanted, 'distinct exact B23 initializer and B24 control')
        yield archive, result


def historical_control(directory, output=None):
    """Validate or copy existing B24 exports; no new control forward is performed."""
    with original_models(directory) as (archive, result):
        receipt = copy.deepcopy(result['challenger'])
        if output is not None:
            output = Path(output)
            need(not output.exists() and output.parent.is_dir(), 'new control export directory only')
            output.mkdir()
            for name in receipt['files']:
                (output / name).write_bytes(archive.read('challenger/models/' + name))
    return dict(receipt, immutable_historical_control=True, new_optimizer_updates=0, new_forwards=0,
        same_job_matched_control=False, original_train_run_id=design()['original_B24_models']['run_id'])


def load_initial_state(directory, frontend, torch, np):
    """Same saved B23 PT/NPZ bytes embedded in B24; no checkpoint forward."""
    with original_models(directory) as (archive, _result):
        stem = 'control/models/mero43-b23-challenger192'
        raw = archive.read(stem + '.pt')
        need(sha(raw) == INITIAL_PT, 'same saved B23 initializer, never B24 or B0')
        initial = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        need(sum(v.numel() for v in initial.values()) == 43876 and all(torch.isfinite(v).all() for v in initial.values()),
             'finite exact original fixed-architecture parent')
        with np.load(io.BytesIO(archive.read(stem + '.npz')), allow_pickle=False) as bundle:
            need(set(bundle.files) == set(initial) | {'mel_filters', 'train_mean', 'train_scale'}
                 and all(np.array_equal(bundle[k], value.numpy()) for k, value in initial.items())
                 and all(np.array_equal(bundle[k], value) for k, value in
                    zip(('mel_filters', 'train_mean', 'train_scale'), frontend, strict=True)),
                 'same exact PT/NPZ state and unchanged original frontend')
    return {key: value.detach().clone() for key, value in initial.items()}


def fit_engine():
    """Keep the existing scientific code object; isolate one scalar global."""
    design()
    original = b24.b23.fit
    profile = SimpleNamespace(**vars(b24.old))
    profile.NATURAL_WEIGHT = NATURAL_WEIGHT
    namespace = dict(original.__globals__, old=profile)
    return FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)


def fit(initial, slots, base_cache, natural_cache, positive_paths, teachers, frontend, output, torch, np, F):
    """Future scientific helper; no public execute entry exists in this preparation."""
    engine = fit_engine()
    original = b24.fit
    namespace = dict(original.__globals__, b23=SimpleNamespace(fit=engine), STEM=STEM)
    wrapped = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    result = wrapped(initial, slots, base_cache, natural_cache, positive_paths, teachers, frontend, output, torch, np, F)
    return dict(result, natural_weight=NATURAL_WEIGHT, historical_control_pt_sha256=CONTROL_PT,
        original_B24_updates_reexecuted=0, new_teacher_forwards=0, new_control_forwards=0)


def source_preflight(inputs):
    """Future TRAIN-only admission remains closed before any acoustic runtime."""
    inputs = Path(inputs)
    proof = previous_failure(inputs)
    assessment_binding()  # Intentionally closed until a separate complete metadata implementation exists.
    import english_v6_b24_source_training_proof as original_train
    original = original_train.verify(inputs / 'original_b24')
    control = historical_control(inputs / 'original_b24')
    report, *values = b24.baseline.source_preflight(inputs / 'training')
    need(sha(encoded(values[-1])) == b24.design()['common_draw_sha256'], 'same complete original 192 draws')
    report = dict(report, schema='english-v6-b26-authenticated-train-source-v1',
        original_B24R1_failed_primary=proof, original_B24_train=original,
        historical_control=control, initialization_pt_sha256=INITIAL_PT, natural_weight=NATURAL_WEIGHT,
        all_original_train_sources_and_teachers_unchanged=True, dev_rows_used_for_training=0,
        new_model_forwards=0, new_source_calls=0)
    return report, *values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check',))
    parser.parse_args()
    design(); assessment_binding(False); opening(False)
    print('PASS: inert B26 single-variable TRAIN preparation; no execution, metadata roster or budget is opened')
