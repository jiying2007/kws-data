"""Bind the Mero MEH candidate to the unchanged saved-PCM C/ARM replay driver.

Only producer identity, artifact protocol names, and source scope are adapted.
The pinned driver performs the original fourteen case selections, twenty-eight
processes, trace comparisons, deadlines, and supervision. Its fresh-case format
remains the internal synthetic192 format also retained by the new training core.
The locked-test replay checks implementation equivalence of already selected
weights; it never selects another checkpoint, threshold, or training recipe.
"""
from pathlib import Path
import hashlib
import importlib.util
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKER = 'automation/english_mero_meh_fullpcm_once.py'
PLAN = 'automation/english_mero_meh_fullpcm_plan.json'
SCHEMA = 'english-mero-meh-fullpcm-once-v1'
CAMPAIGN = 'english-mero-meh-corpus-v1'
TRAIN_SCHEMA = 'english-mero-meh-training-once-v1'
TRAIN_STATUS = 'RETAINED_RESEARCH_MERO_MEH_HEAD'
TRAIN_WORKFLOW = '.github/workflows/english-mero-meh-train-once.yml'
ARTIFACT_PREFIX = 'english-mero-meh-training-'
SOURCE_SCOPE = 'HELD_OUT_KOKORO_VOICEPACKS_WITH_SHARED_PRETRAINING'
LOCKED_USE = 'NUMERICAL_IMPLEMENTATION_EQUIVALENCE_OF_ALREADY_FROZEN_WEIGHTS_ONLY'
PRONUNCIATION_CONTRACT = {'Hey Mero': '/heɪ ˈmɛroʊ/', 'Hey Vosen': '/heɪ ˈvoʊsən/'}
DRIVER = 'automation/english_synthetic192_fullpcm_once.py'
DRIVER_PIN = {'bytes': 38253, 'sha256': 'fd6f1695e059a592b3cb9d7385904f40b115da38c9c3f3d82120ddeec272e861'}


def validate_training_run(run, need):
    need(run.get('repository', {}).get('full_name') == 'jiying2007/kws-data'
        and run.get('repository', {}).get('private') is False
        and run.get('path') == TRAIN_WORKFLOW and run.get('event') == 'push'
        and run.get('head_branch') == 'main' and run.get('run_attempt') == 1
        and run.get('status') == 'completed' and run.get('conclusion') == 'success',
        'ACTUAL_SUCCESSFUL_PUBLIC_MERO_MEH_TRAINING_RUN')


class ReceiptProtocol:
    """Translate result protocol at its first write; saved evidence is not edited."""
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def once(self, path, value):
        if Path(path).name == 'RESULT.json' and value.get('schema') == SCHEMA + '/result':
            value = dict(value)
            if value.get('status') == 'VERIFIED_SYNTHETIC192_HEAD_FULLPCM_NATIVE_ARM':
                value['status'] = 'VERIFIED_MERO_MEH_HEAD_FULLPCM_NATIVE_ARM'
            value.update(source_design_id=CAMPAIGN, fresh_source_scope=SOURCE_SCOPE,
                locked_test_replay_use=LOCKED_USE, new_checkpoint_selections=0,
                threshold_tuning_runs=0, recommended_pronunciation_contract=PRONUNCIATION_CONTRACT,
                recommended_pronunciation_contract_frozen=True, human_native_pronunciation_verified=False)
        return self.original.once(path, value)


def driver():
    path = ROOT / DRIVER
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('REGULAR_FROZEN_FULLPCM_DRIVER')
    raw = path.read_bytes()
    if {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} != DRIVER_PIN:
        raise ValueError('UNCHANGED_FULLPCM_NUMERICAL_DRIVER')
    spec = importlib.util.spec_from_file_location('mero_meh_frozen_fullpcm', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    original_fixed_plan = m.fixed_plan
    original_candidate_gate = m.candidate_gate
    original_helpers = m.helpers
    original_prepare = m.prepare_inputs
    original_download = m.download
    m.ROOT, m.WORKER, m.PLAN, m.SCHEMA = ROOT, WORKER, PLAN, SCHEMA
    m.HELPERS = {**m.HELPERS, DRIVER: DRIVER_PIN}
    # This replay does not issue another training admission. The actual admitted
    # corpus and completed training remain bound by the selected training run.
    m.FALSE_FLAGS = tuple(x for x in m.FALSE_FLAGS if x != 'formal_training_admitted')

    def fixed_plan(worker_sha, candidate=None, main_before=None, head=None, npz=None):
        value = original_fixed_plan(worker_sha, candidate, main_before, head, npz)
        value.update(fresh_source_scope=SOURCE_SCOPE, source_design_id=CAMPAIGN,
            training_result_schema=TRAIN_SCHEMA,
            training_internal_schema='english-synthetic192-training-once-v1',
            locked_test_replay_use=LOCKED_USE, new_checkpoint_selections=0,
            threshold_tuning_runs=0, recommended_pronunciation_contract=PRONUNCIATION_CONTRACT,
            recommended_pronunciation_contract_frozen=True, human_native_pronunciation_verified=False)
        return value

    def source_preflight(plan, h):
        expected = fixed_plan(h.sha(h.regular(ROOT / WORKER)), plan.get('candidate_source'),
            plan.get('main_before'), plan.get('candidate_head'), plan.get('candidate_npz'))
        m.need(h.canonical(plan) == h.canonical(expected), 'EXACT_MERO_MEH_FULLPCM_PLAN')
        if not plan['enabled']:
            m.need(all(plan[k] is None for k in ('candidate_source', 'main_before', 'candidate_head', 'candidate_npz')),
                'DISABLED_WITHOUT_INVENTED_RESULT_PINS')
            return
        m.need(type(plan['main_before']) is str and re.fullmatch('[0-9a-f]{40}', plan['main_before']) is not None,
            'ACTUAL_NEXT_BASE')
        source = plan['candidate_source']
        m.need(type(source) is dict and set(source) == set(m.PREVIOUS)
            and type(source['run_id']) is int and source['run_id'] > 0
            and type(source['artifact_id']) is int and source['artifact_id'] > 0
            and type(source['bytes']) is int and 0 < source['bytes'] <= 512 * 1024**2
            and type(source['head_sha']) is str and re.fullmatch('[0-9a-f]{40}', source['head_sha']) is not None
            and type(source['sha256']) is str and re.fullmatch('[0-9a-f]{64}', source['sha256']) is not None
            and source['name'] == ARTIFACT_PREFIX + str(source['run_id']), 'REAL_MERO_MEH_CANDIDATE_ARTIFACT_PIN')
        for key, size in (('candidate_head', 197908), ('candidate_npz', 199356)):
            weight = plan[key]
            m.need(type(weight) is dict and set(weight) == {'bytes', 'sha256'} and weight['bytes'] == size
                and type(weight['sha256']) is str and re.fullmatch('[0-9a-f]{64}', weight['sha256']) is not None,
                'CANDIDATE_WEIGHT_PIN')

    def candidate_gate(document, plan):
        m.need(document.get('schema') == TRAIN_SCHEMA and document.get('status') == TRAIN_STATUS
            and document.get('source_design_id') == CAMPAIGN and document.get('locked_test_opened') is True,
            'RETAINED_MERO_MEH_CANDIDATE_REQUIRED')
        # Adapt two top-level protocol labels in memory only. The original gate
        # still checks all ten exact booleans, weight pins, run, and model scope.
        compatible = dict(document, schema='english-synthetic192-training-once-v1',
            status='RETAINED_RESEARCH_SYNTHETIC192_HEAD')
        original_candidate_gate(compatible, plan)

    def helpers():
        h, inc = original_helpers()
        return h, ReceiptProtocol(inc)

    def prepare_inputs(previous, candidate, plan, h, inc):
        prepared = original_prepare(previous, candidate, plan, h, inc)
        prepared['summary'].update(status='AUTHENTICATED_MERO_MEH_HEAD_FIXED_FULLPCM_INPUTS',
            source_design_id=CAMPAIGN, locked_test_replay_use=LOCKED_USE,
            new_checkpoint_selections=0, threshold_tuning_runs=0)
        for item in prepared['cases'][:5]:
            item['case']['reference_origin'] = 'NEW_MERO_MEH_CANDIDATE_SAVED_OUTPUT'
        return prepared

    def download(runner, name, source, h):
        local = original_download(runner, name, source, h)
        if name == 'candidate':
            run = h.loads(h.regular(runner.output / 'evidence' / 'candidate-run.stdout'))
            validate_training_run(run, m.need)
        return local

    m.fixed_plan, m.source_preflight, m.candidate_gate = fixed_plan, source_preflight, candidate_gate
    m.helpers, m.prepare_inputs, m.download = helpers, prepare_inputs, download
    return m


def main(argv=None):
    return driver().main(argv)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
