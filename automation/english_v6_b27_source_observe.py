"""B27 native profile; every scientific Q1 observer and quarantine primitive is unchanged."""
from pathlib import Path
import importlib
import english_v6_b27_profile as profile
original = profile.preserved('english_v6_b24q1_source_observe')
profile.inherit(original, globals())
import english_v6_b27_source_ancestry as lineage
PANEL = 'primary'
SOURCE_SCHEMA = 'english-v6-b27-fresh-combined-source-v1'
BATCH = 'english-v6-b27-120-original-quarantine-source-screen-v1'
POLICY_KEYS = original.POLICY_KEYS + ('b27_execution_profile_sha256',)


def source_check(repository=None):
    frozen = primary_observer.source_check(repository)
    scientific = {k: v for k, v in frozen.items() if k not in ('schema', 'observer_sha256', 'source_indices')}
    need(quarantine.equal(scientific, lineage.read_plan()['observer_scientific_contract']),
         'unchanged five-provider native validity and numerical source admission')
    quarantine.orchestration_contract()
    # These two old hashes identify the unchanged successful-empty policy;
    # B27 execution permission is separately bound to its own prospective plan.
    return dict(scientific, schema='english-v6-b27-five-observer-contract-v1', panel=PANEL,
        observer_sha256=sha(Path(__file__).read_bytes()),
        source_indices=lineage.read_plan()[PANEL]['synthetic']['speaker_indices'],
        orchestration_changed=False, successful_empty_remains_invalid=True,
        proposal_sha256=quarantine.PROPOSAL_SHA,
        approval_sha256='6b3a9303874606b17045ff1ee0b78f7a08123c9a1cdc80442056451110b3b13b',
        orchestration_contract_sha256=quarantine.ORCHESTRATION_SHA,
        quarantine_helper_sha256=sha(Path(quarantine.__file__).read_bytes()),
        orchestration_implementation_sha256=sha((HERE / 'english_v6_b24q1_source_observe.py').read_bytes()),
        b27_execution_profile_sha256=sha(profile.PROFILE.read_bytes()))


def owner():
    return importlib.import_module('english_v6_b27_source' if PANEL == 'primary'
                                  else 'english_v6_b27_replication_source')


def expected_registration(source, supplied=None):
    lineage.read_plan()
    expected = owner().registration(Path(source))
    need(supplied is None or quarantine.equal(supplied, expected), 'actual B27 source registration only')
    return expected


def require_original_rows(source, rows):
    need(quarantine.equal(rows, owner().read_source(Path(source))),
         'all complete B27 originals in registered order')

def validate_rows(rows, source_sha, registration):
    count = registration['source_rows']
    need(type(count) is int and source_sha == registration['source_manifest_sha256']
         and 96 <= count <= MAX_COUNT
         and len(rows) == len({r['id'] for r in rows}) == len({r['pcm_sha256'] for r in rows}) == count
         and all(Path(r['id']).name == r['id'] and r['id'].startswith('b27-' + PANEL + '-')
             and r['role'] == 'development' and r['training_allowed'] is False
             and r['observed_class'] is None for r in rows), 'EVERY_PRE_OBSERVER_ELIGIBLE_ORIGINAL_EXACTLY_ONCE')
    synthetic = [r for r in rows if r['cohort'] == 'synthetic']
    natural = [r for r in rows if r['cohort'] == 'natural']
    indices = lineage.read_plan()[PANEL]['synthetic']['speaker_indices']
    need(len(synthetic) == 96 and len(natural) == count - 96
         and {r['speaker_index'] for r in synthetic} == set(indices), 'ALL96_SYNTHETIC_AND_ONLY_ELIGIBLE_NATURAL')
    for index in indices:
        family = [r for r in synthetic if r['speaker_index'] == index]
        need(len(family) == 4 and {r['cell_id']: r['length_scale'] for r in family}
             == {'mero-fast': 1.4, 'mero-slow': 1.8, 'question': 1.0, 'request': 1.08}
             and all(r['requested_class'] == (2 if r['cell_id'].startswith('mero-') else 0) for r in family),
             'UNCHANGED_FOUR_REQUESTS_AND_POSITIVE_RATE_PAIR')
    need(all(r['requested_class'] == 0 and r['whole_original'] is True for r in natural),
         'UNCHANGED_WHOLE_NATURAL_NEGATIVES')
