"""Fresh B24Q1 primary sources after authenticated warm-start training.

Source generation, native models and numerical five-valid admission are unchanged.
The approved Q1 orchestration independently proves successful empty returns,
quarantines those originals across all five, and completes every fixed slot once.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b22_source as original
import english_v6_b22_source_metadata as piper_metadata
from english_v6_b22_replication_reuse import inherit
inherit(original, globals())
import english_v6_b24q1_source_ancestry as lineage
import english_v6_b24q1_source_observe as observers
import english_v6_b24q1_natural_sources as natural
import english_v6_b24q1_quarantine as quarantine
import english_v6_b24q1_capacity as capacity

qa, B13 = lineage.qa, original.B13
PLANS, ASSESSMENT, ASSESSMENT_SHA = lineage.PLANS, lineage.ASSESSMENT, lineage.ASSESSMENT_SHA
SCHEMA = 'english-v6-b24q1-fresh-combined-source-v1'
BATCH = 'english-v6-b24q1-120-original-quarantine-source-screen-v1'
PANEL = 'primary'
FILES = {'primary': '4735d1142c24d74ce1815f924be7b189bcb55643230be73631d899e8b88249ee',
         'replication': '50d8d6ba0828032f2d3585a24f86c8fdfa223fae073761873d8f09c960661e96'}
PIPER_CONFIG = piper_metadata.PANELS / 'piper-config.json'
original_source_binding = source_binding
original_read = _read


def source_binding():
    from english_v6_b24_source_training_proof import SCOPE
    return dict(original_source_binding(), **SCOPE, **lineage.stop_scope())


def _read(directory):
    capacity.primary_source_members(Path(directory))
    return original_read(directory)


def source_policy():
    contract = code_contract()
    return dict(schema='english-v6-b24q1-source-policy-v1', experiment_id='B24Q1',
        **{key: contract[key] for key in observers.POLICY_KEYS})


def registration(source):
    rows, all_rows, history, meta = _read(source)
    contract = code_contract()
    return dict(assessment_sha256=ASSESSMENT_SHA, source_schema=SCHEMA, batch_schema=BATCH,
        source_manifest_sha256=qa.sha(Path(source) / 'MANIFEST.json'), source_rows=len(rows), max_calls=len(rows),
        **meta['source_execution'], **{key: contract[key] for key in observers.POLICY_KEYS})


def attrition_document(all_rows, eligible, decisions, observation):
    """All120 exposure identities and N before observers remain explicit strata."""
    eligible_ids = {row['id'] for row in eligible}
    by_id = {row['id']: row for row in decisions}
    empty = observation['quarantine']
    need = qa.need
    need(len(all_rows) == len({row['id'] for row in all_rows}) == 120
         and len(eligible_ids) == len(eligible) == len(by_id) == len(decisions)
         and set(by_id) == eligible_ids and set(empty) <= eligible_ids, 'ALL120_AND_FIXED_ELIGIBLE_N_PARTITION')
    strata = {}
    for row in all_rows:
        sid = row['id']
        strata[sid] = ('pre_observer_rejected' if sid not in eligible_ids else
            'quarantined_invalid_empty' if sid in empty else
            'qualified' if by_id[sid]['observed_class'] is not None else 'five_valid_unqualified')
    names = ('pre_observer_rejected', 'quarantined_invalid_empty', 'five_valid_unqualified', 'qualified')
    def counts(group, provider=None):
        group_ids = {row['id'] for row in group}
        n = len(group_ids & eligible_ids)
        providers = observers.PROVIDERS if provider is None else (provider,)
        invalid = {p: sum(p in empty.get(sid, {}) for sid in group_ids) for p in providers}
        return dict(original_slots=len(group), observer_eligible=n,
            partition={name: sum(strata[sid] == name for sid in group_ids) for name in names},
            planned_native_slots=len(providers)*n,
            valid_native_payloads=len(providers)*n-sum(invalid.values()),
            invalid_empty_native_payloads=sum(invalid.values()), invalid_empty_by_provider=invalid)
    dimensions = {
        'cohort': lambda row: row['cohort'],
        'requested_class': lambda row: str(row['requested_class']),
        'cell_id': lambda row: row.get('cell_id', 'natural-whole-original'),
        'canonical_person': lambda row: piper_metadata.canonical(
            row['source_speaker_id'] if row['cohort'] == 'synthetic' else row['speaker'])}
    grouped = {}
    for dimension, key in dimensions.items():
        groups = defaultdict(list)
        for row in all_rows:
            groups[key(row)].append(row)
        grouped[dimension] = {label: counts(group) for label, group in sorted(groups.items())}
    grouped['provider'] = {p: counts(all_rows, p) for p in observers.PROVIDERS}
    partition = {name: sum(value == name for value in strata.values()) for name in names}
    need(sum(partition.values()) == 120 and partition['pre_observer_rejected'] == 120-len(eligible),
         'NO_DROPPED_ORIGINAL_OR_CHANGED_PRE_OBSERVER_DENOMINATOR')
    return partition, grouped


def screen_document(source, providers):
    source = Path(source)
    rows, all_rows, history, meta = _read(source)
    observation = observers.load_observations(source, providers, rows, BATCH, registration=registration(source))
    texts, phones, coverage = observation['texts'], observation['phones'], observation['coverage']
    quarantined = set(observation['quarantined_ids'])
    qa.need(quarantined == set(observation['quarantine'])
        and observation['completed_coverage'] == dict.fromkeys(observers.PROVIDERS, len(rows))
        and all(coverage[p] + observation['invalid_empty_coverage'][p] == len(rows) for p in observers.PROVIDERS),
        'ALL_PLANNED_NATIVE_SLOTS_COMPLETE_BEFORE_ANY_SOURCE_DECISION')
    decisions, accepted = [], defaultdict(set)
    for row in rows:
        sid = row['id']
        if sid in quarantined:
            decisions.append(dict(row, observed_class=None, tier='unknown',
                reason=quarantine.EMPTY_REASON, source_admission_eligible=False,
                quarantine_providers=sorted(observation['quarantine'][sid]),
                quarantine_unit='whole_original_id_and_pcm'))
            continue
        qa.need(set(texts[sid]) == set(qa.policy.TEXT_MODELS)
             and set(phones[sid]) == set(qa.policy.PHONE_MODELS), 'ONLY_FIVE_VALID_ORIGINALS_ENTER_OLD_ADMISSION')
        if row['cohort'] == 'synthetic':
            outcome = qa.admit(row, texts[sid], phones[sid])
            value = outcome['class']
            qa.need(outcome['training_allowed'] is False and (value is None or value == row['requested_class']), 'unchanged synthetic source-admission policy')
            if value is not None: accepted[row['source_group_id']].add(value)
            decisions.append(dict(row, observed_class=value, tier=outcome['tier'], reason=outcome['reason'], phone_margins=outcome['margins']))
        else:
            all_obs = set(texts[sid]) == set(qa.policy.TEXT_MODELS) and set(phones[sid]) == set(qa.policy.PHONE_MODELS)
            heard = [qa.policy.norm(text) for text in texts[sid].values()]
            votes = heard.count(qa.policy.norm(row['transcript']))
            veto = any(natural.BLOCK.intersection(text.upper().split()) for text in heard) or any(
                qa.policy.greedy_full_wake(value['logits'], value['vocab']) for value in phones[sid].values())
            tier = 'silver' if all_obs and not veto and votes == 3 else 'weak' if all_obs and not veto and votes >= 2 else 'unknown'
            decisions.append(dict(row, observed_class=0 if tier != 'unknown' else None, tier=tier,
                reason='BLIND_ORDINARY_NO_WAKE' if tier != 'unknown' else 'UNKNOWN_NATURAL_SOURCE',
                exact_votes=votes, full_wake_veto=veto, native_text=texts[sid],
                phone_sha256={p: value['sha256'] for p, value in phones[sid].items()}))
    synthetic = [r for r in decisions if r['cohort'] == 'synthetic']
    admitted_natural = [r for r in decisions if r['cohort'] == 'natural' and r['observed_class'] == 0]
    paired = sorted(group for group, classes in accepted.items() if classes == {0, 2})
    measured = dict(synthetic_positive=sum(r['observed_class'] == 2 for r in synthetic),
        synthetic_ordinary=sum(r['observed_class'] == 0 for r in synthetic), synthetic_paired=len(paired),
        natural_originals=len(admitted_natural), natural_speakers=len({r['speaker'] for r in admitted_natural}))
    minima = lineage.read_plan()['source_gate']
    missing = {k: {'observed': measured[k], 'min': v} for k, v in minima.items() if measured[k] < v}
    partition, attrition = attrition_document(all_rows, rows, decisions, observation)
    return dict(schema=BATCH, panel=PANEL, **source_binding(), source_execution=meta['source_execution'],
        source_manifest_sha256=qa.sha(source / 'MANIFEST.json'), rows=decisions, coverage=coverage,
        source_rows=len(rows), selected_original_slots=120, original_quality_rejections=120-len(rows),
        source_coverage=measured, frozen_minima=minima, source_shortfall=missing, paired_source_groups=paired,
        unknown_count=sum(r['observed_class'] is None for r in decisions), tier_counts=dict(Counter(r['tier'] for r in decisions)),
        source_gate='INSUFFICIENT_SOURCE' if missing else 'SOURCE_GATE_PASS',
        parser_contract_sha256=code_contract()['parser_contract_sha256'], observer_code_contract=code_contract(),
        strict_raw_receipt_readback=True, new_model_forwards=0, optimizer_updates=0, human_gold=False,
        source_policy=source_policy(), completed_coverage=observation['completed_coverage'],
        invalid_empty_coverage=observation['invalid_empty_coverage'], quarantined_ids=observation['quarantined_ids'],
        quarantine=observation['quarantine'], native_slot_accounting=observation['native_slot_accounting'],
        partition=partition, attrition=attrition, orchestration_changed=True,
        qualification_scope='five-valid-admissible-originals-after-complete-fixed-panel-empty-quarantine')


def verify_screen_contract(source, document, providers):
    actual = screen_document(source, providers)
    qa.need(quarantine.equal(actual, document) and actual['source_gate'] == 'SOURCE_GATE_PASS',
            'EXACT_COMPLETE_RAW_Q1_READBACK_QUARANTINE_AND_UNCHANGED_SOURCE_MINIMA')
    return actual


def validate_config(path):
    B13.b12.verify_config(path)
    qa.need(qa.sha(Path(path)) == piper_metadata.INPUTS['piper-config.json'],
            'same complete original Piper map frozen before all B24Q1 source draws')
    mapping = json.loads(Path(path).read_bytes())['speaker_id_map']
    assigned, excluded = set(), set(lineage.read_ancestry()['synthetic_excluded_people'])
    for role in ('primary', 'replication'):
        for index, person in lineage.read_plan()[role]['synthetic']['speaker_id_map'].items():
            canonical = piper_metadata.canonical(person)
            qa.need(mapping.get(person) == int(index) and canonical not in excluded
                    and canonical not in assigned, 'fresh canonical voice; every prior reserve stays excluded')
            assigned.add(canonical)
    qa.need(len(assigned) == 48, 'all48 new independently frozen synthetic identities')


def check():
    lineage.read_plan(); source_plan(); source_plan('replication'); code_contract()
    validate_config(PIPER_CONFIG)
    lineage.training_binding(False)
    lineage.failure.read_binding(False)
    print('PASS: B24Q1 frozen sources; no TTS, decode, observer or KWS calls')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    modes.add_parser('check')
    item = modes.add_parser('guard'); item.add_argument('prepared_parent')
    item = modes.add_parser('authorize'); item.add_argument('out', type=Path); item.add_argument('prepared_parent')
    item = modes.add_parser('verify-corpus'); item.add_argument('corpus', type=Path)
    for mode in ('transport', 'consume'):
        item = modes.add_parser(mode); item.add_argument('corpus', type=Path); item.add_argument('archives', type=Path); item.add_argument('prepared_parent')
    item = modes.add_parser('verify-source'); item.add_argument('source', type=Path)
    item = modes.add_parser('generate'); item.add_argument('out', type=Path); item.add_argument('archives', type=Path)
    item.add_argument('corpus', type=Path); item.add_argument('prepared_parent')
    for mode in ('consume-observer', 'observe'):
        item = modes.add_parser(mode); item.add_argument('provider', choices=observers.PROVIDERS)
        item.add_argument('source', type=Path); item.add_argument('out', type=Path); item.add_argument('prepared_parent')
    item = modes.add_parser('screen'); item.add_argument('source', type=Path); item.add_argument('providers', type=Path); item.add_argument('out', type=Path)
    args = parser.parse_args()
    # One command owns the audit scope. Nested source/ancestry reads still
    # reopen and hash every byte, sharing only a complete original verification.
    from english_v6_b24q1_evidence_scope import scope
    with scope():
        if args.mode == 'check': check()
        elif args.mode == 'guard': lineage.execution_guard(args.prepared_parent)
        elif args.mode == 'authorize': lineage.authorize(args.out, args.prepared_parent)
        elif args.mode == 'verify-corpus': natural.verify_corpus(args.corpus)
        elif args.mode == 'transport':
            from english_v6_b24q1_source_transport import download
            download(args.corpus, args.archives, args.prepared_parent)
        elif args.mode == 'consume': lineage.consumption(args.archives, args.corpus, args.prepared_parent)
        elif args.mode == 'verify-source': registration(args.source)
        elif args.mode == 'generate': generate(args.out, args.archives, args.corpus, args.prepared_parent)
        elif args.mode == 'consume-observer': lineage.consume_observer(args.provider, args.source, args.out, args.prepared_parent)
        elif args.mode == 'observe': observe(args.provider, args.source, args.out, args.prepared_parent)
        else: screen(args.source, args.providers, args.out)
