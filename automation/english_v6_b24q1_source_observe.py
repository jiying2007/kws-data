"""B24Q1 fixed native calls with explicit, independently proved empty quarantine.

Only orchestration changes. Native loaders, valid-payload parsers, finite phone
matrices and scientific admission primitives are the frozen original functions.
Every pre-observer eligible slot is attempted once. A proved empty stays invalid;
every other failure still stops the panel. Saved readback never calls a model.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_v6_b22_source_observe as original
from english_v6_b22_replication_reuse import inherit
inherit(original, globals())
import english_v6_b24q1_source_ancestry as lineage
import english_v6_b24q1_quarantine as quarantine

SOURCE_SCHEMA = 'english-v6-b24q1-fresh-combined-source-v1'
BATCH = 'english-v6-b24q1-120-original-quarantine-source-screen-v1'
PANEL = 'primary'
POLICY_KEYS = ('observer_sha256', 'asr_adapter_sha256', 'parser_contract_sha256',
    'proposal_sha256', 'approval_sha256', 'orchestration_contract_sha256',
    'quarantine_helper_sha256', 'orchestration_implementation_sha256')
row_ledger, ledger_counts = quarantine.row_ledger, quarantine.ledger_counts
complete_ledger = quarantine.complete_ledger


def source_check(repository=None):
    frozen = primary_observer.source_check(repository)
    scientific = {k: v for k, v in frozen.items() if k not in ('schema', 'observer_sha256', 'source_indices')}
    need(quarantine.equal(scientific, lineage.read_plan()['observer_scientific_contract']),
         'UNCHANGED_NATIVE_VALID_PARSER_AND_ADMISSION_PRIMITIVES')
    quarantine.orchestration_contract()
    return dict(scientific, schema='english-v6-b24q1-five-observer-contract-v1', panel=PANEL,
        observer_sha256=sha(Path(__file__).read_bytes()),
        source_indices=lineage.read_plan()[PANEL]['synthetic']['speaker_indices'],
        orchestration_changed=True, successful_empty_remains_invalid=True,
        proposal_sha256=quarantine.PROPOSAL_SHA, approval_sha256=lineage.metadata.APPROVAL_SHA,
        orchestration_contract_sha256=quarantine.ORCHESTRATION_SHA,
        quarantine_helper_sha256=sha(Path(quarantine.__file__).read_bytes()),
        orchestration_implementation_sha256=sha((HERE / 'english_v6_b24q1_source_observe.py').read_bytes()))


def guard(registration, *, env=None, event=None, parent=None):
    lineage.read_plan()
    need(type(registration) is dict, 'ACTUAL_NEW_Q1_REGISTRATION_REQUIRED')
    identity = lineage.execution_guard(registration['prepared_parent'], env, event, parent)
    contract = source_check()
    need(all(quarantine.equal(registration[k], value) for k, value in identity.items())
         and all(quarantine.equal(registration[k], contract[k]) for k in POLICY_KEYS)
         and registration['assessment_sha256'] == lineage.ASSESSMENT_SHA
         and registration['source_schema'] == SOURCE_SCHEMA and registration['batch_schema'] == BATCH
         and type(registration['source_rows']) is int and type(registration['max_calls']) is int
         and 96 <= registration['source_rows'] == registration['max_calls'] <= MAX_COUNT,
         'FIRST_Q1_RUN_EXACT_SOURCE_AND_NEW_ORCHESTRATION_BEFORE_NATIVE_LOAD')
    return {'run': identity['run_id'], 'sha': identity['run_head']}


def validate_rows(rows, source_sha, registration):
    count = registration['source_rows']
    need(type(count) is int and source_sha == registration['source_manifest_sha256']
         and 96 <= count <= MAX_COUNT
         and len(rows) == len({r['id'] for r in rows}) == len({r['pcm_sha256'] for r in rows}) == count
         and all(Path(r['id']).name == r['id'] and r['id'].startswith('b24q1-' + PANEL + '-')
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


def expected_registration(source, supplied=None):
    lineage.read_plan()
    import english_v6_b24q1_source as owner
    expected = owner.registration(Path(source))
    need(supplied is None or quarantine.equal(supplied, expected), 'SOURCE_AUTHORIZED_REGISTRATION')
    return expected


def require_original_rows(source, rows):
    import english_v6_b24q1_source as owner
    need(quarantine.equal(rows, owner.read_source(Path(source))), 'ALL_COMPLETE_ORIGINAL_ROWS_IN_REGISTERED_ORDER')


def started_document(provider, schema, run, source_sha, registration, count):
    return dict(run=run, batch=schema, provider=provider, input_manifest=source_sha,
        registration=registration, parser_contract_sha256=raw_asr.PARSER_CONTRACT_SHA256,
        old_admission_function_sha256=POLICY_FUNCTIONS, denominator=count,
        orchestration_contract_sha256=quarantine.ORCHESTRATION_SHA,
        successful_empty_disposition=quarantine.EMPTY_STATE,
        whole_panel_stop_on_invalid_empty=False, whole_panel_stop_on_other_failure=True,
        continue_all_preplanned_slots_after_proven_empty=True)


def report_document(provider, schema, run, source_sha, rows, captures, ledger,
                    initialization_error=None, row_error=None):
    count, counts = len(rows), ledger_counts(ledger)
    complete = complete_ledger(ledger, count) and initialization_error is None and row_error is None
    contract = source_check()
    return dict(schema=schema, provider=provider, run=run, source_manifest_sha256=source_sha,
        denominator=count, attempted=counts['forward_attempted'],
        actual_native_forward_attempts=counts['forward_attempted'], native_forward_returns=counts['native_returned'],
        valid_native_payloads=counts['payload_valid'], invalid_empty_native_payloads=counts['invalid_empty'],
        completed_native_slots=counts['completed_native_slots'], complete=counts['capture_saved'],
        missing_or_failed=count-counts['completed_native_slots'],
        status='COMPLETE_NATIVE_SLOTS' if complete else 'STOP_OBSERVER_FAILURE', whole_panel_stop=not complete,
        initialization_error=initialization_error, row_error=row_error,
        rows=[captures.get(r['id'], dict(binding(provider, r, source_sha), status='not_started')) for r in rows],
        new_tts_calls=0, new_asr_calls=counts['forward_attempted'] if provider in raw_asr.TEXT_PROVIDERS else 0,
        new_phone_calls=counts['forward_attempted'] if provider in ('xlsr', 'lv60') else 0,
        target_conditioned=False, human_gold=False, training_allowed=False, retry_count=0,
        kws_forward_calls=0, promotion_allowed=False, parser_contract_sha256=raw_asr.PARSER_CONTRACT_SHA256,
        observer_code_sha256=contract['observer_sha256'], asr_adapter_sha256=contract['asr_adapter_sha256'],
        source_orchestration={key: contract[key] for key in POLICY_KEYS if key not in
                             ('observer_sha256', 'asr_adapter_sha256', 'parser_contract_sha256')})


def observe(provider, source, out, *, read_rows, schema, registration):
    lineage.read_plan()
    need(provider in PROVIDERS and schema == BATCH, 'ONLY_REGISTERED_Q1_FIVE_OBSERVER_SOURCE')
    run = guard(registration)
    budget = lineage.require_observer_permit(provider, registration)
    source, root = Path(source), Path(out)
    rows = read_rows(source)
    source_sha = sha((source / 'MANIFEST.json').read_bytes())
    registration = expected_registration(source, registration)
    require_original_rows(source, rows)
    validate_rows(rows, source_sha, registration)
    contract = source_check()
    need(root.is_absolute() and not root.exists() and root.parent.is_dir()
         and not any(p.is_symlink() for p in (root, *root.parents)), 'FRESH_OBSERVER_OUTPUT_ONLY')
    root.mkdir(mode=0o700)
    record(root / 'OBSERVER_BUDGET.json', budget)
    record(root / 'STARTED.json', started_document(provider, schema, run, source_sha, registration, len(rows)))
    captures, initialization_error, row_error = {}, None, None
    try:
        import english_hifi_micro as c
        record(root / 'MODEL_LOAD_STARTED.json', {'provider': provider, 'actual_forward_attempts': 0})
        if provider in ('xlsr', 'lv60'):
            backend, lock = _load_phone(provider, root)
        else:
            assets = Path(os.environ.get('RUNNER_TEMP', '/tmp')) / ('b22-asr-' + provider)
            backend, lock = raw_asr._load(provider, assets, root)
        record(root / 'MODEL.json', lock)
    except BaseException as exc:
        initialization_error = type(exc).__name__ + ': ' + str(exc)[:500]
    if initialization_error is None:
        for row in rows:
            directory = root / 'rows' / row['id']
            directory.mkdir(parents=True, exist_ok=False)
            start = binding(provider, row, source_sha)
            try:
                record(directory / 'STARTED.json', start)
                wav = source / 'wav' / (row['id'] + '.wav')
                measured = c.inspect_wav(wav)
                need(all(measured[k] == row[k] for k in ('wav_sha256', 'pcm_sha256', 'samples')),
                     'UNCHANGED_FULL_ORIGINAL_BEFORE_NATIVE_CALL')
                captured = backend(wav, directory, start)
                result = {**start, **captured, 'status': 'success', 'input_coverage_complete': True}
                record(directory / 'CAPTURE.json', result)
                captures[row['id']] = result
            except BaseException as exc:
                # The error name is only a necessary routing condition. Independent
                # original bytes, actual native success and whole input prove empty.
                proved_empty = None
                if provider in raw_asr.TEXT_PROVIDERS and type(exc) is ValueError and str(exc) == 'EMPTY_NATIVE_TRANSCRIPT':
                    try:
                        proved_empty = quarantine.empty_receipt(source, root, row, start, contract)
                    except BaseException:
                        proved_empty = None
                if proved_empty is not None:
                    record(directory / 'INVALID_EMPTY.json', proved_empty)
                    captures[row['id']] = proved_empty
                    continue  # All remaining fixed slots run once, including this row in other providers.
                row_error = type(exc).__name__ + ': ' + str(exc)[:500]
                failed = dict(start, status='failed', error=row_error, retry_count=0, whole_panel_stop=True)
                record(directory / 'FAILED.json', failed)
                captures[row['id']] = failed
                break
    ledger_doc = quarantine.ledger_document(root, rows, provider, source_sha)
    record(root / 'LEDGER.json', ledger_doc)
    summary = report_document(provider, schema, run, source_sha, rows, captures,
                              ledger_doc['rows'], initialization_error, row_error)
    record(root / 'RESULT.json', summary)
    record(root / 'FILES.json', {'schema': 'english-v6-b18-observer-file-closure-v1',
                                 'self_excluded': 'FILES.json', 'files': file_closure(root)})
    need(summary['status'] == 'COMPLETE_NATIVE_SLOTS', 'WHOLE_PANEL_STOP_OTHER_NATIVE_FAILURE_NO_RETRY')
    return summary


def read_valid(source, root, row, start, contract):
    """Reopen the unchanged original valid payload, including full phone tensors."""
    directory, provider = root / 'rows' / row['id'], start['provider']
    quarantine.complete_input(source, row, start)
    need(quarantine.equal(quarantine.document(directory / 'STARTED.json'), start), 'EXACT_NATIVE_INPUT_START')
    need(not any((directory / name).exists() for name in ('INVALID_EMPTY.json', 'FAILED.json')),
         'VALID_EMPTY_FATAL_EXCLUSIVE')
    capture = quarantine.document(directory / 'CAPTURE.json')
    need(capture['status'] == 'success' and capture['input_coverage_complete'] is True
         and all(quarantine.equal(capture.get(k), v) for k, v in start.items()), 'COMPLETE_ORIGINAL_CAPTURE_IDENTITY')
    # Strictly decode every JSON receipt before the unchanged old reader sees it.
    for path in directory.glob('*.json'):
        quarantine.document(path)
    model = quarantine.document(root / 'MODEL.json')
    if provider in raw_asr.TEXT_PROVIDERS:
        quarantine.validate_text_model(provider, model, contract)
        value = raw_asr.validate_capture(provider, directory, start, capture)
        files = {'STARTED.json', 'FORWARD_STARTED.json', 'FORWARD_RETURNED.json', 'PAYLOAD_VALID.json', 'CAPTURE.json'}
        files.update(('RAW_PROCESS.json', 'NATIVE_COMMAND.json', 'RAW_STDOUT.bin', 'RAW_STDERR.bin')
                     if provider == 'parakeet' else ('RAW_RETURN.json', 'RAW_TEXT.utf8'))
    else:
        import english_v6_primary_source_qa as qa
        import english_hifi_phones_once as phones
        _verify_phone(root, directory, start, capture, model)
        index = qa.policy.PHONE_MODELS.index(provider)
        name, revision = qa.pcm_tools.PHONE_MODELS[index]
        need(model['target_conditioned'] is False and model['model'] == name
             and model['revision'].startswith(revision) and model['weights_sha256'] == phones.WEIGHT_SHA[index],
             'LOCKED_UNCHANGED_PHONE_SOURCE')
        vocab = quarantine.document(root / 'VOCAB.json')
        need(vocab['model'] == name and vocab['revision'] == model['revision']
             and type(vocab['pad_token_id']) is int and vocab['pad_token_id'] == 0, 'NATIVE_VOCAB_LOCK')
        raw = (root / 'logits' / (row['id'] + '.f32')).read_bytes()
        need(type(capture['logit_frames']) is int and capture['logit_frames'] > 0
             and type(capture['vocab_size']) is int and capture['vocab_size'] == 392
             and len(raw) == 4 * capture['logit_frames'] * capture['vocab_size']
             and sha(raw) == capture['logits_sha256'], 'ALL_ORIGINAL_PHONE_LOGITS')
        qa.policy.verified_phone_input(raw, vocab['vocab'], pad_token_id=0)
        ids = [max(range(392), key=frame.__getitem__) for frame in struct.iter_unpack('<392f', raw)]
        need(phones.ctc(ids, vocab['vocab'], 0) == capture['raw_phone_tokens'], 'SAVED_CTC_PARITY')
        value = {'logits': raw, 'vocab': vocab['vocab'], 'sha256': capture['logits_sha256']}
        files = {'STARTED.json', 'FORWARD_STARTED.json', 'FORWARD_RETURNED.json', 'RAW_PHONE.json',
                 'PAYLOAD_VALID.json', 'CAPTURE.json'}
    need({path.name for path in directory.iterdir()} == files
         and all(path.is_file() and not path.is_symlink() for path in directory.iterdir()),
         'EXACT_VALID_ROW_FILE_CLOSURE')
    return capture, value


def load_observations(source, evidence, rows, schema, *, registration=None):
    """Authenticate all 5*N original slots before computing the empty-ID union."""
    lineage.read_plan()
    source, evidence = Path(source), Path(evidence)
    contract = source_check()
    registration = expected_registration(source, registration)
    source_sha = sha((source / 'MANIFEST.json').read_bytes())
    require_original_rows(source, rows)
    validate_rows(rows, source_sha, registration)
    count = len(rows)
    need(schema == BATCH and all(quarantine.equal(registration[k], contract[k]) for k in POLICY_KEYS),
         'EXACT_NEW_SOURCE_ORCHESTRATION_REGISTRATION')
    run = {'run': registration['run_id'], 'sha': registration['run_head']}
    texts, phones = {r['id']: {} for r in rows}, {r['id']: {} for r in rows}
    coverage, empty_coverage, slots, empty_proofs = {}, {}, {}, {r['id']: {} for r in rows}
    for provider in PROVIDERS:
        root = evidence / provider
        lineage.validate_observer_budget(quarantine.document(root / 'OBSERVER_BUDGET.json'), provider, registration)
        need(quarantine.equal(quarantine.document(root / 'FILES.json'), {
            'schema': 'english-v6-b18-observer-file-closure-v1', 'self_excluded': 'FILES.json',
            'files': file_closure(root)}), 'COMPLETE_RAW_OBSERVER_FILE_CLOSURE')
        need(quarantine.equal(quarantine.document(root / 'STARTED.json'),
            started_document(provider, schema, run, source_sha, registration, count)), 'EXACT_PRECALL_POLICY_AND_SOURCE')
        need(quarantine.equal(quarantine.document(root / 'MODEL_LOAD_STARTED.json'),
            {'provider': provider, 'actual_forward_attempts': 0}), 'ONE_NATIVE_INITIALIZATION')
        top = {'OBSERVER_BUDGET.json', 'STARTED.json', 'MODEL_LOAD_STARTED.json', 'MODEL.json',
               'rows', 'LEDGER.json', 'RESULT.json', 'FILES.json'}
        if provider in ('xlsr', 'lv60'):
            top.update(('VOCAB.json', 'logits'))
            need({p.name for p in (root / 'logits').iterdir()} == {r['id'] + '.f32' for r in rows},
                 'EVERY_FIXED_PHONE_MATRIX_AND_NO_EXTRA')
        need({p.name for p in root.iterdir()} == top
             and {p.name for p in (root / 'rows').iterdir()} == {r['id'] for r in rows}
             and all(p.is_dir() and not p.is_symlink() for p in (root / 'rows').iterdir()),
             'ONLY_FIXED_PLANNED_NATIVE_SLOTS')
        ledger_doc = quarantine.ledger_document(root, rows, provider, source_sha)
        need(quarantine.equal(quarantine.document(root / 'LEDGER.json'), ledger_doc)
             and complete_ledger(ledger_doc['rows'], count), 'ALL_FIXED_NATIVE_ATTEMPTS_RETURNS_AND_DISPOSITIONS')
        captures = {}
        for row, ledger_row in zip(rows, ledger_doc['rows'], strict=True):
            start = binding(provider, row, source_sha)
            if ledger_row['state'] == quarantine.EMPTY_STATE:
                value = quarantine.verify_empty(source, root, row, start, contract)
                empty_proofs[row['id']][provider] = value
                captures[row['id']] = value
            else:
                capture, value = read_valid(source, root, row, start, contract)
                captures[row['id']] = capture
                (texts if provider in raw_asr.TEXT_PROVIDERS else phones)[row['id']][provider] = value
        expected = report_document(provider, schema, run, source_sha, rows, captures, ledger_doc['rows'])
        need(quarantine.equal(quarantine.document(root / 'RESULT.json'), expected)
             and expected['status'] == 'COMPLETE_NATIVE_SLOTS', 'EXACT_COMPLETE_REPORT_FROM_ALL_ORIGINAL_RAW_RETURNS')
        slots[provider] = quarantine.slot_counts(ledger_doc['rows'])
        coverage[provider] = slots[provider]['valid']
        empty_coverage[provider] = slots[provider]['invalid_empty']
    # Deliberately after every provider's full raw authentication; an empty never
    # conceals a fatal/missing phone or text return on that same original.
    quarantined = {sid: value for sid, value in empty_proofs.items() if value}
    total = {key: sum(value[key] for value in slots.values()) for key in next(iter(slots.values()))}
    need(total['planned'] == total['attempted'] == total['returned'] == 5 * count <= 600
         and total['valid'] + total['invalid_empty'] == 5 * count
         and total['fatal'] == total['unknown'] == 0, 'COMPLETE_5_TIMES_N_NATIVE_DENOMINATOR')
    return dict(texts=texts, phones=phones, coverage=coverage,
        completed_coverage=dict.fromkeys(PROVIDERS, count), invalid_empty_coverage=empty_coverage,
        quarantined_ids=sorted(quarantined), quarantine=quarantined,
        native_slot_accounting={'per_provider': slots, 'total': total, 'maximum_per_panel': 600})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-check', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(source_check(), indent=2, sort_keys=True))
