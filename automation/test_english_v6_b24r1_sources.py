"""Metadata, closure and rejection contracts; no acoustic execution or network."""
from __future__ import annotations
import ast
import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_v6_b22_source as previous_source
import english_v6_b22_source_observe as previous_observe
import english_v6_b22_source_transport as previous_transport
import english_v6_b22_source_dynamic_preflight as dynamic
import english_v6_b24r1_source as source
import english_v6_b24r1_source_ancestry as lineage
import english_v6_b24r1_source_observe as observe
import english_v6_b24r1_source_transport as transport
import english_v6_b24r1_source_producer as producer
import english_v6_b24_source_training_proof as proof
import english_v6_b24_source_ancestry as original_train
import english_v6_b24r1_b24_source_failure as failure
import english_v6_b24r1_natural_sources as natural


def identity():
    return dict(run_id='123', run_head='b' * 40, prepared_parent='a' * 40,
        run_branch=lineage.BRANCH, run_attempt=1)


def registration():
    return dict(identity(), source_rows=120, max_calls=120, source_manifest_sha256='c' * 64)


class Sources(unittest.TestCase):
    def test_scientific_source_and_observer_functions_are_exact_original_code_objects(self):
        for name in ('generate', 'verify_synthetic', '_read', 'source_plan', 'read_source',
                     'all_original_rows', 'evaluation_preflight', 'registration', 'observe',
                     'screen_document', 'verify_screen_contract'):
            self.assertIs(getattr(source, name).__code__, getattr(previous_source, name).__code__, name)
            self.assertIs(getattr(source, name).__globals__['lineage'], lineage)
        for name in ('guard', 'observe', 'load_observations'):
            self.assertIs(getattr(observe, name).__code__, getattr(previous_observe, name).__code__, name)
            self.assertIs(getattr(observe, name).__globals__['lineage'], lineage)
        self.assertIs(transport.download.__code__, previous_transport.download.__code__)

    def test_natural_decoder_and_original_rejection_change_only_the_id_prefix(self):
        before = ast.parse((HERE / 'english_v6_b22_natural_sources.py').read_text().replace('b22-primary-natural-', 'b24r1-primary-natural-'))
        after = ast.parse((HERE / 'english_v6_b24r1_natural_sources.py').read_text())
        for name in ('extract', 'read_source'):
            old = next(n for n in before.body if isinstance(n, ast.FunctionDef) and n.name == name)
            new = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(old, include_attributes=False), ast.dump(new, include_attributes=False))

    def test_frozen_new_panels_and_original_five_observer_policy(self):
        plan = lineage.read_plan()
        self.assertFalse(plan['source_only_generation_before_model_training_completion_allowed'])
        self.assertEqual(plan['source_gate'], previous_source.lineage.read_plan()['source_gate'])
        contract = source.code_contract()
        self.assertEqual({k: v for k, v in contract.items() if k not in ('schema', 'observer_sha256', 'source_indices')},
                         plan['observer_scientific_contract'])
        self.assertEqual(len(source.source_plan()['rows']), 96)
        self.assertEqual(source.source_plan()['schema'], source.SCHEMA)
        source.validate_config(source.PIPER_CONFIG)

    def test_exact_metadata_only_24_natural_requests(self):
        full = json.loads((source.piper_metadata.PANELS / 'FULL_METADATA_CATALOG.json').read_bytes())
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w') as archive:
            records = [('LibriSpeech/SPEAKERS.TXT', full['speakers_text'])]
            records.extend((row['path'], row['text']) for row in full['transcripts'])
            for name, text in records:
                raw = text.encode()
                member = tarfile.TarInfo(name)
                member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))
        stream.seek(0)
        with tarfile.open(fileobj=stream, mode='r|') as archive:
            rows, catalog = natural.pick(archive, natural.plan())
        self.assertEqual(rows, natural.plan()['roster'])
        self.assertEqual([row['source_id'] for row in rows], natural.plan()['selected_utterance_ids'])
        changed = copy.deepcopy(catalog)
        changed['transcripts'][0]['text'] += '\n'
        with self.assertRaises(ValueError):
            natural.rank_metadata(changed, natural.plan())

    def test_new_voice_roster_rejects_duplicate_old_prefix_or_substitution(self):
        rows = [dict(row, pcm_sha256=hashlib.sha256(row['id'].encode()).hexdigest()) for row in source.source_plan()['rows']]
        reg = dict(registration(), source_rows=96, max_calls=96)
        observe.validate_rows(rows, reg['source_manifest_sha256'], reg)
        changes = (dict(rows[0], id=rows[0]['id'].replace('b24r1-', 'b24-')),
            rows[1], dict(rows[0], speaker_index=-1), dict(rows[0], requested_class=0),
            dict(rows[0], observed_class=2))
        for replacement in changes:
            with self.subTest(replacement=replacement['id']), self.assertRaises(ValueError):
                altered = copy.deepcopy(rows)
                altered[0] = replacement
                observe.validate_rows(altered, reg['source_manifest_sha256'], reg)

    def test_first_public_event_rejects_retry_force_other_branch_or_private(self):
        ident = identity()
        env = dict(GITHUB_ACTIONS='true', GITHUB_REPOSITORY=lineage.REPO, GITHUB_EVENT_NAME='push',
            GITHUB_REF=lineage.BRANCH, GITHUB_RUN_ATTEMPT='1', GITHUB_SHA=ident['run_head'], GITHUB_RUN_ID='123')
        event = dict(before=ident['prepared_parent'], after=ident['run_head'], ref=lineage.BRANCH,
            forced=False, deleted=False, repository=dict(full_name=lineage.REPO, private=False))
        with patch.object(lineage, 'read_plan'), patch.object(failure, 'read_binding'), patch.object(lineage, 'training_binding'):
            self.assertEqual(lineage.execution_guard(ident['prepared_parent'], env, event, ident['prepared_parent']), ident)
            for altered in (dict(env, GITHUB_RUN_ATTEMPT='2'), dict(env, GITHUB_REF='refs/heads/main')):
                with self.assertRaises(ValueError):
                    lineage.execution_guard(ident['prepared_parent'], altered, event, ident['prepared_parent'])
            for altered in (dict(event, forced=True), dict(event, repository=dict(full_name=lineage.REPO, private=True))):
                with self.assertRaises(ValueError):
                    lineage.execution_guard(ident['prepared_parent'], env, altered, ident['prepared_parent'])

    def test_durable_marker_rejects_wrong_provider_run_and_old_namespace(self):
        reg = registration()
        expected = lineage.observer_document('qwen', reg)
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr('OBSERVER_CONSUMPTION.json', json.dumps(expected))
        raw = stream.getvalue()
        artifact = dict(id=1, name=expected['marker_name'], expired=False, size_in_bytes=len(raw),
            digest='sha256:' + lineage.digest(raw), workflow_run=dict(id=123, head_sha=identity()['run_head']))
        receipt = dict(artifact=artifact, original_zip_base64=base64.b64encode(raw).decode(), receipt=expected)
        self.assertEqual(lineage.validate_observer_budget(receipt, 'qwen', reg), expected)
        with self.assertRaises(ValueError):
            lineage.validate_observer_budget(receipt, 'whisper', reg)
        for changed in (dict(artifact, name=artifact['name'].replace('b24r1', 'b24')),
                        dict(artifact, workflow_run=dict(id=124, head_sha=identity()['run_head']))):
            with self.assertRaises(ValueError):
                lineage.validate_observer_budget(dict(receipt, artifact=changed), 'qwen', reg)

    def test_complete_source_capture_includes_all19_helpers_and_excludes_future_b24(self):
        tracked = subprocess.check_output(['git', '-C', str(HERE.parent), 'ls-files', '--cached', '--others', '--exclude-standard'], text=True).splitlines()
        paths = producer.select_paths(tracked)
        self.assertTrue(set(dynamic.required_paths()) <= paths)
        original = json.loads((HERE / 'english_v6_b23_source_training_producer.json').read_bytes())
        self.assertTrue(set(original['files']) <= paths)
        self.assertIn('automation/english_v6_b23_quality_readback.py', paths)
        self.assertIn('automation/english_v6_b23_primary_inputs.json', paths)
        self.assertFalse(any(name.startswith(producer.FUTURE_PREFIXES) for name in paths))
        self.assertNotIn('automation/bind_b24r1_primary.py', paths)
        self.assertNotIn('automation/test_bind_b24r1_primary.py', paths)
        for missing in (dynamic.CAMPAIGN, dynamic.RUNNER):
            with self.assertRaises(ValueError):
                producer.select_paths([name for name in tracked if name != missing])

    def test_actual_isolated_all19_helper_routes_before_asset_boundary(self):
        raw = subprocess.check_output([sys.executable, '-SIB', str(HERE / 'english_v6_b22_source_dynamic_preflight.py')])
        report = json.loads(raw)
        self.assertEqual(report['status'], 'PASS_IMPORT_LOCK_RUNNER_ONLY')
        self.assertEqual(len(report['helper_pins']), 19)
        self.assertTrue(report['code_reads_only'] and report['no_model'] and report['no_network'])
        self.assertFalse(report['model_construction_verified'] or report['acoustic_evidence'])

    def test_selected_complete_producer_runs_source_check_without_checkout_fallback(self):
        tracked = subprocess.check_output(['git', '-C', str(HERE.parent), 'ls-files', '--cached', '--others', '--exclude-standard'], text=True).splitlines()
        paths = producer.select_paths(tracked)
        with tempfile.TemporaryDirectory(prefix='b24r1-isolated-source-producer-') as temporary:
            root = Path(temporary) / 'repository'
            for name in paths:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(HERE.parent / name, target)
            process = subprocess.run([sys.executable, '-SIB', str(root / 'automation/english_v6_b24r1_source.py'), 'check'],
                cwd=temporary, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn('PASS: B24R1 frozen sources', process.stdout)

    def test_inert_template_only_executes_from_capture_and_markers_precede_models(self):
        template = (HERE.parent / producer.TEMPLATE).read_bytes()
        actual = (HERE.parent / lineage.WORKFLOW).read_bytes()
        if os.environ.get('GITHUB_REF') == lineage.BRANCH and os.environ.get('PREPARED_PARENT'):
            self.assertEqual(actual, template.replace(producer.TOKEN, os.environ['PREPARED_PARENT'].encode()))
        else:
            self.assertEqual(actual, template)
        self.assertEqual(template.count(producer.TOKEN), 2)
        text = template.decode()
        self.assertNotIn('workflow_dispatch', text)
        self.assertIn('research/en-v6-b24r1-primary-sources-once-20261124', text)
        self.assertNotIn('refs/heads/research/en-v6-b24-primary-sources-once-20261124', text)
        self.assertEqual(lineage.TRAIN_BRANCH, 'research/en-v6-b24-b23-warm-start-train-once-20261124')
        self.assertIn('provider: [qwen, parakeet, xlsr, lv60, whisper]', text)
        self.assertEqual(text.count('include-hidden-files: true'), text.count('retention-days: 90'))
        for line in text.splitlines():
            if 'source.py' in line and ('python' in line or 'bin/python' in line):
                self.assertIn('$RUNNER_TEMP/b24r1-source-producer/repository/automation/english_v6_b24r1_source.py', line)
        self.assertLess(text.index('name: english-v6-b24r1-primary-source-consumed'), text.index('source.py" generate'))
        self.assertLess(text.index('Check all19 captured dynamic helpers'), text.index("Consume this provider's"))
        self.assertLess(text.index('name: english-v6-b24r1-primary-observer-${{ matrix.provider }}-consumed'),
                        text.index('Install one pinned native observer runtime'))
        self.assertIn('name: b24r1-primary-dev-observer-${{ matrix.provider }}', text)

    def test_unbound_actual_training_has_no_fabricated_pins(self):
        inert = dict(schema='english-v6-b24-completed-train-source-binding-v1', assessment_sha256=original_train.ASSESSMENT_SHA,
                     bound=False, training=None, training_producer_manifest_sha256=None)
        with patch.object(original_train, 'INPUTS') as path:
            path.read_bytes.return_value = json.dumps(inert).encode()
            self.assertFalse(lineage.training_binding(False)['bound'])
            with self.assertRaises(ValueError):
                lineage.training_binding()
            path.read_bytes.return_value = json.dumps(dict(inert, training_producer_manifest_sha256='a' * 64)).encode()
            with self.assertRaises(ValueError):
                lineage.training_binding(False)

    def test_current_layer_does_not_claim_repeated_old_source_admission(self):
        self.assertIs(proof.SCOPE['old_B23_input_archives_reopened_by_source'], False)
        self.assertEqual(proof.SCOPE['original_B23_complete_source_admission_authority'], 'actual_original_B24_TRAIN_entry')
        with patch.object(lineage, 'training_binding_sha', return_value='a' * 64), \
                patch.object(lineage, 'transport_plan_sha', return_value='b' * 64), \
                patch.object(lineage, 'stop_scope', return_value={}):
            binding = source.source_binding()
        self.assertTrue(all(binding[key] == value for key, value in proof.SCOPE.items()))
        self.assertFalse(binding['promotion_allowed'] or binding['training_allowed'] or binding['replication_opened'])

    def test_readback_proof_contains_no_network_model_or_subprocess_imports(self):
        tree = ast.parse(Path(proof.__file__).read_bytes())
        names = {item.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for item in node.names}
        names |= {node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        self.assertFalse(names & {'subprocess', 'socket', 'requests', 'torch', 'numpy', 'onnxruntime', 'soundfile'})
        self.assertFalse({'torch', 'numpy', 'onnxruntime', 'soundfile'} & set(sys.modules))


class OriginalStop(unittest.TestCase):
    def fixture(self):
        lock = {'fixture_only': True}
        train_result = 'd' * 64
        report = dict(schema=failure.REPORT_SCHEMA, status='FAILED_EMPTY_NATIVE_TRANSCRIPT',
            original_lock_sha256=failure.sha(failure.canonical(lock)),
            adapter_sha256=failure.sha(Path(failure.__file__).read_bytes()),
            candidate_pt_sha256=lineage.metadata.CANDIDATE_PT,
            original_assessment_sha256=lineage.metadata.TRAINING_ASSESSMENT_SHA,
            original_train_result_sha256=train_result,
            all14_original_archives_reopened_by_this_helper=True,
            original_source_stop_permanent=True, new_source_registration_performed=False,
            total_valid_payloads_reparsed=528, empty_return_reparsed_by_original_parser=True,
            **dict.fromkeys(failure.FALSE_PERMISSIONS, False))
        opening = dict(original_source_lock=lock,
            expected_failure_report_sha256=failure.sha(failure.canonical(report)))
        return opening, report, {'training': {'result_sha256': train_result}}

    def test_new_evaluation_never_rebinds_original_training_assessment(self):
        self.assertIs(lineage.training_binding, original_train.training_binding)
        self.assertIs(lineage.verify_training, original_train.verify_training)
        self.assertIs(lineage.validate_training_live, original_train.validate_training_live)
        self.assertIs(lineage.training_binding_sha, original_train.training_binding_sha)
        self.assertNotEqual(lineage.ASSESSMENT_SHA, original_train.ASSESSMENT_SHA)
        self.assertEqual(lineage.metadata.TRAINING_ASSESSMENT_SHA, original_train.ASSESSMENT_SHA)
        self.assertEqual(lineage.training_binding()['training']['model_pt_sha256'], lineage.metadata.FROZEN_MODELS)

    def test_stop_report_tamper_candidate_training_identity_or_permission_rejected(self):
        opening, report, training = self.fixture()
        with patch.object(lineage, 'training_binding', return_value=training):
            self.assertEqual(lineage.validate_stop_report(report, opening), report)
            for change in ({'candidate_pt_sha256': 'e' * 64},
                           {'original_assessment_sha256': lineage.ASSESSMENT_SHA},
                           {'original_train_result_sha256': 'f' * 64},
                           {'total_valid_payloads_reparsed': 529}, {'training_allowed': True},
                           {'all14_original_archives_reopened_by_this_helper': False}):
                altered = dict(report, **change)
                with self.assertRaises(ValueError):
                    lineage.validate_stop_report(altered, opening)
                # Even a malicious replacement of the expected report digest
                # cannot turn an altered semantic report into an eligible STOP.
                rebased = dict(opening, expected_failure_report_sha256=failure.sha(failure.canonical(altered)))
                with self.assertRaises(ValueError):
                    lineage.validate_stop_report(altered, rebased)

    def test_saved_report_is_recomputed_from_originals_in_isolated_stdlib(self):
        opening, report, training = self.fixture()
        raw = failure.canonical(report)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / lineage.STOP_REPORT).write_bytes(raw)
            (directory / lineage.STOP_BINDING).write_bytes(b'fixture')
            (directory / 'ORIGINAL_B24_STOP').mkdir()
            with patch.object(failure, 'BINDING') as binding, \
                    patch.object(failure, 'read_binding', return_value=opening), \
                    patch.object(lineage, 'training_binding', return_value=training), \
                    patch.object(lineage.subprocess, 'check_output', return_value=raw) as execute:
                binding.read_bytes.return_value = b'fixture'
                self.assertEqual(lineage.saved_stop(directory), (opening, report))
                command = execute.call_args.args[0]
                self.assertEqual(command[:4], [sys.executable, '-I', '-S', '-B'])
                self.assertEqual(Path(command[4]), Path(failure.__file__).resolve())
                self.assertEqual(command[5], '--verify-worker')
                self.assertEqual(Path(command[6]), directory / 'ORIGINAL_B24_STOP')
                execute.return_value = raw + b' '
                with self.assertRaises(ValueError):
                    lineage.saved_stop(directory)
                execute.side_effect = subprocess.CalledProcessError(1, command)
                with self.assertRaises(subprocess.CalledProcessError):
                    lineage.saved_stop(directory)

    def test_inert_opening_stops_entry_before_network_or_consumption(self):
        inert = dict(schema='english-v6-b24r1-source-stop-opening-v1', enabled=False,
            condition='only_one_separate_B24R1_evaluation_after_authenticated_B24_SOURCE_STOP',
            maximum_new_source_registrations=1, candidate_pt_sha256=failure.CANDIDATE_PT,
            original_train_assessment_sha256=failure.TRAIN_ASSESSMENT,
            original_source_lock=None, expected_failure_report_sha256=None,
            old_source_retry_allowed=False, old_source_continuation_allowed=False,
            new_training_allowed=False, open_old_sealed_reserve=False, second_replacement_allowed=False)
        with tempfile.TemporaryDirectory() as tmp:
            opening = Path(tmp) / 'inert.json'
            opening.write_bytes(failure.encoded(inert))
            with patch.object(failure, 'BINDING', opening), \
                    patch.object(lineage, '_original_execution_guard', return_value=identity()), \
                    patch.object(lineage, 'one_execution') as used, \
                    patch.object(failure, 'fetch') as fetch:
                with self.assertRaises(ValueError):
                    lineage.authorize(Path(tmp) / 'never-created', identity()['prepared_parent'])
                used.assert_not_called()
                fetch.assert_not_called()
                self.assertFalse((Path(tmp) / 'never-created').exists())

    def test_actual_stop_readback_precedes_training_fetch_and_budget_in_authorizer(self):
        tree = ast.parse(Path(lineage.__file__).read_bytes())
        auth = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'authorize')
        text = ast.unparse(auth)
        self.assertLess(text.index('failure.fetch('), text.index('gh('))
        self.assertLess(text.index('validate_stop_report('), text.index('gh('))
        self.assertNotIn('CONSUMPTION.json', text)
        self.assertNotIn('generate(', text)
        retain = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'retain_proof')
        self.assertIn("'ORIGINAL_B24_STOP'", ast.unparse(retain))
        self.assertEqual(failure.MAX_ZIP_BYTES, 128_000_000)


if __name__ == '__main__':
    unittest.main()
