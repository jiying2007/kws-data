"""Strict new-source wrapper fixtures; no real process, audio or model calls."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b20_source_observe as subject


def rows_fixture():
    rows = []
    for index in range(798, 822):
        for cell, scale in (("mero-fast", 1.4), ("mero-slow", 1.8), ("question", 1.0), ("request", 1.08)):
            ident = f"b20-primary-{index}-{cell}"
            rows.append(dict(id=ident, speaker_index=index, cell_id=cell, length_scale=scale,
                requested_class=2 if cell.startswith("mero-") else 0, observed_class=None,
                role="development", training_allowed=False, samples=16000,
                pcm_sha256=subject.sha(ident.encode()), wav_sha256=subject.sha((ident + ":wav").encode())))
    return rows


def registration(source_sha="e" * 64):
    contract = subject.source_check()
    return dict(prepared_parent="b" * 40, run_branch=subject.lineage.BRANCH,
        run_id="123", run_head="a" * 40, run_attempt=1,
        source_schema=subject.SOURCE_SCHEMA, batch_schema=subject.BATCH,
        source_rows=96, max_calls=96, assessment_sha256="c" * 64,
        source_manifest_sha256=source_sha, B19_feasibility_result_sha256="d" * 64,
        **{key: contract[key] for key in ("observer_sha256", "asr_adapter_sha256", "parser_contract_sha256")})


def design_fixture():
    return dict(B19_original_proof=dict(feasibility_result_sha256="d" * 64))


def native_fixture(outcome, calls):
    def backend(wav, directory, start):
        stdout, code = outcome(start["id"])
        command = ["/fixture/assets/nemo-speech", "transcribe", str(wav), "--model",
                   "/fixture/assets/parakeet-tdt-0.6b-v3.q8_0.gguf", "--device", "cpu"]
        env = {"PATH": "/usr/bin:/bin", "HOME": str(directory), "LANG": "C.UTF-8",
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "NEMO_SPEECH_MODEL_DIR": "/fixture/assets", "OMP_NUM_THREADS": "4"}
        class Process:
            def __init__(self, argv, **kwargs):
                calls.append(start["id"])
                kwargs["stdout"].write(stdout)
                kwargs["stderr"].write(b"offline fixture only\n")
            def wait(self, timeout=None): return code
            def poll(self): return code
            def kill(self): raise AssertionError("fixture must not time out")
        return subject.raw_asr.capture_process(command, env, directory, start, popen=Process)
    return backend


class FrozenContracts(unittest.TestCase):
    def test_original_five_backends_parser_and_admission_unchanged(self):
        original = subject.native.source_check()
        current = subject.source_check()
        for key in ("asr_adapter_sha256", "parser_contract_sha256", "old_file_sha256",
                    "source_admission_function_sha256", "asr_models", "asr_runtime_versions", "providers"):
            self.assertEqual(current[key], original[key])
        self.assertIs(subject._load_phone, subject.native._load_phone)
        self.assertIs(subject._verify_phone, subject.native._verify_phone)
        self.assertEqual(current["source_indices"], list(range(798, 822)))
        self.assertEqual(current["maximum_native_calls"], 480)
        self.assertEqual(current["fixed_positive_rate_pair"], [1.4, 1.8])
        self.assertIs(current["diagnostic_empty_exception"], False)

    def test_old_cohort_duplicate_pcm_and_post_result_cell_changes_rejected(self):
        good = rows_fixture()
        subject.validate_rows(good, "e" * 64, registration())
        changes = (("speaker_index", 790), ("pcm_sha256", good[1]["pcm_sha256"]),
                   ("length_scale", .92), ("requested_class", 0), ("role", "train"),
                   ("observed_class", 2), ("training_allowed", True))
        for key, value in changes:
            rows = copy.deepcopy(good); rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                subject.validate_rows(rows, "e" * 64, registration())

    def test_96_returns_cannot_replace_96_distinct_valid_payloads(self):
        ledger = [dict(id=f"r{i}", failed=False,
            **dict.fromkeys(("input_started", "forward_attempted", "native_returned",
                            "raw_payload_saved", "payload_valid", "capture_saved"), True)) for i in range(96)]
        self.assertTrue(subject.complete_ledger(ledger))
        ledger[5].update(failed=True, payload_valid=False, capture_saved=False)
        self.assertEqual(subject.ledger_counts(ledger)["native_returned"], 96)
        self.assertFalse(subject.complete_ledger(ledger))

    def test_exact_first_public_source_identity_and_canary_prerequisite(self):
        reg = registration(); branch = subject.lineage.BRANCH
        env = dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY="jiying2007/kws-data",
                   GITHUB_REF=branch, GITHUB_EVENT_NAME="push", GITHUB_RUN_ATTEMPT="1",
                   GITHUB_RUN_ID="123", GITHUB_SHA="a" * 40)
        event = dict(before="b" * 40, after="a" * 40, ref=branch, forced=False, deleted=False,
                     repository=dict(full_name="jiying2007/kws-data", private=False))
        with patch.object(subject.lineage, "read_plan", return_value=design_fixture()), \
             patch.object(subject.lineage, "ASSESSMENT_SHA", "c" * 64):
            self.assertEqual(subject.guard(reg, env=env, event=event, parent="b" * 40),
                             dict(run="123", sha="a" * 40))
            for key, value in (("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_REF", "refs/heads/main")):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    subject.guard(reg, env=dict(env, **{key: value}), event=event, parent="b" * 40)
            with self.assertRaises(ValueError):
                subject.guard(dict(reg, B19_feasibility_result_sha256="f" * 64), env=env, event=event, parent="b" * 40)
            with self.assertRaises(ValueError):
                subject.guard(reg, env=env, event=dict(event, forced=True), parent="b" * 40)

    def test_closed_preregistration_refuses_before_source_or_backend_access(self):
        read_rows = Mock(side_effect=AssertionError("closed protocol must not read sources"))
        with patch.object(subject.lineage, "read_plan", side_effect=ValueError("closed preregistration")), \
             patch.object(subject.raw_asr, "_load", side_effect=AssertionError("closed protocol must not load model")) as load:
            with self.assertRaisesRegex(ValueError, "closed preregistration"):
                subject.observe("parakeet", Path("missing-source"), Path("missing-output"),
                    read_rows=read_rows, schema=subject.BATCH, registration=registration())
            read_rows.assert_not_called(); load.assert_not_called()

    def test_registered_source_cannot_be_replaced_by_agreed_provider_metadata(self):
        reg = registration()
        owner = SimpleNamespace(registration=lambda _: reg)
        with patch.object(subject.lineage, "read_plan", return_value=design_fixture()), \
             patch.dict(sys.modules, {"english_v6_b20_source": owner}):
            self.assertEqual(subject.expected_registration("source", reg), reg)
            with self.assertRaisesRegex(ValueError, "never provider self-report"):
                subject.expected_registration("source", dict(reg, assessment_sha256="f" * 64))

    def test_supplied_rows_must_equal_actual_source_including_order_and_labels(self):
        rows = rows_fixture()
        owner = SimpleNamespace(read_source=lambda _: rows)
        with patch.dict(sys.modules, {"english_v6_b20_source": owner}):
            subject.require_original_rows("source", copy.deepcopy(rows))
            wrong = copy.deepcopy(rows); wrong[0]["requested_class"] = 0
            for changed in (wrong, list(reversed(rows))):
                with self.assertRaisesRegex(ValueError, "every supplied row"):
                    subject.require_original_rows("source", changed)


class StrictBoundedObserver(unittest.TestCase):
    def simulate(self, invalid_index=None, exit_code=0, forged_rows=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "source"; output = root / "parakeet"
            source.mkdir(); (source / "MANIFEST.json").write_bytes(b"offline-fixture")
            rows = rows_fixture(); by_id = {r["id"]: r for r in rows}
            reg = registration(subject.sha(b"offline-fixture"))
            bad_id = rows[invalid_index]["id"] if invalid_index is not None else None
            calls = []
            backend = native_fixture(lambda ident: (b"\n" if ident == bad_id else b"Hey Mero.\n", exit_code), calls)
            contract = subject.source_check()
            model = dict(observer=contract["asr_models"]["parakeet"], runtime_versions=contract["asr_runtime_versions"]["parakeet"],
                         adapter_sha256=contract["asr_adapter_sha256"], parser_contract_sha256=contract["parser_contract_sha256"],
                         raw_before_validation=True, decode_config_unchanged=True)
            owner = SimpleNamespace(registration=lambda _: reg, read_source=lambda _: rows)
            supplied = copy.deepcopy(rows)
            if forged_rows: supplied[0]["requested_class"] = 0
            import english_hifi_micro as pcm_tools
            with patch.object(subject.lineage, "read_plan", return_value=design_fixture()), \
                 patch.object(subject, "guard", return_value=dict(run="123", sha="a" * 40)), \
                 patch.dict(sys.modules, {"english_v6_b20_source": owner}), \
                 patch.object(subject.raw_asr, "_load", return_value=(backend, model)) as loader, \
                 patch.object(pcm_tools, "inspect_wav", side_effect=lambda wav: by_id[wav.stem]):
                if invalid_index is not None or exit_code or forged_rows:
                    with self.assertRaises(ValueError):
                        subject.observe("parakeet", source, output, read_rows=lambda _: supplied,
                                        schema=subject.BATCH, registration=reg)
                else:
                    subject.observe("parakeet", source, output, read_rows=lambda _: rows,
                                    schema=subject.BATCH, registration=reg)
                if forged_rows:
                    loader.assert_not_called(); self.assertFalse(output.exists())
                    return calls, None
            result = json.loads((output / "RESULT.json").read_bytes())
            ledger = subject.row_ledger(output, rows)
            self.assertEqual(subject.ledger_counts(ledger)["forward_attempted"], len(calls))
            closure = json.loads((output / "FILES.json").read_bytes())
            self.assertEqual(closure["files"], subject.file_closure(output))
            if invalid_index is not None:
                self.assertEqual((output / "rows" / bad_id / "RAW_STDOUT.bin").read_bytes(), b"\n")
                self.assertFalse((output / "rows" / bad_id / "PAYLOAD_VALID.json").exists())
            return calls, result

    def test_complete_96_original_native_calls_have_no_model_promotion_authority(self):
        calls, result = self.simulate()
        self.assertEqual(len(calls), len(set(calls)))
        self.assertEqual((result["actual_native_forward_attempts"], result["native_forward_returns"],
                          result["valid_native_payloads"]), (96, 96, 96))
        self.assertEqual(result["status"], "COMPLETE")
        self.assertIs(result["promotion_allowed"], False)
        self.assertEqual(result["kws_forward_calls"], 0)

    def test_real_exit_zero_empty_stops_whole_panel_without_b19_diagnostic_exception(self):
        calls, result = self.simulate(invalid_index=5)
        self.assertEqual(len(calls), 6)
        self.assertEqual((result["actual_native_forward_attempts"], result["native_forward_returns"],
                          result["valid_native_payloads"]), (6, 6, 5))
        self.assertEqual(result["rows"][5]["status"], "failed")
        self.assertEqual(sum(r["status"] == "not_started" for r in result["rows"]), 90)
        self.assertEqual(result["status"], "STOP_OBSERVER_FAILURE")
        self.assertIs(result["whole_panel_stop"], True)
        self.assertEqual(result["retry_count"], 0)

    def test_nonzero_return_stops_at_one_attempt_with_no_retry(self):
        calls, result = self.simulate(exit_code=2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["native_forward_returns"], 1)
        self.assertEqual(result["valid_native_payloads"], 0)
        self.assertEqual(sum(r["status"] == "not_started" for r in result["rows"]), 95)

    def test_relabelled_caller_rows_cannot_initialize_native_backend(self):
        calls, result = self.simulate(forged_rows=True)
        self.assertEqual(calls, [])
        self.assertIsNone(result)


if __name__ == "__main__": unittest.main()
