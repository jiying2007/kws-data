"""Offline invented provenance fixtures; never generate or score a source."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
definition = importlib.util.spec_from_file_location("b21_fixed_mirror_binding", HERE / "bind_b21_fixed_mirror_primary.py")
binding = importlib.util.module_from_spec(definition)
definition.loader.exec_module(binding)


def proof_fixture():
    first = dict(original_run_id=45352507569,
        original_run_head="d6db2375632adc8a627d094fcf23c5e87b22eddd",
        original_prepared_parent="60f1e6507216ae605d601f3f09ab15177b46aae2",
        original_job_id=138302172086, original_run_attempt=1,
        source_job_log_sha256="c9ecd7399ef7b7b243ea243becfa9b07fbd1ecba402195fc61019581c7db62ac",
        actual_tts_calls=0, natural_original_decodes=0, observer_calls=0, kws_forwards=0,
        source_generation_started=False, transport_continuation=True, original_opened_marker_preserved=True)
    return dict(first, original_transport_origin=first,
        failed_continuation_run_id=45365292922,
        failed_continuation_run_head="2da5cbe0d616f96b21a1bb5665613b9ce8d1c788",
        failed_continuation_prepared_parent="3144028b5db5a7f1314654858b8d4f1343b8cf40",
        failed_continuation_job_id=138345323477,
        failed_continuation_manifest_sha256="a1a0d2acf630fd8b23e773e1e7906401787a8a75cbca81fca2de08dd1d54c4ca",
        failed_continuation_producer_files=246,
        failed_continuation_actual_tts_calls=0, failed_continuation_natural_original_decodes=0,
        failed_continuation_observer_calls=0, failed_continuation_kws_forwards=0,
        failed_continuation_consumption_started=False, failed_continuation_source_generation_started=False,
        failed_continuation_preserved=True, live_refresh_required_for_both_failed_runs=True)


class BindingBoundaryTests(unittest.TestCase):
    def test_driver_pin_and_fixed_mirror_artifact_routing(self):
        driver = HERE / "english_v6_b21_fixed_mirror_primary_eval.py"
        self.assertEqual(hashlib.sha256(driver.read_bytes()).hexdigest(), binding.DRIVER_SHA)
        self.assertEqual(binding.artifact_name("source", 123), "english-v6-b21-primary-fixed-mirror-source-123")
        self.assertEqual(binding.artifact_name("source_producer", 123), "english-v6-b21-primary-fixed-mirror-source-producer-123")
        self.assertEqual(binding.artifact_name("screen", 123), "english-v6-b21-primary-fixed-mirror-frozen-screen-123")
        self.assertEqual(binding.artifact_name("native_qwen", 123), "b21-primary-fixed-mirror-dev-observer-qwen")
        self.assertEqual(binding.artifact_name("pair", 123), "english-v6-b21-competing-margin-weights-123")

    def test_requires_both_exact_failure_identities(self):
        proof = proof_fixture()
        self.assertEqual(binding.validate_transport_origin(proof, copy.deepcopy(proof)), proof)
        first = proof["original_transport_origin"]
        with self.assertRaisesRegex(ValueError, "both original failure identities"):
            binding.validate_transport_origin(first, copy.deepcopy(first))
        changed = copy.deepcopy(proof)
        changed["failed_continuation_run_id"] += 1
        with self.assertRaises(ValueError):
            binding.validate_transport_origin(changed, copy.deepcopy(changed))

    def test_saved_owner_must_equal_independently_verified_complete_proof(self):
        proof = proof_fixture()
        changed = copy.deepcopy(proof)
        changed["unverified_extra"] = "cannot be silently ignored"
        with self.assertRaisesRegex(ValueError, "matches both independently authenticated"):
            binding.validate_transport_origin(changed, proof)
        changed = copy.deepcopy(proof)
        changed["original_transport_origin"]["actual_tts_calls"] = 1
        with self.assertRaises(ValueError):
            binding.validate_transport_origin(changed, copy.deepcopy(changed))

    def test_unknown_or_nonzero_consumption_cannot_authorize_binding(self):
        for field in ("failed_continuation_actual_tts_calls", "failed_continuation_natural_original_decodes",
                      "failed_continuation_observer_calls", "failed_continuation_kws_forwards"):
            for value in (1, False, None):
                proof = proof_fixture()
                proof[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    binding.validate_transport_origin(proof, copy.deepcopy(proof))
        for field in ("failed_continuation_consumption_started", "failed_continuation_source_generation_started"):
            proof = proof_fixture()
            proof[field] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                binding.validate_transport_origin(proof, copy.deepcopy(proof))


if __name__ == "__main__":
    unittest.main()
