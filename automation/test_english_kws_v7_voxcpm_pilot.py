"""Pure stdlib contracts for the bounded VoxCPM V7 source pilot."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_kws_v7_voxcpm_pilot as pilot


def wav16(pcm: bytes, rate: int = 16000) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm)
    return out.getvalue()


class VoxCPMPilotContracts(unittest.TestCase):
    def test_source_check_is_stdlib_only_and_zero_calls(self):
        result = pilot.source_check()
        self.assertEqual(result["source_check"], "PASS")
        self.assertEqual(result["donor_ids"], ["synthetic-af_bella", "synthetic-af_aoede"])
        self.assertEqual(result["donor_roles"], {"train": 1, "development": 1})
        self.assertEqual(result["reserved_donor_audio_opened"], 0)
        self.assertEqual(result["planned_total_model_calls"], 52)
        self.assertEqual(result["new_model_calls"], 0)
        self.assertEqual(result["new_network_calls"], 0)
        self.assertFalse(result["training_allowed"])
        self.assertFalse(result["promotion_allowed"])

    def test_exact_brand_pronunciations_and_independent_candidate_names(self):
        plan, _ = pilot.load_plan()
        self.assertEqual(plan["candidates"]["VOSEN"]["phrase"], "Hey Vosen.")
        self.assertEqual(plan["candidates"]["VOSEN"]["ipa"], "/heɪ ˈvoʊ.sən/")
        self.assertEqual(plan["candidates"]["MERO"]["phrase"], "Hey Mero.")
        self.assertEqual(plan["candidates"]["MERO"]["ipa"], "/heɪ ˈmɛroʊ/")
        self.assertEqual(plan["synthesis"]["deterministic_tempo_variants"], [0.92, 1.0, 1.08])
        self.assertEqual(plan["budgets"]["tts_model_calls"], 4)
        self.assertEqual(plan["budgets"]["total_model_calls"], 52)

    def test_main_guard_rejects_pr_rerun_and_wrong_predecessor(self):
        plan, _ = pilot.load_plan()
        env = {
            "GITHUB_REPOSITORY": plan["repository"],
            "GITHUB_EVENT_NAME": "push",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_SHA": "f" * 40,
            "GITHUB_RUN_ID": "46010000001",
        }
        event = {
            "before": plan["main_before"], "after": env["GITHUB_SHA"],
            "forced": False, "deleted": False,
            "repository": {"full_name": plan["repository"], "private": False},
        }
        self.assertEqual(pilot.guard(plan, env, event)["parent"], plan["main_before"])
        bad = dict(env, GITHUB_EVENT_NAME="pull_request")
        with self.assertRaisesRegex(ValueError, "PUBLIC_MAIN_FIRST_PUSH_ONLY"):
            pilot.guard(plan, bad, event)
        bad_event = dict(event, before="e" * 40)
        with self.assertRaisesRegex(ValueError, "EXACT_PUBLIC_MAIN_SUCCESSOR"):
            pilot.guard(plan, env, bad_event)

    def test_audio_info_checks_exact_mono_pcm16_sample_rate_and_bytes(self):
        pcm = (b"\\x10\\x00\\xf0\\xff") * 4000
        wav = wav16(pcm)
        decoded, samples = pilot.audio_info(wav, expected_rate=16000)
        self.assertEqual(decoded, pcm)
        self.assertEqual(samples, 8000)
        with self.assertRaisesRegex(ValueError, "EXACT_PCM_WAV_GEOMETRY"):
            pilot.audio_info(wav, expected_rate=44100)

    def test_reference_readback_opens_only_two_selected_synthetic_donors(self):
        plan, _ = pilot.load_plan()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "reference.zip"
            donors = []
            manifest_rows = []
            members = []
            roles = (["train"] * 6 + ["development"] * 2
                     + ["calibration"] * 2 + ["locked-test"] * 2)
            selected = {row["donor_id"]: row for row in plan["donors"]}
            for index, role in enumerate(roles):
                donor_id = ("synthetic-af_bella" if index == 0 else
                            "synthetic-af_aoede" if index == 6 else f"unused-{index}")
                if donor_id in selected:
                    spec = selected[donor_id]
                    pcm = (bytes([index + 1, 0]) * 4000)
                    wav = wav16(pcm)
                    row = {
                        "id": donor_id, "role": role,
                        "source_group_id": spec["source_group_id"],
                        "reference_text": spec["reference_text"],
                        "reference_kind": spec["reference_kind"],
                        "sample_rate": 16000, "pcm_samples": 4000,
                        "wav_sha256": pilot.sha(wav),
                        "joined_pcm_sha256": pilot.sha(pcm),
                        "training_allowed": False, "human_gold": False,
                        "independent_human_source_claim": False,
                    }
                    members.append((spec["reference_path"], wav))
                else:
                    row = {
                        "id": donor_id, "role": role,
                        "source_group_id": f"reserved:{index}",
                        "reference_text": "Reserved metadata only",
                        "reference_kind": "SYNTHETIC_NEUTRAL_CLONE_SEED_ONLY",
                        "sample_rate": 16000, "pcm_samples": 4000,
                        "wav_sha256": "0" * 64, "joined_pcm_sha256": "0" * 64,
                        "training_allowed": False, "human_gold": False,
                        "independent_human_source_claim": False,
                    }
                manifest_rows.append(row)
            manifest = {
                "schema": "english-synthetic-clone-reference-12-v1",
                "record_kind": "SYNTHETIC_VOICE_CLONING_REFERENCES_ONLY",
                "reference_audio_count": 12,
                "roles": {"train": 6, "development": 2, "calibration": 2, "locked-test": 2},
                "human_gold": False, "training_allowed": False,
                "source": {
                    "run": plan["donor_reference"]["original_source_run_id"],
                    "artifact_id": plan["donor_reference"]["original_source_artifact_id"],
                },
                "rows": manifest_rows,
            }
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("MANIFEST.json", json.dumps(manifest).encode())
                for name, raw in members:
                    zf.writestr(name, raw)
            test_plan = copy.deepcopy(plan)
            test_plan["donor_reference"]["artifact_sha256"] = pilot.sha(archive.read_bytes())
            output = root / "prepared"
            result = pilot.verify_reference_archive(archive, output, test_plan)
            self.assertEqual(result["status"], "PASS_TWO_SELECTED_SYNTHETIC_DONORS")
            self.assertEqual(result["selected_audio_members_opened"], [
                "wav/synthetic-af_bella.wav", "wav/synthetic-af_aoede.wav"
            ])
            self.assertFalse(result["calibration_audio_opened"])
            self.assertFalse(result["locked_test_audio_opened"])
            self.assertEqual(sorted(p.name for p in (output / "reference").iterdir()),
                             ["synthetic-af_aoede.wav", "synthetic-af_bella.wav"])

    def test_archive_member_paths_cannot_escape_the_selected_donor_root(self):
        with self.assertRaisesRegex(ValueError, "SAFE_RELATIVE_PATH"):
            pilot.safe_relative("../calibration/secret.wav")
        with self.assertRaisesRegex(ValueError, "SAFE_RELATIVE_PATH"):
            pilot.safe_relative("/absolute/locked.wav")


if __name__ == "__main__":
    unittest.main()
