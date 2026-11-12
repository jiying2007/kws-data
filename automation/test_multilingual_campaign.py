"""Offline/no-model regression tests for the bilingual source-first planner."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multilingual_campaign as campaign
import english_qwen_capture as english
import english_blind_asr as blind


def fake_sha(letter: str, length: int = 64) -> str:
    return letter * length


def pin(name: str, digit: int) -> dict:
    return {"provider_id": name, "revision_sha": f"{digit:040x}",
            "model_assets_sha256": f"{digit:064x}",
            "license_review_sha256": f"{digit + 500:064x}"}


def request(locale: str = "zh-CN") -> dict:
    if locale == "zh-CN":
        words = ["你好小窝", "小窝小窝"]
        prompts = [
            ("wake-1", "你好小窝", "wake", "complete_keyword", 1),
            ("wake-2", "小窝小窝", "wake", "complete_keyword", 2),
            ("near-1", "你好小屋", "near_miss", "homophone", None),
            ("other-1", "今天晚上吃什么", "ordinary_negative", "conversation", None),
        ]
        tts = [pin("qwen3-custom-zh", 1)]
        asr = [pin("qwen3-asr-zh", 11), pin("sensevoice-small-zh", 12),
               pin("whisper-small-zh", 13)]
    else:
        words = ["hello orbit"]
        prompts = [
            ("wake-1", "HELLO orbit!", "wake", "complete_keyword", 1),
            ("near-1", "hello orphan", "near_miss", "other_confusable", None),
            ("other-1", "turn off the light", "ordinary_negative", "conversation", None),
        ]
        tts = [pin("kokoro-82m-en", 21), pin("qwen3-custom-en", 22)]
        asr = [pin("qwen3-asr-en", 31), pin("whisper-small-en", 32),
               pin("parakeet-tdt-v3-en", 33)]
    return {
        "schema_version": 1, "policy": campaign.POLICY, "locale": locale,
        "wake_phrases": words,
        "utterances": [
            {"id": name, "text": text, "intent": kind,
             "challenge": challenge, "keyword_index": keyword_id}
            for name, text, kind, challenge, keyword_id in prompts
        ],
        "voices": [
            {"id": "voice-01", "source_family_id": "orig-01", "role": "train-candidate",
             "origin_kind": "owned-synthetic", "reference_pcm_sha256": None,
             "rights_evidence_sha256": None},
            {"id": "voice-02", "source_family_id": "orig-02", "role": "development-only",
             "origin_kind": "owned-synthetic", "reference_pcm_sha256": None,
             "rights_evidence_sha256": None},
        ],
        "tts_providers": tts, "asr_providers": asr,
        "repetitions": 2, "max_generation_cells": 32, "max_paid_usd": 0,
        "model_calls_authorized": False,
    }


def english_market_request() -> dict:
    """Small invented source-only matrix; no WAVs, ASR, model or true label."""
    raw = request("en-US")
    raw["wake_phrases"] = ["Hey Vosen", "Hey Mero"]
    raw["utterances"] = [
        {"id": "wake-vosen", "text": "HEY VOSEN!", "intent": "wake",
         "challenge": "complete_keyword", "keyword_index": 1},
        {"id": "wake-mero", "text": "Hey Mero", "intent": "wake",
         "challenge": "complete_keyword", "keyword_index": 2},
        {"id": "near-vosen", "text": "Hey Wosen", "intent": "near_miss",
         "challenge": "other_confusable", "keyword_index": None},
        {"id": "near-mero", "text": "Hey Miro", "intent": "near_miss",
         "challenge": "other_confusable", "keyword_index": None},
        {"id": "near-prefix", "text": "Hey", "intent": "near_miss",
         "challenge": "partial_prefix", "keyword_index": None},
        {"id": "ordinary", "text": "Turn off the lights", "intent": "ordinary_negative",
         "challenge": "conversation", "keyword_index": None},
    ]
    raw["max_generation_cells"] = 64
    return raw


def english_preset_request() -> dict:
    raw = english_market_request()
    raw["tts_providers"] = [pin("qwen3-custom-en", 22)]
    raw["voices"] = [
        {"id": name, "source_family_id": "qwen3-preset-" + name,
         "role": role, "origin_kind": "owned-synthetic",
         "reference_pcm_sha256": None, "rights_evidence_sha256": None}
        for name, role in (("ryan", "train-candidate"),
                           ("aiden", "development-only"))
    ]
    return raw


class FakeEnglishBackend:
    def __init__(self, fail_first=False):
        self.calls = []
        self.fail_first = fail_first

    def synthesize(self, cell, speaker):
        self.calls.append((cell["cell_id"], speaker))
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("invented test failure")
        return [0.1] * 12000, 24000


def fixture_wave_writer(folder, samples, rate, metrics):
    import wave
    native = folder / "native-f32.wav"
    native.write_bytes(b"invented-fixture-only")
    pcm = b"\x00\x04" * 8000
    wav = folder / "audio-16k.wav"
    with wave.open(str(wav), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(pcm)
    return {"native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
            "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
            "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            "audio_frames": 8000, "sample_rate_hz": 16000,
            "wav_bytes": wav.stat().st_size}


def fake_asr_assets(raw: dict, folder: Path) -> tuple[dict, Path, Path]:
    """Invented files: lock verification only, no pretrained ASR or inference."""
    raw = copy.deepcopy(raw)
    model_root = folder / "asr-models"
    model_root.mkdir()
    bins = folder / "bin"
    bins.mkdir()
    rows = []
    for pin_row in raw["asr_providers"]:
        name = pin_row["provider_id"]
        model_dir = model_root / name
        model_dir.mkdir()
        member = model_dir / "weights.bin"
        member.write_bytes(("invented-asset-" + name).encode())
        manifest = {"schema": blind.MODEL_SCHEMA, "provider_id": name,
                    "revision": pin_row["revision_sha"],
                    "relative_dir": name, "files": [{
                        "path": "weights.bin", "bytes": member.stat().st_size,
                        "sha256": hashlib.sha256(member.read_bytes()).hexdigest()}]}
        manifest_path = folder / (name + ".model-lock.json")
        manifest_path.write_bytes(campaign.canonical(manifest) + b"\n")
        pin_row["model_assets_sha256"] = hashlib.sha256(
            manifest_path.read_bytes()).hexdigest()
        executable = bins / name
        executable.write_bytes(b"#!/bin/sh\nexit 0\n")
        executable.chmod(0o700)
        rows.append({"provider_id": name, "model_revision_sha": pin_row["revision_sha"],
                     "model_assets_sha256": pin_row["model_assets_sha256"],
                     "model_manifest_path": str(manifest_path),
                     "executable_path": str(executable),
                     "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                     "timeout_seconds": 30})
    lock_path = folder / "adapters.json"
    lock_path.write_bytes(campaign.canonical({
        "schema": blind.ADAPTER_SCHEMA, "adapters": rows}) + b"\n")
    return raw, lock_path, model_root


class SourceFirstLanguageTests(unittest.TestCase):
    def test_mandarin_wake_and_homophone_are_distinct(self) -> None:
        p = campaign.plan(request())
        self.assertEqual(p["locale"], "zh-CN")
        self.assertEqual(p["generation_cells_planned"], 16)
        self.assertEqual(p["asr_observations_planned"], 48)
        self.assertEqual(p["source_families_declared"], 2)
        self.assertEqual(len({row["cell_id"] for row in p["cells"]}), 16)
        self.assertEqual({row["role"] for row in p["cells"]},
                         {"train-candidate", "development-only"})
        self.assertFalse(p["training_admitted"])
        self.assertFalse(p["source_rights_authenticated"])
        self.assertFalse(p["independent_acceptance_evidence"])
        self.assertNotIn("shipping_approved", p)
        self.assertNotIn("qualification_allowed", p)
        self.assertTrue(all("qualification_allowed" not in c for c in p["cells"]))
        self.assertEqual(p["model_calls_executed"], 0)
        self.assertEqual(p["requested_intents"], {
            "near_miss": 1, "ordinary_negative": 1, "wake": 2})
        self.assertEqual(
            {x["normalized_text"] for x in [dict(x, normalized_text=campaign.normalize(x["text_requested"],"zh-CN")) for x in p["cells"] if x["intent"] == "wake"]},
            {"你好小窝", "小窝小窝"})

    def test_explicit_english_multiword_and_cross_family_models(self) -> None:
        first = campaign.plan(request("en-US"))
        again = campaign.plan(request("en-US"))
        self.assertEqual(first, again)
        self.assertEqual(first["generation_cells_planned"], 24)
        self.assertEqual(first["asr_observations_planned"], 72)
        self.assertEqual({x["family"] for x in first["asr_pins"]},
                         {"qwen3-asr", "whisper", "nvidia-parakeet"})
        self.assertEqual({x["family"] for x in first["tts_pins"]},
                         {"qwen3-tts", "kokoro"})
        self.assertEqual(first["wake_phrases_requested"], ["hello orbit"])
        self.assertFalse(first["next_campaign_authorized"])

    def test_current_english_market_pair_has_separate_positive_and_near_miss_cells(self) -> None:
        raw = english_market_request()
        campaign.require_market_candidates(raw)
        first = campaign.plan(raw)
        self.assertEqual(first, campaign.plan(raw))
        self.assertEqual(first["locale"], "en-US")
        self.assertEqual(first["wake_phrases_requested"], ["Hey Vosen", "Hey Mero"])
        self.assertEqual(first["generation_cells_planned"], 48)
        self.assertEqual(first["asr_observations_planned"], 144)
        self.assertEqual(first["requested_intents"],
                         {"wake": 2, "near_miss": 3, "ordinary_negative": 1})
        wake = [row for row in first["cells"] if row["intent"] == "wake"]
        self.assertEqual({row["requested_keyword_index"] for row in wake}, {1, 2})
        self.assertEqual({campaign.normalize(row["text_requested"], "en-US")
                          for row in wake}, {"hey vosen", "hey mero"})
        self.assertFalse(first["human_gold"])
        self.assertFalse(first["training_admitted"])
        self.assertFalse(first["independent_acceptance_evidence"])
        self.assertEqual(first["model_calls_executed"], 0)

    def test_market_roster_rejects_missing_reordered_and_cross_market_wakes(self) -> None:
        campaign.require_market_candidates(request("zh-CN"))
        for phrases in (["Hey Mero", "Hey Vosen"], ["Hey Vosen"],
                        ["Hey Vosen", "Hello Orbit"]):
            raw = english_market_request()
            raw["wake_phrases"] = phrases
            with self.assertRaisesRegex(ValueError, "MARKET_CANDIDATE_ROSTER_MISMATCH"):
                campaign.require_market_candidates(raw)
        raw = english_market_request()
        raw["wake_phrases"] = ["你好小窝", "小窝小窝"]
        with self.assertRaisesRegex(ValueError, "ENGLISH_REQUEST_MUST_BE_ASCII"):
            campaign.require_market_candidates(raw)
        raw = english_market_request()
        raw["utterances"][2]["text"] = "hey VOSEN!"
        with self.assertRaisesRegex(ValueError, "NEGATIVE_MUST_NOT_MATCH_WAKE"):
            campaign.plan(raw)

    def test_market_cli_is_opt_in_and_preserves_private_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source, output = work / "request.json", work / "market-plan.json"
            source.write_text(json.dumps(english_market_request(), ensure_ascii=False))
            self.assertEqual(campaign.main(["--request", str(source),
                                            "--output", str(output),
                                            "--market-candidates"]), 0)
            generated = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(generated["market_candidate_roster_verified"])
            self.assertEqual(generated["market_candidate_policy"],
                             campaign.MARKET_CANDIDATE_POLICY)
            self.assertFalse(generated["training_admitted"])
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                campaign.main(["--request", str(source), "--output", str(output),
                               "--market-candidates"])
            source.write_text(json.dumps(request("en-US")))
            absent = work / "must-not-exist.json"
            with self.assertRaisesRegex(ValueError, "MARKET_CANDIDATE_ROSTER_MISMATCH"):
                campaign.main(["--request", str(source), "--output", str(absent),
                               "--market-candidates"])
            self.assertFalse(absent.exists())

    def test_english_qwen_explicit_preset_and_no_cross_market_execution(self) -> None:
        raw = english_preset_request()
        planned = english.prepare(raw)
        self.assertTrue(planned["market_candidate_roster_verified"])
        self.assertEqual(planned["wake_phrases_requested"],
                         ["Hey Vosen", "Hey Mero"])
        self.assertEqual(planned["generation_cells_planned"], 24)
        self.assertFalse(planned["training_admitted"])
        for bad in (
            lambda x: x["voices"][0].update(source_family_id="fake-ryan"),
            lambda x: x["voices"][0].update(id="unknown-clone"),
            lambda x: x["voices"][0].update(origin_kind="consented-human",
                                           reference_pcm_sha256="a" * 64,
                                           rights_evidence_sha256="b" * 64),
            lambda x: x["tts_providers"].append(pin("kokoro-82m-en", 23)),
            lambda x: x["wake_phrases"].reverse(),
        ):
            changed = copy.deepcopy(raw)
            bad(changed)
            with self.assertRaises(ValueError):
                english.prepare(changed)
        with self.assertRaises(ValueError):
            english.prepare(request("zh-CN"))

    def test_english_qwen_model_lock_checks_real_bytes_not_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            model = folder / "assets" / "preset"
            model.mkdir(parents=True)
            material = model / "config.json"
            material.write_bytes(b'{"invented":"fixture-only"}\n')
            source = english_preset_request()
            lock = {"schema": english.MODEL_SCHEMA, "models": [{
                "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                "revision": source["tts_providers"][0]["revision_sha"],
                "relative_dir": "preset", "files": [{
                    "path": "config.json", "bytes": material.stat().st_size,
                    "sha256": hashlib.sha256(material.read_bytes()).hexdigest()}]
            }]}
            lock_path = folder / "lock.json"
            lock_path.write_bytes(campaign.canonical(lock) + b"\n")
            source["tts_providers"][0]["model_assets_sha256"] = (
                hashlib.sha256(lock_path.read_bytes()).hexdigest())
            plan = english.prepare(source)
            self.assertEqual(english.verify_model(plan, lock_path, folder / "assets"),
                             model)
            material.write_bytes(b"unexpected-model-bytes")
            with self.assertRaisesRegex(ValueError, "MODEL_FILE_BYTES_CHANGED"):
                english.verify_model(plan, lock_path, folder / "assets")
            source["tts_providers"][0]["model_assets_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "MODEL_LOCK_SHA_NOT_PINNED"):
                english.verify_model(english.prepare(source), lock_path,
                                     folder / "assets")

    def test_english_qwen_private_capture_is_bounded_and_never_retries(self) -> None:
        plan = english.prepare(english_preset_request())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "private-run"
            backend = FakeEnglishBackend(fail_first=True)
            result = english.capture(plan, out, backend, max_calls=2,
                                     writer=fixture_wave_writer)
            self.assertEqual((result["total_started"], result["total_completed"]),
                             (2, 1))
            self.assertEqual(result["asr_calls"], 0)
            first = out / "cells" / plan["cells"][0]["cell_id"]
            self.assertTrue((first / "failed.json").is_file())
            before = (first / "started.json").read_bytes()
            result = english.capture(plan, out, backend, max_calls=1,
                                     writer=fixture_wave_writer)
            self.assertEqual((result["total_started"], result["total_completed"]),
                             (3, 2))
            self.assertEqual((first / "started.json").read_bytes(), before)
            self.assertEqual(len(backend.calls), 3)
            self.assertFalse(result["training_admitted"])
            self.assertEqual(out.stat().st_mode & 0o777, 0o700)
            other = copy.deepcopy(plan)
            other["wake_phrases_requested"] = ["unbound"]
            with self.assertRaisesRegex(ValueError, "CAPTURE_RESUME_IDENTITY_DRIFT"):
                english.capture(other, out, backend, max_calls=1,
                                writer=fixture_wave_writer)
            with self.assertRaisesRegex(ValueError, "CAPTURE_CALL_BUDGET"):
                english.capture(plan, out, backend, max_calls=17,
                                writer=fixture_wave_writer)

    def test_three_blind_asr_executes_only_pinned_audio_without_text_hints(self) -> None:
        import wave
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            raw, adapter_lock, model_root = fake_asr_assets(
                english_preset_request(), folder)
            plan = english.prepare(raw)
            capture_dir = folder / "generated"
            english.capture(plan, capture_dir, FakeEnglishBackend(),
                            max_calls=1, writer=fixture_wave_writer)
            adapters = blind.verify_adapters(plan, adapter_lock, model_root)
            adapter_sha = hashlib.sha256(adapter_lock.read_bytes()).hexdigest()
            first = plan["cells"][0]
            source = blind.verify_capture(plan, capture_dir, first)
            self.assertIsNotNone(source)
            wav, pcm = source
            self.assertEqual(pcm["frames"], 8000)
            self.assertEqual(pcm["wav_sha256"], hashlib.sha256(
                wav.read_bytes()).hexdigest())
            by_provider = {r["provider_id"]: r for r in plan["asr_pins"]}
            invoked = []

            def fake_native_run(argv, **kwargs):
                self.assertEqual(argv[1], "--audio")
                self.assertEqual(argv[3], "--model-dir")
                self.assertEqual(argv[5], "--output")
                self.assertNotIn("Hey Vosen", " ".join(argv))
                self.assertNotIn("Hey Mero", " ".join(argv))
                self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
                self.assertEqual(kwargs["env"]["TRANSFORMERS_OFFLINE"], "1")
                provider = Path(argv[0]).name
                self.assertIn(provider, blind.NAMES)
                audio = Path(argv[2])
                with wave.open(str(audio), "rb") as wav_input:
                    frames = wav_input.getnframes()
                pin_row = by_provider[provider]
                native = {
                    "schema": blind.RAW_SCHEMA, "provider_id": provider,
                    "model_id": pin_row["model_id"],
                    "model_revision_sha": pin_row["model_revision_sha"],
                    "audio_wav_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                    "processed_start_frame": 0, "processed_end_frame": frames,
                    "decoder_completed": True, "actual_text": "Hey Vosen"}
                Path(argv[-1]).write_bytes(campaign.canonical(native) + b"\n")
                invoked.append(provider)
                return type("NoModelProcess", (), {"returncode": 0})()

            output = folder / "english-asr"
            first_result = blind.dispatch(plan, capture_dir, output, adapters,
                                          adapter_sha, max_calls=2,
                                          executor=fake_native_run)
            self.assertEqual(first_result["asr_attempts_this_call"], 2)
            self.assertIsNone(blind.machine_screen(plan, output, first))
            second_result = blind.dispatch(plan, capture_dir, output, adapters,
                                           adapter_sha, max_calls=1,
                                           executor=fake_native_run)
            self.assertEqual(second_result["asr_attempts_this_call"], 1)
            self.assertEqual(set(invoked), set(blind.NAMES))
            screen = blind.machine_screen(plan, output, first)
            self.assertTrue(screen["three_models_responded"])
            self.assertTrue(screen["all_normalized_agree"])
            self.assertFalse(screen["actual_label_authenticated"])
            self.assertFalse(screen["training_admitted"])
            self.assertTrue(screen["machine_verdict"].endswith("REVIEW_REQUIRED"))
            screen_path = output / "cells" / first["cell_id"] / "machine-screen.json"
            self.assertEqual(campaign.read(screen_path), screen)
            final = blind.dispatch(plan, capture_dir, output, adapters,
                                   adapter_sha, max_calls=1,
                                   executor=fake_native_run)
            self.assertEqual(final["asr_attempts_this_call"], 0)
            self.assertEqual(len(invoked), 3)

    def test_three_blind_asr_fails_closed_on_model_and_source_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            raw, adapter_lock, model_root = fake_asr_assets(
                english_preset_request(), folder)
            plan = english.prepare(raw)
            adapters = blind.verify_adapters(plan, adapter_lock, model_root)
            self.assertEqual(set(adapters), set(blind.NAMES))
            p = model_root / "qwen3-asr-en" / "weights.bin"
            p.write_bytes(b"changed-model")
            with self.assertRaisesRegex(ValueError, "ASR_MODEL_FILE_CHANGED"):
                blind.verify_adapters(plan, adapter_lock, model_root)

            capture_dir = folder / "generated"
            english.capture(plan, capture_dir, FakeEnglishBackend(),
                            max_calls=1, writer=fixture_wave_writer)
            first = plan["cells"][0]
            wav = capture_dir / "cells" / first["cell_id"] / "audio-16k.wav"
            wav.write_bytes(wav.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "CAPTURE_WAV_SHA_DRIFT"):
                blind.verify_capture(plan, capture_dir, first)

    def test_research_licensed_tts_is_plannable_only_with_source_provenance(self) -> None:
        raw = request()
        raw["tts_providers"] = [pin("f5-tts-official-weights", 46)]
        with self.assertRaisesRegex(ValueError, "CLONING_REQUIRES_EACH_ORIGINAL_REFERENCE_AND_RIGHTS"):
            campaign.plan(raw)
        for idx, voice in enumerate(raw["voices"]):
            voice["reference_pcm_sha256"] = f"{400 + idx:064x}"
            voice["rights_evidence_sha256"] = f"{500 + idx:064x}"
        planned = campaign.plan(raw)
        self.assertEqual(planned["tts_pins"][0]["license_status"],
                         "research-terms-review-required")
        self.assertIsNone(planned["tts_pins"][0]["adapter"])
        self.assertFalse(planned["source_rights_authenticated"])
        self.assertFalse(planned["training_admitted"])
        self.assertFalse(planned["independent_acceptance_evidence"])
        raw["tts_providers"][0]["license_review_sha256"] = "unreviewed"
        with self.assertRaisesRegex(ValueError, "MODEL_LICENSE_REVIEW_MISSING"):
            campaign.plan(raw)

    def test_requires_three_distinct_asr_families(self) -> None:
        raw = request()
        raw["asr_providers"][1] = pin("qwen3-asr-zh", 101)
        with self.assertRaisesRegex(ValueError, "MODEL_FAMILY_NOT_INDEPENDENT"):
            campaign.plan(raw)
        raw = request()
        raw["asr_providers"] = raw["asr_providers"][:2]
        with self.assertRaisesRegex(ValueError, "THREE_DISTINCT_ASR_FAMILIES_REQUIRED"):
            campaign.plan(raw)

    def test_pinned_revisions_and_license_review_receipts_are_mandatory(self) -> None:
        raw = request()
        raw["tts_providers"][0]["revision_sha"] = "main"
        with self.assertRaisesRegex(ValueError, "MODEL_REVISION_NOT_PINNED"):
            campaign.plan(raw)
        raw = request()
        raw["asr_providers"][1]["license_review_sha256"] = None
        with self.assertRaisesRegex(ValueError, "MODEL_LICENSE_REVIEW_MISSING"):
            campaign.plan(raw)

    def test_clone_requires_reference_and_evidence_for_every_voice(self) -> None:
        raw = request()
        raw["tts_providers"] = [pin("qwen3-base-clone-zh", 50)]
        with self.assertRaisesRegex(ValueError, "CLONING_REQUIRES_EACH_ORIGINAL_REFERENCE_AND_RIGHTS"):
            campaign.plan(raw)
        for idx, voice in enumerate(raw["voices"]):
            voice["reference_pcm_sha256"] = f"{400 + idx:064x}"
            voice["rights_evidence_sha256"] = f"{500 + idx:064x}"
        planned = campaign.plan(raw)
        self.assertEqual(planned["generation_cells_planned"], 16)
        self.assertFalse(planned["source_rights_authenticated"])

    def test_intended_negative_never_relabels_exact_wake(self) -> None:
        raw = request()
        raw["utterances"][2]["text"] = "你好小窝"
        with self.assertRaisesRegex(ValueError, "NEGATIVE_MUST_NOT_MATCH_WAKE"):
            campaign.plan(raw)
        raw = request()
        raw["utterances"][0]["text"] = "你好小屋"
        with self.assertRaisesRegex(ValueError, "POSITIVE_MUST_MATCH_FULL_KEYWORD"):
            campaign.plan(raw)
        raw = request()
        raw["utterances"][2]["keyword_index"] = 1
        with self.assertRaisesRegex(ValueError, "NEGATIVE_MUST_NOT_MATCH_WAKE"):
            campaign.plan(raw)

    def test_english_does_not_inherit_mandarin_research_words(self) -> None:
        raw = request("en-US")
        raw["wake_phrases"] = ["Orbit"]
        with self.assertRaisesRegex(ValueError, "ENGLISH_WAKE_MUST_BE_EXPLICIT_MULTIWORD"):
            campaign.plan(raw)
        raw = request("en-US")
        raw["tts_providers"] = [pin("qwen3-custom-zh", 1)]
        with self.assertRaisesRegex(ValueError, "PROVIDER_LANGUAGE_OR_ROLE_UNSUPPORTED"):
            campaign.plan(raw)

    def test_synthetic_voices_do_not_masquerade_as_distinct_train_split(self) -> None:
        raw = request()
        raw["voices"][1]["source_family_id"] = raw["voices"][0]["source_family_id"]
        with self.assertRaisesRegex(ValueError, "SOURCE_IDENTITY_SPLIT_LEAK:family"):
            campaign.plan(raw)
        raw = request()
        raw["voices"][0]["role"] = "sealed-evaluation"
        with self.assertRaisesRegex(ValueError, "VOICE_ORIGIN_OR_ROLE"):
            campaign.plan(raw)

    def test_same_reference_pcm_cannot_cross_splits_under_different_names(self):
        raw = request()
        raw["tts_providers"] = [pin("qwen3-base-clone-zh", 1)]
        for voice in raw["voices"]:
            voice.update(origin_kind="licensed-public",
                         reference_pcm_sha256="a" * 64,
                         rights_evidence_sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "SOURCE_IDENTITY_SPLIT_LEAK:pcm"):
            campaign.plan(raw)

    def test_budget_exposure_and_unknown_provider(self) -> None:
        raw = request()
        raw["max_generation_cells"] = 15
        with self.assertRaisesRegex(ValueError, "REQUESTED_CELLS_EXCEED_FIXED_BUDGET"):
            campaign.plan(raw)
        raw = request()
        raw["max_paid_usd"] = 1
        with self.assertRaisesRegex(ValueError, "CAMPAIGN_REQUEST_POLICY_OR_BUDGET"):
            campaign.plan(raw)
        raw = request()
        raw["tts_providers"][0]["provider_id"] = "fake-paid"
        with self.assertRaisesRegex(ValueError, "UNKNOWN_PROVIDER"):
            campaign.plan(raw)

    def test_duplicate_source_json_fields_and_private_create_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source, output = work / "request.json", work / "plan.json"
            source.write_text(json.dumps(request(), ensure_ascii=False))
            self.assertEqual(campaign.main(["--request", str(source), "--output", str(output)]), 0)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertNotIn("shipping_approved", json.loads(output.read_text()))
            with self.assertRaises(FileExistsError):
                campaign.main(["--request", str(source), "--output", str(output)])
            source.write_text('{"policy":"a","policy":"b"}')
            with self.assertRaisesRegex(ValueError, "DUPLICATE_JSON_KEY"):
                campaign.read(source)

    def test_source_doc_and_nonmodel_behavior(self) -> None:
        data = campaign.catalog()
        self.assertIn("f5-tts-official-weights", data)
        self.assertEqual(data["openwakeword-framework-en"]["license_status"],
                         "research-terms-review-required")
        self.assertEqual(len(data), 18)
        plan = campaign.plan(request())
        self.assertTrue(all(row["synthetic"] for row in plan["cells"]))
        self.assertTrue(all(not row["actual_text_verified"] for row in plan["cells"]))


if __name__ == "__main__":
    unittest.main()
