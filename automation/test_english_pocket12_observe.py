"""Stdlib mock tests. All observations in these fixtures are artificial test data."""
import base64
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_pocket12_generate as generate
import english_pocket12_observe as observe
from test_english_pocket12_generate import MockBackend, RUN

# These five immutable, nonnumeric model metadata files are copied byte-for-byte
# from the already saved XLSR artifact; no weights, observations or audio are used.
FIXED_METADATA_B64 = "eNqtWcly40gO/ZWOuvbEtEhJVdZE9EHWZsoiVZIlbpcOLlq4ihK1kRPz74MEkLRVVZ6+zMFhpjMTiQQe8ID0v78Eh3wb7f4Zl4f8y79++7K5HXbaYLp32ssizMYtz+pdbHVZ+e3na2gvonnUv83ifjmI+jstG9+Cyb1rt42baxlbxzZz3zIv4UD7qr0sq9Bay3WVo+6LcGLG4YtZubYOcp59nBv0I1Mdq3rqpk49nfjt6dAc6ZGYs99ur2LesZct1+p+osuIz5heA6XX8tVz6qe9whU6DOWcUTq2UXuWW7j2dBu8TK9e2qs9u0i1+LDT47U66+O9Y19dpmEGek56FZxV+G9CTqDOB6yL2ruEL9N9gPYxt/6kl3v2srbby9S3n9MAbBbaU5I76Nzf5XZVW50WDt29FeRm+jjHZ41Lcc5OG67venT7/3zv4JxxS9ri6ufu1lPNys/MUuhpoc/7O33Ffh30a9IN/lZJOfrPc80ZGvx+Dh/u0zZaQZYm7uOdpIzqFzL+19yD/HBiCLvX4O/KtZaKwKqvdgQeoqBtnrWIfdVexna7AJ/0tp7VzeCnCF8SYf/Mse61y/hyJ6kKuAA8LLu2er8GbWMbqmbhqvvWD35MfKVXh5Nx5E+E78ddgSc/61XouwfMm7/AI98xA6zby61r78FG4zgc9x7iTZypZWHqv6wjPjdzrXHLVs2OwB6s3bqA4WDSUzDWhv2Lzvh0M3MP8tIN6OJYBuzpXoOshbYhfbVHme3nyleLn+W9Pci7h9b4Ek7Sr+K+n5+dXsCun8+3p3uwdernyy3kg9RRzzdfTYXci1vB2om7918MGRcHz1omEPdbcY9f2KSZ/0x3bwL4m5gXwGKxydbCBnd9qLV4f+FnKdh7XIo8EKQ9sFc3R7wKGY2vhH4QK5aZwNqWa77nDWPY7xqMtQcsmL00eNHxPPetdTPeHtYkgMfbD7g6Q3462apxAB0uLtgQ8HVx28sDrpNxmSn7QD0L2+39rJv6494ZML0N2s97R6X7fSoPZAnfP+bvn+TdQLeIdbt9JitQzRLy/TtWc6MFeC6cSnuQi9gR+d7swfpuHk525APGg2+Na0/pIQ5FziQs/lpHwMj5b+SAPkpqq0oB8VYHk3FMvhz9Ss41nOzTIHrE54OcR13ajQ8f1hipLzhtkl59IWswFbkB5nSJ0bPgE7BrawPcQLYK92E8VV1Lr2Se8nPzbKtgA/A5ykp7B7BJElSov+SmS2gp4Kte4tqQP9rLK2A2B38pjLXq03W5sQ0Aw7YaAneahL1au/2w/hKmvZsPuc+1lAjyKuC9m8s9n+jyyz28tpKcSff7KecBJ6dpkKPu7Y9ym7hV7/uNZVYkr4lbXAPclQvMAQfXHCPyPjfHWoKfeyeRGzyL6wSeA3sdActXkK8KDEO9sYWcdMZ4rp0OrxN/j71xLwlf0hviCPIlcM4+HL77TeRiPzcyWHsW/GG3XbiP0XCRAfGtVxBHGdQUcV9iQoH4afgFbNbwCHIX6OFm4zJQmUdy9+qo4+hDDqvnK+11uzj8+eUfv30pTpvidAg2ZXk4/fXrWk5w1kXoCLmZZTzUH+/x984vAlvvmE0Rs5bA7FLG9PrD2uiB15QA6qJABa74GH/5M+R+4OksFHMJ27TyrPAAefunNcCFJXDpR99B3ShkAx881IMirssf71Q7lnLzBa9AjQDrqR5bOTd92G9sVxabIPLSv86HZJOXf2Ve8W66ahoJWBOMyJ3fX/TfwfWP1I5hrz3N2jyXmxfvh7nQ6p6+g4keoSllwpWHnWi7+BNVQlWievOzL6up4mfnHzbDYXUnmolaJ+uJGvQKdR3l5Phe86HpTxcZ9Oq/U0jIDFWoryAGIOcXvpW2BEY/rt0OSIY36QEvKIAtkWPuwCUBzsO6cwj1rpAVTPZXiNkzc20EnC7iMyFeNyFfm3uvQnkCr6XfDsW6GLgn/RgTwrewNha8Le8LzgCOega+SUvClXaZRb22YzuVqOX1SOlAXVHNVut6ZsmzlDiMlCwc9D6R1794uSHyeon5QDg04/pHnjPpKv7kJuah7hHcAz0DcDHzUzSLOtdQ9Da56G2082Zyr4OopeiVkgJnp45Vnp22c3bzxZXAQ+ezjWLICzXmvsHUFncxxV3S5wPU1ynUI9AnGet3vbQGRNdD4PkfkKM9BXWHqoPBu4cxsgYNGrA6/QDXr6LiFzfDzDgQ2mGmVhAZWGk5MD+bZtQhibXYTczF2mwk1nbFWo+yAJ7lETOgHtyhVeKbuqxRLb4rt0Rd8cwy+50iFz0voxjPPaKOozaii2R1UBad0SVZfWRGHfU/0PebkDsnFozRuxl9432P9L0Q3yp3h+KbqqKYzr6gvhqeXR3wrhXeuzy+0rpE7CmqLp5f0/kafYvz84C+I9yjEyPj+Qp9L/CcA56jo81vXx3cY2AKeuH766yDTsyAtt/RN94zW3OlivKS4LKg8UiMxxHaalGRrdA+Lej4QIbN8hctMVefUPeWgbrbswTtvVDF3AZ1XJAe7ou+I5nkiyeyhzEU9pjtzz76VkF7ZFLO+k5n7GiO7vca4flrxERIFY6CmMj+kPsUOnPFZ65Rn8rjtRXeP8DzVzu6P+q6Rmy4+vEN52KyTWbdxD6V/LOj74hksB9VtE05z2kf+ug1QT0digv9OKQ5wkmqs73joLF/5Scoi/CS+bWm0xj9dZxTJdJv9vP6NuH2OKN5xO2e9WqTXplJc6jXOEGbBISdb3ifNmEH7KdnXVpLepV4hwB9tpEYqdFnrynaekfxG2QuxjjiZ1oYaM8d4ieoMKd0UJd8NU1pDvHzvX7COcRP9u2VztuRv9wr+3OHGLq739mfuw75E7vXjsQz36WrU7wtUZ8V6tOqCKdd9tuB5nTy7xPdeb7C2G6xvl3C9Ix1ShhDJ47xhHDtteV8h3JaRyFZCfspOzZVqLAF5zzC0nfeC2PM8muF11K+O8izRjfCdz7NBzxGe43kepV94BtyPeGxwziA7nZHPuN5tFlW7Gk/5eDKbeTfST/ZZY44D8jzKB/7T/JuoxbHTcVjjr9ErlcpBjLTY/moX76sZxl3bBiH0zP7CcZkvxN3DJQXVq8UpyPK1bkv7VdRbrD1XI4Ja6+7J6mTRjpaUicNdfRr7v4xLqdXv8z2bMOK88QyXMk1O8rdjR00zBdOI1NHOzpBduSceddrxN+J7815PXN15jWKz+OC96OO3optUK/RZxGvxRgt33Jei/gLxyWv3VHcAz4zXs95g++/oDjN6S6c56cZn2Ugx08hx+JezvV/SDy0iOeHHI8jyveAR4idgOWjj/22xDDl/c1Knof6KZV743FCccn4NFZ9nJf+MChuE4v8T7k/N6/NfoxdiFV1x+MF51GpD3OAL+Whf4v69cD6rcl+lsm5aqRw3t3yfJd9GcgOgXlseiQbEB9kyaXRiWqGtDoceIw6XtmmKtctC87rMI/+HRYsD/2bDyWmVcRfdm3GxOHfBDvRmPLeG+kXUH7hFxCjJv8m9pr9H2AMh0HGMR3UMuZlvcT6nXie7Ofx3Wq0X7e5K/LDNJLxatTMoxJPyBHTGuyN4znWGdO919hzR9zeCXheo/16NYS8x3l+RJwRj3jNgjG74DHWX93bV1/itEM8NuI77zhHxxK3HbRhvtL5zsQdWXoNG72SG9mpGvIY9fTVb9PD+5qKscG2TGqZf9nXXbbluvLIH3OsU4U/bKlrlzlX5qwu+dsspAzyN+jGNiVOmcZURxKfeE/EyTC+U7xpVMuyPVOq328Uzz3+r0af8/Ul4bFC/J757F/YzzV8IOvWPtpyTS8uN46VPfsf1idN/iT9tTvVtmagyfEb3a85Y6VRfhvwPOXba8e3QjqTeAW6eYViVLtT3ll6PI85JyirsNa4hl8hLotZzHoRr7RY7zv6Ozc2XONTTBctWhuPGOOcz2OtYr8uG50pppUPa1oUdw7Pk5+PL2seO5IzKl7f5jwpx1TLuGnFdqsIk5LbNOoRsjHjQuM+YabnbzzGvGGDrxifsU6x7dPrn+wZMonXmPqG0GIei3WVcwP3ONw7+GwX4paIczWM0ddd5sWKegiRuzUeSzuWLJ/6CF+vCh6zfiWvZ/3sGceGRvxSQqal9VzfVEe2QYtipSf5pJL80uTreEH5+61D88Qv+1kp44V6i00wzxhb3F+sOFfAuCI7H3meajBT/meQ82OhNP8pZD9KeaRjrh9LqdO6Q/2ay3YgjglXhF3qNwC7jY7OnTie70D8cqXaQeOeIzvaZBPiF8AJ71UkvzU2ot5jWLzL597syhygMcfYXENoxDHAcbSeOKbyFySPapvC5Z6XOVrGv+w9lqwv8Uv21GJZTV8kccUcc53zfll/yTHl7vWd96PtVjTXoRgBW8fccxMG9+Ebz6Nu+bDgvDCnNwxlxvupvnmRvqd+JM8/xBT1JBv21ZzeNU7Vqc9jyjPVAWp6lkG2y7lf1agvyY0OvwMQrzwxz8TEKaH1RPK4vmnkU2+iNLVmnKD9nPLI7xXUn4Rv3+QdiE9EXmBOmFOv+xqV1KvBHup3CV+1TnEMnH1iGTr3KanEUE28Mk1YLxijnhXZUec+pZfO5ZjwQz6r+xwjfzTySceU6079xjXOaynr1prfiiD2DdojeSXlOrnWuccLZK1fc7/i73hep9qP6i4Yc61IdUTNb0itD/e+Ezfb/C6gc7+ybrF8jGOXsFBTryLO57iqmVcOMWNB57clwZdst5j4+fSUXXlPLd8emFNrnd8Rdo0tqGexWCb1A5nSvCXhW4LAiOzj9IrrnWkaNDK4rmXc1dSzhGOJG534BXqpXXMfvZL1AfuJ+pbse7OH/J5cKNZhTPf1uRaudYrtpPq6hB+2O9WQSVXMGjnk/4TtrNNbnSfjUm+R79OC38LuhCebuRLmySZbaWeD8Hl7Hy/4zJdmD8V6KrkE1jhkx9WIx2xXxgtxDeTRLG/04lp8we+KtexnPH6n02WdKWXK2kfm4pres5ozKd4LqBPYFsQ3zhPHe71uen62OfGNyFkcF/JtKy2bPRTvje+5pynkG4eu8htocmMfGYzTzNZYD4dr3ctC4o7fu6bH5i7EPaJuYt1UxqobxHw/evcaJw3GHOLGb3v2rdOV/Nqcg29f07g6PceNP7nPyUAfPot4SPhmyGfh++lsfOR5wir0Vbo8m7gIcvFG/PDfKN/LN406kHUb5zJ6E8M3F8JAe85xkzfvEDrxUi7zH/U87kuTZzpcc3zIdzuZB27y3rLvOf0+UfLgQ31c70jv5h7EUcAh8rw2v0mx7+jdzOO8NaeaHes0mXPmXCfxnbq63frzy3/+C8TX+Nk="


def test_plan():
    return {"enabled": True, "main_before": "2" * 40, "training_allowed": False, "retry_count": 0,
        "native_tts_request_budget": 12, "maximum_calls_per_observer": 12, "maximum_total_model_calls": 60,
        "helper_sha256": observe.HELPER_PINS, "observer_code_sha256": observe.sha(observe.regular(Path(observe.__file__))),
        "evaluate_code_sha256": observe.sha(observe.regular(observe.HERE / "english_pocket12_evaluate.py"))}


def environment(root, run_identity=RUN, plan=None):
    plan = test_plan() if plan is None else plan
    event = Path(root) / "mock-event.json"
    event.write_text(json.dumps({"before": plan["main_before"], "after": run_identity["commit"],
        "repository": {"full_name": "jiying2007/kws-data", "private": False}}))
    return {"GITHUB_REPOSITORY": run_identity["repository"], "GITHUB_SHA": run_identity["commit"],
        "GITHUB_RUN_ID": run_identity["run_id"], "GITHUB_RUN_ATTEMPT": run_identity["run_attempt"],
        "GITHUB_EVENT_NAME": run_identity["event"], "GITHUB_REF": "refs/heads/main", "GITHUB_EVENT_PATH": str(event)}


class MockSourceBackend(MockBackend):
    """Longer, unique fixture PCM permits whole phrase CTC tests without a model."""
    def materialize(self, audio, row, directory):
        amplitude = 120 + row["seed"] % 12
        native = struct.pack("<24000f", *((0.0, 0.1, -0.1) * 8000))
        pcm = struct.pack("<16000h", *((amplitude, -amplitude) * 8000))
        (directory / "native_24k.f32").write_bytes(native)
        (directory / "audio.pcm").write_bytes(pcm)
        with wave.open(str(directory / "audio.wav"), "wb") as stream:
            stream.setparams((1, 2, 16000, 16000, "NONE", "not compressed"))
            stream.writeframes(pcm)
        prefix = "rows/" + row["id"] + "/"
        return {"native_f32_path": prefix + "native_24k.f32", "native_f32_sha256": generate.sha(native),
            "native_samples": 24000, "native_sample_rate": 24000, "pcm_path": prefix + "audio.pcm",
            "pcm_sha256": generate.sha(pcm), "pcm_samples": 16000, "wav_path": prefix + "audio.wav",
            "wav_sha256": generate.digest(directory / "audio.wav"), "resample_calls": 1, "quantization_calls": 1,
            "actual_input_record_path": prefix + "NATIVE_REQUEST.json"}


class MockObserverFactory:
    """Patch observe._load only; exercise its real writer and readback protocols."""
    def __init__(self, source_root, fail=None):
        self.source_root, self.fail, self.calls = Path(source_root), fail, []
        self.source, self.rows = generate.validate_source(source_root)
        self.paths = {(self.source_root / r["wav_path"]).absolute(): r for r in self.rows}
        self.texts = {r["id"]: r["requested_text"] for r in self.rows}
        self.phone_tokens = {}
        for row in self.rows:
            cell = row["id"].split("-", 1)[1]
            tokens = {"mero": ["h", "eɪ", "m", "ɛ" if row["voice_id"] == "george" else "iː", "ɹ", "oʊ"],
                "vosen": ["h", "eɪ", "v", "oʊ", "s" if row["voice_id"] == "george" else "z", "ə", "n"],
                "marrow": ["h", "eɪ", "m", "æ", "ɹ", "oʊ"], "bosen": ["h", "eɪ", "b", "oʊ", "s", "ə", "n"],
                "hello": ["h", "ə", "l", "oʊ", "ð", "ɛ", "ɹ"],
                "morning": ["ɡ", "ʊ", "d", "m", "ɔː", "ɹ", "n", "ŋ"]}[cell]
            self.phone_tokens[row["id"]] = tokens

    def failing(self, provider, row, phase):
        if self.fail == (provider, row["id"], phase):
            raise RuntimeError("MOCK_ONLY_" + phase)

    def __call__(self, provider, assets, output):
        assets.mkdir()
        if provider != "phoneme":
            def decode(wav, folder):
                row = self.paths[wav.absolute()]
                self.calls.append((provider, row["id"], observe.sha((self.source_root / row["pcm_path"]).read_bytes())))
                self.failing(provider, row, "forward")
                return self.texts[row["id"]]
            lock = {"observer": observe.identity(provider), "runtime_versions": observe.TEXT_RUNTIME[provider],
                "adapter_sha256": observe.HELPER_PINS["automation/english_seven_voice_asr_once.py"],
                "initialization_model_calls": 0, "target_conditioned": False}
            return decode, lock
        metadata = json.loads(zlib.decompress(base64.b64decode(FIXED_METADATA_B64)))
        folder = output / "model-metadata"
        folder.mkdir()
        for name, payload in metadata.items():
            (folder / name).write_bytes(base64.b64decode(payload))
        vocab = json.loads((folder / "vocab.json").read_bytes())
        native, cursor = observe.native_module(), [0]
        def prepare(raw, geometry):
            row = self.rows[cursor[0]]
            cursor[0] += 1
            if observe.sha(raw) != row["pcm_sha256"] or set(geometry) != {"pcm_samples", "full_to_crop_offset_16k"}:
                raise ValueError("MOCK_REQUIRES_BLIND_SAME_PCM")
            self.failing(provider, row, "prepare")
            n = geometry["pcm_samples"]
            x = [v[0] / 32768 for v in struct.iter_unpack("<h", raw)]
            mean = math.fsum(x) / n
            scale = math.sqrt(math.fsum((v - mean)**2 for v in x) / n + 1e-7)
            normalized = struct.pack("<" + "f" * n, *((v - mean) / scale for v in x))
            facts = {"sample_rate": 16000, "pcm_samples": n, "normalized_shape": [1, n], "normalized_dtype": "torch.float32",
                "attention_mask_shape": [1, n], "attention_mask_dtype": "torch.int32", "attention_mask_sum": n,
                "attention_mask_unique_values": [1], "valid_input_samples": n, "padded_input_samples": n,
                "right_padding_samples": 0}
            return row["id"], facts, normalized, struct.pack("<i", 1) * n
        def forward(identifier):
            row = next(r for r in self.rows if r["id"] == identifier)
            self.calls.append((provider, row["id"], row["pcm_sha256"]))
            self.failing(provider, row, "forward")
            return identifier
        def materialize(identifier, geometry):
            row = next(r for r in self.rows if r["id"] == identifier)
            self.failing(provider, row, "materialize")
            frames = native.frame_geometry(geometry["pcm_samples"], 0)["frames"]
            tokens = self.phone_tokens[identifier]
            ids = [vocab[t] for t in tokens]
            ids += [0] * (frames - len(ids))
            raw = b"".join(struct.pack("<392f", *(1.0 if i == peak else -1.0 for i in range(392))) for peak in ids)
            values = {"shape": [1, frames, 392], "dtype": "float32-little-endian", "order": "C",
                "frame_ids": ids, "phone_tokens": tokens,
                "argmax_special_frame_counts": {str(k): ids.count(k) for k in (0, 1, 2, 3)}}
            return values, raw
        lock = {"model": native.OBSERVERS[provider], "files": native.OBSERVER_FILES[provider],
            "runtime_versions": native.RUNTIME_VERSIONS, "native_forward_attempts_at_initialization": 0,
            "model_parameters_modified": False, "target_conditioned": False}
        return prepare, forward, materialize, lock


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.plan, self.source = test_plan(), self.root / "source"
        MockSourceBackend.calls, MockSourceBackend.failing = [], None
        generate.generate(self.plan, self.source, self.root / "unused", RUN, backend_factory=MockSourceBackend)
        self.env = environment(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def run_observer(self, provider, factory=None):
        factory = factory or MockObserverFactory(self.source)
        output = self.root / provider
        with patch.dict("os.environ", self.env), patch.object(observe, "_load", side_effect=factory):
            result = observe.run(provider=provider, source_root=self.source, output_root=output,
                asset_root=self.root / (provider + "-assets"), plan=self.plan, run_identity=RUN)
        return output, result, factory

    def read(self, provider, output):
        return observe.readback(provider=provider, source_root=self.source, observer_root=output,
            plan=self.plan, run_identity=RUN)

    def test_source_check_is_stdlib_only(self):
        before = set(sys.modules)
        observe.source_check(self.plan)
        self.assertFalse({"torch", "numpy", "scipy", "qwen_asr", "transformers"} & (set(sys.modules) - before))

    def test_blind_text_and_phone_writer_readback_same_pcm(self):
        for provider in ("qwen", "phoneme"):
            output, result, factory = self.run_observer(provider)
            packet = self.read(provider, output)
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(packet["summary"]["actual_forward_attempts"], 12)
            self.assertEqual(len(factory.calls), 12)
            self.assertTrue(all(r["status"] == "success" for r in packet["rows"]))
            self.assertEqual([x[2] for x in factory.calls], [r["pcm_sha256"] for r in factory.rows])

    def test_native_return_survives_materialization_failure(self):
        factory = MockObserverFactory(self.source, ("phoneme", "george-mero", "materialize"))
        output, result, _ = self.run_observer("phoneme", factory)
        packet = self.read("phoneme", output)
        self.assertEqual((result["actual_forward_attempts"], result["native_forward_returns"], result["completed_count"]), (12, 12, 11))
        self.assertEqual(packet["summary"]["completed_count"], 11)
        self.assertEqual(packet["summary"]["failed_count"], 1)
        self.assertEqual(packet["rows"][0]["status"], "failed")
        self.assertEqual(len(factory.calls), 12)

    def test_failed_return_receipt_keeps_actual_native_return(self):
        real_write = observe.write
        def write(path, value):
            if path.name == "FORWARD_RETURNED.json" and path.parent.name == "george-mero":
                raise OSError("MOCK_RECEIPT_FAILURE")
            return real_write(path, value)
        with patch.object(observe, "write", side_effect=write):
            output, result, _ = self.run_observer("qwen")
        packet = self.read("qwen", output)
        self.assertEqual(result["native_forward_returns"], 12)
        self.assertEqual(packet["summary"]["native_forward_returns"], 12)
        self.assertEqual(packet["summary"]["completed_count"], 11)

    def test_missing_terminal_keeps_unknown_actual_count(self):
        output, _, _ = self.run_observer("qwen")
        (output / "RESULT.json").unlink()
        (output / "FILES.json").unlink()
        packet = self.read("qwen", output)
        self.assertIsNone(packet["summary"]["actual_forward_attempts"])
        self.assertEqual(packet["summary"]["attempts_lower_bound"], 12)
        self.assertEqual(len(packet["rows"]), 12)

    def test_native_failure_is_one_call_no_retry(self):
        factory = MockObserverFactory(self.source, ("qwen", "george-mero", "forward"))
        output, result, _ = self.run_observer("qwen", factory)
        self.assertEqual((result["actual_forward_attempts"], result["native_forward_returns"]), (12, 11))
        self.assertEqual(self.read("qwen", output)["summary"]["completed_count"], 11)
        self.assertEqual(len(set((p, i) for p, i, _ in factory.calls)), 12)

    def test_missing_provider_is_unknown_twelve(self):
        packet = self.read("qwen", self.root / "missing")
        self.assertIsNone(packet["summary"]["actual_forward_attempts"])
        self.assertEqual(packet["summary"]["attempts_upper_bound"], 12)
        self.assertEqual(len(packet["rows"]), 12)
        output = self.root / "unobserved"
        output.mkdir()
        observe.write(output / "UNOBSERVED.json", {"stage": "qwen", "run": RUN, "command_step_outcome": "failure",
            "has_terminal_result": False, "diagnostic_only": True, "training_allowed": False, "retry_count": 0,
            "missing_terminal_is_zero": False, "actual_model_calls": None})
        placeholder = self.read("qwen", output)
        self.assertIsNone(placeholder["summary"]["actual_forward_attempts"])
        self.assertIsNone(placeholder["summary"]["not_started_count"])
        self.assertEqual(placeholder["summary"]["unknown_count"], 12)

    def test_initialization_failure_has_zero_calls_with_terminal(self):
        def fail(*args):
            raise RuntimeError("MOCK_INIT")
        output, result, _ = self.run_observer("qwen", fail)
        self.assertEqual(result["not_started_count"], 12)
        summary = self.read("qwen", output)["summary"]
        self.assertEqual(summary["actual_forward_attempts"], 0)
        self.assertEqual(summary["not_started_count"], 12)

    def test_argmax_tamper_rejected_after_manifest_reseal(self):
        output, _, _ = self.run_observer("phoneme")
        path = output / "rows/george-mero/CAPTURE.json"
        capture = observe.read(path)
        capture["values"]["frame_ids"][0] = 0
        path.write_text(json.dumps(capture))
        (output / "FILES.json").unlink()
        observe.write(output / "FILES.json", {"schema": observe.SCHEMA + "/files", "files": observe.file_records(output),
                                              "self_excluded": "FILES.json"})
        with self.assertRaisesRegex(ValueError, "SAVED_LOGITS_ARGMAX_AND_CTC"):
            self.read("phoneme", output)


if __name__ == "__main__":
    unittest.main()
