"""Frozen source design and one-panel Piper generator for B11 independent DEV.

The CLI emits plans and checks speaker ancestry without generating audio. A
separate, evidence-gated workflow may call generate_panel once per cohort.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b10_hardneg_train_dev_source_once as b10

CONFIG_SHA256 = "2efdc6d7f954588b8180132cbd9b8001933fdd00932c92bc92fd0d2028a9eb3d"
GROUPS = {
    "primary": ((546, "p6317"), (547, "p204"), (548, "p8006"), (549, "p6895"),
                (550, "p1264"), (551, "p2494"), (552, "p112"), (553, "p1859"),
                (554, "p398"), (555, "p1052"), (556, "p3294"), (557, "p1460")),
    "replication": ((558, "p8573"), (559, "p5684"), (560, "p8421"), (561, "p5883"),
                    (562, "p7297"), (563, "p246"), (564, "p8057"), (565, "p3835"),
                    (566, "p1748"), (567, "p3816"), (568, "p3357"), (569, "p1053")),
}
ORDINARY = (("question", "What time is it?"), ("statement", "The door is closed."),
            ("request", "Please set a timer."), ("yes-no", "Can you hear me?"))
RATES = (0.92, 1.08)
IPA = "heɪ mˈɛɹoʊ"
SCHEMA = "english-v6-b11-frozen-dev-panel-v1"


def planned(panel):
    rows = []
    for k, (idx, speaker) in enumerate(GROUPS[panel]):
        cells = [("mero-fast", "Hey Mero.", RATES[0], 2),
                 ("mero-slow", "Hey Mero.", RATES[1], 2)]
        for j in range(2):
            name, text = ORDINARY[(k + 2 * j) % len(ORDINARY)]
            cells.append((name, text, RATES[(k // 4 + j) % 2], 0))
        for name, text, rate, target in cells:
            rows.append({"id": f"b11-{panel}-{idx}-{name}", "panel": panel,
                         "speaker_index": idx, "source_speaker_id": speaker,
                         "source_group_id": "piper-libritts-" + speaker,
                         "role": "development", "cell_id": name, "requested_text": text,
                         "length_scale": rate, "requested_class": target,
                         "observed_class": None, "training_allowed": False})
    return rows


def check_mapping(mapping):
    b10.need(len(mapping) == 904 and set(mapping.values()) == set(range(904)),
             "Incomplete native speaker mapping")
    old = b10.previous.old
    excluded_indices = set(range(58)) | set(range(400, 546)) | {i for i, _ in old.NEW_VOICES}
    excluded_speakers = {name for name, idx in mapping.items() if idx in excluded_indices}
    excluded_speakers |= {"p" + str(s) for s in old.OLD_PIPER_IDS + old.OLD_LIBRI_IDS}
    excluded_speakers |= {"p" + str(s) for _, s in old.NEW_VOICES}
    reserved = GROUPS["primary"] + GROUPS["replication"]
    b10.need(len({s for _, s in reserved}) == len({i for i, _ in reserved}) == 24,
             "Duplicate prospective source identity")
    for idx, speaker in reserved:
        b10.need(idx not in excluded_indices and speaker not in excluded_speakers and
                 mapping.get(speaker) == idx, "Exposed or mismatched native speaker: " + speaker)


def verify_config(config_path):
    raw = Path(config_path).read_bytes()
    b10.need(b10.sha(raw) == CONFIG_SHA256, "Official Piper configuration SHA mismatch")
    config = json.loads(raw)
    b10.need(config["num_speakers"] == 904, "Expected LibriTTS 904")
    check_mapping(config["speaker_id_map"])


def source_plan(panel):
    return {"schema": SCHEMA, "panel": panel, "model": b10.plan()["model"],
            "config_sha256": CONFIG_SHA256, "rows": planned(panel), "max_source_calls": 48,
            "max_native_observer_calls": 240, "zero_kws_or_training_calls_at_source_stage": True,
            "unknown_excluded_from_scoring": True, "required_source_review": [
                "qwen", "whisper", "parakeet", "xlsr", "lv60"],
            "replication_open_condition": "first independent DEV comparison passed without retuning"}


def generate_panel(panel, weights_path, config_path, output):
    """Synthesize the predeclared panel after external measured-source admission.

    Call only once for each panel, using the public workflow's durable attempt
    marker. Replication stays sealed until a qualifying primary result.
    """
    import numpy as np
    from piper import PiperVoice, SynthesisConfig
    from scipy.signal import resample_poly

    plan = source_plan(panel)
    model = plan["model"]
    weights = Path(weights_path)
    verify_config(config_path)
    b10.need(version("piper-tts") == model["piper_tts_version"] and
             weights.stat().st_size == model["file_bytes"] and
             b10.previous.digest(weights) == model["file_sha256"], "Wrong source model or engine")
    voice = PiperVoice.load(str(weights), config_path=str(config_path))
    b10.need(voice.config.sample_rate == 22050 and voice.config.num_speakers == 904,
             "Wrong native source geometry")
    output = Path(output)
    output.mkdir()
    b10.write_once(output / "STARTED.json", {"schema": SCHEMA, "panel": panel,
                   "plan_sha256": b10.sha(b10.canon(plan)), "retry_count": 0})
    seen = set()
    completed = []
    for row in plan["rows"]:
        sid = row["id"]
        text = row["requested_text"]
        if row["requested_class"] == 2:
            text = "[[" + IPA + "]]"
            b10.need([s for seq in voice.phonemize(text) for s in seq] == list(IPA),
                     "Canonical direct-IPA wake pronunciation")
        cfg = SynthesisConfig(speaker_id=row["speaker_index"], length_scale=row["length_scale"],
                              noise_scale=0.333, noise_w_scale=0.333,
                              normalize_audio=False, volume=1.0)
        b10.write_once(output / "starts" / (sid + ".json"), {"id": sid, "attempt": 1})
        chunks = list(voice.synthesize(text, cfg))
        b10.need(bool(chunks) and all((c.sample_rate, c.sample_channels, c.sample_width) ==
                 (22050, 1, 2) for c in chunks), "Native PCM format")
        native = b"".join(c.audio_int16_bytes for c in chunks)
        signal = np.frombuffer(native, dtype="<i2").astype(np.float32) / 32768.0
        resampled = resample_poly(signal, 320, 441)
        b10.need(3200 <= len(resampled) <= 240000 and np.isfinite(resampled).all(),
                 "Whole native source duration or nonfinite signal")
        pcm = np.rint(np.clip(resampled, -1, 32767/32768) * 32768).astype("<i2").tobytes()
        wav = output / "wav" / (sid + ".wav")
        b10.previous.wav_write(wav, pcm, 16000)
        facts = b10.previous.c.inspect_wav(wav)
        b10.need(facts["basic_audio_ok"] and facts["pcm_sha256"] not in seen,
                 "Low-quality or duplicate original")
        seen.add(facts["pcm_sha256"])
        completed.append({**row, **facts, "native_pcm_sha256": b10.sha(native),
                          "whole_native_resample": True, "crop_calls": 0,
                          "positive_duration_pass": row["requested_class"] != 2 or
                          facts["samples"] <= b10.MAX_POSITIVE_SAMPLES})
    b10.write_once(output / "MANIFEST.json", {"schema": SCHEMA, "panel": panel,
                   "plan_sha256": b10.sha(b10.canon(plan)), "tts_calls": len(completed),
                   "new_kws_forwards": 0, "human_gold": False, "rows": completed})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", choices=GROUPS)
    parser.add_argument("--verify-config", type=Path)
    args = parser.parse_args()
    if args.verify_config is not None:
        verify_config(args.verify_config)
    print(json.dumps(source_plan(args.panel), ensure_ascii=False, indent=2))
