"""B11's one supplemental TRAIN batch. Commands share the committed cell design."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b10_hardneg_train_dev_source_once as prior
import english_fresh24_ctc_admission as policy

# Frozen before synthesis. Cross every text with six previously qualified TRAIN
# identities, rotating the four rates so text and rate do not encode a class.
SENTENCES = (
    ("question", "Where did my keys go?"),
    ("statement", "The mail came early."),
    ("command", "Turn the radio down."),
    ("request", "Could you open the window?"),
)
RATES = (0.88, 0.96, 1.04, 1.12)
PROVIDERS = policy.TEXT_MODELS + policy.PHONE_MODELS
BATCH = "b11-diverse-train-supplement-v1"
pcm_tools = prior.previous.c
record = pcm_tools.record
need = pcm_tools.need


def planned():
    rows = []
    for i, group in enumerate(prior.plan()["train_groups"]):
        for j, (form, text) in enumerate(SENTENCES):
            rows.append({
                **group, "id": f"b11-train-{group['speaker_index']}-{form}",
                "text": text, "form": form, "rate": RATES[(i + j) % len(RATES)],
                "requested_class": 0, "observed_class": None,
            })
    return rows


def generate(output, old_source, old_screen):
    from huggingface_hub import hf_hub_download
    import numpy as np
    from piper import PiperVoice, SynthesisConfig
    from scipy.signal import resample_poly

    output.mkdir()
    rows = planned()
    original = prior.original_train_groups(old_source, old_screen)
    need(len(rows) == len({r["id"] for r in rows}) == 24 and
         len({r["source_group_id"] for r in rows}) == len(original) == 6,
         "exclusive TRAIN cells")
    record(output / "STARTED.json", {"batch": BATCH, "plan": rows, "limit": 24,
           "code_sha": os.environ["GITHUB_SHA"], "run": os.environ["GITHUB_RUN_ID"]})
    model = prior.plan()["model"]
    model_paths = [Path(hf_hub_download(repo_id=model["repo"], filename=model[key],
                                        revision=model["commit"], token=False))
                   for key in ("file", "config")]
    weights, config = model_paths
    need(weights.stat().st_size == model["file_bytes"] and
         prior.previous.digest(weights) == model["file_sha256"] and
         config.stat().st_size == model["config_bytes"] and
         hashlib.md5(config.read_bytes()).hexdigest() == model["config_md5"], "Piper model lock")
    mapping = json.loads(config.read_bytes())["speaker_id_map"]
    need(all(mapping[r["source_speaker_id"]] == r["speaker_index"] for r in rows),
         "pinned original TRAIN speaker-index map")
    voice = PiperVoice.load(str(weights), config_path=str(config))
    forbidden = {h for family in original.values() for h in family["old_train_pcm_sha256"]}
    seen = set()
    finished = []
    for row in rows:
        sid = row["id"]
        record(output / "starts" / (sid + ".json"), {"id": sid, "attempt": 1})
        cfg = SynthesisConfig(speaker_id=row["speaker_index"], length_scale=row["rate"],
                              noise_scale=0.333, noise_w_scale=0.333,
                              normalize_audio=False, volume=1.0)
        chunks = list(voice.synthesize(row["text"], cfg))
        need(len(chunks) >= 1 and all((a.sample_rate, a.sample_width, a.sample_channels) ==
             (22050, 2, 1) for a in chunks), "complete native 22.05k PCM")
        native = b"".join(a.audio_int16_bytes for a in chunks)
        audio = np.frombuffer(native, dtype="<i2").astype(np.float32) / 32768
        signal = resample_poly(audio, 320, 441)
        pcm = np.rint(np.clip(signal, -1., 32767 / 32768) * 32768).astype("<i2").tobytes()
        path = output / "wav" / (sid + ".wav")
        prior.previous.wav_write(path, pcm, 16000)
        facts = pcm_tools.inspect_wav(path)
        need(facts["basic_audio_ok"] and facts["pcm_sha256"] not in seen | forbidden,
             "distinct whole PCM")
        seen.add(facts["pcm_sha256"])
        finished.append({**row, **facts, "native_pcm_sha256": hashlib.sha256(native).hexdigest()})
    record(output / "MANIFEST.json", {"batch": BATCH, "run": os.environ["GITHUB_RUN_ID"],
           "model": model, "code_sha": os.environ["GITHUB_SHA"], "tts_calls": len(finished),
           "rows": finished})
    read_source(output)


def read_source(source):
    manifest = json.loads((source / "MANIFEST.json").read_bytes())
    rows = manifest["rows"]
    need(manifest["batch"] == BATCH and manifest["tts_calls"] == len(rows) == 24
         and manifest["model"] == prior.plan()["model"] and
         len({r["pcm_sha256"] for r in rows}) == 24, "complete fixed TRAIN batch")
    for actual, frozen in zip(rows, planned(), strict=True):
        need(all(actual[k] == v for k, v in frozen.items()) and actual["role"] == "train",
             "source role/text/rate drift")
        facts = pcm_tools.inspect_wav(source / "wav" / (actual["id"] + ".wav"))
        need(facts["basic_audio_ok"] and all(actual[k] == v for k, v in facts.items()),
             "original bytes and audio quality")
    return rows


def decision(row, texts, phones):
    """Extend the previous single-sentence policy with exact same-sentence agreement."""
    import re
    if set(texts) != set(policy.TEXT_MODELS) or set(phones) != set(policy.PHONE_MODELS):
        return "unknown"
    normalized = [policy.norm(texts[name]) for name in policy.TEXT_MODELS]
    expected = policy.norm(row["text"])
    if expected not in {policy.norm(t) for _, t in SENTENCES}:
        return "unknown"
    if any(re.search(r"\b(?:hey|hay|mero|meero|miro|marrow|merrow)\b", value)
           for value in normalized):
        return "unknown"
    for name in policy.PHONE_MODELS:
        p = phones[name]
        if hashlib.sha256(p["logits"]).hexdigest() != p["sha256"]:
            raise ValueError("inconsistent original phone evidence")
        policy.verified_phone_input(p["logits"], p["vocab"], pad_token_id=0)
        if policy.greedy_full_wake(p["logits"], p["vocab"]):
            return "unknown"
    votes = normalized.count(expected)
    return {3: "silver", 2: "weak"}.get(votes, "unknown")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "generate", "readback"))
    p.add_argument("--out", type=Path)
    p.add_argument("--old-source", type=Path)
    p.add_argument("--old-screen", type=Path)
    args = p.parse_args()
    if args.mode == "plan":
        print(json.dumps(planned(), indent=2))
    elif args.mode == "generate":
        generate(args.out, args.old_source, args.old_screen)
    else:
        print(f"{len(read_source(args.out))} original TRAIN WAVs verified")
