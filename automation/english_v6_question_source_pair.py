"""One source-only, within-speaker comparison of two Piper inputs for the same English question."""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import random
import re
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b12_primary_source_qa as previous
from english_v6_primary_source_qa import admit, load_observations, observers, pcm_tools, policy, record

b10 = previous.b10
need = previous.qa.need
SCHEMA = "english-v6-question-source-paired-diagnostic-v1"
QUESTION = "What time is it?"
IPA = "wˈʌt tˈaɪm ɪz ɪt"
COHORT = ((594, "p1343"), (595, "p7241"), (596, "p451"), (597, "p5401"))
LAST_SOURCE = "e42505142442a735aa3aa4096fda22a9d2ac83bfbba678abfad7011cf3a3ce83"


def history_locks():
    pins = json.loads((previous.PINS / "source_archive_hashes.json").read_bytes())
    return dict(pins, **{"14784071420": LAST_SOURCE})


def historical(directory):
    hashes, groups = set(), set()
    for aid, digest in history_locks().items():
        path = directory / (aid + ".zip")
        need(previous.qa.sha(path) == digest, "historical archive hash " + aid)
        with zipfile.ZipFile(path) as archive:
            rows = json.loads(archive.read("MANIFEST.json"))["rows"]
        hashes.update(r["pcm_sha256"] for r in rows)
        groups.update(r["source_group_id"] for r in rows)
    need((len(hashes), len(groups)) == (1148, 212), "historical source coverage")
    return hashes, groups


def planned():
    rows = []
    for order, (index, speaker) in enumerate(COHORT):
        # Opaque WAV identifiers and counterbalanced collection order. Observers get PCM only.
        modes = ("text", "ipa") if order % 2 == 0 else ("ipa", "text")
        for slot, mode in enumerate(modes):
            rows.append(dict(id=f"qpair-{order * 2 + slot + 1:02d}", speaker_index=index,
                             source_speaker_id=speaker, source_group_id="piper-libritts-" + speaker,
                             mode=mode, submitted_input=QUESTION if mode == "text" else "[[" + IPA + "]]",
                             cell_id="question", requested_text=QUESTION, requested_class=0,
                             role="source_diagnostic_only", training_allowed=False,
                             length_scale=1.0, noise_scale=0.333, noise_w_scale=0.333,
                             seed=20261123 + order))
    return rows


def study_plan():
    return dict(schema=SCHEMA, model=b10.plan()["model"], rows=planned(),
                historical_archive_hashes=history_locks(), max_tts_calls=8, max_native_observers=40,
                promotion_allowed=False, original_probe="B10: 12/12 admitted; B11/B12: 0/12; different speakers",
                qualification="existing exact-sentence majority of three blind transcripts, authentic two-phone veto",
                primary="IPA-only admitted in >=3 of four matched speakers with zero reverse pairs",
                reverse="Text-only admitted in >=3 of four matched speakers with zero reverse pairs",
                otherwise="no consistent directional advantage on this four-speaker diagnostic")


def check_cohort(config, groups):
    previous.verify_config(config)
    mapping = json.loads(config.read_bytes())["speaker_id_map"]
    prior = b10.previous.old
    blacklisted = {str(n) for n in prior.OLD_PIPER_IDS + prior.OLD_LIBRI_IDS}
    blacklisted.update(str(s) for _, s in prior.NEW_VOICES)
    for idx, person in COHORT:
        digits = person.removeprefix("p")
        need(mapping.get(person) == idx and idx >= 594 and digits not in blacklisted and
             idx not in dict(prior.NEW_VOICES), "historically independent source lineage")
        need(not any(re.search(r"(?<!\d)" + digits + r"(?!\d)", gid) for gid in groups),
             "historical speaker identity overlap")


def generate(out, hist):
    import numpy as np
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from piper import PiperVoice, SynthesisConfig
    from scipy.signal import resample_poly

    seen, old_groups = historical(hist)
    model = study_plan()["model"]
    weights, config = [Path(hf_hub_download(repo_id=model["repo"], filename=model[k],
                       revision=model["commit"], token=False)) for k in ("file", "config")]
    check_cohort(config, old_groups)
    need(version("piper-tts") == model["piper_tts_version"] and
         weights.stat().st_size == model["file_bytes"] and b10.previous.digest(weights) == model["file_sha256"],
         "authenticated source engine and checkpoint")
    out.mkdir()
    record(out / "PLAN.json", study_plan())
    rows = []
    for recipe in planned():
        sid = recipe["id"]
        seed = recipe["seed"]
        random.seed(seed)
        np.random.seed(seed)
        ort.set_seed(seed)
        voice = PiperVoice.load(str(weights), config_path=str(config))
        cfg = SynthesisConfig(speaker_id=recipe["speaker_index"], length_scale=recipe["length_scale"],
                              noise_scale=recipe["noise_scale"], noise_w_scale=recipe["noise_w_scale"],
                              normalize_audio=False, volume=1.0)
        record(out / "starts" / (sid + ".json"), dict(id=sid, attempt=1, seed=seed))
        chunks = list(voice.synthesize(recipe["submitted_input"], cfg))
        need(chunks and all((ch.sample_rate, ch.sample_width, ch.sample_channels) == (22050, 2, 1)
                           for ch in chunks), "all native PCM chunks")
        pieces, receipts = [], []
        native_dir = out / "native" / sid
        native_dir.mkdir(parents=True)
        for pos, chunk in enumerate(chunks):
            data = chunk.audio_int16_bytes
            name = f"{pos:03d}.s16le"
            (native_dir / name).write_bytes(data)
            pieces.append(data)
            receipts.append(dict(file=name, sha256=b10.sha(data), samples=len(data)//2,
                                 phonemes=chunk.phonemes, phoneme_ids=chunk.phoneme_ids))
            need(voice.phonemes_to_ids(chunk.phonemes) == chunk.phoneme_ids, "native phoneme IDs")
        if recipe["mode"] == "ipa":
            need([t for c in receipts for t in c["phonemes"]] == list(IPA), "exact B10 direct IPA")
        native = b"".join(pieces)
        signal = np.frombuffer(native, dtype="<i2").astype(np.float32) / 32768.
        derived = resample_poly(signal, 320, 441)
        need(len(derived) > 0 and np.isfinite(derived).all(), "complete finite resampling")
        pcm = np.rint(np.clip(derived, -1, 32767/32768) * 32768.).astype("<i2").tobytes()
        wav = out / "wav" / (sid + ".wav")
        b10.previous.wav_write(wav, pcm, 16000)
        facts = pcm_tools.inspect_wav(wav)
        need(facts["basic_audio_ok"] and facts["pcm_sha256"] not in seen, "original auditable unique PCM")
        seen.add(facts["pcm_sha256"])
        rows.append(dict(recipe, **facts, native_chunks=receipts, native_pcm_sha256=b10.sha(native),
                         native_samples=len(native)//2, whole_native_resample=True, crop_calls=0))
    record(out / "MANIFEST.json", dict(schema=SCHEMA, plan_sha256=b10.sha(b10.canon(study_plan())),
           rows=rows, tts_calls=len(rows), piper=version("piper-tts"), onnxruntime=version("onnxruntime"),
           scipy=version("scipy"), numpy=version("numpy"), fresh_cpu_session_each_input=True,
           normalized_audio=False, volume=1., human_gold=False, kws_forwards=0, optimizer_updates=0))
    read_source(out)


def read_source(source):
    doc = json.loads((source / "MANIFEST.json").read_bytes())
    need(doc["schema"] == SCHEMA and doc["tts_calls"] == len(doc["rows"]) == 8 and
         doc["plan_sha256"] == b10.sha(b10.canon(study_plan())) and
         json.loads((source / "PLAN.json").read_bytes()) == study_plan(), "frozen source comparison")
    for row, recipe in zip(doc["rows"], planned(), strict=True):
        need(all(row[k] == v for k,v in recipe.items()), "registered arm and speaker")
        for k, v in pcm_tools.inspect_wav(source / "wav" / (row["id"] + ".wav")).items():
            need(row[k] == v, "derived audio integrity: " + k)
        combined = hashlib.sha256()
        for i, c in enumerate(row["native_chunks"]):
            need(c["file"] == f"{i:03d}.s16le", "native chunk index")
            data = (source / "native" / row["id"] / c["file"]).read_bytes()
            need(len(data) == 2*c["samples"] and b10.sha(data) == c["sha256"], "native PCM bytes")
            combined.update(data)
        need(row["native_chunks"] and combined.hexdigest() == row["native_pcm_sha256"], "all native chunks")
    return doc["rows"]


def conclude(pairs):
    ipa = sum(p["ipa_admitted"] and not p["text_admitted"] for p in pairs)
    text = sum(p["text_admitted"] and not p["ipa_admitted"] for p in pairs)
    if ipa >= 3 and text == 0:
        return "IPA_ADVANTAGE_ON_MATCHED_SOURCES"
    if text >= 3 and ipa == 0:
        return "TEXT_ADVANTAGE_ON_MATCHED_SOURCES"
    return "NO_CONSISTENT_DIRECTION"


def screen(source, evidence, out):
    rows = read_source(source)
    texts, phones, coverage = load_observations(source, evidence, rows, SCHEMA)
    decisions = []
    for r in rows:
        sid = r["id"]
        label = admit(r, texts[sid], phones[sid])
        decisions.append(dict(id=sid, group=r["source_group_id"], mode=r["mode"],
                              pcm_sha256=r["pcm_sha256"], samples=r["samples"],
                              texts=texts[sid], phones_sha256={p:a["sha256"] for p,a in phones[sid].items()},
                              exact_votes=sum(policy.norm(x) == policy.norm(QUESTION) for x in texts[sid].values()),
                              **label))
    pairs = []
    for _, speaker in COHORT:
        group = "piper-libritts-" + speaker
        matched = {r["mode"]:r for r in decisions if r["group"] == group}
        pairs.append(dict(group=group, text_admitted=matched["text"]["class"] == 0,
                          ipa_admitted=matched["ipa"]["class"] == 0,
                          text_samples=matched["text"]["samples"], ipa_samples=matched["ipa"]["samples"]))
    record(out, dict(schema=SCHEMA, coverage=coverage, rows=decisions, pairs=pairs, result=conclude(pairs),
                     source_manifest_sha256=previous.qa.sha(source/"MANIFEST.json"),
                     kws_forwards=0, optimizer_updates=0, training_allowed=False, promotion_allowed=False))
    print(json.dumps(pairs), conclude(pairs))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    sub.add_parser("history")
    sub.add_parser("plan")
    v=sub.add_parser("generate"); v.add_argument("out",type=Path); v.add_argument("historical",type=Path)
    v=sub.add_parser("observe"); v.add_argument("provider",choices=observers.PROVIDERS); v.add_argument("source",type=Path); v.add_argument("out",type=Path)
    v=sub.add_parser("screen"); v.add_argument("source",type=Path); v.add_argument("evidence",type=Path); v.add_argument("out",type=Path)
    a=p.parse_args()
    if a.mode=="history": print("\n".join(history_locks()))
    elif a.mode=="plan": print(json.dumps(study_plan(), indent=2))
    elif a.mode=="generate": generate(a.out,a.historical)
    elif a.mode=="observe": observers.observe(a.provider,a.source,a.out,read_rows=read_source,schema=SCHEMA)
    else: screen(a.source,a.evidence,a.out)
