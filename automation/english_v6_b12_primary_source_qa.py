"""B12 prospective primary source generation and blind admission."""
from __future__ import annotations
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import re
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_v6_primary_source_qa as qa
from english_v6_b11_dev_panel import b10, IPA
from english_v6_b11_dev_panel import verify_config
from english_v6_b11_dev_panel import GROUPS as B11_GROUPS

PINS=Path(__file__).with_name("english_v6_b12_panels")
PRIMARY_SHA="75abad59c37c1953b74566607ced904f43ed2d65e7c624da64e0c55f6f19c995"
REPLICA_SHA="dd49276732f28c9dcd6af6248b6cdce48ca2dfc18dfd232d72b22d35ab6283b4"
SCHEMA="english-v6-b12-frozen-dev-panel-v1"
QA_SCHEMA="english-v6-b12-primary-blind-source-qa-v1"

def source_plan(name):
    qa.need(name in ("primary","replication"),"B12 panel")
    p=PINS/(name+".json")
    qa.need(qa.sha(p)==(PRIMARY_SHA if name=="primary" else REPLICA_SHA),"pre-outcome B12 recipe SHA")
    plan=json.loads(p.read_bytes())
    qa.need(plan["schema"]==SCHEMA and plan["panel"]==name and len(plan["rows"])==48,"exact B12 recipe")
    return plan

def check_stop(archive):
    qa.need(qa.sha(archive)=="00fbad5f68b2a4db707e5be2d1d6577781a8a2217db40fca20ec66b0318d74b4","B11 terminal outcome archive")
    with zipfile.ZipFile(archive) as z:
        qa.need(z.testzip() is None,"B11 CRC")
        doc=json.loads(z.read("RESULT.json"))
        raw=z.read("ALL_FRAME_LOGITS.npz")
    qa.need(doc["schema"]=="english-v6-4-frozen-panel-v1" and doc["trial"]["hypothesis"]=="B11"
            and doc["trial"]["panel"]=="primary" and doc["panel_gate"]=="NO_PROMOTION"
            and doc["actual_forwards"]==276 and len(doc["improved_groups"])==1
            and hashlib.sha256(raw).hexdigest()==doc["logits_sha256"], "B11 terminal result")

def history(directory):
    locks=json.loads((PINS/"source_archive_hashes.json").read_bytes());hashes=set();groups=set()
    for aid,expected in locks.items():
        p=directory/(aid+".zip")
        qa.need(qa.sha(p)==expected,"previous source archive digest "+aid)
        with zipfile.ZipFile(p) as z:
            entries=json.loads(z.read("MANIFEST.json"))["rows"]
        for row in entries:
            hashes.add(row["pcm_sha256"]);groups.add(row["source_group_id"])
    qa.need(len(hashes)>=500,"previous source originals")
    return hashes,groups

def check_speakers(config_path, historical_groups):
    verify_config(config_path)
    cfg=json.loads(Path(config_path).read_bytes());mapping=cfg["speaker_id_map"]
    prior=b10.previous.old
    used={s for s,i in mapping.items() if i<58 or 400<=i<570 or i in dict(prior.NEW_VOICES)}
    used |= {"p"+str(k) for k in prior.OLD_PIPER_IDS+prior.OLD_LIBRI_IDS}
    used |= {"p"+str(s) for _,s in prior.NEW_VOICES}
    groups=set();indices=set()
    for name in ("primary","replication"):
        plan=source_plan(name)
        for idx,sp in {(r["speaker_index"],r["source_speaker_id"]) for r in plan["rows"]}:
            qa.need(mapping.get(sp)==idx and sp not in used and sp not in groups and
                    idx not in indices and "piper-libritts-"+sp not in historical_groups,"disjoint reserved B12 speaker")
            groups.add(sp);indices.add(idx)
    qa.need(len(groups)==24 and indices==set(range(570,594)),"B12 primary and reserved replication")

def read_source(path):
    return qa.read_panel(path,"primary",source_plan)
def generate_panel(weights_path, config_path, output, historical_hashes, historical_groups):
    panel="primary"
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
    check_speakers(config_path, historical_groups)
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
    seen = set(historical_hashes)
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
                   "registered_plan_file_sha256": PRIMARY_SHA, "new_kws_forwards": 0, "human_gold": False, "rows": completed})


def generate(output, previous):
    from huggingface_hub import hf_hub_download
    hashes,groups=history(previous)
    model=source_plan("primary")["model"]
    paths=[Path(hf_hub_download(repo_id=model["repo"],filename=model[k],revision=model["commit"],token=False)) for k in ("file","config")]
    generate_panel(*paths,output,hashes,groups)
    read_source(output)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    v=sub.add_parser("gate");v.add_argument("archive",type=Path)
    sub.add_parser("history")
    v=sub.add_parser("generate");v.add_argument("out",type=Path);v.add_argument("previous",type=Path)
    v=sub.add_parser("observe");v.add_argument("provider",choices=qa.observers.PROVIDERS);v.add_argument("source",type=Path);v.add_argument("out",type=Path)
    v=sub.add_parser("screen")
    for n in ("source","evidence","out"):v.add_argument(n,type=Path)
    a=p.parse_args()
    if a.mode=="gate":check_stop(a.archive)
    elif a.mode=="history":
        for aid in json.loads((PINS/"source_archive_hashes.json").read_bytes()):print(aid)
    elif a.mode=="generate":generate(a.out,a.previous)
    elif a.mode=="observe":qa.observers.observe(a.provider,a.source,a.out,read_rows=read_source,schema=QA_SCHEMA)
    else:qa.screen(a.source,a.evidence,a.out,schema=QA_SCHEMA,recipe=source_plan)
