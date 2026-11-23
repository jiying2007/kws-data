"""Generate and blind-screen the preregistered B13 synthetic companion primary panel."""
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
import english_v6_b12_primary_source_qa as b12

qa = b12.qa
b10 = b12.b10
PINS = Path(__file__).with_name("english_v6_b13_panels")
FILES = {"primary":"3832279dadc95dabb622af08c61f9d1bd9fe74282b43660421569f7fb15a758a",
         "replication":"f454483ea44e622a3640e28cfc07d60b8e1939a189f874f3b87a5eb218496de6"}
SCHEMA = "english-v6-b13-primary-blind-source-review-v1"
QUESTION_IPA = "wˈʌt tˈaɪm ɪz ɪt"
QUESTION_SOURCE = "12285464b64f6dbe5c37ab3fb17e9fac887c7c1cc84ce11c40d115226c701b1c"
QUESTION_SCREEN = "d3aabb070dd1e98ca5b2c785eeac11c486e77cf0eeb0650dfbc109bc961c6e26"


def source_plan(panel):
    path = PINS / (panel + ".json")
    qa.need(panel in FILES and qa.sha(path) == FILES[panel], "frozen B13 source role")
    doc = json.loads(path.read_bytes())
    qa.need(len(doc["rows"]) == 48 and doc["schema"] == "english-v6-b13-frozen-dev-panel-v1"
            and doc["panel"] == panel, "complete registered source roster")
    return doc


def gate(path):
    qa.need(qa.sha(path) == QUESTION_SCREEN, "authenticated paired question outcome archive")
    with zipfile.ZipFile(path) as z:
        qa.need(z.testzip() is None, "paired study integrity")
        doc = json.loads(z.read("SCREEN.json"))
    qa.need(doc["result"] == "IPA_ADVANTAGE_ON_MATCHED_SOURCES" and len(doc["pairs"]) == 4 and
            all(p["ipa_admitted"] and not p["text_admitted"] for p in doc["pairs"]) and
            doc["coverage"] == dict.fromkeys(qa.observers.PROVIDERS, 8) and
            doc["training_allowed"] is False and doc["kws_forwards"] == 0,
            "prospective four-pair question criterion")


def locks():
    old = json.loads((b12.PINS / "source_archive_hashes.json").read_bytes())
    old["14784071420"] = "e42505142442a735aa3aa4096fda22a9d2ac83bfbba678abfad7011cf3a3ce83"
    old["14788787525"] = QUESTION_SOURCE
    return old


def history(directory):
    hashes, groups = set(), set()
    for aid, digest in locks().items():
        archive = directory / (aid + ".zip")
        qa.need(qa.sha(archive) == digest, "historical source integrity " + aid)
        with zipfile.ZipFile(archive) as z:
            rows = json.loads(z.read("MANIFEST.json"))["rows"]
        hashes.update(r["pcm_sha256"] for r in rows)
        groups.update(r["source_group_id"] for r in rows)
    qa.need((len(hashes),len(groups)) == (1156,216), "historical source identity coverage")
    return hashes, groups


def groups_check(config, history_groups, *, plan_reader=source_plan, first=598, extra_people=()):
    b12.verify_config(config)
    mapping = json.loads(config.read_bytes())["speaker_id_map"]
    prior = b10.previous.old
    old = {str(s) for s in (*prior.OLD_PIPER_IDS, *prior.OLD_LIBRI_IDS, *(v for _,v in prior.NEW_VOICES))}
    owner = json.loads((PINS.parents[1] / "research/english_hardneg_increment_20261117/OWNER_REVIEW.json").read_bytes())
    old.update(r["source_id"].split("-")[0] for r in owner["candidates"])
    old.update(extra_people)
    disjoint = set()
    for name, start in (("primary",first),("replication",first+12)):
        roster = {(r["speaker_index"],r["source_speaker_id"]) for r in plan_reader(name)["rows"]}
        qa.need({i for i, _ in roster} == set(range(start, start+12)) and len(roster)==12, "reserved source range")
        for idx, person in roster:
            digits = person.removeprefix("p")
            qa.need(mapping.get(person) == idx and digits not in old and person not in disjoint
                    and not any(re.search(r"(?<!\d)"+digits+r"(?!\d)", g) for g in history_groups),
                    "speaker ancestry and source-role disjointness")
            disjoint.add(person)


def generate(out, original_archives):
    seen, old_groups = history(original_archives)
    render_panel(out, source_plan("primary"), FILES["primary"], seen,
                 lambda config: groups_check(config, old_groups))
    read_source(out)


def render_panel(out, plan, file_sha, seen, verify_config):
    import numpy as np
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from piper import PiperVoice, SynthesisConfig
    from scipy.signal import resample_poly

    model = plan["model"]
    weights, config = [Path(hf_hub_download(repo_id=model["repo"],filename=model[k],
                       revision=model["commit"],token=False)) for k in ("file","config")]
    verify_config(config)
    qa.need(version("piper-tts") == model["piper_tts_version"] and weights.stat().st_size == model["file_bytes"]
            and b10.previous.digest(weights) == model["file_sha256"], "frozen source engine and weights")
    out.mkdir()
    qa.record(out / "STARTED.json", {"plan_sha256":file_sha, "one_shot":True})
    rows = []
    for row in plan["rows"]:
        sid = row["id"]
        seed = row["seed"]
        random.seed(seed)
        np.random.seed(seed)
        ort.set_seed(seed)
        voice = PiperVoice.load(str(weights),config_path=str(config))
        input_text = row["native_input"]
        target = row["requested_class"]
        cell = row["cell_id"]
        expected = "[["+b12.IPA+"]]" if target == 2 else ("[["+QUESTION_IPA+"]]" if cell == "question" else row["requested_text"])
        qa.need(input_text == expected, "registered native text/IPA input")
        cfg = SynthesisConfig(speaker_id=row["speaker_index"],length_scale=row["length_scale"],
                              noise_scale=.333,noise_w_scale=.333,normalize_audio=False,volume=1.)
        qa.record(out/"starts"/(sid+".json"), {"id":sid,"attempt":1,"seed":seed})
        chunks = list(voice.synthesize(input_text,cfg))
        qa.need(chunks and all((c.sample_rate,c.sample_channels,c.sample_width)==(22050,1,2) for c in chunks),
                "full native source chunks")
        if input_text.startswith("[["):
            qa.need([p for c in chunks for p in c.phonemes] == list(input_text[2:-2]), "direct IPA integrity")
        receipts = []
        native = bytearray()
        native_dir = out/"native"/sid
        native_dir.mkdir(parents=True)
        for i, c in enumerate(chunks):
            data=c.audio_int16_bytes
            name=f"{i:03d}.s16le"
            (native_dir/name).write_bytes(data)
            native.extend(data)
            receipts.append(dict(file=name,samples=len(data)//2,native_pcm_sha256=b10.sha(data),
                            phonemes=c.phonemes,phoneme_ids=c.phoneme_ids))
        signal=np.frombuffer(native,dtype="<i2").astype(np.float32)/32768.
        derived=resample_poly(signal,320,441)
        qa.need(3200<=len(derived)<=240000 and np.isfinite(derived).all(),"whole resampled input")
        pcm=np.rint(np.clip(derived,-1,32767/32768)*32768.).astype("<i2").tobytes()
        wav=out/"wav"/(sid+".wav")
        b10.previous.wav_write(wav,pcm,16000)
        facts=qa.pcm_tools.inspect_wav(wav)
        qa.need(facts["basic_audio_ok"] and facts["pcm_sha256"] not in seen,"unique auditable original")
        seen.add(facts["pcm_sha256"])
        rows.append(dict(row,**facts,native_chunks=receipts,native_pcm_sha256=b10.sha(native),
                         whole_native_resample=True,crop_calls=0,
                         positive_duration_pass=target!=2 or facts["samples"]<=b10.MAX_POSITIVE_SAMPLES))
    qa.record(out/"MANIFEST.json",dict(schema=plan["schema"],panel=plan["panel"],rows=rows,tts_calls=len(rows),
              plan_sha256=b10.sha(b10.canon(plan)),registered_plan_file_sha256=file_sha,
              fresh_cpu_session_each_input=True,new_kws_forwards=0,human_gold=False))


def read_source(path, plan_reader=source_plan, panel="primary"):
    rows=qa.read_panel(path,panel,plan_reader)
    for r in rows:
        total=hashlib.sha256()
        for i,c in enumerate(r["native_chunks"]):
            qa.need(c["file"]==f"{i:03d}.s16le","original chunk index")
            raw=(path/"native"/r["id"]/c["file"]).read_bytes()
            qa.need(len(raw)==2*c["samples"] and b10.sha(raw)==c["native_pcm_sha256"],"original chunk bytes")
            total.update(raw)
        qa.need(total.hexdigest()==r["native_pcm_sha256"],"whole native PCM")
    return rows


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    v=sub.add_parser("gate");v.add_argument("archive",type=Path)
    sub.add_parser("history")
    v=sub.add_parser("generate");v.add_argument("out",type=Path);v.add_argument("original_archives",type=Path)
    v=sub.add_parser("observe");v.add_argument("provider",choices=qa.observers.PROVIDERS);v.add_argument("source",type=Path);v.add_argument("out",type=Path)
    v=sub.add_parser("screen")
    for n in ("source","evidence","out"):v.add_argument(n,type=Path)
    a=p.parse_args()
    if a.mode=="gate":gate(a.archive)
    elif a.mode=="history":print("\n".join(locks()))
    elif a.mode=="generate":generate(a.out,a.original_archives)
    elif a.mode=="observe":qa.observers.observe(a.provider,a.source,a.out,read_rows=read_source,schema=SCHEMA)
    else:qa.screen(a.source,a.evidence,a.out,schema=SCHEMA,recipe=source_plan)
