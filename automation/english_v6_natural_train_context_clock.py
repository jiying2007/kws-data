"""Clock every frozen wake detector on admitted TRAIN utterances at native gain."""
from contextlib import ExitStack
import io
import json
from pathlib import Path
import re
import sys
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_four_model_panel as base

DESIGN = Path(__file__).with_suffix(".json")
DESIGN_SHA = "ae472256363b2d09e85a34434714a34167d6d27e42ed46cd4302c2bc38e69b2e"
NAMES = ("B0", "B9", "B13", "B14")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def decode_events(path, model):
    onsets = [(f, token) for f, token in enumerate(path)
              if token and (f == 0 or token != path[f-1])]
    found = []
    for pos in range(len(onsets)-2):
        candidate = onsets[pos:pos+3]
        if [token for _, token in candidate] == [1, 2, 3]:
            frames = [f for f, _ in candidate]
            ready = [f*160 + 512 for f in frames]
            found.append(dict(model=model, token_frames=frames, token_ready_samples=ready,
                              trigger_sample=ready[-1]))
    return found


def intervals(events, samples):
    ranges = sorted((max(0, e["token_frames"][0]*160-3200),
                     min(samples, e["trigger_sample"]+3200), i) for i,e in enumerate(events))
    merged = []
    for a,b,i in ranges:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
            merged[-1][2].append(i)
        else:
            merged.append([a,b,[i]])
    return merged


def adjacent(clusters, index, samples):
    a,b,_ = clusters[index]
    left_edge = clusters[index-1][1] if index else 0
    right_edge = clusters[index+1][0] if index+1<len(clusters) else samples
    left, right = min(16000,a-left_edge), min(16000,right_edge-b)
    if max(left,right) < 8000:
        return None
    return (a-left,a) if left >= right else (b,b+right)


def clip(output, folder, ident, pcm, a, b):
    name = f"{ident}-{a}-{b}"
    rel = f"{folder}/{name}.wav"
    raw = pcm[2*a:2*b]
    with wave.open(str(output / rel), "wb") as f:
        f.setparams((1,2,16000,0,"NONE","not compressed"))
        f.writeframes(raw)
    return dict(id=name, original_id=ident, wav_path=rel, start_sample=a, end_sample=b,
                samples=b-a, pcm_sha256=base.sha(raw), wav_sha256=base.sha((output/rel).read_bytes()),
                review_status="pending")


def build_segments(out, sid, pcm, events):
    windows, controls = [], {}
    merged = intervals(events, len(pcm)//2)
    for i,(a,b,members) in enumerate(merged):
        window = clip(out,"windows",sid,pcm,a,b)
        window["events"] = sorted((events[j] for j in members),key=lambda e:(e["trigger_sample"], e["model"]))
        ctx = adjacent(merged,i,len(pcm)//2)
        if ctx:
            if ctx not in controls:
                controls[ctx] = clip(out,"controls",sid,pcm,*ctx)
            window.update(control_id=controls[ctx]["id"], control_unavailable_reason=None)
        else:
            window.update(control_id=None, control_unavailable_reason="no adjacent event-free gap of at least 8000 samples")
        windows.append(window)
    return windows,list(controls.values())


def scored_pcm(row, pcm, nets, arrays, torch, np):
    answers, events = {}, []
    with torch.inference_mode():
        for role, historical in zip(NAMES, ("B0","B9","control","challenger"), strict=True):
            net,mel,mean,scale,pt = nets[historical]
            frames = base.b9.v5.features(pcm,row,torch,np,mel)
            x = torch.from_numpy(((frames-mean)/scale).astype(np.float32)).unsqueeze(0)
            y = net(x)
            base.need(y.shape == (1,len(frames),4) and len(frames)==1+(row["samples"]-512)//160
                      and torch.isfinite(y).all().item(), "native causal frame geometry")
            dec = base.b9.v5.ctc_greedy(y,torch)
            key = row["id"]+"__"+role
            arr = y[0].numpy().astype("<f4")
            arrays[key] = arr
            ev = decode_events(arr.argmax(-1).tolist(),role)
            base.need(len(ev)==int(dec["mero_event"])+int(dec["extra_events"]), "canonical decoder parity")
            answers[role] = dict(array=key,raw_f32_sha256=base.sha(arr.tobytes()), frames=len(arr),
                                 model_pt_sha256=pt,collapsed_tokens=dec["collapsed_tokens"],events=ev)
            events.extend(ev)
    return answers,events


def run(spec_path, inputs, output):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)
    design = json.loads(DESIGN.read_bytes())
    base.need(base.sha(DESIGN.read_bytes())==DESIGN_SHA,"registered clock protocol")
    spec=json.loads(spec_path.read_bytes())
    base.need(spec["design_sha256"]==DESIGN_SHA and set(spec["inputs"])=={"B0","B9","pair","source","screen"}
              and all(spec["inputs"][k]==lock for k,lock in design["model_artifacts"]["inputs"].items()),
              "three authentic model archives and two TRAIN source artifacts")
    output.mkdir()
    for folder in ("originals","windows","controls"):
        (output/folder).mkdir()
    originals,windows,controls,scores,arrays,unknown = [],[],[],[],{},[]
    model_spec = {**design["model_artifacts"],"inputs":spec["inputs"]}
    with ExitStack() as s:
        source = s.enter_context(base.opened(inputs/"source.zip",spec["inputs"]["source"]))
        screen = s.enter_context(base.opened(inputs/"screen.zip",spec["inputs"]["screen"]))
        raw = source.read("MANIFEST.json")
        manifest = json.loads(raw)
        raw_screen=screen.read("SCREEN.json")
        judgments = json.loads(raw_screen)
        base.need(manifest["schema"]==judgments["schema"]==design["source_schema"] and
                  manifest["role"]=="train" and judgments["source_manifest_sha256"]==base.sha(raw)
                  and judgments["kws_forwards"]==0,"unscored TRAIN-only source-screen binding")
        labels = {d["id"]:d for d in judgments["decisions"]}
        rows = manifest["rows"]
        base.need(len(rows)==len(labels)==len(judgments["decisions"]) and
                  set(labels)=={r["id"] for r in rows} and
                  len({r["pcm_sha256"] for r in rows})==len(rows),"all distinct preselected source rows")
        nets=base.snapshots(inputs,model_spec,torch,np)
        base.need(nets["control"][-1]==design["pt_locked"]["B13"] and
                  nets["challenger"][-1]==design["pt_locked"]["B14"],"original B13 and B14 weights")
        for row in rows:
            sid = row["id"]; d=labels[sid]
            base.need(re.fullmatch(r"[A-Za-z0-9_-]+",sid) is not None and
                      row["role"]=="train" and row["requested_class"]==0 and
                      all(row[k]==d[k] for k in ("id","speaker","source_group_id","source_id","pcm_sha256"))
                      and d["tier"] in ("silver","weak","unknown"),"ordinary TRAIN identity")
            if d["tier"]=="unknown":
                unknown.append(sid)
                continue
            base.need(d["reviewed_text_votes"] in (2,3) and not d["blocked_native_transcript"],
                      "blind ordinary full-source evidence")
            wav = source.read("wav/"+sid+".wav")
            with wave.open(io.BytesIO(wav)) as f:
                base.need(f.getparams()[:3]==(1,2,16000) and f.getnframes()==row["samples"],"native whole audio")
                pcm=f.readframes(f.getnframes())
            base.need(base.sha(wav)==row["wav_sha256"] and base.sha(pcm)==row["pcm_sha256"]
                      and len(pcm)==row["samples"]*2,"exact unmodified native PCM")
            result,events=scored_pcm(row,pcm,nets,arrays,torch,np)
            ss,cc=build_segments(output,sid,pcm,events)
            windows.extend(ss); controls.extend(cc)
            rel="originals/"+sid+".wav"
            (output/rel).write_bytes(wav)
            record={k:row[k] for k in ("id","speaker","source_group_id","source_id","transcript","samples","pcm_sha256","wav_sha256","flac_sha256")}
            record.update(tier=d["tier"],wav_path=rel,model_event_counts={role:len(x["events"]) for role,x in result.items()},
                          reviewed_text_votes=d["reviewed_text_votes"],raw_text_by_provider=d["raw_text_by_provider"],
                          review_status="pending")
            originals.append(record); scores.append(dict(id=sid,models=result))
    index=dict(schema=design["schema"],source_schema=design["source_schema"],role="train",sample_rate=16000,
               interval_convention="zero-based half-open native PCM sample boundaries",clock=design["clock"],
               human_review_completed=False,proposed_training_eligibility=False,unknown_ids=unknown,
               originals=originals,windows=windows,controls=controls,models=list(NAMES),
               source_manifest_sha256=base.sha(raw),screen_sha256=base.sha(raw_screen))
    write_json(output/"REVIEW_INDEX.json",index)
    np.savez_compressed(output/"ALL_FRAME_LOGITS.npz",**arrays)
    report=dict(schema=design["schema"],protocol_sha256=DESIGN_SHA,inputs=spec["inputs"],scored=scores,
                forwards=len(arrays),admitted=len(originals),unknown=len(unknown),raw_source_rows=len(rows),
                modeled_seconds=sum(r["samples"] for r in originals)/16000,
                events_by_model={m:sum(r["model_event_counts"][m] for r in originals) for m in NAMES},
                triggered_originals_by_model={m:sum(r["model_event_counts"][m]>0 for r in originals) for m in NAMES},
                merged_hit_windows=len(windows),adjacent_controls=len(controls),
                monitor_triggered_readers=len({r["speaker"] for r in originals if any(r["model_event_counts"].values())}),
                review_index_sha256=base.sha((output/"REVIEW_INDEX.json").read_bytes()),
                logits_sha256=base.sha((output/"ALL_FRAME_LOGITS.npz").read_bytes()),optimizer_updates=0,
                threshold_search=False,dev_cal_locked_test_forwards=0,human_labels=0,promotion_allowed=False)
    write_json(output/"RESULT.json",report)


if __name__=="__main__":
    run(*map(Path,sys.argv[1:]))
