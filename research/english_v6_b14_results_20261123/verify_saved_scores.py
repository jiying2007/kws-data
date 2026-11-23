"""Independently check the research decision from sealed source PCM, labels and saved logits."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from contextlib import ExitStack
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import wave
from zipfile import ZipFile

import numpy as np


MODELS = ("B0", "B9", "control", "challenger")
GAINS = ("1.0", "0.75", "0.5")
NATURAL_SHA = "cb34a2b93a4b8e713b37455acc378fd08a1036750f81465bc6d3394e073d6635"
DESIGN_SHA = "944f5df3c7bc3244977a8a33f7e1c393bffff9f7f5c32d2d8cc79e2628e265d9"
PAIR_SHA = "98c725b30e3b8e958096217f9b846f1d8e4646f1e0393f1b5817aac8b8f9c58c"


def require(statement, message):
    if not statement:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def checked_archive(stack, path, expected):
    require(digest(path.read_bytes()) == expected, "SHA256: "+str(path))
    archive = stack.enter_context(ZipFile(path))
    names = archive.namelist()
    require(archive.testzip() is None and len(names) == len(set(names)), "ZIP contents: "+str(path))
    return archive


def read_sources(archives, panel, design, natural):
    eligible = {}
    coverage = {}
    for short, cohort in (("synth", "synthetic"), ("natural", "natural")):
        source = archives[short+"_source"]
        labels = json.loads(archives[short+"_screen"].read("SCREEN.json"))
        manifest_raw = source.read("MANIFEST.json")
        raw = json.loads(manifest_raw)
        require(labels["source_manifest_sha256"] == digest(manifest_raw), "source/label pairing")
        require(raw["panel"] == panel and labels["kws_forwards"] == 0, "prior labels")
        if short == "natural":
            require(raw["design_sha256"] == NATURAL_SHA and raw["corpus_sha256"] == natural["upstream"]["sha256"], "registered natural source")
            roster_raw = source.read("ROSTERS.json")
            require(digest(roster_raw) == raw["rosters_sha256"], "roster")
            roster = json.loads(roster_raw)
            selected_groups = {p:{r["speaker"] for r in rr} for p,rr in roster.items()}
            for p, groups in selected_groups.items():
                require(len(groups) == 12 and len(roster[p]) == 24 and not groups.intersection(natural["excluded_speakers"]), "roster speakers")
            require(not selected_groups["primary"].intersection(selected_groups["replication"]), "no overlapping natural prospective speakers")
            annotation = labels["decisions"]
        else:
            require(raw["registered_plan_file_sha256"] == design["synthetic_plan_sha256"][panel], "fixed synthetic roster")
            annotation = labels["rows"]
        lookup = {r["id"]:r for r in annotation}
        require(len(lookup) == len(annotation) == len(raw["rows"]), "one label per source original")
        for row in raw["rows"]:
            sid = row["id"]
            label = lookup[sid]
            require(label["pcm_sha256"] == row["pcm_sha256"], "original identification")
            if label["tier"] == "unknown":
                continue
            require(label["tier"] in ("weak","silver"), "predefined source tier")
            if short == "synth":
                require(row["requested_class"] == label["observed_class"] in (0,2), "pre-registered type")
                positive = row["requested_class"] == 2
            else:
                require(not label["full_wake_veto"] and label["exact_votes"] >= 2 and label["group"] == row["source_group_id"], "natural admission")
                positive = False
            wav = source.read("wav/"+sid+".wav")
            with wave.open(BytesIO(wav), "rb") as w:
                require(w.getparams()[:3] == (1,2,16000), "complete 16 kHz signed PCM16")
                count = w.getnframes()
                pcm = w.readframes(count)
            require(len(pcm) == 2*count and count == row["samples"] and digest(pcm) == row["pcm_sha256"]
                    and digest(wav) == row["wav_sha256"], "complete authenticated PCM")
            require(sid not in eligible, "unique identity")
            eligible[sid] = dict(cohort=cohort, positive=positive, group=row["source_group_id"],
                                 tier=label["tier"], frames=(count-512)//160+1,
                                 cell=row.get("cell_id","read-speech"), source_id=row.get("source_id",row["id"]),
                                 text=row["requested_text"], samples=count, pcm_sha256=digest(pcm))
    synth = [r for r in eligible.values() if r["cohort"]=="synthetic"]
    pos = [r for r in synth if r["positive"]]
    neg = [r for r in synth if not r["positive"]]
    readers = [r for r in eligible.values() if r["cohort"]=="natural"]
    pos_groups = {r["group"] for r in pos}
    neg_groups = {r["group"] for r in neg}
    coverage = dict(synthetic_min_paired_families=len(pos_groups & neg_groups),
                    synthetic_min_positive_sources=len(pos_groups),
                    synthetic_min_ordinary_forms=len({r["text"] for r in neg}),
                    natural_min_admitted_speakers=len({r["group"] for r in readers}),
                    natural_min_admitted_originals=len(readers),
                    synthetic_min_ordinary_originals=len(neg), synthetic_min_positive_originals=len(pos))
    require(len({r["pcm_sha256"] for r in eligible.values()}) == len(eligible), "unique admitted originals")
    return eligible, coverage


def decode(matrix):
    require(matrix.ndim == 2 and matrix.shape[1] == 4 and matrix.dtype == np.dtype("<f4") and np.isfinite(matrix).all(), "finite raw four-class float32 scores")
    # CTC drops blank after merging equal consecutive winning classes.
    frames = np.argmax(matrix, axis=1)
    changes = np.r_[True, frames[1:] != frames[:-1]]
    starts = np.flatnonzero(changes & (frames != 0)).tolist()
    tokens = frames[starts].tolist()
    return tokens, [starts[i] for i in range(2,len(tokens)) if tokens[i-2:i+1] == [1,2,3]]


def verify(result_zip, result_sha, inputs, design_path):
    design_raw = design_path.read_bytes()
    require(digest(design_raw)==DESIGN_SHA, "immutable research study registration")
    design = json.loads(design_raw)
    with ExitStack() as stack:
        scored = checked_archive(stack, result_zip, result_sha)
        report = json.loads(scored.read("RESULT.json"))
        spec = report["trial"]
        require(spec["design_commit"]=="d0e33039d950e159e14145f8978406180bf45828" and
                spec["inputs"]["pair"]["sha256"]==PAIR_SHA and report["promotion_allowed"] is False and
                report["field_far_per_hour"] is None and report["physical_board_measured"] is False, "registered project stage")
        panel = spec["panel"]
        archives = {k:checked_archive(stack, inputs/(k+".zip"), spec["inputs"][k]["sha256"])
                    for k in ("synth_source","synth_screen","natural_source","natural_screen")}
        nraw=(design_path.parent/"natural-design.json").read_bytes()
        require(digest(nraw)==NATURAL_SHA,"registered dev-other natural source")
        sources, cov = read_sources(archives, panel, design, json.loads(nraw))
        required = design["source_gate"]
        missing = {k:[x,required[k]] for k,x in cov.items() if x < required[k]}
        require(cov == report["source_coverage"] and missing == report["source_shortfall"], "source thresholds and denominators")
        result = dict(panel=panel, result_archive_sha256=result_sha, independent_source_coverage=cov,
                      input_sha256={k:spec["inputs"][k]["sha256"] for k in archives}, promotion_allowed=False,
                      field_far_per_hour=None, physical_board_measured=False)
        if missing:
            require(report["panel_gate"] == "INSUFFICIENT_SOURCE" and report["actual_forwards"] == 0 and "ALL_FRAME_LOGITS.npz" not in scored.namelist(), "fail closed before scoring")
            return dict(result, panel_gate="INSUFFICIENT_SOURCE", actual_forwards=0, source_shortfall=missing)
        fixed_weights = {
            "B0": "1ae8a2b20c8b6991f925dd4c626af2f36b308adee8733a3a7215818716f565d7",
            "B9": "7dba86a3fcda6e5f05cb80b414dc372c70a5a03b3aee783f168441a4871e7655",
            "control": "643fbbc21f5e2ca7a22c8e320966ddd43c3f2985137e1949ae1e8884565db553",
            "challenger": "fb6926cd07beb63283072a7da1b7fe4abe6fbb9b2c1e1082bdb4561fd784a5af",
        }
        require(report["model_pt_sha256"]==fixed_weights, "all exact authentic checkpoint roles")
        result["verified_pt_sha256"] = fixed_weights
        zdata = scored.read("ALL_FRAME_LOGITS.npz")
        require(digest(zdata)==report["logits_sha256"], "raw logits payload SHA")
        arrays = stack.enter_context(np.load(BytesIO(zdata), allow_pickle=False))
        records = {r["id"]:r for r in report["records"]}
        require(len(records) == len(report["records"]) == len(sources) and records.keys()==sources.keys(), "exact evaluation denominator")
        counters = defaultdict(Counter)
        totals = defaultdict(Counter)
        positive_hits = defaultdict(set)
        speaker_counts = defaultdict(Counter)
        measured = set()
        added = []
        row_losses = []
        extra = []
        restored = []
        details = []
        natural_totals = defaultdict(Counter)
        sample_counts = Counter()
        repetitions = defaultdict(Counter)
        for sid, source in sources.items():
            rec = records[sid]
            population = source["cohort"]+"_"+("positive" if source["positive"] else "ordinary")
            require(all(source[k]==rec[k] for k in ("cohort","group","positive","tier","pcm_sha256")), "joined frozen original")
            sample_counts[population] += source["samples"]
            for gain in GAINS:
                events = {}
                decoded = {}
                for arm in MODELS:
                    name = sid+"__"+gain+"__"+arm
                    declared = rec["gains"][gain][arm]
                    matrix = arrays[name]
                    require(declared["sha256"]==digest(matrix.tobytes()) and declared["array"]==name and
                            len(matrix)==declared["frames"]==source["frames"], "raw logits identity and complete frame geometry")
                    tokens, fire = decode(matrix)
                    require(tokens==declared["tokens"] and fire==declared["event_frames"] and len(fire)==declared["events"], "independent event timestamps")
                    measured.add(name)
                    n = len(fire)
                    events[arm] = n
                    decoded[arm] = {"path":tokens,"end_token_frames":fire}
                    if source["cohort"]=="natural":
                        natural_totals[gain][arm]+=n
                    counts = counters[population,gain,arm]
                    counts["originals"]+=1
                    counts["with_event"]+=int(n>0)
                    counts["events"]+=n
                    if source["positive"] and n:
                        positive_hits[gain,arm].add(source["group"])
                        repetitions[arm][gain]+=max(0,n-1)
                    if source["cohort"]=="natural" and gain=="1.0":
                        speaker_counts[source["group"]][arm]+=n
                        totals["natural"][arm]+=n
                reasons=[]
                if source["positive"]:
                    for arm in ("B0","control"):
                        if events[arm] and not events["challenger"]:
                            row_losses.append([sid,gain,arm])
                            reasons.append("lost_vs_"+arm)
                    for arm in ("B9","control"):
                        if max(0,events["challenger"]-1)>max(0,events[arm]-1):
                            extra.append([sid,gain,arm])
                            reasons.append("extra_vs_"+arm)
                    if events["B9"] and not events["control"] and events["challenger"]:
                        restored.append([sid,source["group"],gain])
                        reasons.append("recovered_word")
                else:
                    for arm in ("B9","control"):
                        if events["challenger"] > events[arm]:
                            added.append([sid,gain,arm])
                            reasons.append("more_false_events_vs_"+arm)
                if reasons:
                    details.append(dict(id=sid,group=source["group"],cell=source["cell"],source_text=source["text"],
                                        cohort=source["cohort"],tier=source["tier"],gain=gain,reasons=reasons,
                                        pcm_sha256=source["pcm_sha256"],decoded=decoded))
        require(measured == set(arrays.files) and len(measured)==len(sources)*12==report["actual_forwards"], "all and only registered forwards")
        for (population,gain,arm), counts in counters.items():
            require(dict(counts)==report["summary"][population][arm][gain], "all reported strata")
        improved=sorted(g for g,cnt in speaker_counts.items() if cnt["challenger"]<cnt["B9"])
        above=[g for g,cnt in natural_totals.items() if cnt["challenger"]>cnt["B0"]]
        recovered_groups={r[1] for r in restored}
        passed=(len(recovered_groups)>=design["minimum_recovered_positive_families"] and
                len(improved)>=design["minimum_improved_readers_vs_B9"] and not (extra or added or row_losses or above))
        gate="PASS" if passed else "NO_PROMOTION"
        guards=dict(restored_positives=restored,lost_positives=row_losses,extra_positive_events=extra,
                    added_ordinary_events=added,natural_readers_improved_vs_B9=improved,natural_above_B0_gains=above)
        require(all(sorted(v)==sorted(report[k]) for k,v in guards.items()) and report["panel_gate"]==gate,
                "independent full decision and every reason")
        return dict(result, panel_gate=gate, actual_forwards=len(measured), actual_utterances=len(sources),
                    complete_frames=sum(s["frames"]*12 for s in sources.values()),
                    admitted_seconds_by_cohort={k:v/16000 for k,v in sample_counts.items()},
                    fp32_all_arrays_sha256=digest(zdata),checked_panel_summary=report["summary"],
                    original_gain_natural_events=dict(totals["natural"]),natural_per_speaker=speaker_counts,
                    natural_events_by_gain=natural_totals,positive_recovered_families=sorted(recovered_groups),
                    all_original_gains_causing_block_or_rescue=details,**guards,
                    extra_positive_events_by_arm={a:{g:repetitions[a][g] for g in GAINS} for a in MODELS},
                    positive_detected_family_counts={a:{g:len(positive_hits[g,a]) for g in GAINS} for a in MODELS})


if __name__=="__main__":
    ap=ArgumentParser()
    ap.add_argument("scored", type=Path)
    ap.add_argument("sha256")
    ap.add_argument("inputs", type=Path)
    ap.add_argument("design", type=Path)
    ap.add_argument("output", type=Path)
    args=ap.parse_args()
    args.output.write_text(json.dumps(verify(args.scored,args.sha256,args.inputs,args.design),indent=2)+"\n")
