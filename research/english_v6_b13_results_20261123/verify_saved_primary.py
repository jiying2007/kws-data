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
DESIGN_SHA = "109a3e3a0b74f5c4e48d90323b1d1384294cc3f3b54f0501e34c9c11ca3585d8"
PAIR_SHA = "e66de6d2683c93e434e3d426ef88ffa46c9eba63d89d6940454a81d17e7d29af"


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


def read_sources(archives, panel, design):
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
            require(raw["design_sha256"] == DESIGN_SHA and raw["corpus_sha256"] == design["upstream"]["sha256"], "registered natural source")
            roster_raw = source.read("ROSTERS.json")
            require(digest(roster_raw) == raw["rosters_sha256"], "roster")
            roster = json.loads(roster_raw)
            selected_groups = {p:{r["speaker"] for r in rr} for p,rr in roster.items()}
            for p, groups in selected_groups.items():
                require(len(groups) == 12 and len(roster[p]) == 24 and not groups.intersection(design["excluded_speakers"]), "roster speakers")
            require(not selected_groups["primary"].intersection(selected_groups["replication"]), "no overlapping natural prospective speakers")
            annotation = labels["decisions"]
        else:
            require(raw["registered_plan_file_sha256"] == design["synthetic_panel_sha256"][panel], "fixed synthetic roster")
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
        require(spec["design_commit"]=="24a170a88eb2d08a7e3a5fee62018cbbec8d61a9" and
                spec["inputs"]["pair"]["sha256"]==PAIR_SHA and report["promotion_allowed"] is False and
                report["field_far_per_hour"] is None and report["physical_board_measured"] is False, "registered project stage")
        panel = spec["panel"]
        archives = {k:checked_archive(stack, inputs/(k+".zip"), spec["inputs"][k]["sha256"])
                    for k in ("synth_source","synth_screen","natural_source","natural_screen")}
        sources, cov = read_sources(archives, panel, design)
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
            "control": "40c09e038ad731b8f860087bc5983ff39d993249b749402be140d8439928d555",
            "challenger": "643fbbc21f5e2ca7a22c8e320966ddd43c3f2985137e1949ae1e8884565db553",
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
        sample_counts = Counter()
        repetitions = defaultdict(Counter)
        for sid, source in sources.items():
            rec = records[sid]
            population = source["cohort"]+"_"+("positive" if source["positive"] else "ordinary")
            require(all(source[k]==rec[k] for k in ("cohort","group","positive","tier","pcm_sha256")), "joined frozen original")
            sample_counts[population] += source["samples"]
            for gain in GAINS:
                events = {}
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
                if source["positive"]:
                    row_losses.extend([sid,gain,arm] for arm in ("B0","control") if events[arm]>0 and events["challenger"]==0)
                else:
                    added.extend([sid,gain,arm] for arm in ("B9","control") if events["challenger"] > events[arm])
        require(measured == set(arrays.files) and len(measured)==len(sources)*12==report["actual_forwards"], "all and only registered forwards")
        for (population,gain,arm), counts in counters.items():
            require(dict(counts)==report["summary"][population][arm][gain], "all reported strata")
        improved = sorted(group for group, values in speaker_counts.items()
                          if all(values["challenger"] < values[arm] for arm in ("B9","control")))
        lost = sorted([group,g,arm] for g in GAINS for arm in ("B0","control") for group in positive_hits[g,arm]-positive_hits[g,"challenger"])
        natural = totals["natural"]
        eligible = len(improved)>=design["decision"]["required_new_natural_speakers_at_original_gain"] and natural["challenger"]<min(natural["B9"],natural["control"])
        passed = eligible and natural["challenger"]<=natural["B0"] and not (added or lost)
        gate = "PASS" if passed else "NO_PROMOTION"
        require(improved==report["natural_speakers_improved"] and sorted(added)==sorted(report["added_ordinary_events"])
                and lost==sorted(report["lost_positive_group_gains"]) and gate==report["panel_gate"], "independent registered decision")
        return dict(result, panel_gate=gate, actual_forwards=len(measured), actual_utterances=len(sources),
                    admitted_seconds_by_cohort={k:v/16000 for k,v in sample_counts.items()},
                    fp32_all_arrays_sha256=digest(zdata),
                    checked_panel_summary=report["summary"], natural_original_gain_events=dict(natural),
                    natural_qualifying_speakers=improved, natural_per_speaker=speaker_counts,
                    added_ordinary_event_comparisons=sorted(added), positive_family_loss_comparisons=lost,
                    original_level_positive_loss_comparisons=sorted(row_losses),
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
