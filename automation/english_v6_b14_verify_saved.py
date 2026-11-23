"""Verify a B14 handoff's PCM identities, role assignment, raw scores and decision."""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import wave
from zipfile import ZipFile

import numpy as np


GAINS = ("1.0", "0.75", "0.5")
ARMS = ("B0", "B9", "control", "challenger")
CRITERIA_SHA = "944f5df3c7bc3244977a8a33f7e1c393bffff9f7f5c32d2d8cc79e2628e265d9"


def verify(root, expected):
    def require(ok, why):
        if not ok:
            raise ValueError(why)

    def digest(raw):
        return sha256(raw).hexdigest()

    design = (root / "assessment.json").read_bytes()
    require(digest(design) == CRITERIA_SHA, "assessment digest")
    criteria = json.loads(design)
    with ExitStack() as stack:
        def archive(name, hash_, size=None):
            path = root / name
            require(digest(path.read_bytes()) == hash_ and (size is None or path.stat().st_size == size), name)
            z = stack.enter_context(ZipFile(path))
            require(z.testzip() is None and len(set(z.namelist())) == len(z.namelist()), name + " CRC")
            return z

        primary = archive("primary.zip", expected)
        report = json.loads(primary.read("RESULT.json"))
        spec = report["trial"]
        archives = {k:archive(f"inputs/{k}.zip", v["sha256"], v["bytes"]) for k,v in spec["inputs"].items()}
        pair = json.loads(archives["pair"].read("RESULT.json"))
        require(spec["design_commit"] == "d0e33039d950e159e14145f8978406180bf45828"
                and report["optimizer_updates"] == 0 and report["field_far_per_hour"] is None
                and report["physical_board_measured"] is False and report["promotion_allowed"] is False,
                "fixed independent evaluation")
        require(pair["actual_optimizer_updates"] == 384 and pair["dev_cal_locked_forwards"] == 0, "training isolation")
        sources = {}
        for short, cohort in (("synth", "synthetic"), ("natural", "natural")):
            source = archives[short + "_source"]
            raw = source.read("MANIFEST.json")
            labels = json.loads(archives[short + "_screen"].read("SCREEN.json"))
            require(labels["source_manifest_sha256"] == digest(raw) and labels["kws_forwards"] == 0, "frozen screen")
            manifest = json.loads(raw)
            require(manifest["panel"] == spec["panel"], "panel")
            if short == "synth":
                require(manifest["registered_plan_file_sha256"] == criteria["synthetic_plan_sha256"][spec["panel"]], "synthetic registration")
                decisions = labels["rows"]
            else:
                natural_raw = (root / "natural-design.json").read_bytes()
                nat_design = json.loads(natural_raw)
                require(manifest["design_sha256"] == digest(natural_raw) and manifest["corpus_sha256"] == nat_design["upstream"]["sha256"], "natural provenance")
                roster = json.loads(source.read("ROSTERS.json"))
                require(digest(source.read("ROSTERS.json")) == manifest["rosters_sha256"], "roster provenance")
                groups = [{str(r["speaker"]) for r in roster[p]} for p in ("primary","replication")]
                require(not groups[0].intersection(groups[1]) and all(len(g)==12 and not g.intersection(map(str,nat_design["excluded_speakers"])) for g in groups), "independent natural rosters")
                decisions = labels["decisions"]
            by_id = {x["id"]:x for x in decisions}
            require(len(by_id) == len(manifest["rows"]), "original denominator")
            for row in manifest["rows"]:
                sid = row["id"]
                label = by_id[sid]
                require(row["pcm_sha256"] == label["pcm_sha256"], "label original")
                if label["tier"] == "unknown":
                    continue
                require(label["tier"] in ("silver","weak") and row["role"] == "development", "admitted DEV")
                if short == "natural":
                    require(label["exact_votes"] >= 2 and not label["full_wake_veto"], "human label")
                else:
                    require(label["observed_class"] == row["requested_class"] in (0,2), "synthetic label")
                raw_wav = source.read("wav/" + sid + ".wav")
                with wave.open(BytesIO(raw_wav)) as wav:
                    n = wav.getnframes()
                    pcm = wav.readframes(n)
                    require(wav.getparams()[:3] == (1,2,16000), "source PCM format")
                require(sid not in sources and len(pcm) == 2*n and n == row["samples"]
                        and digest(raw_wav) == row["wav_sha256"] and digest(pcm) == row["pcm_sha256"], "native PCM identity")
                sources[sid] = dict(cohort=cohort, positive=short=="synth" and label["observed_class"]==2,
                                    group=row["source_group_id"], text=row["requested_text"], samples=n, pcm=digest(pcm))
        positives = {k for k,r in sources.items() if r["positive"]}
        natural = {k for k,r in sources.items() if r["cohort"]=="natural"}
        ordinary = sources.keys()-positives-natural
        groups = lambda ids:{sources[k]["group"] for k in ids}
        coverage = dict(synthetic_min_paired_families=len(groups(ordinary)&groups(positives)),
                        synthetic_min_positive_sources=len(groups(positives)), synthetic_min_positive_originals=len(positives),
                        synthetic_min_ordinary_originals=len(ordinary), synthetic_min_ordinary_forms=len({sources[k]["text"] for k in ordinary}),
                        natural_min_admitted_speakers=len(groups(natural)), natural_min_admitted_originals=len(natural))
        require(coverage==report["source_coverage"], "independent cohort sizes")
        if any(coverage[k]<v for k,v in criteria["source_gate"].items()):
            require(report["panel_gate"]=="INSUFFICIENT_SOURCE" and report["actual_forwards"]==0 and "ALL_FRAME_LOGITS.npz" not in primary.namelist(),"stop before numerical scoring")
            return dict(panel_gate=report["panel_gate"],coverage=coverage,arrays=0)
        weights = {m:digest(archives["pair"].read(s+".pt")) for m,s in spec["pair_stems"].items()}
        weights.update(B0="1ae8a2b20c8b6991f925dd4c626af2f36b308adee8733a3a7215818716f565d7",B9="7dba86a3fcda6e5f05cb80b414dc372c70a5a03b3aee783f168441a4871e7655")
        require(report["model_pt_sha256"] == weights, "weight identities")
        logits = primary.read("ALL_FRAME_LOGITS.npz")
        require(digest(logits)==report["logits_sha256"], "array set SHA")
        matrix = stack.enter_context(np.load(BytesIO(logits),allow_pickle=False))
        records={r["id"]:r for r in report["records"]}
        require(records.keys()==sources.keys() and len(records)==len(report["records"]), "no dropped admitted rows")
        raw_counts = {}
        frames = 0
        for sid, src in sources.items():
            rec=records[sid]
            require(all(rec[k]==src[k] for k in ("cohort","group","positive")) and rec["pcm_sha256"]==src["pcm"], "joined original")
            for gain in GAINS:
                for arm in ARMS:
                    key=sid+"__"+gain+"__"+arm
                    a=matrix[key]
                    declared=rec["gains"][gain][arm]
                    require(a.shape==((src["samples"]-512)//160+1,4) and a.dtype==np.dtype("<f4") and np.isfinite(a).all()
                            and digest(a.tobytes())==declared["sha256"] and key==declared["array"], "finite four-class raw array")
                    path=a.argmax(axis=1)
                    onsets=np.flatnonzero((path!=0) & np.r_[True,path[1:]!=path[:-1]]).tolist()
                    tokens=path[onsets].tolist()
                    times=[onsets[i] for i in range(2,len(tokens)) if tokens[i-2:i+1]==[1,2,3]]
                    require((tokens,times,len(times),len(a))==(declared["tokens"],declared["event_frames"],declared["events"],declared["frames"]), "decoded frame clock")
                    raw_counts[sid,gain,arm]=len(times)
                    frames+=len(a)
        require(len(raw_counts)==len(matrix.files)==len(sources)*12==report["actual_forwards"], "complete four-arm grid")
        summaries={}
        populations={"natural_ordinary":natural, "synthetic_ordinary":ordinary, "synthetic_positive":positives}
        for name,ids in populations.items():
            summaries[name]={}
            for arm in ARMS:
                summaries[name][arm]={}
                for gain in GAINS:
                    vals=[raw_counts[k,gain,arm] for k in ids]
                    summaries[name][arm][gain]=dict(originals=len(vals),events=sum(vals),with_event=sum(v>0 for v in vals))
        require(summaries==report["summary"], "independent table")
        restored=sorted([s,sources[s]["group"],g] for s in positives for g in GAINS if raw_counts[s,g,"B9"]>0==raw_counts[s,g,"control"] and raw_counts[s,g,"challenger"]>0)
        positive_losses=sorted([s,g,m] for s in positives for g in GAINS for m in ("B0","control") if raw_counts[s,g,m]>0==raw_counts[s,g,"challenger"])
        extra=sorted([s,g,m] for s in positives for g in GAINS for m in ("B9","control") if max(0,raw_counts[s,g,"challenger"]-1)>max(0,raw_counts[s,g,m]-1))
        added=sorted([s,g,m] for s in natural|ordinary for g in GAINS for m in ("B9","control") if raw_counts[s,g,"challenger"]>raw_counts[s,g,m])
        readers=sorted(speaker for speaker in groups(natural) if sum(raw_counts[s,"1.0","challenger"]-raw_counts[s,"1.0","B9"] for s in natural if sources[s]["group"]==speaker)<0)
        above=[g for g in GAINS if sum(raw_counts[s,g,"challenger"]-raw_counts[s,g,"B0"] for s in natural)>0]
        guards=dict(restored_positives=restored,lost_positives=positive_losses,extra_positive_events=extra,added_ordinary_events=added,
                    natural_readers_improved_vs_B9=readers,natural_above_B0_gains=above)
        require(all(sorted(v)==sorted(report[k]) for k,v in guards.items()), "independently attributed blockers")
        passed=(len({x[1] for x in restored})>=criteria["minimum_recovered_positive_families"] and len(readers)>=criteria["minimum_improved_readers_vs_B9"]
                and not (added or positive_losses or extra or above))
        require(report["panel_gate"]==("PASS" if passed else "NO_PROMOTION"), "registered decision")
        return dict(panel_gate=report["panel_gate"],coverage=coverage,arrays=len(matrix.files),frames=frames,
                    seconds={k:sum(sources[s]["samples"] for s in v)/16000 for k,v in populations.items()},
                    model_pt_sha256=weights,summary=summaries,**guards)


if __name__ == "__main__":
    args=argparse.ArgumentParser()
    args.add_argument("folder",type=Path)
    args.add_argument("score_zip_sha256")
    args.add_argument("output",type=Path)
    opt=args.parse_args()
    opt.output.write_text(json.dumps(verify(opt.folder,opt.score_zip_sha256),indent=2)+"\n")
