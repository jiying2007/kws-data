"""Prospective B17 quality assessment of a B0-initialized, TRAIN-only matched pair.

Source gates run before model loading. Historical B14/B15/B16 panels and every
previously used or reserved speaker remain excluded. No optimizer or threshold
search is available here; passing primary only permits independent replication.
"""
from contextlib import ExitStack
import io
import json
from pathlib import Path
import sys
import tempfile
import re

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_four_model_panel as common
import english_v6_b17_source as synth
import english_v6_b17_natural_sources as natural
import english_v6_b14_decision as old_decision

SCHEMA = "english-v6-b17-unseen-primary-four-frozen-models-v1"
PAIR_SCHEMA = "english-v6-b17-b0-ordinary-margin-pair-v1"
REGISTERED = Path(__file__).with_name("english_v6_b17_primary_inputs.json")
ASSESSMENT = Path(__file__).with_name("english_v6_b17_panels") / "assessment.json"
ASSESSMENT_SHA = "7fdfe9cd47b18f64429fece944468af8adfde6c6ba012a57236a8a33aae1a948"
DESIGN_SHA = "f9ba0ce59b0deed805b313063e648470e61af6c9122d08f6fe9aa5ea9e324831"
B0_PT = "1ae8a2b20c8b6991f925dd4c626af2f36b308adee8733a3a7215818716f565d7"
UNCHANGED_QUALITY_SHA = "944f5df3c7bc3244977a8a33f7e1c393bffff9f7f5c32d2d8cc79e2628e265d9"
GAINS = ("1.0", "0.75", "0.5")
GATE = {"synthetic_positive": 12, "synthetic_ordinary": 24, "synthetic_paired": 8,
        "natural_originals": 12, "natural_speakers": 8}
STEMS = {arm: arm + "/models/mero43-b17-" + arm + "192" for arm in ("control", "challenger")}
INPUT_KEYS = {"B0", "B9", "pair", "synth_source", "synth_screen", "natural_source", "natural_screen"}


def protocol():
    import re
    common.need(re.fullmatch(r"[0-9a-f]{64}", ASSESSMENT_SHA) is not None and
                common.sha(ASSESSMENT.read_bytes()) == ASSESSMENT_SHA and
                common.sha(old_decision.ASSESSMENT.read_bytes()) == UNCHANGED_QUALITY_SHA,
                "frozen prospective B17 assessment and unchanged acoustic criteria")
    doc = json.loads(REGISTERED.read_bytes())
    common.need(doc["schema"] == SCHEMA and doc["source_gate"] == GATE and
                common.sha(Path(old_decision.__file__).read_bytes()) == "6aa5f999598e47b41eb0a946acd82fdc5b47d30aea8990871c1c1a01065e4023" and
                type(doc["scoring_allowed"]) is bool and doc["pair_schema"] == PAIR_SCHEMA and
                doc["pair_stems"] == STEMS and doc["parent_pt_sha256"] == B0_PT and
                doc["assessment_sha256"] == ASSESSMENT_SHA and doc["training_design_sha256"] == DESIGN_SHA and
                all(doc[k] is False for k in ("promotion_allowed", "physical_ssc305", "cal_locked_access",
                      "B14_replication_opened", "B15_replication_opened", "B16_replication_opened")) and
                set(doc["inputs"]) == INPUT_KEYS,
                "B0-parent four-model experiment, no old replication or release")
    common.need(synth.source_plan("primary")["frozen_source_gate"] ==
                {"positive_originals_min": 12, "ordinary_originals_min": 24, "paired_source_groups_min": 8},
                "no post-generation source gate change")
    return doc


def validate_pair_receipt(receipt):
    common.need(receipt["schema"] == PAIR_SCHEMA and receipt["parent_pt_sha256"] == B0_PT and
                receipt["actual_optimizer_updates"] == 384 and receipt["trained_model_parameters"] == 43876 and
                receipt["dev_cal_locked_forwards"] == 0 and receipt["promotion_allowed"] is False and
                receipt["assessment_sha256"] == ASSESSMENT_SHA and receipt["design_sha256"] == DESIGN_SHA,
                "exact independently registered B0-parent completed TRAIN pair")
    for arm in ("control", "challenger"):
        common.need(receipt[arm]["optimizer_updates"] == 192 and receipt[arm]["stem"] == STEMS[arm],
                    "same-job arm has expected 192 steps and native model stem")


def snapshots(directory, spec, torch, np):
    """Keep legacy B9-parent validators immutable; authenticate B17's actual B0 parent."""
    models = {}
    with common.opened(directory / "B0.zip", spec["inputs"]["B0"]) as archive, tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp)
        for name in ["RESULT.json"] + ["models/mero-fsmn-ctc-expanded256." + e for e in ("pt", "npz", "onnx")]:
            dest = target / name
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(archive.read(name))
        model, mel, mean, scale, hashes = common.b0.model(torch, np, target, "B0")
        common.need(hashes["pt"] == B0_PT, "installed B0 identity")
        models["B0"] = (model, mel, mean, scale, hashes["pt"])
    with common.opened(directory / "B9.zip", spec["inputs"]["B9"]):
        prior, _ = common.b9.verify_b9_archive(directory / "B9.zip", torch, np)
    models["B9"] = (*prior["B9_control"], common.b9.MODELS["B9_control"]["pt"])
    common.need(all(np.array_equal(a, b) for a, b in zip(models["B0"][1:4], models["B9"][1:4])),
                "B0/B9 use identical frozen acoustic frontend and normalization")
    with common.opened(directory / "pair.zip", spec["inputs"]["pair"]) as archive:
        receipt = json.loads(archive.read("RESULT.json"))
        validate_pair_receipt(receipt)
        sources = json.loads(archive.read("SOURCE.json"))
        for arm in ("control", "challenger"):
            members = {}
            for ext in ("pt", "npz", "onnx"):
                filename = STEMS[arm] + "." + ext
                raw = archive.read(filename)
                expected = receipt[arm]["files"][Path(filename).name]
                common.need(common.sha(raw) == expected["sha256"] and len(raw) == expected["bytes"],
                            "complete trained B17 artifact member " + filename)
                members[ext] = raw
            weights = torch.load(io.BytesIO(members["pt"]), map_location="cpu", weights_only=True)
            net = common.b9.v5.build_model(torch)
            net.load_state_dict(weights, strict=True)
            net.eval()
            common.need(sum(p.numel() for p in net.parameters()) == 43876 and
                        all(bool(torch.isfinite(p).all()) for p in net.parameters()), "finite actual B17 weights")
            with np.load(io.BytesIO(members["npz"]), allow_pickle=False) as npz:
                common.need(set(npz.files) == set(weights) | {"mel_filters", "train_mean", "train_scale"} and
                            all(np.array_equal(npz[k], t.cpu().numpy()) for k, t in weights.items()),
                            "B17 PT and NPZ learned tensor identity")
                frontend = tuple(npz[k].copy() for k in ("mel_filters", "train_mean", "train_scale"))
            common.need(all(np.array_equal(a, b) for a, b in zip(frontend, models["B0"][1:4])),
                        "candidate retains B0 frozen acoustic frontend")
            models[arm] = (net, *frontend, common.sha(members["pt"]))
    common.need(set(models) == set(common.NAMES), "four explicitly authenticated frozen roles")
    return models, sources


def reject_train_overlap(dev_rows, sources):
    """Check actual paired TRAIN originals again at the final evaluation boundary."""
    training = sources["base_train"] + sources["natural_train"]
    shas = {r.get("pcm_sha256", r.get("source_pcm_sha256")) for r in training}
    groups = {r.get("group", r.get("source_group_id")) for r in training}
    common.need(len(sources["base_train"]) == 79 and len(sources["natural_train"]) == 9 and
                len(shas) == 88 and None not in shas and None not in groups,
                "complete original paired TRAIN identity inventory")
    common.need(not shas.intersection(r["pcm_sha256"] for r in dev_rows) and
                not groups.intersection(r["source_group_id"] for r in dev_rows),
                "new assessment cannot contain an actual paired TRAIN source or waveform")


def verified_synthetic(sources,screen):
    planned=synth.source_plan("primary")
    raw=sources.read("MANIFEST.json")
    doc=json.loads(raw)
    lab=json.loads(screen.read("SCREEN.json"))
    common.need(doc["schema"]==planned["schema"]==synth.SCHEMA
                and doc["source_ancestry_sha256"] == lab["source_ancestry_sha256"] == synth.lineage.ANCESTRY_SHA
                and doc["assessment_sha256"] == lab["assessment_sha256"] == ASSESSMENT_SHA
                and doc["all_prior_replication_sealed"] is True
                and doc["panel"]=="primary" and doc["tts_calls"]==96 and
                doc["registered_plan_file_sha256"]==common.sha((synth.PLANS/"primary.json").read_bytes())
                and doc["new_kws_forwards"]==0 and doc["human_gold"] is False
                and lab["schema"]==synth.BATCH and lab["panel"]=="primary"
                and lab["source_manifest_sha256"]==common.sha(raw)
                and lab["registered_plan_file_sha256"]==doc["registered_plan_file_sha256"]
                and lab["coverage"]==dict.fromkeys(synth.qa.observers.PROVIDERS,96)
                and lab["tts_calls_at_screen"]==lab["new_model_forwards"]==lab["optimizer_updates"]==0
                and lab["training_allowed"] is lab["replication_opened"] is lab["promotion_allowed"] is False,
                "frozen 96 original B17 DEV source + all five unprompted native observers")
    original=doc["rows"]; decisions={r["id"]:r for r in lab["rows"]}
    common.need(len(original)==len(planned["rows"])==len(decisions)==96 and
                len({r["pcm_sha256"] for r in original})==96 and
                len({r["source_group_id"] for r in original})==24,
                "96 full-source unique original PCM and 24 speakers")
    chosen=[];groups={}
    for r,locked in zip(original,planned["rows"],strict=True):
        q=decisions[r["id"]]
        common.need(all(r[k]==v for k,v in locked.items()) and r["role"]=="development"
                    and r["training_allowed"] is False and r["whole_native_resample"] is True
                    and r["crop_calls"]==0 and r["basic_audio_ok"] is True,
                    "source/speaker/IPA/audio original immutable")
        common.need(all(q[k]==v for k,v in r.items() if k!="observed_class"),
                    "machine label did not mutate original source bytes or split")
        if q["observed_class"] is None:
            common.need(q["tier"]=="unknown","reject unlabeled sound")
            continue
        common.need(q["tier"] in ("silver","weak")
                    and q["observed_class"]==r["requested_class"],"blind source label independent")
        groups.setdefault(r["source_group_id"],set()).add(q["observed_class"])
        chosen.append(q)
    pairs=sorted(g for g,c in groups.items() if c=={0,2})
    counts=dict(synthetic_positive=sum(r["observed_class"]==2 for r in chosen),
                synthetic_ordinary=sum(r["observed_class"]==0 for r in chosen),
                synthetic_paired=len(pairs))
    common.need(counts=={"synthetic_positive":lab["source_coverage"]["positive_originals"],
                         "synthetic_ordinary":lab["source_coverage"]["ordinary_originals"],
                         "synthetic_paired":lab["source_coverage"]["paired_source_groups"]}
                and pairs==lab["paired_source_groups"],
                "no source screening omissions or distorted paired numerator")
    return chosen,counts


def validate_natural_roster(doc, rosters, assessment, ancestry):
    fixed = assessment["primary"]["natural"]["speakers"]
    planned = rosters["primary"]
    ids = [r["source_id"] for r in planned]
    people = [str(r["speaker"]) for r in planned]
    common.need(len(ids) == len(set(ids)) == 24 and len(set(fixed)) == 12 and
                people == [person for person in fixed for _ in range(2)] and
                not set(people).intersection(ancestry["natural_excluded_speakers"]),
                "exact new 12-reader/two-utterance roster, no old exposure or duplicate slot")
    observed = [r["source_id"] for r in doc["rows"] + doc["source_rejections"]]
    common.need(len(observed) == len(set(observed)) == 24 and set(observed) == set(ids) and
                all(r["reason"] in ("duration", "audio_quality") for r in doc["source_rejections"]),
                "selected originals and documented quality rejects cover all slots without substitution")


def verified_human(sources,screen):
    raw=sources.read("MANIFEST.json")
    doc=json.loads(raw);lab=json.loads(screen.read("SCREEN.json"))
    roster_raw=sources.read("ROSTERS.json")
    rosters=json.loads(roster_raw)
    design=natural.plan()
    common.need(doc["schema"]==lab["schema"]==natural.SCHEMA and
                doc["panel"]==lab["panel"]=="primary"
                and doc["source_ancestry_sha256"] == lab["source_ancestry_sha256"] == synth.lineage.ANCESTRY_SHA
                and doc["design_sha256"]==ASSESSMENT_SHA
                and doc["all_prior_replication_sealed"] is lab["all_prior_replication_sealed"] is True
                and doc["corpus_sha256"]==design["upstream"]["sha256"]
                and doc["rosters_sha256"]==common.sha(roster_raw)
                and lab["source_manifest_sha256"]==common.sha(raw)
                and lab["coverage"]==dict.fromkeys(natural.qa.observers.PROVIDERS,len(doc["rows"]))
                and doc["kws_forwards"]==lab["kws_forwards"]==0
                and lab["training_allowed"] is False
                and set(rosters)=={"primary"} and len(rosters["primary"])==24,
                "new real human source, five observers, no model selection exposure")
    assessment = json.loads(ASSESSMENT.read_bytes())
    ancestry_bytes = (ASSESSMENT.parent / "source-ancestry.json").read_bytes()
    common.need(common.sha(ancestry_bytes) == assessment["source_ancestry_sha256"],
                "full historical canonical speaker exclusion was frozen before optimizer")
    ancestry = json.loads(ancestry_bytes)
    validate_natural_roster(doc, rosters, assessment, ancestry)
    ref={r["source_id"]:r for r in rosters["primary"]}
    labels={r["id"]:r for r in lab["decisions"]}
    common.need(0 < len(doc["rows"]) == len(labels) <= 24 and
                len({r["pcm_sha256"] for r in doc["rows"]}) == len(doc["rows"]),
                "24 complete human originals")
    admitted=[]
    for row in doc["rows"]:
        q=labels[row["id"]]
        common.need(row["source_id"] in ref and
                    all(row[k]==v for k,v in ref[row["source_id"]].items())
                    and row["role"]=="development" and row["source_group_id"]=="librispeech-"+natural.CORPUS_SPLIT+"-"+str(row["speaker"])
                    and q["pcm_sha256"]==row["pcm_sha256"] and
                    q["source_id"]==row["source_id"] and q["group"]==row["source_group_id"],
                    "unchanged human source and five-observer identity")
        if q["tier"]=="unknown": continue
        common.need(q["tier"] in ("silver","weak") and not q["full_wake_veto"]
                    and q["exact_votes"]>=2,"admit only independent verified human nonwake")
        admitted.append(dict(row,cell_id="read-speech",observed_class=0,tier=q["tier"]))
    common.need(len(admitted)==lab["admitted"] and
                len({r["speaker"] for r in admitted})==lab["admitted_speakers"],
                "true source counts and independent human family denominator")
    return admitted,dict(natural_originals=len(admitted),
                         natural_speakers=len({r["speaker"] for r in admitted}))


def source_gate(observed):
    common.need(set(observed) == set(GATE), "all independently registered source dimensions")
    common.need(all(type(v) is int and v >= 0 for v in observed.values()), "nonnegative integer source counts")
    missing = {k: {"observed": observed[k], "minimum": v} for k, v in GATE.items() if observed[k] < v}
    return ("INSUFFICIENT_SOURCE" if missing else "SOURCE_GATE_PASS"), missing


def run(specfile, inputs, out):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    meta = json.loads(specfile.read_bytes())
    common.need(meta == protocol() and meta["scoring_allowed"] is True,
                "scoring requires authentic registered new sources and completed TRAIN weights")
    out.mkdir()
    report = dict(schema=SCHEMA, assessment_sha256=ASSESSMENT_SHA, training_design_sha256=DESIGN_SHA,
                  source_gate="NOT_RUN", panel_gate="NOT_RUN", actual_forwards=0, optimizer_updates=0,
                  threshold_search=False, promotion_allowed=False, product_released=False,
                  B14_replication_opened=False, B15_replication_opened=False, B16_replication_opened=False,
                  ssc305_physical=False, natural_field_far_per_hour=None, field_3_5m_frr=None,
                  installed_software_baseline="B0", replication_open_allowed=False)
    try:
        with ExitStack() as stack:
            archives = {key: stack.enter_context(common.opened(inputs/(key+".zip"), meta["inputs"][key]))
                        for key in ("synth_source", "synth_screen", "natural_source", "natural_screen")}
            synthetic, synthetic_counts = verified_synthetic(archives["synth_source"], archives["synth_screen"])
            humans, human_counts = verified_human(archives["natural_source"], archives["natural_screen"])
            counts = {**synthetic_counts, **human_counts}
            gate, missing = source_gate(counts)
            report.update(source_coverage=counts, source_gate=gate, source_shortfall=missing,
                          panel_gate="INSUFFICIENT_SOURCE")
            if not missing:
                models, training_sources = snapshots(inputs, meta, torch, np)
                reject_train_overlap(synthetic+humans, training_sources)
                rows, arrays = [], {}
                for key, cohort, admitted in (("synth_source", "synthetic", synthetic),
                                               ("natural_source", "natural", humans)):
                    for row in admitted:
                        scores = common.infer(archives[key].read("wav/"+row["id"]+".wav"),
                                              row, models, arrays, torch, np)
                        rows.append(dict(id=row["id"], group=row["source_group_id"], cohort=cohort,
                                         cell=row["cell_id"], positive=row["observed_class"]==2,
                                         tier=row["tier"], pcm_sha256=row["pcm_sha256"], gains=scores))
                        report["actual_forwards"] = len(arrays)
                common.need(len({r["pcm_sha256"] for r in rows}) == len(rows) and len(arrays) == 12*len(rows),
                            "complete four-model/three-gain original PCM assessment")
                target = out/"ALL_FRAME_LOGITS.npz"
                np.savez_compressed(target, **arrays)
                decision = old_decision.decide(rows)
                report.update(decision, records=rows, logits_sha256=common.sha(target.read_bytes()),
                              model_pt_sha256={k:v[-1] for k,v in models.items()},
                              replication_open_allowed=decision["panel_gate"]=="PASS")
    except Exception as exc:
        report.update(panel_gate="STOP", error=str(exc))
        raise
    finally:
        (out/"RESULT.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: english_v6_b17_primary_eval.py REGISTERED_SPEC INPUT_ARCHIVES OUTPUT")
    run(*map(Path, sys.argv[1:]))
