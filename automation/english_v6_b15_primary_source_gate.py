"""Authenticate frozen B15 independent PRIMARY source and STOP if admission shortfalls.

Zero KWS/model forwards, zero optimizer updates, no threshold search, no CAL/LOCKED.
The saved source gate never repairs, substitutes, or relaxes any frozen DEV row.
"""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import io
import json
from pathlib import Path
import wave
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ASSESSMENT = ROOT / "automation/english_v6_b15_panels/assessment.json"
ASSESSMENT_SHA = "daa45200d226dd004a41e74a3cf5ff876a1f3fc41dd2def7c5ed937efbf085f8"
SYNTH_PLAN = ROOT / "automation/english_v6_b15_panels/primary-synthetic.json"
SYNTH_PLAN_SHA = "afde489a5a87a7ea2cc037ab21e38d779270cf1bfa3a2d496064f3a779910f1b"
PROVIDERS = {"qwen", "whisper", "parakeet", "xlsr", "lv60"}
LOCKS = {
    "pair": {
        "id": 14877145654, "run": 45296300913,
        "head": "b3d2a6580eba1f2249b3df3ab3c9f1cd4c2b5a46",
        "bytes": 1026648, "sha256": "9607876591b155f728b4f525029fee933cca98eb0921f4bd84c7d6288f4e75c4",
    },
    "synth_source": {
        "id": 14878153098, "run": 45300438645,
        "head": "755da1b9aa9fe3a4b643a07aef9831c74e0291be",
        "bytes": 2639166, "sha256": "8b8ad3109cf35a80455a369aeaa65389b45f4d1aee3403c6286cba9ea7c12ed7",
    },
    "synth_screen": {
        "id": 14878927172, "run": 45300438645,
        "head": "755da1b9aa9fe3a4b643a07aef9831c74e0291be",
        "bytes": 10318, "sha256": "b4ca67a646e8a4f8d4de2c19f0fca360bb3276255ce7ed998f998ec253eb20d9",
    },
    "natural_source": {
        "id": 14878351529, "run": 45297962939,
        "head": "3da0d5a35d8dcd7d1c79bbf870a1df776c9d7c8d",
        "bytes": 3983738, "sha256": "a5b0e07ebd583bce0fd561a738fcd540619541a22534bbdf3f20b2550d2eda97",
    },
    "natural_screen": {
        "id": 14877684389, "run": 45297962939,
        "head": "3da0d5a35d8dcd7d1c79bbf870a1df776c9d7c8d",
        "bytes": 5104, "sha256": "c1ccf0182110daf65d77e2ca3c800f46afc78fccb09057d49895b393ace7e992",
    },
}

def need(ok, reason):
    if not ok:
        raise ValueError(reason)

def digest(data):
    return sha256(data).hexdigest()

def open_pinned(path, role):
    lock = LOCKS[role]
    raw = path.read_bytes()
    need(len(raw) == lock["bytes"] and digest(raw) == lock["sha256"], "exact archived ZIP: " + role)
    archive = zipfile.ZipFile(io.BytesIO(raw))
    need(len(archive.namelist()) == len(set(archive.namelist())) and archive.testzip() is None,
         "archived source CRC and unique names: " + role)
    return archive

def verify_wav(archive, row):
    wav = archive.read("wav/" + row["id"] + ".wav")
    need(digest(wav) == row["wav_sha256"], "original WAV digest " + row["id"])
    with wave.open(io.BytesIO(wav)) as f:
        need((f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getnframes()) ==
             (1, 2, 16000, row["samples"]), "exact 16k mono original " + row["id"])
        pcm = f.readframes(f.getnframes())
    need(digest(pcm) == row["pcm_sha256"] and len(pcm) == 2 * row["samples"]
         and row["basic_audio_ok"] is True, "complete source PCM identity " + row["id"])

def read_synth(source, screen):
    raw = source.read("MANIFEST.json")
    manifest = json.loads(raw)
    labels = json.loads(screen.read("SCREEN.json"))
    need(digest(SYNTH_PLAN.read_bytes()) == SYNTH_PLAN_SHA, "frozen primary source plan")
    plan = json.loads(SYNTH_PLAN.read_bytes())
    need(manifest["schema"] == plan["schema"] == "english-v6-b15-frozen-dev-panel-v1"
         and manifest["panel"] == plan["panel"] == "primary"
         and manifest["registered_plan_file_sha256"] == SYNTH_PLAN_SHA
         and manifest["tts_calls"] == 48 and manifest["new_kws_forwards"] == 0
         and manifest["human_gold"] is False
         and labels["source_manifest_sha256"] == digest(raw)
         and manifest["plan_sha256"] == labels["plan_sha256"]
         and labels["frozen_before_kws"] is True and labels["kws_forwards"] == labels["optimizer_updates"] == 0
         and labels["promotion_allowed"] is False, "blind synthetic role and evidence bindings")
    need(labels["coverage"] == dict.fromkeys(PROVIDERS, 48), "complete five native synthetic observers")
    originals = manifest["rows"]
    rows = {r["id"]: r for r in labels["rows"]}
    need(len(originals) == len(rows) == 48 and len({r["pcm_sha256"] for r in originals}) == 48,
         "48 independent synthetic original PCM")
    classes = defaultdict(set)
    class_counts = Counter()
    for original, planned in zip(originals, plan["rows"], strict=True):
        need(all(original[k] == v for k, v in planned.items())
             and original["training_allowed"] is False
             and original["role"] == "development" and original["whole_native_resample"] is True
             and original["crop_calls"] == 0, "no synthetic source substitution")
        verdict = rows[original["id"]]
        need(all(verdict[k] == v for k, v in original.items() if k != "observed_class"),
             "blind source label identity")
        verify_wav(source, original)
        native = b""
        for chunk in original["native_chunks"]:
            data = source.read("native/" + original["id"] + "/" + chunk["file"])
            need(len(data) == 2 * chunk["samples"] and digest(data) == chunk["native_pcm_sha256"],
                 "original TTS native chunk identity")
            native += data
        need(digest(native) == original["native_pcm_sha256"], "native whole-source PCM")
        if verdict["observed_class"] is None:
            need(verdict["tier"] == "unknown", "UNKNOWN must not be labeled")
            continue
        need(verdict["tier"] in ("silver", "weak")
             and verdict["observed_class"] == original["requested_class"], "blind source class match")
        class_counts[verdict["observed_class"]] += 1
        classes[original["source_group_id"]].add(verdict["observed_class"])
    paired = sorted(k for k, v in classes.items() if v == {0, 2})
    need(paired == labels["paired_source_groups"] and len(paired) >= 4
         and labels["meets_four_pair_minimum"] is True
         and len({r["source_group_id"] for r in originals}) == 12, "whole source-paired count")
    return {
        "synthetic_qualified_positive_originals": class_counts[2],
        "synthetic_qualified_ordinary_originals": class_counts[0],
        "synthetic_paired_voice_families": len(paired),
    }

def read_natural(source, screen):
    raw = source.read("MANIFEST.json")
    manifest = json.loads(raw)
    labels = json.loads(screen.read("SCREEN.json"))
    roster_raw = source.read("ROSTERS.json")
    rosters = json.loads(roster_raw)
    need(manifest["schema"] == labels["schema"] == "english-v6-b15-ordinary-human-dev-clean-source-v1"
         and manifest["panel"] == labels["panel"] == "primary"
         and manifest["design_sha256"] == ASSESSMENT_SHA
         and manifest["rosters_sha256"] == digest(roster_raw)
         and labels["source_manifest_sha256"] == digest(raw)
         and labels["coverage"] == dict.fromkeys(PROVIDERS, 24)
         and manifest["kws_forwards"] == labels["kws_forwards"] == 0
         and manifest["training_allowed"] is labels["training_allowed"] is False,
         "frozen natural DEV and native five-observer roles")
    need(set(rosters) == {"primary"} and len(rosters["primary"]) == 24
         and len({r["speaker"] for r in rosters["primary"]}) == 12,
         "new 12-reader prospective natural source roster")
    selected = {r["source_id"]: r for r in rosters["primary"]}
    labels_by_id = {r["id"]: r for r in labels["decisions"]}
    originals = manifest["rows"]
    need(len(originals) == len(labels_by_id) == labels["source_rows"] == 24
         and len({r["pcm_sha256"] for r in originals}) == 24, "unique 24 human originals")
    accepted = 0
    groups = set()
    for row in originals:
        verdict = labels_by_id[row["id"]]
        need(row["source_id"] in selected and all(row[k] == v for k, v in selected[row["source_id"]].items())
             and row["role"] == "development" and row["requested_class"] == 0
             and row["source_group_id"] == "librispeech-dev-clean-" + str(row["speaker"])
             and verdict["pcm_sha256"] == row["pcm_sha256"]
             and verdict["group"] == row["source_group_id"]
             and verdict["source_id"] == row["source_id"], "original human reader lineage")
        verify_wav(source, row)
        need(digest(source.read("flac/" + row["id"] + ".flac")) == row["flac_sha256"],
             "source original FLAC digest")
        if verdict["tier"] == "unknown":
            continue
        need(verdict["tier"] in ("silver", "weak") and verdict["exact_votes"] >= 2
             and verdict["full_wake_veto"] is False
             and set(verdict["native_text"]) == {"qwen", "whisper", "parakeet"}
             and set(verdict["phone_sha256"]) == {"xlsr", "lv60"},
             "unprompted human ordinal admission")
        accepted += 1
        groups.add(row["speaker"])
    need(accepted == labels["admitted"] and len(groups) == labels["admitted_speakers"],
         "human DEV admitted source groups")
    return {"natural_qualified_speakers": len(groups), "natural_qualified_originals": accepted}

def gate(counts, requirements):
    need(set(counts) == set(requirements), "only the five pre-registered metrics")
    need(all(type(v) is int and v >= 0 for v in counts.values()), "nonnegative source counts")
    short = {k: {"observed": counts[k], "required": v, "missing": v-counts[k]}
             for k, v in requirements.items() if counts[k] < v}
    return ("INSUFFICIENT_SOURCE" if short else "SOURCE_GATE_PASS", short)

def self_test():
    mins = {"synthetic_qualified_positive_originals": 8, "synthetic_qualified_ordinary_originals": 12,
            "synthetic_paired_voice_families": 5, "natural_qualified_speakers": 8,
            "natural_qualified_originals": 12}
    obs = {"synthetic_qualified_positive_originals": 6, "synthetic_qualified_ordinary_originals": 22,
           "synthetic_paired_voice_families": 4, "natural_qualified_speakers": 12,
           "natural_qualified_originals": 18}
    state, deficits = gate(obs, mins)
    need(state == "INSUFFICIENT_SOURCE"
         and set(deficits) == {"synthetic_qualified_positive_originals", "synthetic_paired_voice_families"}
         and deficits["synthetic_qualified_positive_originals"]["missing"] == 2
         and deficits["synthetic_paired_voice_families"]["missing"] == 1,
         "source gate fail-closed diagnostics")
    good, none = gate(mins, mins)
    need(good == "SOURCE_GATE_PASS" and not none, "independent source gate pass contract")
    print("PASS: fixed original-source denominators and fail-closed incomplete admission")

def run(inputs, output):
    need(digest(ASSESSMENT.read_bytes()) == ASSESSMENT_SHA, "immutable pre-optimizer B15 gates")
    study = json.loads(ASSESSMENT.read_bytes())
    need(study["schema"] == "english-v6-b15-prospective-nonblank-contrast-assessment-v1"
         and study["freeze_before_optimizer"] is True
         and study["constraints"]["retained_b14_replication_sealed"] is True,
         "no previously scored DEV re-use")
    with open_pinned(inputs/"pair.zip","pair") as pair, \
         open_pinned(inputs/"synth_source.zip","synth_source") as ss, \
         open_pinned(inputs/"synth_screen.zip","synth_screen") as sc, \
         open_pinned(inputs/"natural_source.zip","natural_source") as ns, \
         open_pinned(inputs/"natural_screen.zip","natural_screen") as nc:
        learned = json.loads(pair.read("RESULT.json"))
        need(learned["actual_optimizer_updates"] == 384 and learned["trained_model_parameters"] == 43876
             and learned["dev_cal_locked_forwards"] == 0 and learned["promotion_allowed"] is False
             and learned["independent_assessment_sha256"] == ASSESSMENT_SHA,
             "real B15 TRAIN and no opened DEV in optimizer")
        observed = {**read_synth(ss, sc), **read_natural(ns, nc)}
    requirements = study["primary_gate"]["source_minimums"]
    state, deficit = gate(observed, requirements)
    result = {
        "schema": "english-v6-b15-original-primary-source-gate-v1",
        "keyword": "Hey Mero",
        "preregistered_assessment_sha256": ASSESSMENT_SHA,
        "status": state,
        "observed": observed,
        "minimum": requirements,
        "source_shortfall": deficit,
        "genuine_paired_training_updates": 384,
        "model_parameters": 43876,
        "zip_provenance": LOCKS,
        "full_original_pcm_identity_verified": True,
        "blind_observer_source_coverage": {"natural": 120, "synthetic": 240},
        "kws_model_forwards": 0, "optimizer_updates": 0,
        "threshold_search": False, "opened_cal_locked": False,
        "B14_replication_opened": False, "B15_replication_opened": False,
        "ssc305_physical_measured": False,
        "model_promotion_allowed": False, "field_far_per_hour": None,
        "field_frr": None, "software_baseline": "B0",
    }
    output.mkdir()
    (output/"RESULT.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ("status","observed","source_shortfall",
                                               "kws_model_forwards","model_promotion_allowed")}, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        need(args.inputs and args.output, "original source archives and new output required")
        run(args.inputs, args.output)
