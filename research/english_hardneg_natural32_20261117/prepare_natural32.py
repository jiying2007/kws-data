"""Freeze 32 natural training-pool utterances from existing metadata only.

No model imports, audio decoding, network, training, or job dispatch. This is a
source plan, not a negative-label producer or an execution authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


POLICY = "english-coverage-hardneg-natural32-20261117-v1"
DOMAIN = POLICY + "/"
RATE = 16000
MIN_SAMPLES = 37760  # first ready plus two confirmation hops; no tail padding
MAX_SAMPLES = 240000
COUNT = 32
TRAINING = {
    "run_id": 43867764366,
    "head_sha": "d8be1c24b12cac0ba554e9a4a41e4b6ee0db18db",
    "artifact_id": 14225893574,
    "name": "snapshot160-weak-embedding-research-43867764366",
    "bytes": 61290521,
    "sha256": "2aa398cd11b787342d91c056fed188d8e3607929da9fdaf873451b5cf05b07f1",
}
QUALITY = {
    "run_id": 44042856615,
    "head_sha": "4f4d3f15214613c0c3a8f92c871da0b419b3f7f7",
    "artifact_id": 14310138100,
    "name": "english-coverage-quality-44042856615",
    "bytes": 50040751,
    "sha256": "56b7e9181cc0e8662b00dd1344ef414a1466a668428fec8bc709b600ba8e97e1",
}
HEAD_SHA = "d3e39adf72495034de0405abb997b76744885f91868eccfa97ba09b929bc63e4"
CORPUS = {
    "dataset": "LibriSpeech dev-clean",
    "url": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "bytes": 337926286,
    "sha256": "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3",
    "official_md5": "42e2234ba48799c1f50f24a7926300a1",
    "license": "CC-BY-4.0",
    "attribution": "V. Panayotov, G. Chen, D. Povey, S. Khudanpur; LibriSpeech / OpenSLR 12",
}
OLD_DEVELOPMENT = {"1462", "1993", "2412", "251", "3081", "3853", "6241", "6295"}
FALSE_FLAGS = {
    "human_gold": False,
    "trusted_negative": False,
    "training_admitted": False,
    "formal_training_admitted": False,
    "pronunciation_verified": False,
    "model_promotion_allowed": False,
    "qualification_allowed": False,
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def unique(pairs):
    out = {}
    for key, value in pairs:
        need(key not in out, "duplicate JSON key: " + key)
        out[key] = value
    return out


def read_json(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def archive(path, expected):
    path = Path(path).absolute()
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), "regular archive required")
    raw = path.read_bytes()
    need(len(raw) == expected["bytes"] and sha(raw) == expected["sha256"], "exact archive identity required")
    opened = zipfile.ZipFile(path)
    names = opened.namelist()
    need(len(names) == len(set(names)), "duplicate ZIP member")
    return opened


def build_plan(training_path, quality_path):
    with archive(training_path, TRAINING) as z:
        source_raw = z.read("LIBRISPEECH_SOURCES.json")
        roster_raw = z.read("STREAM_ROSTER.json")
        source, roster = read_json(source_raw), read_json(roster_raw)
    with archive(quality_path, QUALITY) as z:
        need(z.read("sources/STREAM_ROSTER.json") == roster_raw, "coverage source roster drift")
        need(sha(z.read("models/coverage-head.bin")) == HEAD_SHA, "coverage head identity")
        result = read_json(z.read("evidence/RESULT.json"))
    need(result["status"] == "COMPLETED_WEAK_COVERAGE_QUALITY_RESEARCH" and
         result["source_synthetic_cases"] == 77, "completed fixed coverage evidence")
    need(source["source_pool_is_previously_exposed"] is True, "historical source exposure must remain explicit")
    tracks, originals = source["actual_research_tracks"], source["decoded_originals"]
    need(len(tracks) == 40 and len(originals) == 517 and
         sum(r["frames"] for r in originals) == 58646164, "fixed original corpus roster")
    by_id = {r["id"]: r for r in originals}
    need(len(by_id) == 517, "unique original IDs")
    speakers = {r["speaker"] for r in originals}
    need(len(speakers) == 40 and speakers == {r["speaker"] for r in tracks}, "fixed 40 speakers")
    old_order = sorted(speakers, key=lambda s: (sha(("en63-bg48-speaker-v1/" + s).encode()), s))
    old_dev = set(old_order[32:])
    need(old_dev == OLD_DEVELOPMENT, "old study-development boundary")
    current_order = sorted(speakers, key=lambda s: (sha(("en-final-embedding-v1/" + s).encode()), s))
    current_dev = set(current_order[:10])
    need({t["speaker"] for t in tracks if t["role"] == "development"} == current_dev and
         {t["speaker"] for t in tracks if t["role"] == "train"} == speakers - current_dev,
         "current training/development speaker boundary")
    eligible_speakers = speakers - old_dev - current_dev
    need(len(eligible_speakers) == 24, "24-speaker intersection of original train roles")
    consumed_ids = {f["source_id"] for t in tracks for f in t["fragments"]}
    need(len(consumed_ids) == 114 and consumed_ids <= set(by_id), "all 114 consumed original members excluded")
    remaining = [r for r in originals if r["speaker"] in eligible_speakers and r["id"] not in consumed_ids]
    need(len(remaining) == 241 and sum(r["frames"] for r in remaining) == 27649040,
         "remaining source denominator")
    bounded = [r for r in remaining if MIN_SAMPLES <= r["frames"] <= MAX_SAMPLES]
    speaker_order = sorted(eligible_speakers, key=lambda s: (sha((DOMAIN + "speaker/" + s).encode()), s))
    per_speaker = {
        s: sorted([r for r in bounded if r["speaker"] == s],
                  key=lambda r: (sha((DOMAIN + "source/" + r["id"]).encode()), r["id"]))
        for s in speaker_order
    }
    need(all(per_speaker.values()), "at least one bounded original per training speaker")
    selected = []
    depth = 0
    while len(selected) < COUNT:
        before = len(selected)
        for speaker in speaker_order:
            if depth < len(per_speaker[speaker]):
                selected.append(per_speaker[speaker][depth])
                if len(selected) == COUNT:
                    break
        need(len(selected) > before, "insufficient distinct originals")
        depth += 1
    rows = []
    for i, row in enumerate(selected):
        frames = row["frames"]
        need(row["trusted_negative"] is False, "do not upgrade source labels")
        embeddings = 1 + (frames - RATE) // 1280
        rows.append({
            "index": i,
            "recording_id": "hardneg-natural32-" + sha(row["id"].encode())[:24],
            "source_member": row["id"],
            "speaker_id": row["speaker"],
            "source_family_id": "librispeech-dev-clean-speaker-" + row["speaker"],
            "source_role": "train",
            "old_bg48_role": "train",
            "current_coverage_role": "train",
            "flac_sha256": row["flac_sha256"],
            "pcm_sha256": row["pcm_sha256"],
            "transcript_sha256": row["transcript_sha256"],
            "frames": frames,
            "sample_rate": RATE,
            "channels": 1,
            "sample_width": 2,
            "duration_seconds": frames / RATE,
            "expected_embedding_rows": embeddings,
            "expected_ready_rows": embeddings - 15,
            "current_coverage_original_source_consumed": False,
            "historically_exposed_to_other_models": True,
            "label_status": "UNKNOWN_REVIEW_REQUIRED",
            "expected_events": None,
            "allowed_use": ["one_frozen_head_research_scan", "candidate_acoustic_review"],
            **FALSE_FLAGS,
        })
    samples = sum(r["frames"] for r in rows)
    need(samples <= 480 * RATE and len({r["source_member"] for r in rows}) == COUNT,
         "32 distinct sources, at most 480 seconds")
    return {
        "schema": POLICY,
        "status": "METADATA_FROZEN_PLAN_NO_EXECUTION",
        "data_repository_commit": "61e524e89c86950ef8e284ecaf18fddf82d11ca1",
        "pipeline_repository_commit": "aec192dd7c5955a7fa2b709564cdacb185a590be",
        "training_source": TRAINING,
        "quality_source": QUALITY,
        "coverage_head_binary_sha256": HEAD_SHA,
        "head_profile_status": "UNCHANGED_HISTORICAL_PROFILE_NO_IPA_RELABEL",
        "corpus": CORPUS,
        "metadata_inputs": {"LIBRISPEECH_SOURCES.json": sha(source_raw), "STREAM_ROSTER.json": sha(roster_raw)},
        "selection": {
            "domain": DOMAIN,
            "rule": "train-role intersection; exclude every source_id in all 40 consumed tracks; whole-utterance duration bounds; hash speaker/source then round-robin",
            "old_bg48_study_development_excluded": sorted(old_dev),
            "current_coverage_development_excluded": sorted(current_dev),
            "eligible_train_speakers": sorted(eligible_speakers),
            "consumed_original_members_excluded": sorted(consumed_ids),
            "remaining_original_count": 241,
            "remaining_original_seconds": 1728.065,
            "bounded_original_count": len(bounded),
            "minimum_samples": MIN_SAMPLES,
            "maximum_samples": MAX_SAMPLES,
            "selected_originals": COUNT,
            "selected_speakers": len({r["speaker_id"] for r in rows}),
            "observed_source_samples": samples,
            "observed_source_seconds": samples / RATE,
            "selection_uses_any_candidate_model_scores": False,
        },
        "execution_request_required": True,
        "proposed_execution": {
            "environment": "public GitHub Actions native CPU only",
            "full_pcm_passes": COUNT,
            "cpu_threads": 1,
            "maximum_execution_seconds": 1200,
            "input_chunk_samples": 320,
            "expected_embedding_rows": sum(r["expected_embedding_rows"] for r in rows),
            "expected_ready_rows": sum(r["expected_ready_rows"] for r in rows),
            "threshold": 0.8,
            "confirm_embeddings": 3,
            "cooldown_embeddings": 13,
            "stream_policy": "one fresh stream per original utterance; reset to sample 0 only between originals",
            "eof_padding_samples": 0,
            "gain_changes": 0,
            "resampling": False,
            "concatenation": False,
            "new_tts_asr_phone_calls": 0,
            "optimizer_updates": 0,
            "retry_count": 0,
            "original_audio_seconds_are_not_natural_continuous_far_exposure": True,
        },
        "model_forward_calls_performed": 0,
        "optimizer_updates_performed": 0,
        "new_independent_speaker_count": 0,
        "verified_negative_seconds": 0,
        "independent_far": "NOT_MEASURED",
        "independent_frr": "NOT_MEASURED",
        "rows": rows,
        **FALSE_FLAGS,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    need(not output.exists() and output.parent.is_dir() and
         not any(p.is_symlink() for p in (output, *output.parents)), "create-only output with existing parent")
    plan = build_plan(args.training, args.quality)
    raw = canonical(plan)
    with output.open("xb") as target:
        target.write(raw)
    print(json.dumps({"status": plan["status"], "path": str(output), "sha256": sha(raw),
                      "selected_originals": COUNT, "selected_speakers": plan["selection"]["selected_speakers"],
                      "seconds": plan["selection"]["observed_source_seconds"],
                      "embedding_rows": plan["proposed_execution"]["expected_embedding_rows"],
                      "ready_rows": plan["proposed_execution"]["expected_ready_rows"],
                      "model_forward_calls": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
