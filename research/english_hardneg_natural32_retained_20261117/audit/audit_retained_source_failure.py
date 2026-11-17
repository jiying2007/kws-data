#!/usr/bin/env python3
"""Independent saved-byte audit of retained FLAC acquisition with zero compute.

Reads ZIP/JSON/FLAC STREAMINFO only. Never imports an execution worker, launches
a subprocess, decodes audio, constructs a model, compiles, or accesses a network.
The expected PCM SHA values are authenticated metadata, not fresh PCM results.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

ARTIFACT = {"artifact_id": 14317913716, "run_id": "44060185229",
            "head_sha": "cb6baa703f46451af73ec2cc0f85edfd1b159874", "bytes": 16339221,
            "sha256": "3b7a4ff6a59ba397eebac59a4f89feba8cc075a4f8881c9d531a3739bc1ec9a7"}
FIRST = {"artifact_id": 14317102541, "run_id": 44057968882,
         "head_sha": "d4edc97e4efb5612d16018f5cdeb42002d9eedfa", "bytes": 4257652,
         "sha256": "514b70f69470620612d6ea3dfa84a53369468d0f86a659863a4301c108266e12"}
MANIFEST_SHA = "f4bee5d2d80cd26e5bd58f661644198e9550329b288d3bdb557f5d7bb60132e6"
FIRST_MANIFEST_SHA = "818e160774e8885ca4efa36bdaff13971e819079b66190781aae7c2fe3cd7a63"
RESEARCH = "research/english_hardneg_natural32_20261117/"
ROSTER_SHA = "31457faf8efd0cf77629c1a6f8de67bc101190eb118bcfe9ac2f256acca41239"
CONFIG_SHA = "13ff76c46b35d8d1c427bc237cfc3153771f281379ad86df8203f685aa2e7af4"
HEAD = {"bytes": 197908, "sha256": "d3e39adf72495034de0405abb997b76744885f91868eccfa97ba09b929bc63e4"}
LIBRARY = {"bytes": 2441224, "sha256": "040089f781d87366a3d8a27933043aa96e891849f2bf9ce6bbd2321d0d70152a"}
WORKER_SHA = "56fb6ed780cf569b994bb46f35b4b5909a15837ef90818ea5388c550694e62ae"
PLAN_SHA = "81cb7eff374225ee948d2f44acc2a18c5785b729a496ff0638a81fc2f5a79e56"
WORKFLOW_SHA = "f5a369afeec687921cefb1e5fd6d215031cba8e2ae61d6f53632c59df505f113"
OLD_WORKFLOW = "source/.github/workflows/english-hardneg-natural32-once.yml"
OLD_WORKFLOW_PIN = {"bytes": 4423, "sha256": "2709d3cb7197624ca35e270974cd43bee2132d931d28dc109dda00d98b084d88"}
CORPUS = {"url": "https://www.openslr.org/resources/12/dev-clean.tar.gz", "bytes": 337926286,
          "sha256": "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3"}
FALSE_FLAGS = ("human_gold", "trusted_negative", "training_admitted", "formal_training_admitted",
               "pronunciation_verified", "model_promotion_allowed", "qualification_allowed")


def need(ok, why):
    if not ok:
        raise ValueError(why)


def ident(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def load(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            need(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def read_zip(raw, count):
    out = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        need(len(infos) == count and sum(i.file_size for i in infos) < 64 * 1024**2, "ZIP bounds")
        for item in infos:
            n = item.filename
            need(not item.is_dir() and n not in out and not PurePosixPath(n).is_absolute() and
                 "\\" not in n and "\0" not in n and all(p not in ("", ".", "..") for p in n.split("/")), "safe unique regular ZIP names")
            need(not item.flag_bits & 1 and stat.S_IFMT(item.external_attr >> 16) in (0, stat.S_IFREG), "regular nonencrypted ZIP member")
            out[n] = archive.read(item)
    return out


def verify_zero(files, second):
    result, supervisor, start = (load(files[p]) for p in ("evidence/RESULT.json", "evidence/SUPERVISOR_EXIT.json", "STARTED.json"))
    pin = ARTIFACT if second else FIRST
    binding = {"repository": "jiying2007/kws-data", "run_id": str(pin["run_id"]), "head_sha": pin["head_sha"],
               "main_before": FIRST["head_sha"] if second else "61e524e89c86950ef8e284ecaf18fddf82d11ca1",
               "run_attempt": 1, "event": "push", "public_repository": True}
    for item in (result, supervisor, start["binding"]):
        need(all(type(item[k]) is type(v) and item[k] == v for k, v in binding.items()), "exact failed run binding")
    need(result["passed"] is False and result["status"] == "FAILED_NO_RETRY" and result["reports"] == [], "failed zero-report result")
    for key in ("consumer_build_attempts", "decoder_process_attempts", "model_process_attempts", "completed_validated_traces",
                "new_onnx_calls", "python_head_forwards", "new_tts_asr_phone_calls", "optimizer_updates", "retry_count"):
        need(type(result[key]) is int and result[key] == 0, "zero prior compute counter: " + key)
    need(all(result[k] is False for k in FALSE_FLAGS) and result["quality_improvement_claim"] is False and
         result["verified_negative_count"] == result["verified_negative_seconds"] == 0 and
         result["independent_far"] == result["independent_frr"] == "NOT_MEASURED", "UNKNOWN quality boundary")
    need(result["candidate_head"] == HEAD and result["native_library"] == LIBRARY and result["configuration_sha256"] == CONFIG_SHA,
         "same fixed head/library/configuration")
    need(supervisor["passed"] is False and supervisor["returncode"] == 1 and supervisor["timed_out"] is False and
         supervisor["error"] is None and supervisor["partial_receipt_errors"] == [] and supervisor["retry_count"] == 0 and
         supervisor["cleanup"] == {"descendants_found": False, "terminated": False, "killed": False}, "clean failed supervisor")
    wall = 537.4946303590001 if second else 307.016853929
    need(supervisor["wall_seconds"] == wall and start["maximum_seconds"] == 1200 and
         start["monotonic_deadline"] - start["monotonic_started"] == 1200, "supervisor measured time/deadline")
    commands = {"current-run": ["gh", "api", "repos/jiying2007/kws-data/actions/runs/" + str(pin["run_id"])]}
    sources = {"training": (43867764366, 14225893574), "quality": (44042856615, 14310138100), "sdk": (43984579132, 14285590605)}
    if second:
        sources["prior-failure"] = (FIRST["run_id"], FIRST["artifact_id"])
    for source, (run_id, artifact_id) in sources.items():
        commands[source + "-run"] = ["gh", "api", "repos/jiying2007/kws-data/actions/runs/" + str(run_id)]
        commands[source + "-artifact"] = ["gh", "api", "repos/jiying2007/kws-data/actions/artifacts/" + str(artifact_id)]
        commands[source + "-zip"] = ["gh", "api", "repos/jiying2007/kws-data/actions/artifacts/" + str(artifact_id) + "/zip"]
    commands["corpus-download"] = ["curl", "--fail", "--location", "--proto", "=https", "--proto-redir", "=https",
                                   "--connect-timeout", "20", "--max-time", "600" if second else "300", "--retry", "0",
                                   "--silent", "--show-error", CORPUS["url"]]
    stages = {"source-binding", "authenticate-artifacts", "select-original-corpus"}
    if second:
        commands["ffmpeg-version"] = ["ffmpeg", "-version"]
        stages |= {"verify-prior-acquisition-failure", "decode-originals"}
    tags = set(commands) | {"stage-" + s for s in stages}
    need({n for n in files if n.startswith("evidence/") and n.endswith(".started.json")} ==
         {"evidence/" + s + ".started.json" for s in tags}, "exact command/stage start set")
    need({n for n in files if n.startswith("evidence/") and n.endswith(".exit.json")} ==
         {"evidence/" + s + ".exit.json" for s in tags}, "exact command/stage exit set")
    pairs = []
    missing_ffmpeg = "FileNotFoundError:[Errno 2] No such file or directory: 'ffmpeg'"
    for tag in sorted(tags):
        begin = load(files["evidence/" + tag + ".started.json"])
        end = load(files["evidence/" + tag + ".exit.json"])
        need(end["retry_count"] == 0 and end["wall_seconds"] >= 0 and
             start["monotonic_started"] <= begin["monotonic_started"] < start["monotonic_deadline"] and
             begin["monotonic_started"] + end["wall_seconds"] < start["monotonic_deadline"], "complete no-retry bounded step")
        if tag in commands:
            need(begin["argv"] == end["argv"] == commands[tag] and begin["kind"] is end["kind"] is None and
                 begin["attempt"] == end["attempt"] == 1 and begin["retry_count"] == 0 and
                 begin["new_session"] is False and begin["cpu_threads"] == 1 and
                 begin["global_deadline"] == start["monotonic_deadline"] and end["timed_out"] is False and
                 end["descendants_supervised_by_parent_process_group"] is True, "noncompute acquisition/probe command pair")
            expected_error = missing_ffmpeg if second and tag == "ffmpeg-version" else (
                "ValueError:COMMAND_NONZERO_EXIT" if not second and tag == "corpus-download" else None)
            expected_rc = None if second and tag == "ffmpeg-version" else (28 if expected_error else 0)
            need(end["error"] == expected_error and end["returncode"] == expected_rc, "exact failing acquisition/dependency probe")
        else:
            need(begin["stage"] == end["stage"] == tag[len("stage-"):], "stage pairing")
            failed = "stage-decode-originals" if second else "stage-select-original-corpus"
            error = "ValueError:" + missing_ffmpeg if second else "ValueError:ValueError:COMMAND_NONZERO_EXIT"
            need(end["error"] == (error if tag == failed else None), "exact failure stage")
        pairs.append({"tag": tag, "kind": begin.get("kind"), "argv": begin.get("argv"),
                      "returncode": end.get("returncode"), "error": end["error"], "wall_seconds": end["wall_seconds"]})
    prohibited = ("selectedinputs/pcm/", "traces/", "scores/", "candidates/", "bin/",
                  "evidence/decode-", "evidence/scan-", "evidence/build-native", "evidence/native-")
    need(not any(n.startswith(prohibited) for n in files), "no PCM/trace/score/compute output")
    if second:
        need(result["error"] == "ValueError:" + missing_ffmpeg and result["prior_compute_attempts"] == 0 and
             result["acquisition_execution_attempt"] == 2 and result["aggregate_maximum_model_processes"] == 32, "second-run failure and aggregate scope")
    return {"run_binding": binding, "build_attempts": 0, "decode_attempts": 0, "model_attempts": 0,
            "completed_validated_traces": 0, "command_pairs": len(commands), "stage_pairs": len(stages),
            "all_command_kinds_are_null": True, "start_exit_pairs": pairs, "supervisor_wall_seconds": wall,
            "global_deadline_reached": False, "descendants_found": False, "zero_compute_verified": True}


def audit(path):
    raw = path.read_bytes()
    need(ident(raw) == {k: ARTIFACT[k] for k in ("bytes", "sha256")}, "fixed retained failure ZIP")
    files = read_zip(raw, 292)
    manifest = load(files["MANIFEST.json"])
    need(ident(files["MANIFEST.json"])["sha256"] == MANIFEST_SHA and len(manifest["files"]) == 291 and
         set(files) == set(manifest["files"]) | {"MANIFEST.json"}, "strict closed second artifact")
    for name, pin in manifest["files"].items():
        need(ident(files[name]) == pin, "all retained member bytes/hash: " + name)
    need(manifest["supervised_success"] is False and manifest["downloads_excluded"] is True and
         manifest["authoritative_completion_receipt"] == "evidence/SUPERVISOR_EXIT.json" and
         all(manifest[k] is False for k in FALSE_FLAGS), "failed manifest authority/labels")
    for name, value in {"source/automation/english_hardneg_natural32_recovery_once.py": WORKER_SHA,
                        "source/automation/english_hardneg_natural32_recovery_plan.json": PLAN_SHA,
                        "source/.github/workflows/english-hardneg-natural32-recovery-once.yml": WORKFLOW_SHA,
                        "source/CONFIG.json": CONFIG_SHA}.items():
        need(ident(files[name])["sha256"] == value, "frozen recovery source/config")
    need(ident(files[OLD_WORKFLOW]) == OLD_WORKFLOW_PIN, "hidden old workflow retained")
    plan = load(files["source/automation/english_hardneg_natural32_recovery_plan.json"])
    for name, pin in plan["source_pins"].items():
        need(ident(files["source/" + name]) == pin, "all original source pins preserved")
    need(ident(files["models/coverage-head.bin"]) == HEAD and
         ident(files["installed-base/sdk-native/lib/libekws_research_stream.a"]) == LIBRARY, "retained unchanged model/library")
    zero = verify_zero(files, True)

    first_raw = files["prior-failure/original-artifact.zip"]
    need(ident(first_raw) == {k: FIRST[k] for k in ("bytes", "sha256")}, "first ZIP retained byte exact")
    first = read_zip(first_raw, 85)
    fm = load(first["MANIFEST.json"])
    need(ident(first["MANIFEST.json"])["sha256"] == FIRST_MANIFEST_SHA and len(fm["files"]) == 85 and
         set(fm["files"]) - set(first) == {OLD_WORKFLOW} and set(first) - set(fm["files"]) == {"MANIFEST.json"} and
         fm["files"][OLD_WORKFLOW] == OLD_WORKFLOW_PIN, "only first exact historical workflow omission")
    for name in first:
        if name != "MANIFEST.json":
            need(ident(first[name]) == fm["files"][name], "first retained payload hash")
        need(files["prior-failure/retained/" + name] == first[name], "nested first file preserved byte exact")
    need({n[len("prior-failure/retained/"):] for n in files if n.startswith("prior-failure/retained/")} == set(first), "nested first exact file inventory")
    need(ident(files["prior-failure/supplemental-source/.github/workflows/english-hardneg-natural32-once.yml"]) == OLD_WORKFLOW_PIN,
         "separate first workflow supplement retained")
    first_zero = verify_zero(first, False)
    proof_raw = files["evidence/PRIOR_ACQUISITION_FAILURE_VERIFIED.json"]
    need(proof_raw == files["prior-failure/OMISSION_AND_ZERO_COMPUTE_PROOF.json"], "same first-failure proof retained twice")
    proof = load(proof_raw)
    need(proof["closed_manifest"] is False and proof["original_zip_unchanged"] is True and
         proof["prior_decode_attempts"] == proof["prior_build_attempts"] == proof["prior_compute_attempts"] == proof["prior_trace_count"] == 0,
         "first proof reports omission and zero computation")
    prior_run, prior_artifact = (load(files[n]) for n in ("evidence/prior-failure-run.stdout", "evidence/prior-failure-artifact.stdout"))
    need(prior_run["id"] == FIRST["run_id"] and prior_run["head_sha"] == FIRST["head_sha"] and prior_run["run_attempt"] == 1 and
         prior_run["status"] == "completed" and prior_run["conclusion"] == "failure", "saved first run API identity")
    need(prior_artifact["id"] == FIRST["artifact_id"] and prior_artifact["size_in_bytes"] == FIRST["bytes"] and
         prior_artifact["digest"] == "sha256:" + FIRST["sha256"] and prior_artifact["workflow_run"]["id"] == FIRST["run_id"] and
         prior_artifact["workflow_run"]["head_sha"] == FIRST["head_sha"], "saved first artifact API identity")

    roster_raw = files["source/" + RESEARCH + "natural32-roster.json"]
    need(ident(roster_raw)["sha256"] == ROSTER_SHA, "same frozen 32 original roster")
    roster = load(roster_raw)
    rows = roster["rows"]
    need(len(rows) == len({r["recording_id"] for r in rows}) == 32 and len({r["speaker_id"] for r in rows}) == 24 and
         sum(r["frames"] for r in rows) == 3270080 and sum(r["expected_embedding_rows"] for r in rows) == 2174 and
         sum(r["expected_ready_rows"] for r in rows) == 1694, "unchanged source and future observation budgets")
    for name, pin in roster["metadata_inputs"].items():
        need(ident(files["source-metadata/" + name])["sha256"] == pin, "frozen original source metadata")
    originals = load(files["source-metadata/LIBRISPEECH_SOURCES.json"])["decoded_originals"]
    original_by_id = {r["id"]: r for r in originals}
    need(len(originals) == len(original_by_id) == 517, "517 original metadata identities")
    tracks = load(files["source-metadata/STREAM_ROSTER.json"])
    background = [r for r in tracks if r.get("source", {}).get("kind") == "exposed-read-speech-splice"]
    consumed = {f["source_id"] for t in background for f in t["source"]["fragments"]}
    need(len(background) == 40 and len(consumed) == 114 and not consumed & {r["source_member"] for r in rows}, "exclude every original previously used by coverage")
    speaker_roles = {r["source"]["speaker"]: r["role"] for r in background}
    speakers = set(speaker_roles)
    old_order = sorted(speakers, key=lambda s: (hashlib.sha256(("en63-bg48-speaker-v1/" + s).encode()).hexdigest(), s))
    old_train = set(old_order[:32])
    need(len(speakers) == 40 and all(r["speaker_id"] in old_train and speaker_roles[r["speaker_id"]] == "train" for r in rows), "both train speaker sets")
    selected = load(files["evidence/SELECTED_ORIGINALS.json"])
    need(selected["status"] == "EXACT_SELECTED_FLAC_AND_TRANSCRIPTS" and selected["corpus"] == CORPUS and
         selected["archive_members_inspected"] == 2943 and selected["archive_paths_extracted"] == 0 and
         selected["decodes_performed"] == selected["model_calls"] == 0, "completed source-only corpus selection")
    need(set(selected["selected_members"]) == {r["source_member"] for r in rows} and
         set(selected["selected_transcripts"]) == {r["recording_id"] for r in rows}, "exact source receipt identities")
    expected_paths, per_source = set(), []
    for index, row in enumerate(rows):
        rid, member = row["recording_id"], PurePosixPath(row["source_member"])
        original = original_by_id[row["source_member"]]
        need(row["index"] == index and row["source_role"] == row["old_bg48_role"] == row["current_coverage_role"] == "train" and
             row["current_coverage_original_source_consumed"] is False and row["label_status"] == "UNKNOWN_REVIEW_REQUIRED" and
             row["expected_events"] is None and all(row[k] is False for k in FALSE_FLAGS), "unmodified UNKNOWN train source")
        need(all(row[k] == original[k] for k in ("flac_sha256", "pcm_sha256", "transcript_sha256", "frames")) and
             row["speaker_id"] == original["speaker"] and original["trusted_negative"] is False, "expected hashes bound to original historical source")
        fp, tp = "selectedinputs/flac/" + rid + ".flac", "selectedinputs/transcripts/" + rid + ".txt"
        cp = "selectedinputs/transcript-sources/" + member.parts[-3] + "-" + member.parts[-2] + ".trans.txt"
        expected_paths |= {fp, tp, cp}
        flac, transcript, chapter = files[fp], files[tp], files[cp]
        need(ident(flac)["sha256"] == row["flac_sha256"] and len(flac) >= 42 and flac[:4] == b"fLaC" and
             flac[4] & 127 == 0 and int.from_bytes(flac[5:8], "big") == 34, "original FLAC hash/STREAMINFO")
        value = int.from_bytes(flac[18:26], "big")
        info = {"sample_rate": value >> 44, "channels": ((value >> 41) & 7) + 1,
                "bits_per_sample": ((value >> 36) & 31) + 1, "frames": value & ((1 << 36) - 1)}
        need(info == {"sample_rate": 16000, "channels": 1, "bits_per_sample": 16, "frames": row["frames"]}, "original sample geometry")
        need(selected["selected_members"][str(member)] == {"flac": ident(flac), "streaminfo": info}, "source receipt FLAC identity")
        need(ident(transcript)["sha256"] == row["transcript_sha256"] and selected["selected_transcripts"][rid] == ident(transcript), "original transcript hash/receipt")
        pairs = [line.strip().split(maxsplit=1) for line in chapter.decode("utf-8").splitlines()]
        need(all(len(x) == 2 for x in pairs) and len({x[0] for x in pairs}) == len(pairs) and
             dict(pairs)[member.stem].encode("utf-8") == transcript, "selected transcript exact original chapter utterance")
        per_source.append({"recording_id": rid, "source_member": str(member), "speaker_id": row["speaker_id"],
                           "flac": ident(flac), "streaminfo": info, "transcript": ident(transcript),
                           "chapter_transcript": {"path": cp, **ident(chapter)}, "expected_pcm_sha256": row["pcm_sha256"],
                           "pcm_digest_status": "FROZEN_HISTORICAL_METADATA_ONLY_NO_NEW_PCM_DECODE", "source_role": "train",
                           "label_status": "UNKNOWN_REVIEW_REQUIRED", "current_coverage_original_source_consumed": False})
    need({n for n in files if n.startswith("selectedinputs/")} == expected_paths and
         len([n for n in expected_paths if n.startswith("selectedinputs/transcript-sources/")]) == 27, "32 FLAC + 32 text + 27 chapter exact inventory")
    corpus_receipt = load(files["evidence/CORPUS_DOWNLOAD.json"])
    need(corpus_receipt == {"actual": {k: CORPUS[k] for k in ("bytes", "sha256")}, "source": CORPUS, "retry_count": 0}, "complete corpus download pinned receipt")
    corpus_exit = load(files["evidence/corpus-download.exit.json"])
    need(corpus_exit["returncode"] == 0 and corpus_exit["wall_seconds"] == 517.931449793, "successful measured corpus acquisition")
    return {"schema": "independent-natural32-retained-source-failure-audit-v1", "audit_completed": True,
            "status": "VERIFIED_RETAINED_ORIGINALS_WITH_ZERO_AGGREGATE_COMPUTE", "artifact": ARTIFACT,
            "api_verification": "Root separately verified the second API identity; this auditor performs no network call.",
            "manifest": {"identity": ident(files["MANIFEST.json"]), "zip_files_including_manifest": 292,
                         "declared_and_verified_payload_members": 291, "closed_set": True, "missing": [], "extra": [],
                         "hidden_workflows_retained": sorted(n for n in files if "/.github/" in n)},
            "second_execution": zero, "first_execution": first_zero,
            "aggregate_compute": {"build_attempts": 0, "decode_attempts": 0, "model_attempts": 0,
                                  "completed_validated_traces": 0, "observed_embedding_rows": 0, "observed_ready_rows": 0,
                                  "model_budget_remaining": 32},
            "nested_first_failure": {"original_zip": ident(first_raw), "retained_files_byte_identical": 85,
                                     "original_closed_manifest": False, "sole_original_omission": {OLD_WORKFLOW: OLD_WORKFLOW_PIN},
                                     "separate_supplement_retained": True, "identical_proof_copies": True,
                                     "saved_API_run_and_artifact_fields_verified": True},
            "retained_sources": {"roster_sha256": ROSTER_SHA, "originals": 32, "speakers": 24, "sample_frames": 3270080,
                                 "source_seconds": 204.38, "chapter_transcripts": 27, "flac_hashes_verified": 32,
                                 "selected_transcript_hashes_verified": 32, "streaminfo_geometry_verified": 32,
                                 "historical_expected_PCM_hash_bindings_verified": 32, "actual_PCM_hashes_recomputed": 0,
                                 "originals_excluded_as_previously_consumed": 114, "overlap_with_excluded_originals": 0,
                                 "source_receipt": ident(files["evidence/SELECTED_ORIGINALS.json"]), "per_source": per_source},
            "acquisition": {"corpus_download_succeeded": True, "corpus_bytes_and_sha256_receipt": corpus_receipt,
                            "corpus_command_seconds": 517.931449793, "full_corpus_tar_retained": False,
                            "dependency_probe_launch_failed": "ffmpeg executable absent", "dependency_probe_returncode": None},
            "quality_boundary": {"scan_completed": False, "candidates_produced": False, "quality_improvement_claim": False,
                                 "verified_negative_seconds": 0, "independent_far": "NOT_MEASURED", "independent_frr": "NOT_MEASURED",
                                 **dict.fromkeys(FALSE_FLAGS, False)},
            "audit_execution": {"models_executed": 0, "audio_decodes_executed": 0, "compiler_calls": 0,
                                "network_calls": 0, "subprocess_calls": 0, "artifact_modified": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists(), "create-only audit output")
    report = audit(args.artifact)
    report["auditor"] = ident(Path(__file__).read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as out:
        out.write((json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode())
    print(json.dumps({"output": str(args.output), "closed_manifest": True, "verified_payload_members": 291,
                      "verified_original_FLAC": 32, "aggregate_model_attempts": 0, "actual_PCM_decodes": 0}, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
