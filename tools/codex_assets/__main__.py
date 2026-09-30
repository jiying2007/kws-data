import argparse
import hashlib
import json
import pathlib
import shutil
import wave

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPLITS = {"Vivian": "train", "Uncle_Fu": "train", "Dylan": "train",
          "Serena": "development_a", "Eric": "development_b"}
SCRIPT_NAMES = ["kws_qwen3_probe_20260928.py", "kws_qwen3_cohort_20260928.py",
                "kws_qwen3_stage2_20260928.py",
                "kws_qwen3_software_closure_batch_20260928.py",
                "kws_qwen3_software_holdout_20260928.py"]
ASR_IDENTITY = {
    "asr_binary_sha256": "3cca8d3f4a7edc19717fe49aecceab632727ed6a11f230fb64037cdb284c184a",
    "asr_model_sha256": "e291b9c468b651e2697caa09bc684326c3addc6a019e78eb537cfd1a8248ca07",
    "asr_tokens_sha256": "6fed8c6c248516f38e7faa19404b57413e8ce259f1cbc1fa4aebc86eac32fdfd",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def silence_ms(pcm):
    import array
    import math
    import sys
    samples = array.array("h", pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    levels = [math.sqrt(sum(float(x * x) for x in samples[i:i + 160]) / 160)
              for i in range(0, len(samples) - 159, 160)]
    if not levels or max(levels) < 100:
        return len(levels) * 10
    active = [i for i, value in enumerate(levels) if value > max(levels) * .01]
    longest = run = 0
    for i in range(active[0], active[-1] + 1):
        run = run + 1 if levels[i] <= max(levels) * .01 else 0
        longest = max(longest, run)
    return longest * 10


def inspect_audio(path, row):
    if sha(path) != row["file_sha256"]:
        raise ValueError("file SHA mismatch: " + str(path))
    with wave.open(str(path), "rb") as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(),
                stream.getcomptype()) != (16000, 1, 2, "NONE"):
            raise ValueError("expected 16 kHz mono PCM16: " + str(path))
        frames = stream.getnframes()
        pcm = stream.readframes(frames)
    if frames <= 0 or len(pcm) != frames * 2:
        raise ValueError("empty or truncated WAV: " + str(path))
    if hashlib.sha256(pcm).hexdigest() != row["pcm_sha256"]:
        raise ValueError("PCM SHA mismatch: " + str(path))
    if "frames" in row and frames != row["frames"]:
        raise ValueError("frame count mismatch: " + str(path))
    return frames, silence_ms(pcm)


def import_existing(source, scripts, dry_run):
    reviewed = source / "build/dataset-archive-candidates/qwen3-reviewed-v1"
    original = json.loads((reviewed / "manifest.json").read_text())
    receipt = reviewed / "audio-review.jsonl"
    if sha(receipt) != original["review"]["receipt_sha256"]:
        raise ValueError("human receipt SHA mismatch")
    datasets = []
    destination = ROOT / "datasets/qwen3-reviewed-v1"
    for row in original["recordings"]:
        inspect_audio(reviewed / row["path"], row)
    datasets.append((destination, original, read_rows(receipt), reviewed, "human"))
    for name, count, selection in [("qwen3-train", 18, "accepted-train.jsonl"),
                                   ("qwen3-holdout", 4, None)]:
        folder = source / "build/software-closure-20260928" / name
        all_rows = read_rows(folder / "manifest.jsonl")
        reviews = read_rows(folder / "asr-review.jsonl")
        accepted = {r["source_id"]: r for r in reviews if r["verdict"] == "accepted"}
        if selection:
            allowed = {r["source_id"] for r in read_rows(folder / selection)}
        else:
            allowed = set(accepted)
        selected = []
        for row in all_rows:
            if row["source_id"] not in allowed:
                continue
            review = accepted[row["source_id"]]
            if (review["file_sha256"], review["intended_text"], review["kind"], review["keyword_id"]) != (
                    row["file_sha256"], row["text"], row["kind"], row["keyword_id"]):
                raise ValueError("ASR receipt label mismatch")
            audio = folder / pathlib.Path(row["audio_path"]).name
            frames, pause = inspect_audio(audio, row)
            if row["kind"] == "positive" and pause >= 120:
                raise ValueError("discontinuous accepted positive: " + row["recording"])
            selected.append({k: row[k] for k in ["recording", "source_id", "file_sha256",
                                                 "pcm_sha256", "speaker_id", "seed", "kind", "keyword_id"]})
            selected[-1].update(path="audio/" + audio.name, frames=frames,
                                sample_rate_hz=16000, intended_text=row["text"],
                                review_verdict="accepted", internal_silence_ms=pause)
        if len(selected) != count:
            raise ValueError("unexpected curated count: " + name)
        manifest = {"schema_version": 1, "dataset_id": name + "-asr-development-20260928",
                    "evidence_class": "asr-selected-synthetic-kws-dataset-v1",
                    "source_model": original["source_model"], "raw_human_audio": False,
                    "qualification_allowed": False, "scope": "research-development-only",
                    "source_manifest_sha256": sha(folder / "manifest.jsonl"),
                    "source_asr_review_sha256": sha(folder / "asr-review.jsonl"),
                    "screening": {"generated": len(all_rows), "published": len(selected),
                                  "excluded": len(all_rows) - len(selected)}, "recordings": selected}
        datasets.append((ROOT / "datasets" / (name + "-asr-v1"), manifest,
                         [accepted[r["source_id"]] for r in selected], folder, "asr"))
    source_scripts = [(scripts / name) for name in SCRIPT_NAMES]
    script_hashes = {p.name: sha(p) for p in source_scripts}
    required = original["generation"]
    for name, field in zip(SCRIPT_NAMES[:3], ["first_probe_script_sha256",
                                            "first_cohort_script_sha256", "stage2_script_sha256"]):
        if script_hashes[name] != required[field]:
            raise ValueError("historical generation script SHA mismatch: " + name)
    report = {"datasets": len(datasets), "recordings": sum(len(m["recordings"]) for _, m, _, _, _ in datasets),
              "dry_run": dry_run}
    if dry_run:
        return report
    if (ROOT / "catalog.json").exists():
        raise ValueError("catalog already exists; immutable import cannot overwrite")
    catalog = {"schema_version": 1, "publication_date": "2026-09-30",
               "pipeline_commit": "c4025ee2e686c0c6cfd1ba078e6d1dabc3f70ee8", "datasets": []}
    for dest, manifest, reviews, folder, evidence in datasets:
        dest.mkdir(parents=True, exist_ok=False)
        for row in manifest["recordings"]:
            target = dest / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            src = folder / row["path"] if evidence == "human" else folder / pathlib.Path(row["path"]).name
            shutil.copyfile(src, target)
        review_name = "audio-review.jsonl" if evidence == "human" else "asr-review.jsonl"
        if evidence == "human":
            shutil.copyfile(receipt, dest / review_name)
        else:
            (dest / review_name).write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                                                   for r in reviews), encoding="utf-8")
        write_json(dest / "manifest.json", manifest)
        split_rows = [{"recording": r["recording"], "source_id": r["source_id"],
                       "split": SPLITS[r["speaker_id"]], "observed_development": True}
                      for r in manifest["recordings"]]
        write_json(dest / "splits.json", {"policy": "historical-development-speaker-split-v1",
                                          "formal_qualification_allowed": False, "recordings": split_rows})
        catalog["datasets"].append({"dataset_id": manifest["dataset_id"],
                                   "path": dest.relative_to(ROOT).as_posix(),
                                   "review_file": review_name, "review_type": evidence,
                                   "recordings": len(manifest["recordings"]),
                                   "manifest_sha256": sha(dest / "manifest.json"),
                                   "review_sha256": sha(dest / review_name),
                                   "splits_sha256": sha(dest / "splits.json")})
    recipe = ROOT / "recipes/qwen3/historical"
    recipe.mkdir(parents=True, exist_ok=True)
    for p in source_scripts:
        shutil.copyfile(p, recipe / p.name)
    write_json(ROOT / "recipes/qwen3/historical-scripts.json", script_hashes)
    write_json(ROOT / "catalog.json", catalog)
    return report


def verify():
    catalog = json.loads((ROOT / "catalog.json").read_text())
    seen_sources, seen_pcm, speaker_splits = set(), set(), {}
    seconds = positives = negatives = 0
    for entry in catalog["datasets"]:
        folder = (ROOT / entry["path"]).resolve()
        if ROOT not in folder.parents:
            raise ValueError("unsafe dataset path")
        for name, key in [("manifest.json", "manifest_sha256"),
                          (entry["review_file"], "review_sha256"), ("splits.json", "splits_sha256")]:
            if sha(folder / name) != entry[key]:
                raise ValueError("catalog SHA mismatch: " + name)
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["qualification_allowed"] is not False or manifest["raw_human_audio"] is not False:
            raise ValueError("unexpected dataset scope")
        if len(manifest["recordings"]) != entry["recordings"]:
            raise ValueError("catalog count mismatch")
        reviews = read_rows(folder / entry["review_file"])
        lookup = {(r["source_id"], r["file_sha256"]): r for r in reviews}
        if len(lookup) != len(reviews) or len(lookup) != entry["recordings"]:
            raise ValueError("receipt coverage mismatch")
        splits = json.loads((folder / "splits.json").read_text())["recordings"]
        split_map = {r["source_id"]: r["split"] for r in splits}
        if len(split_map) != len(splits) or set(split_map) != {r["source_id"] for r in manifest["recordings"]}:
            raise ValueError("split coverage mismatch")
        for row in manifest["recordings"]:
            audio = (folder / row["path"]).resolve()
            if folder not in audio.parents:
                raise ValueError("unsafe audio path")
            frames, pause = inspect_audio(audio, row)
            seconds += frames / 16000
            review = lookup[(row["source_id"], row["file_sha256"])]
            expected = ("speech-like-audio-review-v1" if entry["review_type"] == "human"
                        else "speech-like-asr-review-v1")
            if review["evidence_class"] != expected or review["verdict"] != "accepted":
                raise ValueError("wrong review evidence")
            if (review["intended_text"], review["kind"], review["keyword_id"]) != (
                    row["intended_text"], row["kind"], row["keyword_id"]):
                raise ValueError("review label mismatch")
            if entry["review_type"] == "asr":
                if any(review.get(key) != digest for key, digest in ASR_IDENTITY.items()):
                    raise ValueError("ASR asset identity mismatch")
                import unicodedata
                normalize = lambda s: "".join(c for c in unicodedata.normalize("NFKC", s)
                                               if not c.isspace() and not unicodedata.category(c).startswith("P"))
                if review["asr_standard"] != "exact-normalized-text-v1" or normalize(review["asr_text"]) != normalize(row["intended_text"]):
                    raise ValueError("ASR accepted text mismatch")
                if row["kind"] == "positive" and pause >= 120:
                    raise ValueError("ASR positive continuity failed")
            if row["source_id"] in seen_sources or row["pcm_sha256"] in seen_pcm:
                raise ValueError("duplicate source or PCM")
            seen_sources.add(row["source_id"])
            seen_pcm.add(row["pcm_sha256"])
            split = split_map[row["source_id"]]
            if split != SPLITS[row["speaker_id"]]:
                raise ValueError("speaker split mismatch")
            previous = speaker_splits.setdefault(row["speaker_id"], split)
            if previous != split:
                raise ValueError("speaker leakage")
            positives += row["kind"] == "positive"
            negatives += row["kind"] != "positive"
    for name, digest in json.loads((ROOT / "recipes/qwen3/historical-scripts.json").read_text()).items():
        if sha(ROOT / "recipes/qwen3/historical" / name) != digest:
            raise ValueError("generation script identity changed")
    return {"verified": True, "datasets": len(catalog["datasets"]),
            "recordings": len(seen_sources), "positives": positives, "negatives": negatives,
            "seconds": round(seconds, 3), "speaker_splits": speaker_splits}


def main():
    parser = argparse.ArgumentParser(description="导入和核验固定的合成开发数据")
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import-existing")
    imp.add_argument("--source-root", required=True, type=pathlib.Path)
    imp.add_argument("--scripts-root", required=True, type=pathlib.Path)
    imp.add_argument("--dry-run", action="store_true")
    sub.add_parser("verify")
    args = parser.parse_args()
    result = verify() if args.command == "verify" else import_existing(
        args.source_root.resolve(), args.scripts_root.resolve(), args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
