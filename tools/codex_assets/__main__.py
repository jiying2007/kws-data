"""Single validation/export authority for the public synthetic asset catalog."""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import unicodedata
import wave

ROOT = pathlib.Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_int(value):
    return type(value) is int and value > 0


def hash_value(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def safe_path(base, name):
    require(isinstance(name, str) and name, "invalid asset path")
    path = pathlib.Path(name)
    require(not path.is_absolute() and ".." not in path.parts, "unsafe asset path")
    candidate = base / path
    require(base.resolve() in candidate.resolve().parents, "unsafe asset path")
    cursor = base
    for part in path.parts:
        cursor /= part
        require(not cursor.is_symlink(), "symlink asset path forbidden")
    return candidate


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
    require(hash_value(row.get("file_sha256")) and hash_value(row.get("pcm_sha256")),
            "invalid audio SHA")
    require(sha(path) == row["file_sha256"], "file SHA mismatch: " + str(path))
    require(positive_int(row.get("frames")), "invalid frame count")
    require(type(row.get("sample_rate_hz")) is int and row["sample_rate_hz"] == 16000,
            "invalid declared sample rate")
    with wave.open(str(path), "rb") as stream:
        require((stream.getframerate(), stream.getnchannels(), stream.getsampwidth(),
                 stream.getcomptype()) == (16000, 1, 2, "NONE"),
                "expected 16 kHz mono PCM16: " + str(path))
        frames = stream.getnframes()
        pcm = stream.readframes(frames)
    require(frames > 0 and len(pcm) == frames * 2, "empty or truncated WAV")
    require(hashlib.sha256(pcm).hexdigest() == row["pcm_sha256"], "PCM SHA mismatch")
    require(frames == row["frames"], "frame count mismatch")
    return frames, silence_ms(pcm)


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text)
                   if not c.isspace() and not unicodedata.category(c).startswith("P"))


def catalog_specs(catalog):
    require(type(catalog.get("schema_version")) is int and catalog["schema_version"] == 2,
            "expected catalog schema 2; use historical Git for historical catalogs")
    for name in ("sources", "split_policies", "review_policies", "keyword_texts"):
        require(isinstance(catalog.get(name), dict) and catalog[name], "missing catalog " + name)
    require(isinstance(catalog.get("datasets"), list) and catalog["datasets"], "empty dataset catalog")
    for key, text in catalog["keyword_texts"].items():
        require(isinstance(key, str) and key.isdigit() and int(key) > 0 and
                isinstance(text, str) and text, "invalid keyword definition")
    for source in catalog["sources"].values():
        require(source.get("source_kind") == "synthetic-preset-voice", "unsupported source kind")
        require(isinstance(source.get("generator_family"), str) and source["generator_family"],
                "missing generator family")
        model = source.get("source_model", {})
        require(isinstance(model, dict) and model.get("repository") and model.get("revision") and
                hash_value(model.get("model_sha256")) and hash_value(model.get("speech_tokenizer_sha256")),
                "invalid source model identity")
        rights = source.get("rights", {})
        require(rights.get("model_license") and rights.get("model_license_url") and
                rights.get("publication_status") == "maintainer-authorized-fixed-batches" and
                rights.get("commercial_output_license") == "not-established", "unsupported rights scope")
        require(safe_path(ROOT, rights.get("publication_evidence")).is_file(), "missing rights evidence")
    for policy in catalog["split_policies"].values():
        require(policy.get("observed_development") is True and
                policy.get("formal_qualification_allowed") is False, "unsupported split scope")
        roles = policy.get("roles")
        require(isinstance(roles, list) and roles and len(set(roles)) == len(roles) and
                all(isinstance(x, str) and x in {"train", "development_a", "development_b"} for x in roles),
                "invalid development roles")
        assignments = policy.get("speaker_assignments")
        require(isinstance(assignments, dict) and assignments and
                all(isinstance(k, str) and k and v in roles for k, v in assignments.items()),
                "invalid speaker assignments")
    for policy in catalog["review_policies"].values():
        method = policy.get("method")
        require(method in {"human", "asr"}, "unknown review method")
        expected = "speech-like-audio-review-v1" if method == "human" else "speech-like-asr-review-v1"
        require(policy.get("evidence_class") == expected, "invalid review evidence class")
        if method == "asr":
            assets = policy.get("assets", {})
            require(set(assets) == {"asr_binary_sha256", "asr_model_sha256", "asr_tokens_sha256"} and
                    all(hash_value(v) for v in assets.values()), "invalid ASR identity")
            require(policy.get("normalization") == "exact-normalized-text-v1", "unsupported ASR normalization")
            require(positive_int(policy.get("max_positive_internal_silence_ms")), "invalid continuity policy")


def load_verified():
    catalog_path = safe_path(ROOT, "catalog.json")
    catalog_sha = sha(catalog_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog_specs(catalog)
    required_paths = {"catalog.json", "tools/codex_assets/__main__.py", "tools/codex_assets/__init__.py",
                      "recipes/qwen3/historical-scripts.json"}
    required_paths.update(s["rights"]["publication_evidence"] for s in catalog["sources"].values())
    seen_ids, seen_sources, seen_pcm, speaker_splits = set(), set(), set(), {}
    seconds = positives = negatives = 0
    bundles = []
    for entry in catalog["datasets"]:
        dataset_id = entry.get("dataset_id")
        require(isinstance(dataset_id, str) and dataset_id and dataset_id not in seen_ids,
                "duplicate or invalid dataset ID")
        seen_ids.add(dataset_id)
        require(positive_int(entry.get("recordings")), "invalid catalog count")
        require(entry.get("storage") == {"kind": "git", "privacy": "public-synthetic"},
                "unsupported storage; materialization adapter not implemented")
        for key, group in (("source_ref", "sources"), ("split_policy_ref", "split_policies"),
                           ("review_policy_ref", "review_policies")):
            require(isinstance(entry.get(key), str) and entry[key] in catalog[group], "unknown " + key)
        source = catalog["sources"][entry["source_ref"]]
        split_policy = catalog["split_policies"][entry["split_policy_ref"]]
        review_policy = catalog["review_policies"][entry["review_policy_ref"]]
        require(entry.get("review_type") == review_policy["method"], "review type mismatch")
        folder = safe_path(ROOT, entry["path"])
        for name, key in (("manifest.json", "manifest_sha256"),
                          (entry["review_file"], "review_sha256"), ("splits.json", "splits_sha256")):
            require(hash_value(entry.get(key)), "invalid catalog SHA")
            asset = safe_path(folder, name)
            require(sha(asset) == entry[key], "catalog SHA mismatch: " + name)
            required_paths.add(asset.relative_to(ROOT).as_posix())
        manifest = json.loads(safe_path(folder, "manifest.json").read_text(encoding="utf-8"))
        require(type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1,
                "invalid manifest schema")
        require(manifest.get("dataset_id") == dataset_id, "dataset ID mismatch")
        require(manifest.get("qualification_allowed") is False and manifest.get("raw_human_audio") is False,
                "unexpected dataset scope")
        require(manifest.get("source_model") == source["source_model"], "source model identity mismatch")
        expected_dataset_class = ("reviewed-synthetic-kws-dataset-v1" if review_policy["method"] == "human"
                                  else "asr-selected-synthetic-kws-dataset-v1")
        require(manifest.get("evidence_class") == expected_dataset_class, "manifest evidence class mismatch")
        rows = manifest["recordings"]
        require(len(rows) == entry["recordings"], "catalog count mismatch")
        reviews = read_rows(safe_path(folder, entry["review_file"]))
        require(all(isinstance(r.get("source_id"), str) and r["source_id"] and
                    hash_value(r.get("file_sha256")) for r in reviews), "invalid review source/hash")
        lookup = {(r["source_id"], r["file_sha256"]): r for r in reviews}
        require(len(lookup) == len(reviews) == entry["recordings"], "receipt coverage mismatch")
        split_doc = json.loads(safe_path(folder, "splits.json").read_text(encoding="utf-8"))
        require(split_doc.get("formal_qualification_allowed") is False, "split qualification forbidden")
        splits = split_doc["recordings"]
        split_map = {r["source_id"]: r for r in splits}
        require(len(split_map) == len(splits) and set(split_map) == {r["source_id"] for r in rows},
                "split coverage mismatch")
        exported = []
        recording_ids = set()
        for row in rows:
            require(isinstance(row.get("recording"), str) and row["recording"], "invalid recording ID")
            require(row["recording"] not in recording_ids, "duplicate recording ID")
            recording_ids.add(row["recording"])
            require(isinstance(row.get("source_id"), str) and row["source_id"], "invalid source ID")
            require(row.get("kind") in {"positive", "confusable"}, "unsupported recording kind")
            require(isinstance(row.get("intended_text"), str) and row["intended_text"], "invalid intended text")
            keyword = row.get("keyword_id")
            if row["kind"] == "positive":
                require(positive_int(keyword) and catalog["keyword_texts"].get(str(keyword)) == row["intended_text"],
                        "positive keyword/text mismatch")
            else:
                require(keyword is None and row["intended_text"] not in catalog["keyword_texts"].values(),
                        "negative keyword/text mismatch")
            require(row.get("review_verdict") == "accepted", "manifest review not accepted")
            audio = safe_path(folder, row["path"])
            required_paths.add(audio.relative_to(ROOT).as_posix())
            frames, pause = inspect_audio(audio, row)
            seconds += frames / 16000
            review = lookup[(row["source_id"], row["file_sha256"])]
            require(type(review.get("schema_version")) is int and review["schema_version"] == 1,
                    "invalid review schema")
            require("keyword_id" in review and (positive_int(review["keyword_id"]) if row["kind"] == "positive"
                    else review["keyword_id"] is None), "invalid review keyword ID")
            if review_policy["method"] == "human":
                require(isinstance(review.get("reviewer_id"), str) and review["reviewer_id"].strip(),
                        "missing human reviewer identity")
                require(hash_value(review.get("review_archive_sha256")), "invalid human review archive SHA")
                require(review["review_archive_sha256"] == manifest.get("review", {}).get("copied_review_archive_sha256"),
                        "human review archive identity mismatch")
            require(review.get("evidence_class") == review_policy["evidence_class"] and
                    review.get("verdict") == "accepted", "wrong review evidence")
            require((review["intended_text"], review["kind"], review["keyword_id"]) ==
                    (row["intended_text"], row["kind"], row["keyword_id"]), "review label mismatch")
            if review_policy["method"] == "asr":
                require(all(review.get(k) == v for k, v in review_policy["assets"].items()),
                        "ASR asset identity mismatch")
                require(review.get("asr_standard") == review_policy["normalization"] and
                        normalize(review["asr_text"]) == normalize(row["intended_text"]), "ASR accepted text mismatch")
                require(row["kind"] != "positive" or pause < review_policy["max_positive_internal_silence_ms"],
                        "ASR positive continuity failed")
            require(row["source_id"] not in seen_sources and row["pcm_sha256"] not in seen_pcm,
                    "duplicate source or PCM")
            seen_sources.add(row["source_id"])
            seen_pcm.add(row["pcm_sha256"])
            split = split_map[row["source_id"]]
            require(split.get("observed_development") is True, "observed development flag required")
            require(split.get("recording") == row["recording"], "split recording mismatch")
            role = split.get("split")
            require(role == split_policy["speaker_assignments"].get(row.get("speaker_id")) and
                    role in split_policy["roles"], "speaker split mismatch")
            speaker_key = source["generator_family"] + ":" + row["speaker_id"]
            require(speaker_splits.setdefault(speaker_key, role) == role, "speaker leakage")
            positives += row["kind"] == "positive"
            negatives += row["kind"] == "confusable"
            exported.append(dict(row, path=(pathlib.Path(entry["path"]) / row["path"]).as_posix(),
                                 split=role, observed_development=True, duration_s=frames / 16000,
                                 review_method=review_policy["method"],
                                 review_evidence_class=review_policy["evidence_class"]))
        if review_policy["method"] == "human":
            batch_review = manifest.get("review", {})
            require(batch_review.get("receipt_sha256") == entry["review_sha256"],
                    "human manifest receipt SHA mismatch")
            require(all(type(batch_review.get(k)) is int and batch_review[k] >= 0
                        for k in ("accepted", "rejected", "uncertain")) and
                    batch_review["accepted"] == len(rows) and batch_review["rejected"] == 0 and
                    batch_review["uncertain"] == 0, "human review count mismatch")
        identity = {key: entry[key] for key in ("dataset_id", "manifest_sha256", "review_sha256", "splits_sha256")}
        bundles.append({"dataset_id": dataset_id, "content_id": "sha256:" + digest(identity),
                        "identity": identity, "source_ref": entry["source_ref"],
                        "generator_family": source["generator_family"],
                        "qualification_allowed": False, "recordings_sha256": digest(exported),
                        "recordings": exported})
    scripts_path = safe_path(ROOT, "recipes/qwen3/historical-scripts.json")
    for name, value in json.loads(scripts_path.read_text(encoding="utf-8")).items():
        required_paths.add((pathlib.Path("recipes/qwen3/historical") / name).as_posix())
        require(hash_value(value) and sha(safe_path(ROOT / "recipes/qwen3/historical", name)) == value,
                "generation script identity changed")
    require(sha(catalog_path) == catalog_sha, "catalog changed during verification")
    summary = {"verified": True, "catalog_schema": 2, "catalog_sha256": catalog_sha,
               "datasets": len(bundles), "recordings": len(seen_sources),
               "positives": positives, "negatives": negatives, "seconds": round(seconds, 3),
               "speaker_splits": speaker_splits,
               "content_ids": {b["dataset_id"]: b["content_id"] for b in bundles}}
    return summary, bundles, required_paths


def verify():
    return load_verified()[0]


def git_identity():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT)] + list(args), text=True).strip()
    return git("rev-parse", "HEAD"), git("status", "--porcelain", "--untracked-files=all")


def git_tracked_paths():
    output = subprocess.check_output(["git", "-C", str(ROOT), "ls-tree", "-rz", "--name-only", "HEAD"])
    return {p.decode("utf-8") for p in output.split(b"\0") if p}


def export_receipt(dataset_ids, expected_commit, expected_catalog_sha256):
    require(isinstance(expected_commit, str) and re.fullmatch(r"[0-9a-f]{40}", expected_commit),
            "expected full pinned data commit")
    require(hash_value(expected_catalog_sha256), "expected pinned catalog SHA")
    head, dirty = git_identity()
    require(head == expected_commit, "data commit mismatch")
    require(not dirty, "consumer export requires clean data checkout")
    summary, bundles, required_paths = load_verified()
    require(required_paths <= git_tracked_paths(), "consumer asset absent from pinned Git tree")
    require(summary["catalog_sha256"] == expected_catalog_sha256, "pinned catalog SHA mismatch")
    available = {b["dataset_id"]: b for b in bundles}
    require(dataset_ids and len(set(dataset_ids)) == len(dataset_ids) and
            all(x in available for x in dataset_ids), "unknown or duplicate dataset selection")
    require(git_identity() == (head, ""), "data checkout changed during export")
    return {"schema_version": 1, "evidence_class": "kws-data-native-consumer-receipt-v1",
            "data_repository_commit": head, "catalog_sha256": summary["catalog_sha256"],
            "qualification_allowed": False, "labels": "native-text-no-token-or-event-alignment",
            "datasets": [available[x] for x in sorted(dataset_ids)]}


def main():
    parser = argparse.ArgumentParser(description="核验长期数据资产并导出固定版本原生消费收据")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    exp = sub.add_parser("export")
    exp.add_argument("--dataset", action="append", required=True)
    exp.add_argument("--expected-commit", required=True)
    exp.add_argument("--expected-catalog-sha256", required=True)
    args = parser.parse_args()
    result = verify() if args.command == "verify" else export_receipt(
        args.dataset, args.expected_commit, args.expected_catalog_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
