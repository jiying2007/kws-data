"""Mine a bounded TRAIN-only natural near-phonetic cohort from pinned LibriSpeech transcripts.

Transcript metadata only: no audio decoding, ASR/KWS, optimizer updates, or DEV access.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import time
import unicodedata

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
REPO = "openslr/librispeech_asr"
REV = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
PARTITIONS = ("clean/train.100", "clean/train.360", "other/train.500")
SALT = "heym43-b15-train-nearphonetic-20261124-v1"
PATTERNS = {
    "they_may_row": re.compile(r"\bTHEY MAY ROW\b"),
    "he_may_row": re.compile(r"\bHE MAY ROW\b"),
    "may_row": re.compile(r"\bMAY ROW\b"),
    "me_or_my": re.compile(r"\bME OR MY\b"),
    "me_or": re.compile(r"\bME OR\b"),
    "mayor": re.compile(r"\bMAYOR\b"),
    "merrily": re.compile(r"\bMERRILY\b"),
    "mere_old": re.compile(r"\bMERE OLD\b"),
    "mere": re.compile(r"\bMERE\b"),
    "hero": re.compile(r"\bHERO\b"),
    "mirror": re.compile(r"\bMIRROR\b"),
    "mary_marry": re.compile(r"\b(?:MARY|MARRY)\b"),
    "morrow": re.compile(r"\bMORROW\b"),
    "arrow": re.compile(r"\bARROW\b"),
}
PRIORITY = {
    "they_may_row": 0, "he_may_row": 1, "may_row": 2, "me_or_my": 3,
    "me_or": 4, "mayor": 5, "merrily": 6, "mere_old": 7, "mere": 8,
    "hero": 9, "mirror": 10, "mary_marry": 11, "morrow": 12, "arrow": 13,
}
DIGEST = re.compile(r"[0-9a-f]{64}")


def normalized(text):
    raw = unicodedata.normalize("NFKD", text).upper()
    return " ".join(re.findall("[A-Z]+", raw))


def families(text):
    words = normalized(text)
    return sorted((name for name, pattern in PATTERNS.items() if pattern.search(words)),
                  key=lambda name: (PRIORITY[name], name))


def inspect(entry):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=False)
    key = f"datasets/{REPO}@{REV}/{entry.path}"
    with fs.open(key, "rb", block_size=65536, cache_type="none") as stream:
        parquet = pq.ParquetFile(stream, pre_buffer=False)
        seen, digest, hits = 0, sha256(), []
        for row_group in range(parquet.num_row_groups):
            texts = parquet.read_row_group(row_group, columns=["text"]).column("text").to_pylist()
            matched = []
            for row_index, raw in enumerate(texts):
                if not isinstance(raw, str):
                    raise ValueError("missing official transcript text")
                encoded = raw.encode("utf-8")
                digest.update(len(encoded).to_bytes(4, "little"))
                digest.update(encoded)
                found = families(raw)
                if found:
                    matched.append((row_index, raw, found))
            if matched:
                identity = parquet.read_row_group(
                    row_group, columns=["id", "speaker_id", "chapter_id"]
                ).to_pylist()
                for row_index, raw, found in matched:
                    item = identity[row_index]
                    hits.append({
                        "source_id": item["id"], "speaker_id": str(item["speaker_id"]),
                        "chapter_id": str(item["chapter_id"]), "parquet_path": entry.path,
                        "row_group": row_group, "row_in_group": row_index,
                        "transcript": raw, "match_families": found,
                        "partition": next(p for p in PARTITIONS if entry.path.startswith(p + "/")),
                    })
            seen += len(texts)
        if seen != parquet.metadata.num_rows or not seen:
            raise ValueError("incomplete official transcript shard")
    return {
        "shard": entry.path, "file_bytes": entry.size,
        "upstream_lfs_sha256": entry.lfs.sha256, "rows": seen,
        "text_stream_sha256": digest.hexdigest(), "hits": hits,
    }


def retry(entry, attempts=4, inspect_fn=None, sleep_fn=time.sleep):
    import requests
    operation = inspect if inspect_fn is None else inspect_fn
    transient = (requests.exceptions.RequestException, TimeoutError, ConnectionError, OSError)
    for attempt in range(attempts):
        try:
            return operation(entry)
        except transient:
            if attempt + 1 == attempts:
                raise
            sleep_fn(min(2 ** (attempt + 1), 16))
    raise AssertionError("unreachable retry state")


def rank(row):
    priority = min(PRIORITY[name] for name in row["match_families"])
    key = f"{SALT}/{row['speaker_id']}/{row['source_id']}".encode()
    return priority, sha256(key).hexdigest(), row["source_id"]


def select(candidates, target_speakers=24, per_speaker=2, per_partition=8):
    grouped = defaultdict(list)
    for row in candidates:
        grouped[row["speaker_id"]].append(row)
    eligible = {speaker: sorted(rows, key=rank)[:per_speaker]
                for speaker, rows in grouped.items()}
    partition_speakers = defaultdict(list)
    for speaker, rows in eligible.items():
        partition_speakers[rows[0]["partition"]].append((speaker, rows))
    selected, used = [], set()
    for partition in PARTITIONS:
        ranked = sorted(
            partition_speakers[partition],
            key=lambda pair: (rank(pair[1][0]),
                              sha256(f"{SALT}/{partition}/{pair[0]}".encode()).hexdigest())
        )
        for speaker, rows in ranked[:per_partition]:
            selected.append((speaker, rows))
            used.add(speaker)
    remaining = sorted(
        ((speaker, rows) for speaker, rows in eligible.items() if speaker not in used),
        key=lambda pair: (rank(pair[1][0]),
                          sha256(f"{SALT}/fill/{pair[0]}".encode()).hexdigest())
    )
    for item in remaining:
        if len(selected) >= target_speakers:
            break
        selected.append(item)
    selected = selected[:target_speakers]
    if len(selected) < 16:
        raise ValueError(f"only {len(selected)} independent speakers matched; minimum is 16")
    result = []
    for speaker, rows in selected:
        for ordinal, row in enumerate(rows):
            result.append({
                **row, "source_group_id": "librispeech-train-" + speaker,
                "selection_rank": rank(row), "speaker_sample_ordinal": ordinal,
                "role": "train", "requested_class": 0,
                "human_review_completed": False, "proposed_training_eligibility": False,
            })
    if len({r["source_id"] for r in result}) != len(result):
        raise ValueError("duplicate official utterance identity")
    if len({r["speaker_id"] for r in result}) != len(selected):
        raise ValueError("source-speaker grouping mismatch")
    return result


def self_test():
    checks = {
        "they_may_row": "They may row the boat tomorrow.",
        "he_may_row": "He may row the boat.",
        "may_row": "We may row.",
        "me_or_my": "It was me or my brother.",
        "mayor": "The mayor arrived.",
        "merrily": "They sang merrily.",
        "mere_old": "A mere old story.",
        "hero": "He became a hero.",
        "mirror": "Look in the mirror.",
        "mary_marry": "Mary will marry him.",
        "morrow": "On the morrow.",
        "arrow": "An arrow flew.",
    }
    for expected, sentence in checks.items():
        if expected not in families(sentence):
            raise AssertionError(f"pattern failed: {expected}")
    rows = []
    for speaker, partition in (("101", PARTITIONS[0]), ("202", PARTITIONS[1]), ("303", PARTITIONS[2])):
        for i in range(3):
            rows.append({"speaker_id": speaker, "source_id": f"{speaker}-{i}",
                         "match_families": ["they_may_row"], "partition": partition})
    one = select(rows, target_speakers=3, per_speaker=2, per_partition=1)
    two = select(rows, target_speakers=3, per_speaker=2, per_partition=1)
    if one != two or len(one) != 6:
        raise AssertionError("selection is not deterministic or bounded")
    import requests
    calls, delays = [], []
    def flaky(_entry):
        calls.append(1)
        if len(calls) < 3:
            raise requests.exceptions.ReadTimeout("synthetic transient")
        return {"ok": True}
    if retry(object(), attempts=3, inspect_fn=flaky, sleep_fn=delays.append) != {"ok": True}:
        raise AssertionError("transient retry did not recover")
    if delays != [2, 4]:
        raise AssertionError("retry backoff is not bounded and deterministic")
    print("PASS: near-phonetic taxonomy, deterministic speaker selection, bounded retry")


def scan(output):
    from huggingface_hub import HfApi
    api = HfApi(token=False)
    shards = []
    for partition in PARTITIONS:
        entries = sorted(
            (e for e in api.list_repo_tree(
                repo_id=REPO, path_in_repo=partition, revision=REV, repo_type="dataset"
            ) if re.fullmatch(re.escape(partition) + r"/\d+\.parquet", e.path)),
            key=lambda e: e.path,
        )
        if not entries or any(not e.lfs or not DIGEST.fullmatch(e.lfs.sha256) for e in entries):
            raise ValueError("missing or unversioned official shard: " + partition)
        shards.extend(entries)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(retry, shards))
    candidates = [row for result in results for row in result["hits"]]
    if len({row["source_id"] for row in candidates}) != len(candidates):
        raise ValueError("duplicate source ID across immutable shards")
    selected = select(candidates)
    counts, selected_counts = defaultdict(int), defaultdict(int)
    for row in candidates:
        for family in row["match_families"]:
            counts[family] += 1
    for row in selected:
        for family in row["match_families"]:
            selected_counts[family] += 1
    report = {
        "schema": "english-v6-b15-nearphonetic-train-transcript-roster-v1",
        "upstream_repo": REPO, "upstream_revision": REV,
        "partition_scope": list(PARTITIONS),
        "role": "TRAIN only; transcript-derived candidates; no audio or model evidence yet",
        "selection": {
            "salt": SALT, "target_speakers": 24, "minimum_speakers": 16,
            "maximum_utterances_per_speaker": 2, "maximum_initial_speakers_per_partition": 8,
            "patterns_priority": PRIORITY,
            "pattern_counts_all_candidates": dict(sorted(counts.items())),
            "pattern_counts_selected": dict(sorted(selected_counts.items())),
        },
        "total_official_utterances_scanned": sum(r["rows"] for r in results),
        "shard_count": len(results), "candidate_utterances": len(candidates),
        "candidate_speakers": len({r["speaker_id"] for r in candidates}),
        "selected_utterances": len(selected),
        "selected_speakers": len({r["speaker_id"] for r in selected}),
        "selected_by_partition": {p: sum(r["partition"] == p for r in selected) for p in PARTITIONS},
        "decoded_audio": 0, "asr_calls": 0, "kws_forwards": 0,
        "optimizer_updates": 0, "training_admitted_rows": 0,
        "candidates": selected,
        "shards": [{k: r[k] for k in (
            "shard", "file_bytes", "upstream_lfs_sha256", "rows", "text_stream_sha256"
        )} for r in results],
        "attribution": "LibriSpeech ASR corpus by V. Panayotov, G. Chen, D. Povey and S. Khudanpur, OpenSLR SLR12.",
        "license": "CC BY 4.0",
    }
    output.mkdir()
    (output / "ROSTER.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (output / "SOURCE_CREDITS.txt").write_text(
        "LibriSpeech ASR corpus by V. Panayotov, G. Chen, D. Povey and S. Khudanpur, OpenSLR SLR12.\n"
        "License: CC BY 4.0. Upstream revision: " + REV + "\n"
        "This artifact contains transcript metadata only; it is not admitted training audio.\n"
    )
    print(json.dumps({
        "scanned": report["total_official_utterances_scanned"],
        "candidates": report["candidate_utterances"],
        "candidate_speakers": report["candidate_speakers"],
        "selected": report["selected_utterances"],
        "selected_speakers": report["selected_speakers"],
        "selected_by_partition": report["selected_by_partition"],
        "patterns": report["selection"]["pattern_counts_selected"],
    }, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        self_test()
    elif len(sys.argv) == 2:
        scan(Path(sys.argv[1]))
    else:
        raise SystemExit("usage: english_v6_b15_nearphonetic_train_scan.py OUTPUT_DIR | --self-test")
