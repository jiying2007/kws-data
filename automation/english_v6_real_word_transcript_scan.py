"""Look for existing whole-word English wake-phrase leads in LibriSpeech TRAIN transcripts."""
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
SPELLINGS = ("MERO", "MEERO", "MIRO", "MERROW", "MARROW")
PAIR = re.compile(r"\bHEY (" + "|".join(SPELLINGS) + r")\b")
ALONE = re.compile(r"\bMERO\b")
DIGEST = re.compile(r"[0-9a-f]{64}")


def normalized(text):
    raw = unicodedata.normalize("NFKD", text).upper()
    return " ".join(re.findall("[A-Z]+", raw))


def inspect(entry):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=False)
    key = f"datasets/{REPO}@{REV}/{entry.path}"
    with fs.open(key, "rb", block_size=65536, cache_type="none") as stream:
        pf = pq.ParquetFile(stream, pre_buffer=False)
        seen, digest, leads, mero_only = 0, sha256(), [], 0
        for rg in range(pf.num_row_groups):
            texts = pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
            matches = []
            for i, raw in enumerate(texts):
                if not isinstance(raw, str):
                    raise ValueError("missing source transcript")
                utf = raw.encode()
                digest.update(len(utf).to_bytes(4, "little"))
                digest.update(utf)
                words = normalized(raw)
                mero_only += bool(ALONE.search(words))
                variants = sorted(set(PAIR.findall(words)))
                if variants:
                    matches.append((i, raw, variants))
            if matches:
                ids = pf.read_row_group(rg, columns=["id", "speaker_id"]).to_pylist()
                for i, raw, variants in matches:
                    leads.append(dict(id=ids[i]["id"], speaker_id=str(ids[i]["speaker_id"]),
                                      text=raw, spellings=variants, row_group=rg, row=i))
            seen += len(texts)
        if seen != pf.metadata.num_rows or seen == 0:
            raise ValueError("incomplete metadata scan")
    return dict(shard=entry.path, file_bytes=entry.size, upstream_lfs_sha256=entry.lfs.sha256,
                rows=seen, text_stream_sha256=digest.hexdigest(),
                contains_standalone_mero=mero_only, transcript_leads=leads)


def retry(entry, attempts=4, inspect_fn=None, sleep_fn=time.sleep):
    """Retry only transient transport/I/O failures, never data or policy errors."""
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


def self_test():
    import requests

    calls, delays = [], []

    def flaky(_entry):
        calls.append(1)
        if len(calls) < 3:
            raise requests.exceptions.ReadTimeout("synthetic transient timeout")
        return {"ok": True}

    result = retry(object(), attempts=3, inspect_fn=flaky, sleep_fn=delays.append)
    if result != {"ok": True} or len(calls) != 3 or delays != [2, 4]:
        raise AssertionError("transient retry contract failed")

    calls.clear()

    def invalid(_entry):
        calls.append(1)
        raise ValueError("synthetic non-transient data error")

    try:
        retry(object(), attempts=4, inspect_fn=invalid, sleep_fn=delays.append)
    except ValueError:
        pass
    else:
        raise AssertionError("non-transient error was swallowed")
    if len(calls) != 1:
        raise AssertionError("non-transient error was retried")
    if int(os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]) < 60:
        raise AssertionError("download timeout is below the reviewed bound")
    print("PASS: bounded transient retry, fail-fast data errors, 60 s HF range timeout")


def scan(output):
    from huggingface_hub import HfApi

    api = HfApi(token=False)
    shards = []
    for partition in PARTITIONS:
        items = sorted(
            (entry for entry in api.list_repo_tree(repo_id=REPO, path_in_repo=partition,
                                                   revision=REV, repo_type="dataset")
             if re.fullmatch(re.escape(partition) + r"/\d+\.parquet", entry.path)),
            key=lambda entry: entry.path)
        if not items or any(not entry.lfs or not DIGEST.fullmatch(entry.lfs.sha256) for entry in items):
            raise ValueError("missing or unversioned shards: " + partition)
        shards.extend(items)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(retry, shards))

    leads = [dict(shard=result["shard"], **item)
             for result in results for item in result["transcript_leads"]]
    if len({row["id"] for row in leads}) != len(leads):
        raise ValueError("duplicate official utterance id")

    report = dict(
        schema="english-v6-librispeech-train-word-transcript-search-v1",
        upstream_repo=REPO,
        upstream_revision=REV,
        normalization="NFKD, uppercase, whole consecutive ASCII word tokens within one official utterance",
        searches=["HEY " + spelling for spelling in SPELLINGS],
        role="upstream TRAIN only; leads require provenance and direct audio review",
        totals={partition: sum(result["rows"] for result in results
                               if result["shard"].startswith(partition + "/"))
                for partition in PARTITIONS},
        total_utterances=sum(result["rows"] for result in results),
        matched_utterances_by_spelling={
            spelling: sum(spelling in item["spellings"] for item in leads)
            for spelling in SPELLINGS
        },
        mero_standalone_utterances=sum(result["contains_standalone_mero"] for result in results),
        shard_count=len(results),
        shards=results,
        transcript_leads=leads,
        decoded_audio=0,
        model_calls=0,
        admitted_human_positives=0,
    )
    output.mkdir()
    (output / "RESULT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(report["totals"], report["matched_utterances_by_spelling"])


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        self_test()
    elif len(sys.argv) == 2:
        scan(Path(sys.argv[1]))
    else:
        raise SystemExit("usage: english_v6_real_word_transcript_scan.py OUTPUT_DIR | --self-test")
