#!/usr/bin/env python3
"""Validate saved public evidence; optionally verify its exact source projection."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path)
    parser.add_argument("--labels-source", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = load(root / "MANIFEST.json")
    assert {p.name for p in root.iterdir() if p.is_file()} == {
        "MANIFEST.json", *manifest["files"]
    }, "Unexpected or missing public file"
    for name, pin in manifest["files"].items():
        data = (root / name).read_bytes()
        data.decode("utf-8")
        assert len(data) == pin["bytes"] and digest(data) == pin["sha256"], name
    labels = load(root / "ACTUAL-LABELS.json")
    results = load(root / "RESULTS.json")
    aliases = {f"Z{i}" for i in range(1, 7)}
    label_rows = {r["recording"]: r for r in labels["rows"]}
    rows = {r["recording"]: r for r in results["all6"]}
    assert set(label_rows) == set(rows) == aliases
    assert labels["labels_before_predictions"] is True
    assert labels["final"] is True
    assert {k: v["actual_text"] for k, v in label_rows.items()} == {
        "Z1": "你好", "Z2": "小屋小屋", "Z3": "小挖",
        "Z4": "小窝小窝", "Z5": "你好小窝", "Z6": "你好小屋",
    }
    for alias, label in label_rows.items():
        assert label["uncertainty"] is None
        assert rows[alias]["actual_text"] == label["actual_text"]
        assert rows[alias]["actual_keyword_presence"] == label["keyword_presence"]
    assert results["qualification"] is False
    assert results["population_rate_estimates"] is None
    for arm, pin in manifest["raw_projections"].items():
        data = (root / pin["public_file"]).read_bytes()
        raw = [json.loads(line) for line in data.splitlines()]
        assert len(raw) == pin["public_records"] == 100
        assert Counter(r["kind"] for r in raw) == pin["public_kind_counts"]
        assert Counter(r["recording"] for r in raw) == pin["public_alias_counts"]
        assert all(r["recording"] in aliases for r in raw)
        assert all(set(r) == set(manifest["raw_fields_by_kind"][r["kind"]]) for r in raw)
        starts = [r for r in raw if r["kind"] == "clip_start"]
        ends = [r for r in raw if r["kind"] == "clip_end"]
        callbacks = [r for r in raw if r["kind"] == "callback"]
        assert [r["recording"] for r in starts] == results["execution"]["recording_order"]
        assert [r["recording"] for r in ends] == [r["recording"] for r in starts]
        assert sum(r["frames"] for r in starts) == 184320
        assert len(callbacks) == 41
        assert sum(len(r["logits"]) for r in callbacks) == 378
        assert sum(len(v) for r in callbacks for v in r["logits"]) == 2268
        assert sum(r["fbank_rows"] for r in ends) == 1140
        assert sum(r["decoder_rows_decoded"] for r in ends) == pin["decoder_rows_decoded"]
        counts = {"correct_positive_recordings": 0, "missed_positive_recordings": 0,
                  "wrong_nonwake_recordings": 0, "repeat_events": 0}
        for start, end in zip(starts, ends):
            alias = start["recording"]
            label = label_rows[alias]
            for key in ("frames", "wav_sha256", "pcm_sha256"):
                assert start[key] == end[key] == label[key]
            assert end["complete"] is True and end["finish_calls"] == 1
            observed = [r for r in callbacks if r["recording"] == alias and
                        r["valid"] == 1 and r["state"] == 1 and r["keyword"] in (1, 2)]
            expected = rows[alias][arm]
            assert len(observed) == end["event_count"] == len(expected["events"])
            event_counts = Counter(f"K{r['keyword']}" for r in observed)
            assert expected["counts"] == {k: event_counts[k] for k in ("K1", "K2")}
            for event, saved in zip(observed, expected["events"]):
                assert all(event[k] == saved[k] for k in manifest["event_fields"])
                assert saved["available_audio_s"] == event["available_samples"] / 16000
            present = [k for k, v in label["keyword_presence"].items() if v == "present"]
            if present:
                hit = all(event_counts[k] > 0 for k in present)
                counts["correct_positive_recordings"] += int(hit)
                counts["missed_positive_recordings"] += int(not hit)
            else:
                counts["wrong_nonwake_recordings"] += int(bool(observed))
            counts["repeat_events"] += sum(max(0, n - 1) for n in event_counts.values())
        assert counts == results["descriptive_counts"][arm]
        assert rows["Z3"][arm]["actual_text_edits"] is None
        assert rows["Z3"][arm]["actual_text_exact"] is None
        if args.original_dir:
            source = (args.original_dir / pin["source_basename"]).read_bytes()
            assert len(source) == pin["source_bytes"] and digest(source) == pin["source_sha256"]
            lines = source.splitlines(keepends=True)
            selected = [line for line in lines if json.loads(line).get("recording") in aliases]
            dropped = [json.loads(line)["kind"] for line in lines
                       if json.loads(line).get("recording") not in aliases]
            assert len(lines) == pin["source_records"] == 103
            assert b"".join(selected) == data, "Clip line bytes differ from source"
            assert dropped == pin["omitted_global_kinds"]
    if args.labels_source:
        source = args.labels_source.read_bytes()
        assert digest(source) == manifest["label_projection"]["source_sha256"]
        parsed = json.loads(source)
        projected = [{k: row[k] for k in manifest["label_projection"]["row_fields"]}
                     for row in sorted(parsed["rows"], key=lambda r: r["recording"])]
        assert labels["rows"] == projected
    print("PASS: saved public hashes, clip records, labels, geometry, and event counts")
    if args.original_dir:
        print("PASS: complete source hashes and exact unchanged clip-line projection")
    if args.labels_source:
        print("PASS: frozen label source hash and explicit technical-field projection")


if __name__ == "__main__":
    main()
