#!/usr/bin/env python3
"""Offline, full-dictionary, reversible export. Never imports pypinyin.

The source distribution is read only as an archive. No setup.py, package,
installer, model code, or generator from that archive is executed.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import unicodedata

sys.dont_write_bytecode = True
VERSION = "0.55.0"
ARCHIVE_SHA256 = "b5711b3a0c6f76e67408ec6b2e3c4987a3a806b7c528076e7c7b86fcf0eaa66b"
ARCHIVE_BYTES = 839836
ARCHIVE_URL = "https://files.pythonhosted.org/packages/b4/a4/784cf98c09e0dc22776b0d7d8a4a5b761218bcae4608c2416ce1e167c8af/pypinyin-0.55.0.tar.gz"
UPSTREAM_COMMIT = "df101577145af2eb1abe5656e592e34e3bb56d23"
SUBMODULES = {
    "pinyin-data": "fa9761fff402f8560196b1ba085c437c52b56d7c",
    "phrase-pinyin-data": "cee0ed6e6e4898580cafd2bd5e3723e20b214aa0",
}
TARGETS = ["你好小窝", "小窝小窝"]
SCORER_SHA256 = "2bd566983f9846e19c3ac24d59cffe155cc4b45c39c1d64a9f1fae7772af104a"
TONE_MARKS = {"\u0304": "1", "\u0301": "2", "\u030c": "3", "\u0300": "4"}
SOURCE_HASHES = {
    "pypinyin/pinyin_dict.json": "5f294c01e6c6c0a1c8e329c79335a3f8e0b27d06bf1de7a99244b765892d1e5b",
    "pypinyin/phrases_dict.json": "a45ff140a6b631ca9c82127b280a2f414e0aba6bb2824a0e9d1e77fff359c665",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def tone_digits(reading):
    """Encode the single lexical tone at suffix; preserve ü, ê and no-tone.

    Only the four tone diacritics move. No neutral tone is invented. Conversion
    must be injective over the complete source vocabulary or the freeze fails.
    """
    decomposed = unicodedata.normalize("NFD", reading)
    tones = [TONE_MARKS[c] for c in decomposed if c in TONE_MARKS]
    if len(tones) > 1 or any(c.isdigit() for c in reading):
        raise ValueError("Unexpected source reading tone encoding")
    return unicodedata.normalize("NFC", "".join(c for c in decomposed if c not in TONE_MARKS)) + "".join(tones)


def load_scorer(path):
    if sha256(path.read_bytes()) != SCORER_SHA256:
        raise ValueError("Pinned scorer SHA mismatch; review and explicitly refreeze")
    spec = importlib.util.spec_from_file_location("frozen_official_lexicon_scorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_archive(path):
    raw = path.read_bytes()
    if len(raw) != ARCHIVE_BYTES or sha256(raw) != ARCHIVE_SHA256:
        raise ValueError("Pinned official source archive bytes/SHA mismatch")
    wanted = [*SOURCE_HASHES, "LICENSE.txt", "README.rst", "PKG-INFO", "pypinyin/phonetic_symbol.py"]
    result = {}
    with tarfile.open(path, "r:gz") as archive:
        names = [m.name for m in archive.getmembers()]
        for name in wanted:
            full = f"pypinyin-{VERSION}/{name}"
            if names.count(full) != 1:
                raise ValueError("Missing/duplicate archive member")
            member = archive.getmember(full)
            if not member.isfile() or not 0 < member.size <= 8 * 1024 * 1024:
                raise ValueError("Invalid archive member")
            result[name] = archive.extractfile(member).read()
            if name in SOURCE_HASHES and sha256(result[name]) != SOURCE_HASHES[name]:
                raise ValueError("Pinned source dictionary mismatch")
    return result


def export(source, scorer):
    characters = scorer.decode_json(source["pypinyin/pinyin_dict.json"], "characters")
    phrases = scorer.decode_json(source["pypinyin/phrases_dict.json"], "phrases")
    readings = {r for variants in characters.values() for r in variants.split(",")}
    readings.update(r for rows in phrases.values() for variants in rows for r in variants)
    reverse = {}
    for reading in sorted(readings):
        encoded = tone_digits(reading)
        if encoded in reverse and reverse[encoded] != reading:
            raise ValueError("Tone export is not injective; no merging allowed")
        scorer.validate_readings([encoded])
        if scorer.toneless(reading) != scorer.toneless(encoded):
            raise ValueError("Export changes existing scorer toneless behavior")
        reverse[encoded] = reading

    # Check the Unicode implementation against official literal symbol data,
    # without executing the downloaded Python module or any package code.
    tree = ast.parse(source["pypinyin/phonetic_symbol.py"].decode("utf-8"))
    nodes = [n.value for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "phonetic_symbol" for t in n.targets)]
    if len(nodes) != 1:
        raise ValueError("Official tone symbol literal missing")
    symbol_map = ast.literal_eval(nodes[0])
    if not all(tone_digits(symbol) == expected.replace("v", "ü") for symbol, expected in symbol_map.items()):
        raise ValueError("Tone encoding differs from official symbol table")

    exported_characters = {k: ",".join(tone_digits(r) for r in variants.split(",")) for k, variants in characters.items()}
    exported_phrases = {k: [[tone_digits(r) for r in variants] for variants in rows] for k, rows in phrases.items()}
    # Exact semantic round trip, preserving keys, row order, variant order,
    # all entries, all alternatives, and original Unicode spelling.
    if {k: ",".join(reverse[r] for r in v.split(",")) for k, v in exported_characters.items()} != characters:
        raise ValueError("Character semantic round trip failed")
    if {k: [[reverse[r] for r in v] for v in rows] for k, rows in exported_phrases.items()} != phrases:
        raise ValueError("Phrase semantic round trip failed")
    return characters, phrases, exported_characters, exported_phrases, reverse, len(symbol_map)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = read_archive(args.archive)
    scorer = load_scorer(args.scorer)
    original_chars, original_phrases, characters, phrases, reverse, symbols = export(source, scorer)
    exported = {"characters.json": json_bytes(characters), "phrases.json": json_bytes(phrases), "reading-roundtrip.json": json_bytes(reverse)}
    config = {
        "schema_version": "kws-asr-evidence-rules-v1",
        "rule_version": "conservative-text-evidence-v1",
        "target_order": TARGETS,
        "lexicons": {kind: {"sha256": sha256(exported[kind + ".json"]), "entries": len(value)}
                     for kind, value in (("characters", characters), ("phrases", phrases))},
        "uncertainty_markers": list(scorer.REQUIRED_MARKERS),
    }
    exported["rules.json"] = json_bytes(config)
    rules = scorer.Rules(config, sha256(exported["rules.json"]), characters, phrases, TARGETS)
    for key in ("characters", "phrases"):
        if len(exported[key + ".json"]) > scorer.FILE_LIMITS[key]:
            raise ValueError("Full export exceeds scorer byte limit; no truncation allowed")
    invalid_source = []
    for key, value in original_chars.items():
        try:
            scorer.validate_readings(value.split(","))
        except scorer.ContractError as exc:
            invalid_source.append({"codepoint": key, "character": chr(int(key)), "readings": value,
                                   "rejection": str(exc), "exported_readings": characters[key]})
    report = {
        "schema_version": "official-pypinyin-lexicon-freeze-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "outputs_collected_before_freeze": False,
        "collection_timing_basis": "Contemporaneous execution status states actual ASR outputs have not been collected; exporter does not inspect or collect model output",
        "source": {"project": "pypinyin", "version": VERSION, "archive_url": ARCHIVE_URL,
                   "archive_bytes": ARCHIVE_BYTES, "archive_sha256": ARCHIVE_SHA256,
                   "pypi_release": "https://pypi.org/project/pypinyin/0.55.0/",
                   "git_commit": UPSTREAM_COMMIT, "submodule_commits": SUBMODULES,
                   "license": "MIT", "license_file": "upstream/LICENSE.txt",
                   "license_basis": "License distributed with immutable official source archive; upstream submodule MIT notices retained separately"},
        "source_members": {k: {"bytes": len(v), "sha256": sha256(v)} for k, v in source.items()},
        "scorer": {"path": str(args.scorer), "bytes": args.scorer.stat().st_size, "sha256": sha256(args.scorer.read_bytes())},
        "exporter": {"path": "freeze_lexicons.py", "sha256": sha256(Path(__file__).read_bytes())},
        "representation": "Full TONE3-style suffix digits, preserving ü and ê; no-tone stays unmarked; only four tone diacritics move",
        "all_entries_and_variants_preserved": True, "filtered_entries": 0, "removed_readings": 0,
        "added_entries": 0, "added_readings": 0, "source_code_executed": False,
        "unique_source_readings": len(reverse), "roundtrip_exact": True,
        "official_symbol_mapping_checks": symbols, "unmodified_source_contract_rejections": invalid_source,
        "target_order": TARGETS,
        "target_resolution": [{"target": target, "tokens": rules.tokenize(target)[0], "units": rules.tokenize(target)[1]} for target in TARGETS],
        "artifacts": {k: {"bytes": len(v), "sha256": sha256(v)} for k, v in exported.items()},
        "validation": "Existing unmodified scorer Rules constructor accepts full exports, mandatory markers, target order, and unambiguous targets",
        "limitations": ["Full means every entry in the official pypinyin 0.55.0 shipped dictionaries, not all Chinese vocabulary or optional extension lexicons",
                        "Dictionary pronunciation is text evidence, not acoustic tone verification; polyphony and OOV abstentions remain",
                        "This local timestamp/hash record is not an independently timestamped preregistration",
                        "No package installation, ASR inference, model weights, or real transcript scoring occurred"],
    }
    exported["freeze.json"] = json_bytes(report)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in exported.items():
        with (args.output_dir / name).open("xb") as output:
            output.write(value)
    print(json.dumps({"freeze": str(args.output_dir / "freeze.json"), "rules_sha256": sha256(exported["rules.json"]), "counts": config["lexicons"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
