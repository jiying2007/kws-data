# Official pronunciation lexicon freeze for real ASR text scoring

Frozen at **2026-10-01T23:50:59Z**, before collection of actual ASR output
according to the task declaration. This is a local timestamped freeze, not an
independently certified preregistration. No ASR inference or real transcript
scoring was performed here.

## Scoring inputs

Use only these three frozen files as scoring inputs:

| File | Entries | Bytes | SHA256 |
|---|---:|---:|---|
| `frozen/characters.json` | 41,923 | 705,099 | `2800d2c3b3dde2ee476d095149cc9a0f40d475729704fb085ac705a6413d508d` |
| `frozen/phrases.json` | 47,111 | 1,973,232 | `5bdc1773b7af068bfb6ba34687e08c95fa37b0cc2468f25025d57e776f4ca648` |
| `frozen/rules.json` | n/a | 500 | `1495ae540c56a20de1ba78074d10f488f741acca82be9e358d2a2b4c68828f8b` |

The unchanged scorer is
`/workspace/shared/kws-pipeline-restored/research/asr_evidence/calibrate.py`,
30,305 bytes, SHA256
`2bd566983f9846e19c3ac24d59cffe155cc4b45c39c1d64a9f1fae7772af104a`.
Its existing `conservative-text-evidence-v1` algorithm is unchanged. Both
exports fit the scorer's 8 MiB file and 100,000-entry limits.

`target_order` is exactly `['你好小窝', '小窝小窝']`:

- `你好小窝` resolves to `ni3 hao3 xiao3 wo1`, using official `你好` phrase
  readings, then official character readings for `小` and `窝`
- `小窝小窝` resolves to `xiao3 wo1 xiao3 wo1` via character readings

No target-specific entry was added. All twelve mandatory uncertainty markers
are frozen: `[unk]`, `<unk>`, `[inaudible]`, `[不清]`, `[缺字]`, `[不全]`,
`听不清`, `不确定`, `�`, `□`, `…`, `...`.

## Immutable official source and provenance

- Release: [pypinyin 0.55.0 on PyPI](https://pypi.org/project/pypinyin/0.55.0/)
- Archive: `upstream/pypinyin-0.55.0.tar.gz`, 839,836 bytes, SHA256
  `b5711b3a0c6f76e67408ec6b2e3c4987a3a806b7c528076e7c7b86fcf0eaa66b`
- Official download:
  https://files.pythonhosted.org/packages/b4/a4/784cf98c09e0dc22776b0d7d8a4a5b761218bcae4608c2416ce1e167c8af/pypinyin-0.55.0.tar.gz
- Resolved upstream tag `v0.55.0` commit:
  [`df101577145af2eb1abe5656e592e34e3bb56d23`](https://github.com/mozillazg/python-pinyin/tree/df101577145af2eb1abe5656e592e34e3bb56d23)
- Character data submodule at that commit:
  [`fa9761fff402f8560196b1ba085c437c52b56d7c`](https://github.com/mozillazg/pinyin-data/tree/fa9761fff402f8560196b1ba085c437c52b56d7c)
- Phrase data submodule at that commit:
  [`cee0ed6e6e4898580cafd2bd5e3723e20b214aa0`](https://github.com/mozillazg/phrase-pinyin-data/tree/cee0ed6e6e4898580cafd2bd5e3723e20b214aa0)

The Git commit's JSON dictionaries match the PyPI archive members byte for
byte. Both pinned submodules' `pinyin.txt` data also reconstruct the shipped
JSON dictionaries exactly, including reading order, under the upstream
generator's documented parsing/duplicate-readings convention. The upstream
Makefile and generator source are retained for inspection but were not run.

Original archive dictionary members:

- `pypinyin/pinyin_dict.json`: 788,780 bytes, SHA256
  `5f294c01e6c6c0a1c8e329c79335a3f8e0b27d06bf1de7a99244b765892d1e5b`
- `pypinyin/phrases_dict.json`: 2,545,585 bytes, SHA256
  `a45ff140a6b631ca9c82127b280a2f414e0aba6bb2824a0e9d1e77fff359c665`

The original archive, raw JSON, PyPI release metadata, commit tree, immutable
source copies, complete upstream license notices, READMEs, and exact source
URL/bytes/hash lists are under `upstream/`. `SHA256SUMS` and
`artifact-inventory.json` bind the complete local deliverable.

## License notices and ancestry

The release and both data submodules are distributed under the MIT License:

- `upstream/LICENSE.txt`: pypinyin, copyright 2016 mozillazg and 闲耘
- `upstream/pinyin-data/LICENSE`: character data, copyright 2016 mozillazg
- `upstream/phrase-pinyin-data/LICENSE`: phrase data, copyright 2017 mozillazg
- `THIRD-PARTY-NOTICES.txt`: complete retained notices, suitable to accompany
  copies of the exported data

The pinned character-data README identifies Unicode/Unihan 16.0.0 and other
dictionary/manual-correction sources. The pinned phrase-data README identifies
its historical starting datasets and additional reference sources. Their full
provenance statements are retained, rather than asserting newly invented
per-entry authorship. Unicode's versioned UCD 16.0.0 README and published
Unicode License V3 notice are retained as supplementary ancestry notices with
their retrieval URLs and exact byte hashes. The Unicode notice URL is not an
immutable release URL; its downloaded bytes are explicitly pinned locally.

This uses the licenses actually published with the source. It is not an
independent legal audit of every historical reference named by upstream.
Retain the complete notices with redistributed exports. No optional
`large_pinyin.txt`, CC-CEDICT extension, or `pypinyin-dict` package was imported.

## Why every reading uses tone digits

The raw character dictionary contains four entries (`呣`, `嘸`, `欸`, `誒`)
with valid pinyin readings containing combining tone marks, including `m̀`,
`m̄`, `ê̄`, and `ê̌`. The scorer's letter-or-tone-digit contract rejects those
raw spellings. Silently dropping those entries or readings would be wrong.

Instead, **every** reading in **both** dictionaries is exported consistently in
a reversible TONE3-style suffix-digit representation, a format the existing
scorer explicitly accepts. Only macron, acute, caron, and grave tone marks move
to terminal digits 1–4. Umlaut `ü` and circumflex `ê` are retained. Unmarked
readings remain unmarked; no neutral tone is invented. Examples:
`nǐ → ni3`, `m̀ → m4`, `ê̄ → ê1`, `lǜ → lü4`.

All 1,559 distinct source readings map one-to-one, with no collisions. All
dictionary entries, alternatives, and alternative order round-trip exactly to
the source. `frozen/reading-roundtrip.json` stores the complete inverse map.
The transform was checked against all 35 entries of the official literal
`phonetic_symbol` table without importing or executing pypinyin. Each reading
also preserves the existing scorer's toneless-comparison result. No entries or
readings were removed, merged, normalized away, guessed, or added.

"Full" means the complete two dictionaries shipped in pypinyin 0.55.0; it does
not mean exhaustive coverage of Chinese or the optional larger extension
dictionaries. Polyphony and OOV abstentions remain. Dictionary readings are not
acoustic pronunciation or tone measurements.

## Reproduce and validate offline

Run from this `scoring/` directory:

```sh
PYTHONDONTWRITEBYTECODE=1 python test_frozen_lexicons.py
sha256sum -c SHA256SUMS
```

`test-results.txt` records the test run. It includes a successful full CLI
import using explicitly invented, temporary **not-run** envelopes, zero
transcripts, and unknown human labels; that test cannot be interpreted as
model execution or acoustic evidence. Constructed string probes in unit tests
are likewise test cases, not ASR outputs. No package installation, source
package execution, model download, or inference is needed.

To reproduce exports into a **new** directory (never overwrite a freeze):

```sh
PYTHONDONTWRITEBYTECODE=1 python freeze_lexicons.py \
  --archive upstream/pypinyin-0.55.0.tar.gz \
  --scorer /workspace/shared/kws-pipeline-restored/research/asr_evidence/calibrate.py \
  --output-dir reproduced
```

The source archive, source dictionary hashes, and scorer hash are pinned in
the exporter. Reproduction of the dictionaries, reverse map, and rules is
byte-deterministic. A new `freeze.json` records a new timestamp and is therefore
not expected to have the original freeze record's hash.

Before collecting actual outputs, independently record the frozen rules hash
above and the separate final manifest hash in the experiment plan. During
import, pass those recorded expected hashes rather than calculating new
"expected" hashes from possibly changed files. Bind both model envelopes to
the same frozen rules and manifest. Never use the repository's invented tiny
fixture lexicons for actual ASR outputs.
