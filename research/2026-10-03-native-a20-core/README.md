# A20 model and training-core publication checkpoint

This is a temporary, independently verifiable subset of the complete research
archive. **Overall archive completion is false.** All 65 selected
logical assets are present and hash-verifiable here; 276
other logical assets remain assigned to the full archive. The complete target
has 341 assets and archive-index SHA-256
`3435110711dc158d20ccd3eb1393227090d4a5939e791292e69030a90d78f7e7`. Its reviewed publication-manifest SHA-256 is
`d0c265888b8e4799101a226eb568f4592f5e63d824493fae4214b0d577b5ed01`. See `CORE_STATUS.json` for machine-readable scope.

## Exact contents

- A20 native FP32 payload and tensor identity/manifest; exact A20 and F20 training
  checkpoints and separately attributed upstream donor checkpoint
- All selected F/A D20 training records, stored logits, fit/update history,
  protocol/recipe, endpoint/initialization metadata and twenty exact synthetic
  D20 source WAVs, with original/public source-hash mappings
- Historical training architecture, sanitized scientific trainer/contract
  excerpts, exporter, pinned upstream sources, CMVN/config/token resources,
  applicable source licenses, and the numerical contract
- Model-byte verification and the full target's original/public transformation
  map, retained as provenance metadata; references in that map do not imply that
  every full-target asset is present in this subset

Every selected asset metadata record and original payload identity is identical
to its full-target record. The 64 stored content-addressed objects
and their gzip shards are byte-identical to the full target's objects/shards.
There is no separate model export, rerun, training, scientific regeneration, or
new result. Checkpoints are opaque bytes; the archive tool never unpickles them.

## What is pending

The remaining full archive contains the fixed twelve CosyVoice clips, eighteen
exposed Serena diagnostic clips, replacement-eight non-speech DSP fixtures,
their reference/native arrays, failed and recovered numerical evidence, and raw
saved benchmark timing records. These remain required. This checkpoint must not
be described as the complete model/data/evidence delivery. The full archive is
intended to supersede this temporary research prefix after verified publication.

The D20 audio's historical role is weak-labeled training input, not human-gold,
blind qualification or a new default catalog admission. Current
`training_catalog_admission`, `qualification_allowed` and `shipping_promotion`
are false. `commercial_output_license` remains `not-established`.

Historical metadata snapshots can describe formerly missing training artifacts.
The restored scientific recipe, trainer excerpt, D20 manifest, checkpoint/logit
records and source audio are now retained. Historical source excerpts omit
private orchestration and are explicitly modified; byte restoration is not a
claim that end-to-end retraining or benchmark replay has been executed or is
already turnkey. The native runtime source is separately published in the draft
[code PR](https://github.com/jiying2007/kws-pipeline/pull/463), at reviewed head
`804553286fd9caa294769cc4d44cc0c43dc66e88`.

## Offline verification

From this directory, with Python 3.9+ and no additional packages:

```sh
python3 tools/archive.py verify . --expected-index-sha256 c9812fad12fc12e8f3a76afeeff7b0b8881f236aba640b3fc400319f3c70f0f5
python3 tools/archive.py materialize . /new/empty/output --expected-index-sha256 c9812fad12fc12e8f3a76afeeff7b0b8881f236aba640b3fc400319f3c70f0f5
python3 -m unittest discover -s tools -p 'test_archive.py' -v
```

See `ARCHIVE_FORMAT.md` for byte identities, bounded storage, safety checks and
checkpoint loading cautions. See `LICENSES/SCOPE.md` for distinct code, model and
synthetic-input provenance. No product registry, default asset catalog, build,
ABI or qualification state is changed by this research checkpoint.
