# PR450 existing public trace retention

Preserves the original GitHub Actions artifact as one immutable object, not a new diagnostic run. Source: [run 36689641752](https://github.com/jiying2007/kws-pipeline/actions/runs/36689641752), artifact 11085476672, head `0595b248b19b856df80902a6656bbea94ddd5a94`.

Original ZIP: 23,590,873 bytes; SHA-256 `dafaa468c0d9d6ffb323dca2845df74128ea128f0fc3870ce2d4224fcfe26672`. GitHub retention expires 2026-10-30T08:31:33Z. There are 44 original members: 13 KWTRACE1 traces, 6 PCM WAVs, and 25 metadata/script files. SOURCE-MEMBERS.json lists their byte identities, including the empty captures.jsonl.

## Offline verification and restoration

`python3 verify.py`

`python3 ../2026-10-08-d20-d90-host-research/verify_archive.py --root . --restore /existing/parent/new-directory`

The output is the exact original `frozen-far-source-replay-11085476672.zip`. Restoration never executes the historical script, model, replay, training, ASR or TTS. The original ZIP is wrapped in the repository's existing content-addressed ZIP layout and split at 8 MiB; its members are not duplicated as separate stored objects. ARCHIVE.json describes the wrapper, CATALOG.json the original ZIP, and SOURCE-MEMBERS.json its original contents. Do not confuse the two ZIP hashes.

The shared storage verifier/tests remain unchanged at kws-data commit d9a65cc616cbb0e55a99f4c0e77c16e29bc6622a, research/2026-10-08-d20-d90-host-research. The shared helper SHA-256 is pinned by verify.py. The existing retained-publication-bytes job verifies this package and all 44 inner member identities as well as the shared safety regression tests.

## Provenance and scope

The pinned workflow uses configs/nightly.xiaowo-frozen-model.json (SHA-256 a515749b2987e1df23f588a1db8f1e9a7f710662ed6f8b4df92c09aacf40f5f6). Its TTS backend is procedural `tone`. The saved render receipt reports synthetic-domain, a proxy AFE, no RIR manifest, and simulated scenes. There is no human speech recording, external corpus, model weight, credential file, or private home path identified in the archive. The original public GitHub runner paths are retained unchanged. Basic text scans are not a universal guarantee of absence of sensitive data.

The historical Python script is retained as inert evidence under the source repository's Apache-2.0 license (included). Synthetic audio and numerical results are not relicensed as third-party speech or represented as real-world evaluation. No gate or scientific conclusion is changed. This package contains only PR450's pre-existing public artifact; it does not contain the cancelled PR500 deliverable.
