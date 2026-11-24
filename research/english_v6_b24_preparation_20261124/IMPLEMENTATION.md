# B24 implementation and evidence authority

The frozen hypothesis is in `TRAIN_ONLY_WARM_START_MEMO.md` and design SHA
`4fa88cf4050aa276323a7306f21d53163d47c27e3c011c2547d5eeebe6980309`.
This implementation adds no scientific factor: it reads the exact original B23
checkpoint, then calls the existing, hash-fixed `english_v6_b23_prefix_peak.fit`
function. That function constructs a fresh AdamW after loading the supplied
weights. All 192 draws and all 1847 newly budgeted forwards are unchanged. The
wrapper only renames three new exports and corrects their parent/phase metadata.
It never recreates B23 optimizer moments or repeats the old B23 updates.

The B23 failure reader has two distinct authorities. The independent verifier
algorithm is the final reviewed `english_v6_b23_quality_readback.py`, SHA
`a8b4297d868ad4b3b7b7b9adca07e7a8b20873f42b09cbc7dcd1cd4ea1680f45`,
fixed before the B23 PRIMARY result. It is not claimed to be a member of the
original PRIMARY producer archive. The actual evaluator, input registration,
complete producer manifests/Git files, source admission and full original
returned arrays are authenticated from that original execution. The B24
fail-only adapter changes the required verdict to `NO_PROMOTION` and keeps every
downstream permission false. It cannot accept PASS, STOP, a partial return,
unknown consumption, a forged summary or an incomplete original producer.

The new public source-exposure metadata contains only the whitelist needed to
exclude every prior original PCM and every exposed or reserved canonical
speaker. Full downloaded ZIPs and GitHub API responses are not copied into public
Git. Metadata verification can regenerate the whitelist from those exact
original bytes without opening audio or reading model scores. The future TRAIN
receipts artifact also retains exact original `MANIFEST.json` and `HISTORY.json`
members as `ORIGINAL_B23_SOURCE_MANIFEST.json` and
`ORIGINAL_B23_SOURCE_HISTORY.json`. Their original member hashes are already in
the public origin whitelist. SOURCE can authenticate old whole-frame lengths and
identities without reopening old audio/provider ZIPs or substituting a summary
for an original.

`english_v6_b24_opening.json` remains disabled until a complete actual B23
PRIMARY failure lock is verified and bound. Independently frozen new panels are
also required before any budget. The runtime workflow is inert; only a reviewed
workflow-only first-attempt activation may upload the one-use marker. The marker
precedes installation of the model runtime and the first optimization forward.
No B24 model, TTS or ASR call has been made during this preparation.
