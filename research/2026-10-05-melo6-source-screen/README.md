# Melo6 initial generation evidence

Research only. This archive records six synthesis calls and two session
constructions across the two documented attempts. The first attempt constructed
one session but failed the loaded-output-signature check before any synthesis
call; all six prior rows remain `not_run`. The recovery constructed one session
and made one call per phrase, with no warmup or generation retry. These counts
come from the supplied attempt identities, unchanged control flow, saved claims,
and terminal receipts; they are not a system-wide historical call census.

Six native 44.1 kHz mono float WAVs and six 16 kHz mono PCM16 derivatives are
preserved byte for byte. The independent audit verified the full resampling
reconstruction. Neither waveform determinism nor acoustic correctness is claimed.
Machine evidence is weak, human listening truth is `UNKNOWN`, ASR is pending,
CTC labels are null, and training admission is false. No training was performed.

The only inputs authorized for the blind ASR handoff are:

- `research/2026-10-05-melo6-source-screen/recovery/blind/blind-inputs.zip`
- `research/2026-10-05-melo6-source-screen/recovery/blind/blind-input-freeze.json`

The ZIP has seven members: a strict six-clip job and six digest-named audio files.
The rest of this archive contains generation provenance and must not be supplied
as blind recognizer input. Legacy `qwen6` schema names in the freeze are retained
for parser compatibility, not a source-model claim. Later ASR evidence belongs
on the same research branch; it has not been added to this initial snapshot.

`recovery/` preserves the 32 independently allowlisted recovery files.
`prior-failure/` preserves the 12 independently allowlisted prior files and their
relative structure. `GENERATION-ATTEMPT2.json` is the frozen prelaunch identity.
The independent audit is included unchanged. Raw diagnostic causes, logs,
runtime environments, wheel bodies, and model bodies are excluded. Safe cause
retention metadata contains only its hash, size, and capture/retention flags.

All audio and resource evidence is intended for `jiying2007/kws-data` on
`research/melo6-results-archive-v1`. Scientific source code belongs in the
matching `jiying2007/kws-pipeline` source snapshots, not this data archive.

`content-manifest.json` declares the size, SHA-256, and Git blob SHA-1 of every
file except itself. Verify the manifest's own SHA-256 against the publication
receipt, then run Python 3.10 or newer from this directory:

```sh
python verify_archive.py --prior-source PRIOR_SOURCE_DIRECTORY --recovery-source RECOVERY_SOURCE_DIRECTORY
```

Each source directory must be the corresponding pipeline `research/melo6_tts`
snapshot. The verifier reads every plan binding and adapter-declared source,
checks both generation freezes, all six started claims and tensor boundaries,
all WAV headers and sample hashes, and all seven ZIP members against the blind
freeze and generated derivatives. It uses only the Python standard library,
does not import the model or runner, and makes no model calls or writes.
It does not recompute resampling; the pinned independent audit records that
separate byte-exact reconstruction. Hash integrity does not establish speech
quality, pronunciation correctness, voice rights, or training eligibility.

Resource numbers are observations: RSS and disk usage are sampled rather than
continuous peaks or hard caps; CPU limits are per process, not an aggregate
process-tree cap; saved per-call time includes conversion.
