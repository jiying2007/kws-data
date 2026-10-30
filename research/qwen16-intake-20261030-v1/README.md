# New Qwen16 source intake — 2026-10-30

This is a new **unadmitted audio-generation** campaign, not model training or
production qualification. Source implementation and tests do not establish an
executed batch. The source tests use invented audio and metadata only.

The matrix is 4 authorized model presets × 2 keywords × 2 seeds = at most 16
calls. Vivian, Uncle_Fu and Dylan retain their original train voice role (12
pending samples); Serena remains development_a (4 pending samples). New seeds
are not independent speakers. No calibration, test, or qualification split is
created. All actual transcripts stay null and all admission flags stay false.

The Qwen3-TTS 0.6B CustomVoice model, revision
`85e237c12c027371202489a0ec509ded67b5e4b5`, uses original retained asset and source
locks from kws-pipeline commit `983e8082f712bd5099feb32bfb099e9f6df22d3a`.
Only named public presets are used: no personal reference audio, paid endpoint,
Codex, Work, or new unpinned model. Original donor files and closed historical
campaigns are never modified or executed through their old main entry points.

## Execution and recovery

Source development branch: `research/qwen16-intake-20261030-v1`.
After ordinary source tests pass, its narrowly scoped registration job records
all 10 new Git paths in the existing retention manifest with their **actual
source commit**, preserving every old row. It can write only that development
branch's manifest; it cannot change protected main or any gate. Open the PR at
the resulting registered head, pass all existing required checks, merge normally,
and verify main again. Source registration is not data or model admission.

The separate generation workflow supports either one main workflow_dispatch
with expected_source_sha or the **first creation** of the fixed branch
`research/run-qwen16-intake-20261030-v1` at the exact reviewed main commit. The
latter provides the same explicit launch through the available branch connector;
ordinary source pushes, PRs, and main merges cannot start generation. Both paths
share workflow run_number=1 and run_attempt=1, and the launcher independently
compares the checkout to live main before any acquisition. A failed execution is
consumed: no repeat dispatch, branch recreation, or silent recovery.

Budget: one model load, at most 16 TTS calls, zero ASR, zero training, zero retries;
4 CPU, 12 GiB container memory, no swap, 256 PIDs, plus 2 GiB host memory reserve.
Setup is at most 1200 seconds, generation stage 4800 seconds, each cell 300 seconds,
whole launcher 6000 seconds, outer job 110 minutes. Actual host memory and 16 GiB
free disk are checked; these resources are not guaranteed by a runner label.
Only public standard ubuntu-24.04 is used. Inference is non-root, read-only code
and runtime, network none. No GitHub token is passed to containers.

Original 24 kHz FLOAT and derived 16 kHz PCM16 WAVs are retained without gain or
trimming. Complete RIFF extents, sample geometry, original float payload and PCM
hashes, native EOS observation and signal metrics are checked before publication.
Claims retain GENERATED_PENDING_REVIEW / FAILED_NO_RETRY / NOT_RUN denominators.
A stale temporary file cannot prevent a final receipt; invalid output evidence
produces a failure-only artifact and nonzero exit, not a successful batch.

Allowlisted artifacts retain for 30 days. Before expiry, approved public preset
audio and exact receipts must be durably archived. Weights, installation logs,
private audio and private candidates are never uploaded. A lost runner may still
lose unuploaded evidence; that is not permission to regenerate consumed cells.

## Source acceptance

```sh
python3 -I -B -m unittest discover -s research/qwen16-intake-20261030-v1 -p 'test_*.py' -v
python3 -I -B -O -m unittest discover -s research/qwen16-intake-20261030-v1 -p 'test_*.py' -v
python3 -I -B -OO -m unittest discover -s research/qwen16-intake-20261030-v1 -p 'test_*.py' -v
```

Tests are software contracts, not real Docker/model results. The actual batch
must subsequently receive genuine human-review-v2 or reviewed native three-ASR
capture evidence. This positive-only batch is not a complete new training base:
negative/continuous audio, split/ancestry isolation, bounded candidate training,
weight retention, numerical equivalence, human final-AFE and physical SSC305
qualification remain separate requirements. Production approval stays false.
