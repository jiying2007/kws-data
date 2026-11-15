# Persistent V4 research executor

This executor accepts three explicit, bounded English63 research task types after
a reviewed, **new versioned request** reaches public `main`. The original online
weak-intent task produces GRU and DS-TCN weights and reloads them. The separate
saved-weight native C comparison consumes the existing successful checkpoints
with zero optimizer updates; its protocol is described below.
The third task measures new C discontinuity and state-isolation contracts using
the retained native header and original PCM, with no Python model forward.
PRs verify source and request boundaries only. They do not train, generate
audio, call ASR or download the original audio.

## Submit an execution

1. Review and merge the model source in `jiying2007/kws-pipeline` first. The
   only entrypoint is `research/english63_online_pilot/online_pilot.py`.
2. Copy `.github/research-v4/request.template.json` into
   `.github/research-v4/requests/<request_id>.json`. Set a new descriptive
   request ID, execution reason, and the reviewed **40-character commit SHA**.
   The placeholder is intentionally invalid. Repository, entrypoint, input
   artifacts and budgets cannot be overridden by a request.
3. Add that one request through the existing PR checks. Keep consumed requests
   unchanged. The main merge triggers `research-v4-executor.yml`; changing
   executor code or its documentation alone never launches training.

There is no schedule, patrol, workflow dispatch, TTS generation or external
agent task. The repository remains public. Execution uses this repository's
Actions token to read this repository's original artifacts; the pipeline
repository's token is not assumed to have cross-repository artifact access.
No Codex or ChatGPT Work compute is requested by this workflow.

## Fixed inputs and budget

| Item | Fixed value |
| --- | --- |
| Original audio | Artifact `13954744587`, source run `43242278962`; 63 original mono 16 kHz clips |
| Joined observations | Artifact `14001747172`, member `english63-three-evidence-review.json` |
| Original audio ZIP SHA256 | `f180004a27ea14c0575f4dd0e2a8476ca048442c76c2dcc186ede6d731cdffcc` |
| Review ZIP SHA256 | `f83d6a2f399c20af5bbc920ef6c38c42ac1e44b516d2b7291f4510557491cec8` |
| Review JSON SHA256 | `89f20b7ac5a672840013b3cf780422dcf68451b6f5d618e8f494a0e5f5dc6b12` |
| Candidates | GRU and DS-TCN; 48 optimizer updates each; seeds fixed by the reviewed source |
| Roles | 30 weak-intent training rows; 12 exposed development rows; 21 quarantined nearphone rows |
| Model wall time | At most 1,200 seconds for both candidates and the driver's first replay |
| Independent saved-weight replay | At most 300 seconds; zero optimizer updates |
| GitHub job | Ubuntu 24.04, CPU, one compute thread, at most 45 minutes including setup |
| Runtime | Python 3.12.16, NumPy 2.3.5, PyTorch 2.10.0+cpu; actual runtime also retained |
| Output/archive size | At most 64 MiB; terminal artifacts retained 90 days |

This is a **new** experiment environment. It does not claim byte-identical
training reproduction of the earlier Python 3.13.5 pilot. The explicit saved
weight replay checks the new experiment in its own recorded environment.
Expired/missing original inputs cause a failure; audio and ASR are not
regenerated. The joined ASR, IPA and acoustic observations remain observations,
not human-authenticated pronunciation labels.

## Consumed requests and durable readback

The execution key is the SHA256 of the exact request bytes. A concurrency group
serializes this key. Before any model work, the workflow retains
`research-v4-claim-<request_sha256>` as an Actions artifact. The terminal artifact
is named `research-v4-result-<request_sha256>`.

On a rerun the executor first checks these durable artifacts. A prior success
is downloaded and verified against its exact request, pipeline commit, complete
file hash manifest, both actual NPZ checkpoints, the 48-update receipts and
successful no-training replay. It is reused without downloading the inputs or
executing the models. A prior failed terminal, a claim without a valid terminal,
an expired receipt or an ambiguous artifact set stops execution. It cannot be
interpreted as permission to repeat training. `run_attempt == 1` is an additional
fallback against missing/deleted receipts, **not** the idempotency mechanism.

Failures after a claim retain the terminal, logs, `FAILED.json` and any partial
weights that exist. They remain failures. If the runner is killed before its
terminal can be uploaded, the earlier claim still prevents automatic replay.
Recovery requires a reviewed new request with the failure diagnosis and changed
execution reason; the consumed request is never edited. Requests with new IDs
are explicit new execution authorizations, so they should not be added merely
to repeat an unchanged successful experiment.

The Actions download step alone receives `GH_TOKEN`. Authentication is removed
before following the signed blob download URL. Checkout credentials are not
persisted. The model and replay child processes receive a small environment
allowlist with offline flags and one-thread settings; GitHub, artifact and
provider credentials are absent.

## Result contents and scientific boundary

The terminal artifact retains the exact `REQUEST.json`, `TERMINAL.json`,
`EXECUTION.json`, `runtime.txt`, process logs, word-level `keyword-evidence/`,
and the model's `SUMMARY.json`, `REPLAY.json`, `events.jsonl`,
`gru-weights.npz` and `dsc_tcn-weights.npz`. It does not duplicate the original
WAVs or input archives. The word-level report reuses the 189 existing ASR
observations and creates 126 keyword rows; it runs no additional recognizer.

Success means the bounded experiment executed and its saved predictions were
reproduced. Zero proposals is a valid diagnostic outcome. It is not evidence
of independent English FRR, continuous-negative FAR/hour, word-end latency or
SSC305 performance; all four remain `NOT_RUN`. Formal training admission,
human gold and model promotion remain false. The exposed 12 rows never enter
optimization, and the 21 ambiguous nearphone rows never become trusted negative
training labels. Finite clip and diagnostic continuous-sequence results do not
constitute an independent real-world continuous-stream qualification.

After downloading and unpacking a successful terminal artifact, its checkpoint
manifest can be verified without models or tokens. Put the artifact under a
new absolute directory's `result/` subdirectory, retain the original named
request file, then run:

```sh
python3 -I -S -B automation/research_v4_executor.py check \
  --request .github/research-v4/requests/<request_id>.json \
  --state /absolute/downloaded-run
```

For a separately requested no-training numerical replay, reuse the exact
original inputs, checked-out pipeline commit and retained runtime:

```sh
python -I -B research/english63_online_pilot/online_pilot.py \
  --tts-zip /absolute/original-tts.zip \
  --review-json /absolute/english63-three-evidence-review.json \
  --verify-result-dir /absolute/downloaded-run/result/model
```

No local project-model execution is needed for source verification. The
executor tests use invented receipts/checkpoint bytes and mocked APIs; the
pipeline owns actual numeric/model tests. Register new files in the existing
branch-retention inventory as integration files when integrating this change.

## Independent task: saved-weight native C comparison

`english63-saved-native-c-parity` uses the same persistent workflow and immutable
request directory. Its separate consumer is `automation/research_v4_native_c.py`.
The existing `en63-online-20261114-v1.json` request remains unchanged. There is
no call to the online training entrypoint in this task.

Start from `.github/research-v4/native-c-request.template.json`, pin the reviewed
pipeline commit containing `research/english63_online_c/c_replay.py`, and add
one new named request through the normal PR checks. A code/documentation update
without a new request does not perform model computation. The only execution
entrypoint is:

```sh
python -I -B research/english63_online_c/c_replay.py \
  --tts-zip /absolute/original-tts.zip \
  --review-json /absolute/english63-three-evidence-review.json \
  --trained-result-dir /absolute/original-successful-artifact-root \
  --output-dir /absolute/new-comparison
```

The original successful checkpoint is
[run 43595338774](https://github.com/jiying2007/kws-data/actions/runs/43595338774),
artifact `14114695344`, from data commit
`14547faa42299ae9d819ba4bff9824c2471c412e`. Its exact byte identities are fixed:

| Original checkpoint component | SHA256 |
| --- | --- |
| Complete artifact ZIP | `bc76805934df8eb158fa4259a75c26a4ea4276c15b83542acad5948a1f954b54` |
| Original consumed request | `ffa74269a0537115462da17b5050462e867cdb9980690105d2a92a4bdc0b7e53` |
| Successful terminal | `7c840f2877c0d80a147b5d1287261faeb01495e61f8a2d8ab8cf9053c6cc91a4` |
| GRU NPZ | `71f731155a19538b533e4253d2e4b19d5c03328bd26da5263542159df066e623` |
| DS-TCN NPZ | `167282c1037a28152bed4afaf49e53a4307bd8d5576a486b41c0a2291fdca935` |

Before compiling or forwarding a model, the consumer verifies that the original
Actions run succeeded, the artifact belongs to that run, the ZIP digest matches,
and the **whole original terminal/checkpoint/replay manifest** passes the
existing online validator. It checks the fixed NPZ identities again after the
comparison and evidence verification. A missing or changed checkpoint never
causes retraining or re-creation of the original WAVs.

This task runs on one CPU thread with the same pinned Python/NumPy/Torch
versions. C compilation and reference/C comparison have a shared 1,200-second
wall limit. A second `--verify-result-dir` call has a 300-second limit and uses
only saved evidence: zero optimizer updates and zero model forward calls.
The full Actions job is capped at 40 minutes including setup. Timeout handling
kills the execution process group, including compiler/native descendants.

The comparison covers both existing models, all 63 isolated clips and one
continuous concatenation of the same source PCM, with fixed `160` and `mixed`
chunk plans. It requires 13,010 isolated frames and 13,134 continuous frames,
64 reference sequences per model, feature/logit/probability absolute errors
at most `0.0002`, identical chunk probabilities/events, and identical Python/C
events. EOF padding, EOF-only events and continuous-state resets must all be
zero. These checks establish equivalence for these retained inputs. They do
not establish independent real-world continuous-stream performance.

The 17 comparison files are `SUMMARY.json`, `REPLAY.json`, `BUILD.json`,
`RESOURCES.json`, the Linux `en63_replay` executable, two reference NPZ files,
eight aggregate binary traces, and `export/EXPORT.json` plus
`export/en63_generated.h`. Original PCM/WAVs and copies of the trained NPZ
weights are not included. All files are covered by the evidence manifest.
The consumer also retains its request, execution receipt, runtime and logs,
within the existing 24-file/64-MiB accepted-result limits.

Native C claims and terminal artifacts use independent names:
`research-v4-native-c-claim-<request_sha256>` and
`research-v4-native-c-result-<request_sha256>`. A successful terminal is verified
and reused without input downloads, compilation or model calls. A previous
failure, an incomplete claim or expired receipt blocks automatic repetition.
Failures retain available comparison evidence and logs. Any necessary correction
uses a reviewed new versioned request, preserving the earlier terminal.

No model or threshold is selected. Formal training admission, human gold and
model promotion remain false; English FRR, continuous-negative FAR/hour,
word-end latency and SSC305 profile remain `NOT_RUN`. Recorded Linux native
resources are host observations. The original synthetic source and exposed
development limitations continue to apply.

## Native C discontinuity and state contracts

`english63-native-c-state-contracts` addresses a different question from the
consumed parity experiment: after a dropped block, can the C stream clear its
acoustic, encoder and decoder history while preserving the session's supplied
sample clock? A full `en63_reset` still starts a new session at zero. A separate
`en63_discontinuity` starts a new acoustic segment, repeats the 50-frame warm-up,
and preserves cumulative supplied samples, produced frames and emitted events.
Missing capture time is not known and is never filled with silence or invented
samples. This research clock is not a hardware capture clock.

The fixed consumer is `automation/research_v4_native_c_state.py`, dispatched by
the same V4 executor and workflow. The only new pipeline entrypoint is
`research/english63_online_c/state_replay.py`. The request template is
`.github/research-v4/native-c-state-request.template.json`; its placeholder source
SHA cannot execute. Existing request bytes and claim names remain unchanged.

Inputs are the original two public audio/review artifacts above and the exact
successful native artifact `14117638750` from run `43605922752`, whose SHA256 is
`e85d14c87727f93b6e33edc196af00a6c9ce023d9e7d41d844109617fa5fb397`.
The baseline request, terminal, source commit, full evidence manifest and C
header are checked before reuse. Its exported weights are copied unchanged;
there is no weight export, optimizer update or Python reference-model forward.

Each of GRU and DS-TCN has ten fixed cases: saved baseline prefix, midstream
reset, reset after EOF, two interleaved instances, discontinuity at samples
399/400/32401, EOF without padding, invalid-call atomicity and invalid-model
atomicity. The source inventory remains 63 clips and 2,101,760 samples. The test
uses the first **128,246 unique existing samples**, divided into two 64,123-sample
streams. These are controlled state regressions, not a new 63-clip acoustic
evaluation. Repeated suffixes and resets add **zero independent audio exposure**.
The saved no-discontinuity prefix must remain byte-identical to its prior C
trace; any failure is retained rather than accepted with a looser threshold.

The new task uses Python 3.12.16 standard library and GCC on public Ubuntu 24.04
Actions. It does not install NumPy or PyTorch. Execution is capped at 600 seconds,
followed by a separate 180-second saved-evidence verification that has zero model
calls. Requests cannot override the source entrypoint, original inputs, model
roster, budget or scientific boundary. The child environment is the same
credential-free, offline, single-thread allowlist as the existing executor.

The workflow uploads `research-v4-c-state-claim-<request_sha256>` before executing
the native models and retains `research-v4-c-state-result-<request_sha256>` for
90 days. A successful saved terminal is reusable without model execution. A
failed terminal, orphan claim, ambiguous result or missing receipt on an Actions
rerun blocks repetition. Results retain actual case observations, raw numeric
traces, reviewed source snapshot, build/runtime identities, summary and complete
hash manifest; failed partial outputs remain failures. All successful output
files still fit the existing 24-file and 64-MiB outer budget.

Source PR checks use invented JSON and analytic fixtures only. The new state
contract proves neither target pronunciation nor independent FRR, continuous
negative FAR/hour, true word-end latency or SSC305 resources. These remain
`NOT_RUN`; model selection, formal admission and promotion remain false.

## Fixed paired C pool-access profile

`english63-native-c-pool-profile` measures one implementation variable:
`EN63_POOL_TIME_MAJOR=0/1` changes traversal of the 200-slot, 32-channel sliding
pool while preserving each channel's FP32 addition and maximum order. Its fixed
consumer is `automation/research_v4_native_c_pool.py`; the only pipeline entry is
`research/english63_online_c/pool_profile.py`. The same existing workflow handles
source checks and the public main request. The template
`.github/research-v4/native-c-pool-request.template.json` cannot execute until its
placeholder commit is replaced by the reviewed full source SHA. The sole request
ID is `en63-native-c-pool-20261115-v1`; this lane cannot create another campaign by
renaming a request. Existing requests, consumers and claims remain unchanged.

Inputs are the original TTS/review artifacts and the successful state result
artifact `14141171320` from run `43663731129`, attempt 1. The 1,597,111-byte state
ZIP has SHA256
`2fdd0c951d6cf964d6b73afff5726e7d82b365dae539876822e513d7af0ccaa7`.
Its request, data/source commits, terminal, file manifest and original C header
are checked without executing its binaries. The new paired baseline and
candidate are both compiled on the same current runner from one reviewed source
commit and the same retained header, driver and flags; the macro value is the
only algorithm variable. Historical timing or RSS is not the new baseline.

Each model has eight pairs: one trace pair, one warm-up pair and six timed pairs.
Timed order alternates baseline/candidate and candidate/baseline, three of each.
Each full pass supplies the same 2,101,760 original PCM samples in 13,136 calls of
160 samples, producing 13,134 frames. Both models together have exactly 32 full
passes, 420,288 frames and 67,256,320 supplied samples including repeats. The
unique original sample positions remain 2,101,760; repeated profiling adds zero
independent audio exposure. The original 30 train-role, 12 exposed-development
and 21 quarantine identities are not promoted to labels or optimizer inputs.

The profile uses the lowest allowed single CPU, an overall 180-second native
process-group budget and a 30-second budget for each child. The outer execution
and saved-only verification remain bounded at 600 and 180 seconds. It uses only
Python 3.12.16 standard library and GCC, with no NumPy or PyTorch installation.
The child receives the existing single-thread environment allowlist and five
public Actions identity fields, never download or checkout credentials.

Four complete traces must exactly reproduce the retained original continuous
trace hashes, including candidate versus baseline. Each trace has magic
`EN63TRC1`, version 1, 168-byte records and 2,206,528 total bytes. Resource timing
is recorded separately from trace writing. The fixed benefit rule for each
model requires at least 5% median paired CPU reduction, at least five positive
pairs out of six, a median paired p99 ratio no greater than 1.10, exact numerical
equality and no increase beyond 38,192 bytes per state. An otherwise valid
`NO_BENEFIT` result is still a successful experiment; no extra pairs or changed
thresholds may be used to turn it into a benefit result. It does not select or
promote a model, and it does not measure independent FRR, natural FAR/hour,
acoustic word-end latency or SSC305 resources.

The independent artifacts are `research-v4-c-pool-claim-<request_sha256>` and
`research-v4-c-pool-result-<request_sha256>`. The claim is uploaded before native
execution. A prior successful terminal is verified without model execution;
failed, incomplete, ambiguous or expired receipts do not authorize repetition.
The result's `pool-profile` directory has exactly 15 fixed files, within the
existing 24-file and 64-MiB outer budget. Original raw process measurements,
source/build/input identities, native binaries, traces and saved verification
are retained; the saved verifier has zero model calls. Partial failures keep
their failure terminal and available evidence. Source PR tests use only
invented JSON/binary fixtures and mocked execution.


## Frozen English63 screen on the retained LibriSpeech sources

`english63-native-c-librispeech-screen` supplies the previously selected 517
LibriSpeech dev-clean utterances from 40 source speakers to the two trained
English63 native models. Each model receives one continuous PCM stream in
lexical source-member order. GRU and DS-TCN reuse the fixed state ancestor's
exported header, original thresholds and baseline pool macro zero. There is
no optimization, threshold selection, extra warm-up replay or model promotion.

The request `en63-native-c-librispeech-20261115-v1` is handled by the existing
V4 workflow and `automation/research_v4_native_c_librispeech.py`. Only a new
immutable request on public main can execute. The source-only PR checks cannot
run the original corpus or models. The template's placeholder source commit
is deliberately invalid. Existing consumed requests and claims remain frozen.

The fixed reference is artifact 14090383839 from successful first-attempt run
43538998200 (source commit `5da87fc5785cfa0b5ad26c451fc366b80e3d491b`). Its
47,199-byte ZIP contains only `english-natural-read-speech-result.json`. The
reference binds the selected FLAC bytes, transcript hashes, speaker source
identities and sample counts. It contains no historical decoded-PCM hashes.
The original OpenSLR dev-clean archive is fetched once from its fixed HTTPS
URL, without redirects, credentials or retries, under its separate exact
337,926,286-byte and SHA256 constraint. This does not enlarge the existing
64-MiB GitHub artifact or output bound. A failed transfer leaves its partial
file and stops. The fixed transfer deadline is 300 seconds, with 30-second
socket timeouts; no second request is issued.

Actual decoding uses NumPy 2.3.5 and SoundFile 0.13.1, records libsndfile's
version, and verifies every original FLAC identity and PCM16 geometry. It
records each newly decoded PCM hash and the concatenated PCM hash. It does
not claim comparison against nonexistent earlier PCM hashes. No source
samples are resampled, normalized, inserted or discarded. The stream has
58,646,164 samples / 3,665.38525 seconds, with 366,539 feeds of at most 160
samples and a final 84-sample feed. Each model produces 366,537 frames with
no reset between source utterances and no EOF padding. Both models together
have two passes, 733,074 frames and 117,292,328 supplied samples including
replay; reuse of the old source set adds zero independent audio duration.

The native group is limited to 420 seconds, each pass to 180 seconds, and the
whole entrypoint to 900 seconds. The two processes use one fixed allowed CPU.
The runtime installs NumPy/SoundFile only; no Torch, Sherpa, ASR or TTS runtime
is introduced. The workflow uploads a consumed claim before native execution.
Failed, orphaned, ambiguous or unproven rerun requests stop instead of repeating
model work. Success remains a completed diagnostic even when proposals occur.

The twelve-file inner result preserves compact 32-byte probability/event
traces, sources, build/runtime identity, exact inputs, raw process observations
and the saved verification receipt. The consumer first matches every member
of `SOURCE.zip` against a fixed reviewed path/SHA256 allowlist, rejecting
missing, extra, duplicate or symlink entries before any archived code runs.
Only that frozen standard-library saved reader is then invoked with
`--verify-result-dir`, isolated Python, a credential-free environment with
no Actions execution identity, and a 180-second limit. It recomputes decoder
events from saved probabilities without downloading, compiling, loading models
or running forward passes. The consumer separately checks the outer request,
claim and run identities, file closure, header, trace sizes, fixed counters
and research scope. A successful saved terminal is reusable without inference.

These are previously exposed read-speech sources with machine text filtering,
not human acoustic negative gold or household audio. The result reports
proposals, source-splice proximity and host resource observations. It does
not report an independently qualified FAR/hour, FRR, acoustic word-end latency
or SSC305 measurement. Those fields stay `NOT_RUN`; trusted-negative status,
human gold, training admission, model selection and promotion stay false.


## Paired natural-background continuation from the frozen online weights

`english63-natural-background-continuation` is a separate weak-training
experiment, handled by `automation/research_v4_background_continuation.py`
inside the same V4 workflow. Its immutable request is
`en63-background48-20261115-v1`; it cannot consume any earlier training or
LibriSpeech-screen claim. The request pins the new source commit and all
protocol values. Source PR checks use invented fixtures only. Actual model
work is restricted to the first public-main Actions attempt after the new
claim has been uploaded. There is no scheduler, dispatch retry or generic
command field.

The warm-start source is original online training artifact 14114695344 from
run 43595338774, not the older whole-clip checkpoints. Both GRU arms start
from the same frozen GRU NPZ, and both DS-TCN arms from the same frozen DS-TCN
NPZ. There is no saved optimizer state: all four arms use fresh AdamW,
48 updates per arm, learning rate 0.001, weight decay 0.01 and gradient clip
1. Each model's control and treatment use the same fixed seed. The control
loss is the original 30-clip weak MIL cross-entropy. Treatment adds 0.5 times
the mean weak-background MIL cross-entropy of 128 natural windows. Each step
backpropagates the original loss and four 32-window natural microbatches,
then clips once and updates once. The total budget is exactly 192 updates.
The original train-only normalizer, decoder and pool macro zero stay fixed.

All 40 source speakers from the already exposed 517-clip LibriSpeech roster
are ordered by SHA256 of `en63-bg48-speaker-v1/` plus speaker identity. The
first 32 supply training windows and the last eight are study development.
Within each training speaker, eligible utterances have at least 48,000
samples and are ordered by SHA256 of `en63-bg48-record-v1/` plus source-member
path; the first four yield centered 48,000-sample windows. Thus treatment
uses 128 three-second windows, 6,144,000 samples / 384 seconds. The 102
study-development source clips contain 11,124,244 samples and produce 69,525
continuous frames. The twelve original English63 development clips are
evaluated separately with resets, producing 2,440 frames. Original development
and quarantined nearphone rows never enter training.

Baseline, control and treatment for both model kinds each receive the same
13 development streams exactly once in native C: one natural continuous
stream and twelve isolated original English63 clips. Across all six model
passes this is 431,790 forward frames. There is no repeated verification
forward. The consumer independently checks 78 initializations, 78 finishes,
431,946 feeds and 69,110,904 supplied samples across those six passes. Repeated
supplies are not counted as new independent audio. For each model kind, the predefined benefit rule requires positive
baseline and control natural-proposal totals, at least a halving by treatment
against both, and no increase in either keyword's natural proposals. Each
keyword's four positive clips must have correct-trigger coverage at least
the larger of baseline, control and two clips, with wrong-other-keyword clip
counts no greater than either comparator. The four ordinary-background
clips' total proposals must not exceed either comparator. All valid benefit
outcomes, including `NO_BENEFIT`, produce a successful scientific terminal;
the gate never authorizes extra training or evaluation.

Each training arm is limited to 180 seconds and the training group to 720;
each native child is limited to 120 seconds and the native group to 180.
The source entrypoint has a 1,200-second limit and the saved verifier 180.
Python 3.12.16, NumPy 2.3.5, CPU Torch 2.10.0 and SoundFile 0.13.1 are pinned.
Input downloads reuse the fixed artifact and corpus safety helpers. The
337,926,286-byte original corpus remains outside the 64-MiB result limit.

The distinct artifacts are `research-v4-background-claim-<request_sha256>`
and `research-v4-background-result-<request_sha256>`. The claim records an
update budget of 192 and zero updates before model start. Successful execution
and terminal records require the actual 192 updates and 431,790 native frames.
Failed executions retain available weights, progress and evidence; an unknown
completed count is not reported as zero or as a full run. Failed, orphaned,
ambiguous and unproven attempts remain consumed and cannot repeat.

The inner result has exactly 17 files: eight JSON/archive receipts plus
`native.stderr`, `WEIGHTS.zip`, `RUNTIME.zip` and six variant/model trace
containers. `WEIGHTS.zip` contains the four new NPZs; `RUNTIME.zip` retains
three generated headers and their native binaries. The outer result remains
within the unchanged 24-file and 64-MiB limits. Before any archived reader
executes, every `SOURCE.zip` path and SHA256 must match the reviewed closure.
The isolated standard-library reader then recomputes saved events, counts
and benefit results without network access, compilation, model loading,
forward calls or optimizer updates. Outer request/claim/run identity and
actual work counts are independently checked by the data consumer.

This is a paired weak-background correction study on already exposed source
material. Development observations are not independent acoustic gold or
household FAR. The benefit gate does not select or promote a model, change
runtime defaults or grant formal-training admission. Those permissions,
human-gold and trusted-negative status remain false.
