# Persistent V4 research executor

This executor accepts two explicit, bounded English63 research task types after
a reviewed, **new versioned request** reaches public `main`. The original online
weak-intent task produces GRU and DS-TCN weights and reloads them. The separate
saved-weight native C comparison consumes the existing successful checkpoints
with zero optimizer updates; its protocol is described below.
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
