> Continuation update: original SOURCE run 45352507569 stopped at the whole-corpus download with zero TTS, natural decoding, native observation or KWS calls. The original evaluator remains unchanged; the versioned continuation-primary evaluator and its separately retained zero-call origin proof govern the next actual primary execution. See `research/english_v6_b21_transport_continuation_20261124/` for the scientific-equivalence record. All replication and engineering locks remain null.

# B21 original quality readback and inert replication evaluator

This preparation implements the next boundaries while the actual B21 primary experiment is pending. It supplies no quality result, replication opening, engineering permission, promotion, or product release. The original primary evaluator remains byte-for-byte unchanged.

After the original source run failed at corpus transport before any TTS, natural decode, native observation, or KWS call, the primary readback target was explicitly routed to `english_v6_b21_continuation_primary_eval.py`. The original failed run and original primary file remain preserved. The separate transport-continuation routing record documents this change; all original quality-lock/report schemas and scientific decision rules remain unchanged.

## Shared original quality reader

`automation/english_v6_b21_quality_readback.py` exports two stdlib-only functions:

```python
verify_quality_original(directory: Path, lock: dict) -> dict
fetch_quality_original(directory: Path, lock: dict) -> dict
```

The first reads saved originals only. The second collects the exact locked archives with read-only GitHub API calls and invokes the first. Neither entry can run with an incomplete lock. Both return only after an original complete PASS has been independently reproduced. A NO_PROMOTION, STOP, missing return, consumed-unknown result, altered source identity, substituted model, or incomplete producer closes the boundary.

The complete lock contains exactly `schema`, `panel`, `result`, `producer`, `consumed`, `evaluator_sha256`, `inputs_sha256`, `producer_manifest_sha256`, and `original_result_sha256`. Its schema is `english-v6-b21-original-quality-lock-v1`; its panel is `primary` or `replication`. Each of the three artifact pins contains exactly `id`, `run_id`, `run_head`, `bytes`, and `sha256`. The three IDs must be distinct and belong to the same actual first run and head. `original_result_sha256` hashes the original RESULT.json member, while `result.sha256` hashes the complete original ZIP.

The retained directory contains:

- `originals/result.zip`, `originals/producer.zip`, `originals/consumed.zip`
- `api/run.json`, `api/commit.json`, `api/artifacts.json`
- `inputs/B0.zip`, `inputs/B9.zip`, `inputs/pair.zip`, `inputs/source.zip`, `inputs/screen.zip`
- The five `inputs/native_{provider}.zip` originals for qwen, whisper, parakeet, xlsr, and lv60.

Verification runs in a fresh isolated Python process, authenticates the exact original evaluator and its complete Python/resource closure against both SHA256 and the actual original Git trees, and reconstructs the original source admission. All 120 raw source originals remain in the exclusion inventory, including UNKNOWN and quality-rejected rows. The five original native archives must reproduce the exact original SCREEN. Source identity, TRAIN independence, complete model exports, all four PT identities, and the originally uploaded consumption marker are checked before reading model outputs.

The reader then requires every forward-entry journal, every complete returned `(1, frames, 4)` FP32 NPY tensor, and every retained `(frames, 4)` NPZ member. It compares every tensor byte, recomputes greedy tokens/events/sample clocks, verifies all counters and source joins, and invokes the unchanged B14 decision code and assessment. No NumPy, Torch, ONNX Runtime, ASR, TTS, training, or model inference is imported or executed by verification.

The verified report has schema `english-v6-b21-verified-original-quality-v1`. Stable integration fields are `panel`, `panel_gate`, `model_pt_sha256`, `assessment_sha256`, `training_design_sha256`, `original_result_sha256`, `pair_pin`, `admission`, `source_inventory`, and `original_identity`. It also records `original_lock_sha256`, `logits_sha256`, `actual_forwards`, and explicit verification booleans. `source_inventory` contains exactly the sorted unique arrays `pcm_sha256`, `source_group_id`, and `canonical_people`, plus `original_count: 120`. `original_identity` contains the string `run_id`, `run_head`, integer `run_attempt: 1`, and `prepared_parent`.

## Independent replica evaluator

`automation/english_v6_b21_replication_eval.py` preserves the original four roles, three gains, 1440 maximum forward entries, raw outputs, per-entry consumed journal, whole original feature grid, and unchanged B14 decision. It uses the explicit replica source module and separately registered replica source/scoring workflows. Frozen primary files and primary source behavior are not modified.

The new evaluator independently verifies the original primary PASS before using the replica inputs and again before scoring. The primary result proof is retained under `primary_quality/` in the original replica producer upload. Every model PT hash and the original paired-weights artifact must equal the primary PASS. The complete replica source PCM, source families, and canonical people must be disjoint from all primary raw originals. The new replica source module independently retains and verifies the same original primary proof.

The replica registration remains inert: `scoring_allowed` is false; the primary quality lock, reader SHA, source/native artifacts, new model pins, and producer closure remain unset. The separately reviewed binding may occur only after the actual primary PASS and the actual original replica source gate. The workflow template has no dispatch path and requires a first-attempt public workflow-only child of its exact reviewed parent. It preserves hidden producer files explicitly.

A complete replica PASS allows subsequent candidate engineering validation. It does not install a model, change B0, open old sealed replication/calibration data, claim physical SSC305 latency, claim field FAR/FRR, or release a product. Any other original quality outcome keeps downstream work closed.

## Validation completed for this preparation

Ten original-quality fixtures and twelve replica boundary fixtures pass under normal Python, `-O`, and `-OO`. Complete fabricated 576-return fixtures exercise both panel shapes; negative fixtures reject altered raw tensors, clocks, models, summaries, missing journals, partial/unknown consumption, and NO_PROMOTION. Replica fixtures also verify same-primary pair/weights, the hard ceiling before forward entry, original identity joins, no replay, and null-lock inert behavior. These are implementation fixtures and are never scientific evidence or opening authority.
