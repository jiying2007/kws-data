# Unexecuted option: compact-head probability distillation

The supervised40-clip admission remains0 after the frozen Whisper screen. This proposal is a different unlabeled-training objective, not a retry of that gate, repaired transcript, or claim of human/ASR-verified OTHER labels.

## Scientific validity

Keep the pretrained encoder frozen. For each permitted native TRAIN audio frame, compute the intact teacher2599-class distribution p with its complete softmax. Targets for the compact six-class head are:
- q_blank=p_blank
- q_你,p_好,p_小,p_窝 are the corresponding original class probabilities
- q_OTHER is the sum of every remaining class probability

The six targets sum to1. Do not crop five logits and renormalize; do not turn OTHER into blank. Use a fixed cross-entropy/KL objective to fit these complete probability targets at temperature1. A new140→6 affine cannot generally represent logsumexp of2594 affine logits exactly, so mismatch is a real and falsifiable compression risk. It transfers teacher biases and mistakes rather than adding verified semantic knowledge.

## Small bounded proposed study

One pretrained frozen encoder, one initialized compact head, one fixed update budget/seed and fixed unlabeled native TRAIN manifest, chosen before outputs. Existing12+18 Qwen TRAIN and fixed40 historical HI-MIA TRAIN could be eligible as unlabeled audio only after provenance/license admission; the40 remain supervised-admitted=false. No development audio in fitting, no transcripts consumed by the loss. Freeze script, teacher checkpoint, frontend, frame grid, input hashes, KL aggregation and optimizer before execution. This note does not approve a run or supersede the halted CTC spec.

Primary fidelity comparison should be full teacher probabilities aggregated to6 versus student6 at the same frame endpoints. Report KL/absolute probability error pooled and per target/OTHER, not just target argmax. On the fixed observed Qwen development clips, report full-teacher and compact quality readouts plus resource measurements with their exact scope. Use the same six-symbol search alphabet including OTHER and the same decoder in the teacher-aggregation and compact paths for a meaningful compression comparison; historical2599 keyword-filter readout stays a separate reference. Thresholds remain uncalibrated unless calibrated on an independently defined collection; no threshold sweep to rescue the compact result.

A paired teacher-aggregation reference is essential: probability aggregation plus OTHER-aware decoding can itself change events compared with the original keyword-only decoder, before any head compression. Separate this change from approximation error. An exact learned OTHER linear row may be impossible, and fixed small training clips can overfit the teacher.

Fail if the frozen fidelity/quality-retention criteria are not met; do not widen training, change calibration or add rounds based on observed failures. Acceptance criteria must be numerical and approved in a new preregistration, not invented after outputs. The old supervised-admission failure remains in reports.

## Expected benefit and limit

Same analytically supported compact model:390,520 parameters,388,984 MAC/frame (~12.97M MAC/s), versus756,133 and752,004 MAC/frame for the intact head. Roughly48.3% model-MAC reduction and about1.565MB FP32 payload with CMVN; encoder/frontend/delay remain. No automatic48% whole-pipeline speedup or SSC305 target claim. This is a plausible way to decouple representation compression from unavailable trustworthy transcripts while preserving complete teacher probability mass. It does not improve teacher accuracy by itself or establish calibrated FAR/product quality.

## Concrete minimal preregistration proposal for review (still unexecuted)

- One trainable6-way head, no random-encoder arm: this tests compression fidelity, not representation pretraining. Freeze encoder/CMVN; isolated head seed7331,846 trainable parameters. Retain full donor teacher in eval/no-grad mode
- Fixed unlabeled TRAIN set:30 Qwen nativeTRAIN clips plus the preselected40 HI-MIA historicalTRAIN clips, only if waveform provenance/license hashes are admitted for unlabeled use. All40 keep supervised-admitted=false; no transcript or Whisper output participates in loss/selection. Do not substitute examples from development or replace rejected supervised items
- Complete teacher temperature1 softmax aggregated to6 with known dictionary bindings; enforce finite q>=0 and sum(q)=1 within1e-6. No clipping-away or renormalization of omitted mass
- Match semantic frame grids by using the same Hamming/frontend/encoder output at each clip. Cache140-dim frozen features and six-class q, discard full2599 logits after aggregation to bound memory
- Objective KL(q_teacher || p_student); compute frame KL mean within each clip and then equal mean across70 clips. This prevents long clips from gaining extra sample weight, though silence within a clip can still dominate. No target-confidence weighting chosen after readout
- AdamW lr1e-3,weight_decay0,gradient_norm_cap1,100 fixed full-set updates; no schedule/early-stop/validation selection. CPU1thread, hard process-CPU limit600s. No extra rounds if loss or metrics disappoint
- Teacher-aggregation reference and student share six-symbol beam alphabet with OTHER, two contiguous targets, inherited suffix/first-hit chunk behavior, all fixed upstream beam/duration/pruning defaults. threshold0 remains diagnostic; no calibrated FAR claim
- Primary Qwen evaluation: all12 native development clips (6positive/6confusable), never used in fitting. All42 original plus two frozen padding conditions remain fully reported with train/development labels, but are not independent replications or the primary quality criterion
- Non-vacuity gate: teacher-aggregation reference must hit at least one development positive for EACH target word. Otherwise this decoder/vocabulary reference is unusable for the stated retention hypothesis; do not claim zero-vs-zero success
- Proposed fixed fidelity gates on the12 raw development clips: equal-clip-mean KL<=0.05 nats/frame; each of four target classes MAE<=0.05 on frames where that teacher target probability>=0.05, with nonzero eligible counts; report every class including blank/OTHER, all per-clip means and p95 errors. These are proposed engineering tolerances, not recognized product standards, and require review before execution
- Proposed quality retention gate: lose none of the teacher-reference hit development clips, add no confusable-trigger clip and no wrong-keyword event. Show raw events/scores even if a gate fails. Loss advantage alone is insufficient
- Resource evidence: exact parameter/cache payload and analytical MACs first. If the fixed final checkpoint meets neither fidelity nor retention, stop, keeping all failures. Any matched host runtime comparison must use the same300ms frontend/decoder workload and clearly count teacher aggregation overhead; no A32 extrapolation

These criteria are a one-time observed-development compression feasibility test. They do not turn already-inspected Qwen development into fresh qualification data. No criterion, data weighting or threshold may be changed after outputs to turn a failed result into a pass. A later product operating point needs independent calibration and evaluation data.

Synthetic preflight precision: full2599 softmax and OTHER summation use float64 from the frozen FP32 logits, then cast the six probabilities toFP32 for the shared decoder. This avoids an observed synthetic-only FP32 probability-sum error beyond1e-6; no corpus readout was run before freezing this choice. Probability mass is never selectively cropped.
