# Fixed compact-head distillation: failed fidelity gate

The one approved100-update experiment completed its checkpoint and all126 clip readouts. It fails the preregistered compression criterion. No extra optimization, threshold change, C model export or candidate promotion followed.

## Results

- Full teacher probabilities were aggregated without dropping mass into blank/你/好/小/窝/OTHER, then decoded with OTHER retained in search
- This shared decoder's teacher reference hit3/6 native development positives (keyword1:2, keyword2:1),0/6 confusable clips. All hits were Serena
- Student on that same development set hit4/6, retained all3 teacher hit clips and introduced no confusable or wrong-word events. All4 hits were still Serena; this is not cross-voice success
- Conditional target-class MAE:你0.178271 (9 eligible frames),好0.422120 (10),小0.358496 (10),窝0.279108 (9). Every class exceeds the fixed0.05 ceiling
- BOTH fidelity and retention were required. Retention passed, fidelity failed, so the candidate failed despite the extra development hit
- Complete observed42-clip readouts: raw9/20 positive hits; tail500ms10/20; head+tail9/20. Each had0/22 confusable-trigger clips and no wrong-word events. These include training examples and are not independent qualification trials

## Numerical reporting failure and repair boundary

The original script aborted only in final KL reporting, after writing the checkpoint and all three output JSON files. It took log of rounded FP32 student probabilities:270 entries were zero on development,269 with teacher probability zero and1 with positive teacher probability. The269 zero/zero terms should contribute zero mathematically; the remaining positive-q/zero-rounded-p term makes the literal rounded-distribution KL infinite. This reporting failure is not evidence training diverged.

Ideal softmax KL from original finite logits cannot be reliably recovered from the saved rounded probabilities. The logits themselves were hashed but not saved. We did not rerun inference to recover them after the candidate had already failed every target-MAE gate. The ideal KL is therefore null/unavailable, not replaced by epsilon and not called a pass.

The corrected stable_kl function masks q=0 and uses float64 log_softmax. Four synthetic tests cover extreme logits, probability underflow, zero target mass and invalid input. The optional checkpoint replay code exists for audit but its main was NOT executed. Original driver, outputs and exception log remain unchanged. summarize_failure.py reads JSON using only the standard library and writes failure-readback.json.

Training loss history and exact optimization CPU/wall time were stored only in process memory and were not persisted before the late assertion. They remain explicitly missing. The immutable code reaches checkpoint writing only after100 optimizer updates; no second training run occurred.

## Artifact and scope

Checkpoint compact-final.pt:1,578,849 bytes, SHA256 b61fed9ef590a9c922ee2a6ddc6584b3f29330c45bd4c07d386db459a5514bba. It is a PyTorch research checkpoint, not a native C export. Original driver SHA d4a5e80b47451aaecea5e77a55c1053fa7a7a91ec07bdff856cfc080a50ef85d; approved student preflight fd667f419d10df6c93a84980d95fd69ad99d37cc1e2917cc556944565e368cf3.

Training used70 fixed native/project TRAIN audio files without consuming transcripts:30 Qwen and40 HI-MIA-CW. The40 remain supervised-admitted=false after the independentASR screen; distillation did not overturn that failure. This objective transfers teacher distributions, not verified transcription/background truth. Underlying source licenses and provenance remain those documented in the input receipts; no universal commercial clearance is inferred.

A single affine OTHER output cannot generally reproduce logsumexp of2594 teacher affine outputs exactly. That, the fixed optimization budget and limited data are possible explanations, not proven causes of this failure. No follow-up head/optimizer sweep was started. Reusable native frontend parity work and independently prepared fuller-sentence data are separate tasks.
