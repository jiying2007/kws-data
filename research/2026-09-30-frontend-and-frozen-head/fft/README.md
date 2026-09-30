# Two-window frontend diagnostic

Scope: exactly two previously identified 400-sample PCM16 windows, each evaluated once through the unchanged arithmetic of the pinned Python frontend and an instrumented copy of the original C frontend. No full-clip or acoustic-model inference, no revised numerical gates, no correction variant. Inputs and original NPZ/WAV/PCM identities are in preflight.json. These selected-window results cannot establish complete frontend conformance or target performance.

## Finding

DC subtraction, preemphasis and all512 window/padding values are bit-identical between Python and C for both windows. Their complex FFT bins differ. Thus the first observed arithmetic divergence is FFT, upstream of power/mel/log.

- Serena frame91, mel4: FFT bin4's weighted power difference is -0.0043882368241; bin5 contributes +0.0000993847432. Together they account for the mel difference of about -0.00428867. The original log difference -0.0012373924255 is reproduced exactly. Window L1 is106490.41, while selected FFT amplitudes are about2.63 and1.20, demonstrating the large cancellation scale.
- Uncle_fu frame269, mel9: FFT bin9 contributes +0.0733690047878, bin8 only +0.0000622775940. Together they account for the mel difference of about +0.07343292. The original log difference +0.0010147094727 is reproduced exactly. Window L1 is243989.27, with selected FFT amplitudes about18.27 and8.56.
- Absolute mel-reduction residual relative to double sum of each implementation's saved powers is at most1.52e-7 for Serena and1.52e-6 for Uncle_fu. Final log rounding residual is at most3.16e-8 and2.38e-7 respectively. They are much smaller than the observed mismatch. Changing only mel accumulation or log cannot address its dominant source.

The independent direct libm double DFT uses the identical saved window, modulo512 angles and math.fsum. It gives mel/log3.46548291779/1.24285199314 and72.3692452562/4.28178141986. Both Python and C lie on opposite sides; this is FP32 FFT cancellation sensitivity, not evidence that the PyTorch FP32 result is exact ground truth. This engineering reference is not the Decimal80 v2 acceptance oracle.

## Historical reproduction

C reproduces all80 saved historical log values bit-for-bit in both windows. Python reproduces all80 for Serena; for Uncle_fu only mel39 differs by9.536743e-7 under single-frame versus historical batched execution. The selected failing mel9 remains bit-exact. Therefore the selected failures are reproduced, but the isolated Python trace is not claimed to reconstruct every original batched intermediate.

## Execution and preservation

The reviewed run.py executed the two windows once and wrote traces.npz. Its final JSON serialization then failed because a NumPy float32 error scalar was not serializable. execution-status.json records exit1; original run.py is preserved. The separately reviewed summarize_saved.py reads existing traces only, explicitly computes complex errors in Python double precision, and reconstructs result.json without loading a native library, invoking Torch or re-executing a frontend. Its result binds the original script, preflight, library, traces and summary script hashes. execution.log is the redirected stdout and is empty because serialization failed before printing. The TypeError and stage are retained in execution-status.json; the original traceback was returned by the execution tool.

Preparation's instrumentation diffs are in source/. They only copy existing intermediates and store the existing rfft expression before abs; they do not change FFT, coefficients or numerical gates. C compiled with O2, C11, strict warnings and -ffp-contract=off; resolved compiler binary/version/hash and Torch/NumPy versions are frozen in preflight.json. No installed/released source changed.

## Minimal next experiment, not executed

A future separately reviewed variant could retain identical preprocessing and coefficients while changing FFT arithmetic precision, checking these two fixed windows first against both the frozen Python reference and independent DFT. More accurate DFT proximity does not by itself guarantee universal agreement with the FP32 donor. Preserve the1e-3 gate, prior failures and full synthetic regression. No FFT correction or broader claim is made here.
