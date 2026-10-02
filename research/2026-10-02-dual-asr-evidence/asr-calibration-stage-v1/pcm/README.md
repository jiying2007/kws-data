# Strict PCM16-to-float32 binding stage

This is scratch-only preparation. It imports only the Python standard library. It does not import either ASR runtime, run NumPy, download models, install packages, run inference, or calibrate anything.

## Result

- 39 synthetic stdlib tests pass
- All 16 recovered WAVs match their predeclared source SHA-256 and byte count
- Exactly 1,004,800 mono frames at 16,000 Hz, totaling 62.8 seconds
- All sample values convert exactly to IEEE-754 binary32 using signed PCM16 / 32768.0; every actual sample was checked for exact reversibility
- The raw PCM bytes, converted float32 bytes, each sample descriptor, and the combined binding descriptor have separate SHA-256 identities
- No waveform bytes or transcription labels were saved in the output manifests or logs

## API

`WaveExpectation(opaque_id, basename, file_sha256, file_bytes, frame_count, sample_rate_hz=16000, channels=1, bits_per_sample=16)` is the private predeclared source binding. Opaque IDs must match `clip-0001` through `clip-9999` (the pattern also permits `clip-0000`). The source filename is confined to this private object.

`bind_wave(absolute_root, expectation)` returns `BoundPCM` with immutable `bytes` fields `pcm16_le`, `pcm_float32_le`, and `descriptor_json`, plus `opaque_id` and `descriptor_sha256`. Its `.descriptor` property returns a fresh dict, so mutations cannot change the binding. The raw bytes are little-endian signed 16-bit samples; the float bytes are little-endian binary32 samples in the same mono sequence.

`bind_batch(absolute_root, expectations)` returns bindings in exactly the supplied order, rejecting duplicate opaque IDs and filenames. A batch stops at 1,001 declarations and rejects more than 1,000; each WAV has an independent strict 16 MiB input cap. Partial results are never returned after a failure. The orchestrator should retain its smaller corpus-level resource limit when integrating.

`decoder_manifest(bindings)` produces a separate manifest with only opaque IDs, sample/container descriptors, hashes, and frozen order. Its canonical SHA is over sorted-key compact ASCII JSON without a trailing newline. `decoder-inputs.json` is that canonical JSON plus one newline, so the receipt reports both the canonical and file hashes.

Source order is the order of `files` in the pinned existing `recovery-verification.json`, with IDs assigned `clip-0001` to `clip-0016`. The private source-name mapping is not copied into the decoder manifest. Do not hand `WaveExpectation`, source paths, recovery metadata, reference text, filename-derived labels, or evaluation annotations to decoder adapters.

## Strict acceptance rules

The inspected 16 recovered files all use an ordinary 44-byte RIFF/WAVE header: `RIFF`, exact file-size-minus-eight, `WAVE`, one `fmt ` chunk of length 16, PCM encoding 1, mono, 16000 Hz, byte rate 32000, block alignment 2, 16 bits, then one `data` chunk occupying the exact remainder.

Only this format is accepted. RF64, RIFX, extensible WAV, float WAV, fmt extensions, metadata/ancillary/duplicate chunks, reordered chunks, padding or trailing data, inconsistent lengths, empty audio, and any other encoding fail closed. There is no broad `wave`/ffmpeg/sox/torchaudio fallback. No channel mix, resampling, trimming, normalization beyond the exact PCM16 scale, or reordering occurs.

The POSIX reader opens every directory component using `O_DIRECTORY | O_NOFOLLOW` relative to an already opened directory fd. File reads use `O_NOFOLLOW | O_NONBLOCK`, require a regular file with one link, compare declared and observed sizes, read at most declared-size-plus-one in chunks of at most 65,536 bytes, compare before/after/named-file identity and metadata, then verify the whole-file SHA before parsing. It rejects symlink components, hard links, noncanonical root spellings, source-name traversal/aliases, FIFOs, directories, missing files, replacements, and detectable in-place modifications. Unsupported POSIX nofollow/dir-fd capabilities fail closed. The returned immutable byte snapshot is the binding; the original pathname is not assumed unchanged afterward.

## Remaining runtime integration gap

Actual NumPy/ASR integration has **not run**. The later authorized adapter must:

1. Re-bind the predeclared sources in that runtime environment, and require the expected raw PCM SHA, float32 SHA, descriptor SHA, frame count, shape, and frozen ID order to match this stage
2. Create the input from the immutable bound bytes with `np.frombuffer(bound.pcm_float32_le, dtype='<f4')`; this view should be nonwriteable and must have shape `(frame_count,)`, one dimension, exact little-endian float32 dtype, and contiguous storage
3. If an ASR API requires writable input, make an independent `.copy(order='C')` for **each** decoder and verify its `tobytes(order='C')` SHA before the call. Never share a mutable array between models. If the API accepts read-only input, preserve the immutable view and do not enable writes
4. Verify both decoders received the same immutable source binding and exact full-frame sample sequence. Record runtime/adapter input evidence at the actual inference boundary; this preparation cannot establish that either model consumes a supplied array unchanged internally
5. Pass explicit sample rate 16000 where the API needs it. A runtime that only accepts a path must not silently replace this array binding with a path decoder; that requires an explicitly reviewed alternative binding

The two decoder adapters and their runtime behavior are outside this module. Descriptor hashes alone are not proof that an ASR backend was fed those samples.

## Reproduce

From `/workspace/scratch/6c2ef8b5a46e`:

```
python -m unittest discover -s asr-calibration-stage-v1/pcm -p 'test_*.py' -v
python asr-calibration-stage-v1/pcm/verify_recovered.py
```

All synthetic test fixtures are created and removed under this directory. Verification writes only `decoder-inputs.json` and `verification-report.json` here. The manifest reader pins the inspected recovery receipt to 3,082 bytes and SHA-256 `8a162581ed88bb933663f27aafe0cc1721d9119222d05ac46ec12f435e839f87`; replacing the recovery receipt is not silently accepted.

## Files

- `pcm_binding.py`: binding and descriptor API
- `test_pcm_binding.py`: synthetic failure, conversion, identity, and immutability tests
- `verify_recovered.py`: pinned all-16 verification without inference or audio output
- `decoder-inputs.json`: separate opaque-ID decoder manifest
- `verification-report.json`: concise all-16 outcome and unrun-stage receipt
- `tests.log`, `actual-verification.log`: observed test/verification output
