"""Verify the frozen 16-file recovery using stdlib only; never print PCM/labels.

Writes label-free descriptors and a concise verification receipt beside this
script. The source mapping stays in the already existing recovery JSON only.
"""
from decimal import Decimal
import json
import os
from pathlib import Path
import stat
import struct

import pcm_binding as p

MANIFEST_NAME = "recovery-verification.json"
MANIFEST_BYTES = 3082
MANIFEST_SHA256 = "8a162581ed88bb933663f27aafe0cc1721d9119222d05ac46ec12f435e839f87"


def read_source_manifest(root):
    """Bounded regular, nofollow read, pinned to the inspected recovery receipt."""
    root_fd = p._open_root(root)
    fd = None
    try:
        fd = os.open(MANIFEST_NAME, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                     dir_fd=root_fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != MANIFEST_BYTES:
            raise p.BindingError("recovery metadata type/size/alias mismatch")
        data = bytearray()
        while len(data) <= MANIFEST_BYTES:
            chunk = os.read(fd, MANIFEST_BYTES + 1 - len(data))
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(fd)
        named = os.stat(MANIFEST_NAME, dir_fd=root_fd, follow_symlinks=False)
        if p._stamp(before) != p._stamp(after) or p._stamp(after) != p._stamp(named):
            raise p.BindingError("recovery metadata changed during read")
        if len(data) != MANIFEST_BYTES or p.sha256(data) != MANIFEST_SHA256:
            raise p.BindingError("recovery metadata SHA256 mismatch")
        return json.loads(data)
    finally:
        if fd is not None:
            os.close(fd)
        os.close(root_fd)


def main():
    here = Path(__file__).absolute().parent
    root = str(here.parent.parent / "kws-calibration-recovered")
    source = read_source_manifest(root)
    if source["wav_count"] != 16 or len(source["files"]) != 16:
        raise p.BindingError("expected exactly 16 recovery entries")
    expectations = []
    for index, entry in enumerate(source["files"], 1):
        frames = Decimal(str(entry["duration_s"])) * p.SAMPLE_RATE
        if frames != frames.to_integral_value():
            raise p.BindingError("source duration is not an integral frame count")
        expectations.append(p.WaveExpectation(
            opaque_id=f"clip-{index:04d}", basename=entry["file"],
            file_sha256=entry["sha256"], file_bytes=entry["bytes"], frame_count=int(frames)))
    bindings = p.bind_batch(root, expectations)
    # Independent full-sample reversibility check, without printing samples.
    for bound in bindings:
        for (raw,), (converted,) in zip(struct.iter_unpack("<h", bound.pcm16_le),
                                        struct.iter_unpack("<f", bound.pcm_float32_le)):
            if converted * 32768 != raw:
                raise p.BindingError("conversion was not lossless/exact")
    total_frames = sum(e.frame_count for e in expectations)
    if Decimal(str(source["total_seconds"])) * p.SAMPLE_RATE != total_frames:
        raise p.BindingError("aggregate source duration mismatch")
    manifest = p.decoder_manifest(bindings)
    canonical = p._canonical(manifest)
    (here / "decoder-inputs.json").write_bytes(canonical + b"\n")
    report = {
        "schema": "asr-pcm-verification-v1", "status": "PASS", "input_count": len(bindings),
        "source_manifest_sha256": MANIFEST_SHA256,
        "source_manifest_bytes": MANIFEST_BYTES,
        "wav_total_bytes": sum(e.file_bytes for e in expectations),
        "total_frames": total_frames, "sample_rate_hz": p.SAMPLE_RATE,
        "total_seconds_exact_decimal": str(Decimal(total_frames) / p.SAMPLE_RATE),
        "pcm16_total_bytes": sum(len(b.pcm16_le) for b in bindings),
        "float32_total_bytes": sum(len(b.pcm_float32_le) for b in bindings),
        "decoder_manifest_canonical_sha256": p.sha256(canonical),
        "decoder_manifest_file_sha256": p.sha256(canonical + b"\n"),
        "verified": ["predeclared_file_sha256_and_size", "exact_canonical_header",
                     "16000_hz_mono_pcm16", "predeclared_frame_counts", "full_sample_order",
                     "exact_float32_conversion_all_samples", "raw_and_float_descriptor_hashes",
                     "immutable_return_bytes", "no_labels_in_decoder_manifest"],
        "not_run": ["numpy_array_adapter_in_actual_ASR_environment", "ASR_runtime_imports",
                    "model_downloads", "inference", "calibration"],
    }
    (here / "verification-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
