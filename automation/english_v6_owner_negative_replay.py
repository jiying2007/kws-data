"""Replay only the nine specifically owner-reviewed natural TRAIN intervals."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_four_model_panel as scorer

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "research/english_hardneg_increment_20261117/OWNER_REVIEW.json"
SPEC = ROOT / "automation/english_v6_owner_negative_replay.json"


def load_json(path):
    return json.loads(path.read_bytes())


def check(test, message):
    if not test:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def originals(archive, spec):
    check(digest(AUDIT.read_bytes()) == spec["review_sha256"], "owner review identity")
    review = load_json(AUDIT)
    check(review["owner_statement_verbatim"] == "没有唤醒词" and not review["full_original_utterance_labels_asserted"]
          and review["reviewed_negative_samples"] == 488880 and review["candidate_count"] == 9
          and not review["independent_test_admitted"], "owner review scope")
    eligible = set(spec["eligible_speakers"])
    rows = review["candidates"]
    check(len(rows) == len({r["candidate_id"] for r in rows}) == 9
          and len({r["recording_id"] for r in rows}) == 8, "reviewed intervals")
    for r in rows:
        if r["speaker_id"] not in eligible:
            continue
        check(r["source_role"] == "train" and r["research_training_admitted"] and r["trusted_negative"]
              and r["review_decision"] == "N" and "Hey Mero" in r["negative_targets"], "reviewed research TRAIN")
        stem = r["candidate_id"]
        wav = archive.read("candidates/wav/" + stem + ".wav")
        pcm = archive.read("candidates/pcm/" + stem + ".pcm")
        check(digest(wav) == r["wav_sha256"] and digest(pcm) == r["window_pcm_sha256"]
              and len(pcm) // 2 == r["window_samples"] == r["end_sample_exclusive"] - r["start_sample"],
              "exact reviewed window SHA and boundaries")
        with wave.open(io.BytesIO(wav)) as stream:
            check(stream.getparams()[:3] == (1, 2, 16000) and stream.readframes(stream.getnframes()) == pcm,
                  "whole approved PCM16 mono, 16 kHz")
        yield r, pcm


def models(inputs, torch, np):
    nets = {}
    for trial, name in (("B11", "english_v6_four_model_panel_run.json"),
                        ("B12", "english_v6_b12_four_model_panel_run.json")):
        four = scorer.snapshots(inputs / trial, load_json(ROOT / "automation" / name), torch, np)
        for k, v in four.items():
            label = k if k in ("B0", "B9") else trial + "_" + k
            if label in nets:
                check(nets[label][-1] == v[-1], "base checkpoint identity")
            else:
                nets[label] = v
    return nets


def decode_frames(logits):
    path = logits.argmax(axis=1).tolist()
    onsets = [(i, x) for i, x in enumerate(path) if x and (i == 0 or path[i-1] != x)]
    tokens = [x for _, x in onsets]
    events = [onsets[i+2][0] for i in range(len(tokens)-2) if tokens[i:i+3] == [1, 2, 3]]
    return tokens, events


def run(inputs, output):
    import numpy as np
    import torch
    spec = load_json(SPEC)
    torch.set_num_threads(1)
    nets = models(inputs, torch, np)
    arrays = {}
    records = []
    with scorer.opened(inputs / "reviewed.zip", spec["inputs"]["reviewed"]) as archive, torch.inference_mode():
        for r, pcm in originals(archive, spec):
            conditions = {}
            for label, (net, bank, mean, scale, pt) in nets.items():
                feats = scorer.b9.v5.features(pcm, {"samples": r["window_samples"]}, torch, np, bank)
                y = net(torch.from_numpy(((feats - mean) / scale).astype(np.float32)).unsqueeze(0))
                logits = y[0].cpu().numpy().astype("<f4")
                check(logits.shape == (1 + (r["window_samples"] - 512)//160, 4) and np.isfinite(logits).all(),
                      "native frame geometry")
                tokens, events = decode_frames(logits)
                old = scorer.b9.v5.ctc_greedy(y, torch)
                check(tokens == old["collapsed_tokens"] and len(events) == int(old["mero_event"]) + old["extra_events"],
                      "native decoder agreement")
                key = r["candidate_id"] + "__" + label
                arrays[key] = logits
                conditions[label] = dict(key=key, tokens=tokens, event_frames=events,
                                         available_after_sample=[512 + 160*i for i in events],
                                         frame_count=len(logits), logits_sha256=digest(logits.tobytes()))
            records.append(dict(candidate_id=r["candidate_id"], recording_id=r["recording_id"], speaker_id=r["speaker_id"],
                                samples=r["window_samples"], source=r["source_id"], start_sample=r["start_sample"],
                                end_sample_exclusive=r["end_sample_exclusive"], pcm_sha256=r["window_pcm_sha256"], models=conditions))
    check(records and len(arrays) == len(records)*6, "complete nonempty cohort")
    summary = {}
    for label in nets:
        fired = [r for r in records if r["models"][label]["event_frames"]]
        summary[label] = {"windows": len(fired), "sources": len({r["recording_id"] for r in fired}),
                          "speakers": sorted({r["speaker_id"] for r in fired}),
                          "events": sum(len(r["models"][label]["event_frames"]) for r in fired)}
    output.mkdir()
    frames = output/"TRAIN_FRAME_LOGITS.npz"
    np.savez_compressed(frames, **arrays)
    report = {"schema":"english-v6-owner-negative-window-replay-v1","scope":"exposed research TRAIN",
              "records":records,"summary":summary,"actual_forwards":len(arrays),"gain":1.0,
              "model_pt_sha256":{k:v[-1] for k,v in nets.items()},"all_logits_sha256":digest(frames.read_bytes()),
              "optimizer_updates":0,"full_utterance_model_forwards":0,"dev_cal_locked_forwards":0,
              "field_far_per_hour":None,"promotion_allowed":False,
              "two_group_hardness_diagnostic":len(summary["B9"]["speakers"]) >= 2}
    (output/"RESULT.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.inputs, args.output)
