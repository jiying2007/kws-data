"""Score one newly registered B14 panel with frozen models and whole source PCM."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b14_natural_sources as natural
from english_v6_b13_independent_eval import human_rows
from english_v6_b14_decision import ASSESSMENT, decide
import english_v6_four_model_panel as base


DECISION_COMMIT = "d0e33039d950e159e14145f8978406180bf45828"
DECISION_SHA = "944f5df3c7bc3244977a8a33f7e1c393bffff9f7f5c32d2d8cc79e2628e265d9"


def run(spec_path, inputs, output):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    spec = json.loads(spec_path.read_bytes())
    panel = spec["panel"]
    base.need(panel in ("primary", "replication") and spec["hypothesis"] == "B14"
              and spec["design_commit"] == DECISION_COMMIT, "registered B14 assessment")
    base.need(base.sha(ASSESSMENT.read_bytes()) == DECISION_SHA, "frozen criteria")
    design = json.loads(ASSESSMENT.read_bytes())
    plan_bytes = (ASSESSMENT.parent / (panel + ".json")).read_bytes()
    plan_sha = design["synthetic_plan_sha256"][panel]
    base.need(base.sha(plan_bytes) == plan_sha, "registered prospective synthetic roster")
    output.mkdir()
    with ExitStack() as stack:
        opened = {k:stack.enter_context(base.opened(inputs / (k + ".zip"), spec["inputs"][k]))
                  for k in ("synth_source", "synth_screen", "natural_source", "natural_screen")}
        synth_manifest = json.loads(opened["synth_source"].read("MANIFEST.json"))
        base.need(synth_manifest["registered_plan_file_sha256"] == plan_sha, "authentic synthetic panel")
        screen = json.loads(opened["synth_screen"].read("SCREEN.json"))
        selected = [r for r in screen["rows"] if r["tier"] in ("silver", "weak")]
        positives = [r for r in selected if r["observed_class"] == 2]
        ordinary = [r for r in selected if r["observed_class"] == 0]
        humans = human_rows(opened["natural_source"], opened["natural_screen"], panel, contract=natural)
        coverage = dict(synthetic_min_paired_families=len(screen["paired_source_groups"]),
                        synthetic_min_positive_sources=len({r["source_group_id"] for r in positives}),
                        synthetic_min_positive_originals=len(positives), synthetic_min_ordinary_originals=len(ordinary),
                        synthetic_min_ordinary_forms=len({r["requested_text"] for r in ordinary}),
                        natural_min_admitted_speakers=len({r["source_group_id"] for r in humans}),
                        natural_min_admitted_originals=len(humans))
        missing = {k:[n,design["source_gate"][k]] for k,n in coverage.items() if n < design["source_gate"][k]}
        report = dict(schema="english-v6-b14-recovery-decision-v1", trial=spec, source_coverage=coverage,
                      source_shortfall=missing, actual_forwards=0, optimizer_updates=0, promotion_allowed=False,
                      threshold_search=False, field_far_per_hour=None, physical_board_measured=False)
        if missing:
            report["panel_gate"] = "INSUFFICIENT_SOURCE"
        else:
            synthetic, _, _ = base.select(opened["synth_source"], opened["synth_screen"], panel, json.loads(plan_bytes))
            nets = base.snapshots(inputs, spec, torch, np)
            records = []
            arrays = {}
            for source, cohort, cohort_rows in (("synth_source","synthetic",synthetic), ("natural_source","natural",humans)):
                for row in cohort_rows:
                    gains = base.infer(opened[source].read("wav/" + row["id"] + ".wav"), row, nets, arrays, torch, np)
                    records.append(dict(id=row["id"], group=row["source_group_id"], cohort=cohort, cell=row["cell_id"],
                                        positive=row["observed_class"] == 2, tier=row["tier"], pcm_sha256=row["pcm_sha256"],
                                        gains=gains))
            base.need(len({r["pcm_sha256"] for r in records}) == len(records) and len(arrays) == 12 * len(records),
                      "unique PCM and complete model/gain grid")
            path = output / "ALL_FRAME_LOGITS.npz"
            np.savez_compressed(path, **arrays)
            report.update(decide(records), records=records, actual_forwards=len(arrays),
                          model_pt_sha256={m:v[-1] for m,v in nets.items()}, logits_sha256=base.sha(path.read_bytes()))
    (output / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    run(*map(Path, sys.argv[1:]))
