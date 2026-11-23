"""Apply the predeclared B14 positive-preservation and false-event conditions."""
import json
from pathlib import Path

ARMS = ("B0", "B9", "control", "challenger")
GAINS = ("1.0", "0.75", "0.5")
ASSESSMENT = Path(__file__).with_name("english_v6_b14_panels") / "assessment.json"


def decide(rows):
    design = json.loads(ASSESSMENT.read_bytes())
    positives = [row for row in rows if row["positive"]]
    ordinary = [row for row in rows if not row["positive"]]
    readers = [row for row in ordinary if row["cohort"] == "natural"]

    def events(row, gain, arm):
        return row["gains"][gain][arm]["events"]

    restored = []
    lost = []
    extra = []
    false_additions = []
    for row in positives:
        for gain in GAINS:
            observed = {arm: events(row, gain, arm) for arm in ARMS}
            if observed["B9"] and not observed["control"] and observed["challenger"]:
                restored.append([row["id"], row["group"], gain])
            if not observed["challenger"]:
                lost.extend([row["id"], gain, ref] for ref in ("B0", "control") if observed[ref])
            for ref in ("B9", "control"):
                if max(0, observed["challenger"] - 1) > max(0, observed[ref] - 1):
                    extra.append([row["id"], gain, ref])
    for row in ordinary:
        for gain in GAINS:
            for ref in ("B9", "control"):
                if events(row, gain, "challenger") > events(row, gain, ref):
                    false_additions.append([row["id"], gain, ref])
    summary = {}
    for name, cohort in (("natural_ordinary", readers), ("synthetic_ordinary", [r for r in ordinary if r["cohort"] == "synthetic"]), ("synthetic_positive", positives)):
        summary[name] = {}
        for arm in ARMS:
            summary[name][arm] = {}
            for gain in GAINS:
                values = [events(row, gain, arm) for row in cohort]
                summary[name][arm][gain] = dict(originals=len(values), with_event=sum(v > 0 for v in values), events=sum(values))
    improved = []
    for group in sorted({row["group"] for row in readers}):
        subset = [row for row in readers if row["group"] == group]
        if sum(events(row, "1.0", "challenger") for row in subset) < sum(events(row, "1.0", "B9") for row in subset):
            improved.append(group)
    natural = summary["natural_ordinary"]
    exceeds_b0 = [gain for gain in GAINS if natural["challenger"][gain]["events"] > natural["B0"][gain]["events"]]
    passed = (len({item[1] for item in restored}) >= design["minimum_recovered_positive_families"]
              and len(improved) >= design["minimum_improved_readers_vs_B9"]
              and not (lost or extra or false_additions or exceeds_b0))
    return dict(summary=summary, restored_positives=restored, lost_positives=lost,
                extra_positive_events=extra, added_ordinary_events=false_additions,
                natural_readers_improved_vs_B9=improved, natural_above_B0_gains=exceeds_b0,
                panel_gate="PASS" if passed else "NO_PROMOTION")
