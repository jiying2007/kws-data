"""B24 identities around unchanged strict five-observer execution and readback."""
from __future__ import annotations
import english_v6_b22_source_observe as original
from english_v6_b22_replication_reuse import inherit
inherit(original, globals())
import english_v6_b24_source_ancestry as lineage
SOURCE_SCHEMA = 'english-v6-b24-fresh-combined-source-v1'
BATCH = 'english-v6-b24-120-original-strict-source-screen-v1'


def source_check(repository=None):
    frozen = primary_observer.source_check(repository)
    scientific = {k: v for k, v in frozen.items() if k not in ("schema", "observer_sha256", "source_indices")}
    need(scientific == lineage.read_plan()["observer_scientific_contract"],
         "unchanged native observers, raw parsers, source admission and runtime versions")
    return dict(scientific, schema="english-v6-b24-five-observer-contract-v1",
        observer_sha256=sha(Path(__file__).read_bytes()),
        source_indices=lineage.read_plan()["primary"]["synthetic"]["speaker_indices"])

def validate_rows(rows, source_sha, registration):
    count = registration["source_rows"]
    need(source_sha == registration["source_manifest_sha256"] and 96 <= count <= MAX_COUNT
         and len(rows) == len({r["id"] for r in rows}) == len({r["pcm_sha256"] for r in rows}) == count
         and all(Path(r["id"]).name == r["id"] and r["id"].startswith("b24-primary-")
                 and r["role"] == "development" and r["training_allowed"] is False
                 and r["observed_class"] is None for r in rows), "every original eligible source exactly once")
    synthetic = [r for r in rows if r["cohort"] == "synthetic"]
    natural = [r for r in rows if r["cohort"] == "natural"]
    need(len(synthetic) == 96 and len(natural) == count - 96
         and {r["speaker_index"] for r in synthetic} == set(lineage.read_plan()["primary"]["synthetic"]["speaker_indices"]), "all96 synthetic and only retained natural originals")
    for index in lineage.read_plan()["primary"]["synthetic"]["speaker_indices"]:
        family = [r for r in synthetic if r["speaker_index"] == index]
        need(len(family) == 4 and {r["cell_id"]: r["length_scale"] for r in family}
             == {"mero-fast": 1.4, "mero-slow": 1.8, "question": 1.0, "request": 1.08}
             and all(r["requested_class"] == (2 if r["cell_id"].startswith("mero-") else 0) for r in family),
             "unchanged positive rate pair and ordinary sentences")
    need(all(r["requested_class"] == 0 and r["whole_original"] is True for r in natural), "whole unmodified natural negatives")

def expected_registration(source, supplied=None):
    lineage.read_plan()
    import english_v6_b24_source as owner
    expected = owner.registration(Path(source))
    need(supplied is None or supplied == expected, "source-authorized registration, never provider self-report")
    return expected

def require_original_rows(source, rows):
    import english_v6_b24_source as owner
    need(rows == owner.read_source(Path(source)),
         "every supplied row equals the authenticated full original source, in registered order")
