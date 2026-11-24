"""Offline permission/ancestry fixtures only: no audio, downloads or model calls."""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import io
import json
from pathlib import Path
import struct
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b17_replication_gate as gate
import english_v6_b17_replication_history as chain
import english_v6_b17_replication_source as synthetic
import english_v6_b17_replication_natural_sources as natural


@contextmanager
def closed_opening():
    """Test the closed state without preventing a later evidence-only activation."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "UNIT_TEST_CLOSED_OPENING.json"
        path.write_text('{"enabled": false}\n')
        with patch.object(gate, "OPENING", path):
            yield


def unit_npy(labels):
    """Small explicitly synthetic numeric fixture; never a scored audio input."""
    body = b"".join(struct.pack("<4f", *(2.0 if i == label else 0.0 for i in range(4))) for label in labels)
    header = repr(dict(descr="<f4", fortran_order=False, shape=(len(labels), 4))).encode()
    header += b" " * ((64 - ((10 + len(header) + 1) % 64)) % 64) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + body, body


def zip_bytes(members):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    return out.getvalue()


def unit_primary(improved=2):
    """A 48-row saved-array parser fixture, never authorization or model evidence."""
    selected = [r for r in synthetic.primary.source_plan("primary")["rows"]
                if r["speaker_index"] < 730 and r["cell_id"] != "mero-slow"]
    rows = [dict(id=r["id"], group=r["source_group_id"], cell=r["cell_id"], cohort="synthetic",
                 positive=r["requested_class"] == 2, tier="silver", pcm_sha256=gate.sha(("fixture:" + r["id"]).encode()))
            for r in selected]
    people = synthetic.primary.lineage.read_plan()[0]["primary"]["natural"]["speakers"]
    rows.extend(dict(id=f"b17-natural-{sp}-100-0000", group="librispeech-test-clean-" + sp,
                     cell="read-speech", cohort="natural", positive=False, tier="silver",
                     pcm_sha256=gate.sha(("fixture:" + sp).encode())) for sp in people)
    arrays = {}
    for row in rows:
        row["gains"] = {}
        for gain in gate.GAINS:
            row["gains"][gain] = {}
            for arm in gate.ARMS:
                event = bool(row["positive"] and not (arm == "control" and row is rows[0]))
                if row["cohort"] == "natural":
                    event = arm == "B9" and row["group"].removeprefix("librispeech-test-clean-") in people[:improved]
                raw, body = unit_npy([0, 1, 2, 3] if event else [0, 0, 0, 0])
                key = row["id"] + "__" + gain + "__" + arm
                arrays[key + ".npy"] = raw
                row["gains"][gain][arm] = dict(events=int(event), tokens=[1, 2, 3] if event else [],
                    frames=4, event_frames=[3] if event else [], array=key, sha256=gate.sha(body))
    models = dict(B0=gate.B0_PT, B9=gate.B9_PT, control="c" * 64, challenger="d" * 64)
    result = dict(schema=gate.PRIMARY_SCHEMA, assessment_sha256=gate.ASSESSMENT_SHA,
        training_design_sha256=gate.DESIGN_SHA, source_gate="SOURCE_GATE_PASS", panel_gate="PASS",
        replication_open_allowed=True, optimizer_updates=0, installed_software_baseline="B0",
        threshold_search=False, promotion_allowed=False, product_released=False, B14_replication_opened=False,
        B15_replication_opened=False, B16_replication_opened=False, ssc305_physical=False,
        model_pt_sha256=models, source_shortfall={}, records=rows, actual_forwards=576,
        source_coverage=dict(synthetic_positive=12, synthetic_ordinary=24, synthetic_paired=12,
                             natural_originals=12, natural_speakers=12))
    result.update(gate.decision.decide(rows))
    result["panel_gate"] = "PASS"  # A forged PASS on a failing fixture must still be rejected.
    return result, arrays, models


def package(result, arrays, models):
    result = deepcopy(result)
    saved = zip_bytes(arrays)
    result["logits_sha256"] = gate.sha(saved)
    raw = zip_bytes({"RESULT.json": json.dumps(result).encode(), "ALL_FRAME_LOGITS.npz": saved})
    return raw, dict(primary_result=dict(bytes=len(raw), sha256=gate.sha(raw)), model_pt_sha256=models)


class ReplicationContracts(unittest.TestCase):
    def test_original_quality_code_and_assessment_are_hash_pinned(self):
        self.assertEqual(gate.sha(Path(gate.decision.__file__).read_bytes()), gate.DECISION_SHA)
        self.assertEqual(gate.sha(gate.decision.ASSESSMENT.read_bytes()), gate.QUALITY_SHA)
        self.assertEqual(gate.sha(Path(synthetic.primary.__file__).with_name("english_v6_b17_primary_eval.py").read_bytes()),
                         gate.PRIMARY_EVAL_SHA)

    def test_inert_lock_precedes_any_github_or_plan_access(self):
        with closed_opening(), patch.object(gate, "gh", side_effect=AssertionError("no network")) as gh, \
             patch.object(gate.ancestry, "read_plan", side_effect=AssertionError("closed before plan access")):
            with self.assertRaisesRegex(ValueError, "sealed"):
                gate.fetch(Path("must-not-exist"))
            gh.assert_not_called()

    def test_source_event_requires_first_public_exact_parent_push(self):
        branch = "refs/heads/research/en-v6-b17-synthetic-replication-once-20261124"
        parent, head = "a" * 40, "b" * 40
        env = dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY=gate.REPO, GITHUB_EVENT_NAME="push",
                   GITHUB_REF=branch, GITHUB_RUN_ATTEMPT="1", GITHUB_SHA=head)
        event = dict(before=parent, after=head, ref=branch, forced=False, deleted=False,
                     repository=dict(full_name=gate.REPO, private=False))
        gate.execution_guard(parent, branch, env, event, parent)
        for wrong_env, wrong_event, actual_parent in (
            ({**env, "GITHUB_RUN_ATTEMPT": "2"}, event, parent),
            (env, {**event, "forced": True}, parent),
            (env, {**event, "repository": dict(full_name=gate.REPO, private=True)}, parent),
            (env, event, "c" * 40)):
            with self.assertRaises(ValueError):
                gate.execution_guard(parent, branch, wrong_env, wrong_event, actual_parent)
        with self.assertRaisesRegex(ValueError, "placeholder"):
            gate.execution_guard("__INERT__", branch, env, event, parent)

    def test_failed_or_retried_primary_cannot_download_original_zip(self):
        pin = dict(id=1, run_id=2, run_head="a" * 40, bytes=3, sha256="b" * 64)
        lock = dict(primary_result=pin, required_primary_branch="unit-test-fixture",
                    required_primary_workflow="unit-test-fixture.yml")
        artifact = dict(id=1, expired=False, size_in_bytes=3, digest="sha256:" + "b" * 64,
                        workflow_run=dict(id=2, head_sha="a" * 40))
        run = dict(status="completed", conclusion="success", run_attempt=1, head_sha="a" * 40,
                   event="push", head_branch="unit-test-fixture", path="unit-test-fixture.yml",
                   repository=dict(full_name=gate.REPO, private=False))
        for changes in (dict(conclusion="failure"), dict(run_attempt=2), dict(head_sha="c" * 40)):
            with patch.object(gate, "opening", return_value=(lock, "d" * 64)), \
                 patch.object(gate, "gh", side_effect=[json.dumps({**run, **changes}).encode(),
                                                       json.dumps(artifact).encode()]) as remote:
                with self.assertRaisesRegex(ValueError, "successful first public"):
                    gate.fetch(Path("must-not-exist"))
                self.assertEqual(remote.call_count, 2)
                self.assertFalse(any(call.args[0].endswith("/zip") for call in remote.call_args_list))

    def test_all_source_entrypoints_are_closed_before_io_or_models(self):
        missing = Path("missing-unit-test-only")
        calls = [lambda: synthetic.required_archives(missing), lambda: synthetic.history(missing, missing),
                 lambda: synthetic.generate(missing, missing, missing), lambda: synthetic.read_source(missing, missing),
                 lambda: synthetic.observe("qwen", missing, missing, missing),
                 lambda: synthetic.screen(missing, missing, missing, missing),
                 lambda: natural.required_archives(missing), lambda: natural.history(missing, missing),
                 lambda: natural.pick(None, None, missing), lambda: natural.verify_corpus(missing, missing),
                 lambda: natural.extract(missing, missing, missing, missing), lambda: natural.read_source(missing, missing),
                 lambda: natural.observe("qwen", missing, missing, missing),
                 lambda: natural.screen(missing, missing, missing, missing)]
        with closed_opening(), patch.object(synthetic.B13, "render_panel", side_effect=AssertionError("no TTS")), \
             patch.object(synthetic.qa.observers, "observe", side_effect=AssertionError("no observer model")), \
             patch.object(synthetic.qa.pcm_tools, "inspect_wav", side_effect=AssertionError("no audio")):
            for i, call in enumerate(calls):
                with self.subTest(boundary=i), self.assertRaisesRegex(ValueError, "sealed"):
                    call()

    def test_frozen_replication_geometry_and_primary_reader_exclusion(self):
        plan = synthetic.source_plan()
        self.assertEqual(len(plan["rows"]), 96)
        self.assertEqual({r["speaker_index"] for r in plan["rows"]}, set(range(742, 766)))
        self.assertEqual(plan["frozen_source_gate"],
                         dict(positive_originals_min=12, ordinary_originals_min=24, paired_source_groups_min=8))
        design = natural.plan()
        self.assertEqual(design["source_slots"], 18)
        self.assertEqual(design["speakers"], ["5849", "8173", "6599", "6455", "1686", "1585", "6267", "4831", "7641"])
        primary_people = gate.ancestry.read_plan()[0]["primary"]["natural"]["speakers"]
        self.assertLessEqual(set(primary_people), set(design["excluded_speakers"]))
        self.assertFalse(set(design["speakers"]) & set(design["excluded_speakers"]))

    def test_npy_parser_recomputes_ctc_blank_and_repeat_boundaries(self):
        raw, body = unit_npy([0, 1, 1, 0, 1, 2, 3, 3, 0, 1, 2, 3])
        self.assertEqual(gate.read_npy(raw), (body, [1, 1, 2, 3, 1, 2, 3], [6, 11], 12))

    def test_npy_parser_rejects_nonfinite_or_incomplete_saved_array(self):
        raw, body = unit_npy([0, 1, 2, 3])
        with self.assertRaisesRegex(ValueError, "frame bytes"):
            gate.read_npy(raw[:-1])
        with self.assertRaisesRegex(ValueError, "finite"):
            gate.read_npy(raw[:-len(body)] + struct.pack("<f", float("nan")) + body[4:])

    def test_saved_array_parser_positive_fixture_without_inference(self):
        result, arrays, models = unit_primary()
        raw, lock = package(result, arrays, models)
        proof = gate.verify_primary(raw, lock)
        self.assertEqual(proof["primary_quality"], "PASS")
        self.assertEqual(proof["actual_primary_forwards"], 576)

    def test_forged_pass_fails_original_b14_quality_recomputation(self):
        result, arrays, models = unit_primary(improved=1)
        raw, lock = package(result, arrays, models)
        with self.assertRaisesRegex(ValueError, "B14 stringent quality"):
            gate.verify_primary(raw, lock)

    def test_saved_event_claim_must_equal_all_original_frame_bytes(self):
        result, arrays, models = unit_primary()
        result["records"][0]["gains"]["1.0"]["challenger"]["events"] = 0
        raw, lock = package(result, arrays, models)
        with self.assertRaisesRegex(ValueError, "CTC event/frame"):
            gate.verify_primary(raw, lock)

    def test_orphan_saved_array_and_model_identity_change_both_rejected(self):
        result, arrays, models = unit_primary()
        arrays["unrelated.npy"] = unit_npy([0])[0]
        raw, lock = package(result, arrays, models)
        with self.assertRaisesRegex(ValueError, "saved primary frame arrays"):
            gate.verify_primary(raw, lock)
        result, arrays, models = unit_primary()
        result["model_pt_sha256"]["B0"] = "e" * 64
        raw, lock = package(result, arrays, {**models, "B0": gate.B0_PT})
        with self.assertRaisesRegex(ValueError, "genuine completed B17 primary"):
            gate.verify_primary(raw, lock)

    def transcript_fixture(self, missing=False):
        design = deepcopy(natural.plan())
        metadata = "".join(f"{sp} | M | dev-other | 1.0 | Unit fixture\n" for sp in design["speakers"]).encode()
        design["metadata_speakers_sha256"] = gate.sha(metadata)
        out = io.BytesIO()
        with tarfile.open(fileobj=out, mode="w") as archive:
            def add(name, raw):
                member = tarfile.TarInfo(name); member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))
            add("LibriSpeech/SPEAKERS.TXT", metadata)
            for speaker in design["speakers"]:
                count = 1 if missing and speaker == design["speakers"][0] else 3
                lines = [f"{speaker}-100-{i:04d} PLEASE READ THIS ORDINARY SENTENCE\n" for i in range(count)]
                add(f"LibriSpeech/dev-other/{speaker}/100/{speaker}-100.trans.txt", "".join(lines).encode())
        out.seek(0)
        return out, design

    def test_natural_eighteen_slots_are_deterministic_without_audio(self):
        raw, design = self.transcript_fixture()
        with patch.object(gate, "require", return_value={}), tarfile.open(fileobj=raw) as archive:
            roster = natural.pick(archive, design, None)
            self.assertEqual(len(roster["replication"]), 18)
            self.assertEqual([r["speaker"] for r in roster["replication"]], [s for s in design["speakers"] for _ in range(2)])
            self.assertEqual(roster, natural.pick(archive, design, None))

    def test_no_natural_reader_or_utterance_replacement(self):
        raw, design = self.transcript_fixture(missing=True)
        with patch.object(gate, "require", return_value={}), tarfile.open(fileobj=raw) as archive:
            with self.assertRaisesRegex(ValueError, "STOP without replacement"):
                natural.pick(archive, design, None)

    def source_fixture(self, directory, change):
        permit = dict(primary_result_sha256="a" * 64, opening_lock_sha256="b" * 64,
                      primary_source_inputs={"synth_source": {}, "natural_source": {}})
        design = natural.plan()
        roster = [dict(speaker=sp, source_id=f"{sp}-100-{i:04d}", chapter="100",
                       transcript="PLEASE READ THIS ORDINARY SENTENCE", transcript_file_sha256="0" * 64)
                  for sp in design["speakers"] for i in range(2)]
        rows = [dict(row, pcm_sha256=gate.sha(row["source_id"].encode()),
                     source_group_id="librispeech-dev-other-" + row["speaker"]) for row in roster]
        change(roster, rows)
        raw = natural.roster_bytes({"replication": roster})
        (directory / "ROSTERS.json").write_bytes(raw)
        meta = dict(schema=natural.SCHEMA, panel="replication", **chain.binding(permit),
                    corpus_split=natural.CORPUS_SPLIT, corpus_sha256=design["upstream"]["sha256"],
                    design_sha256=gate.ASSESSMENT_SHA, rosters_sha256=gate.sha(raw),
                    selected_original_slots=18, selected_speakers=9, kws_forwards=0,
                    metadata_speakers_sha256=design["metadata_speakers_sha256"],
                    training_allowed=False, human_gold=False, all_prior_replication_sealed=True,
                    source_rejections=[], rows=rows)
        (directory / "MANIFEST.json").write_text(json.dumps(meta))
        return permit

    def test_old_reader_cannot_hide_behind_replica_group_prefix(self):
        def change(roster, rows):
            for row in (roster[0], rows[0]):
                row.update(speaker="1462", source_id="1462-100-0000", source_group_id="librispeech-dev-other-1462")
        with tempfile.TemporaryDirectory() as tmp:
            permit = self.source_fixture(Path(tmp), change)
            with patch.object(gate, "require", return_value=permit), self.assertRaisesRegex(ValueError, "nine readers"):
                natural.read_source(Path(tmp), None)

    def test_duplicate_or_missing_natural_slot_stops_before_original_audio(self):
        def duplicate(roster, rows):
            roster[1], rows[1] = dict(roster[0]), dict(rows[0])
        for change in (duplicate, lambda roster, rows: rows.pop()):
            with tempfile.TemporaryDirectory() as tmp:
                permit = self.source_fixture(Path(tmp), change)
                with patch.object(gate, "require", return_value=permit), \
                     patch.object(natural.qa.pcm_tools, "inspect_wav", side_effect=AssertionError("no audio")), \
                     self.assertRaisesRegex(ValueError, "nine readers|every original slot"):
                    natural.read_source(Path(tmp), None)


if __name__ == "__main__":
    unittest.main()
