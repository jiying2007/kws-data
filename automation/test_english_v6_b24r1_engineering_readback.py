"""Invented saved-evidence mutation contracts; no model, compiler or network."""
from __future__ import annotations
import copy
from contextlib import nullcontext
import importlib.util
import io
import json
from pathlib import Path
import re
import stat
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_v6_b24r1_engineering_readback as reader
import english_v6_b24r1_engineering as engineering


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def pin(raw):
    return dict(bytes=len(raw), sha256=reader.sha(raw))


def zipped(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    return stream.getvalue()


def lock_fixture():
    value = dict(schema=reader.LOCK_SCHEMA, repository=reader.REPO,
        artifacts={role: dict(id=100 + index, run_id=1234, run_head="a" * 40, bytes=200, sha256="b" * 64)
                   for index, role in enumerate(sorted(reader.ROLES))},
        api={role: dict(bytes=10, sha256="c" * 64) for role in reader.API_ROLES},
        **{key: "d" * 64 for key in reader.HASH_FIELDS})
    value.update(evaluator_sha256=reader.ENGINEERING_SHA, quality_reader_sha256=reader.QUALITY_READER_SHA)
    return value


def api_fixture():
    lock = lock_fixture()
    run = dict(id=1234, head_sha="a" * 40, status="completed", conclusion="success", run_attempt=1,
        event="push", repository=dict(full_name=reader.REPO, private=False), head_branch=reader.BRANCH, path=reader.WORKFLOW)
    names = dict(result="english-v6-b24r1-software-sdk-evidence-1234", producer="english-v6-b24r1-engineering-producer-1234",
        consumed=reader.MARKER, input_proof="english-v6-b24r1-engineering-input-proof-1234")
    artifacts = [dict(id=value["id"], size_in_bytes=value["bytes"], digest="sha256:" + value["sha256"],
        name=names[role], expired=False, workflow_run=dict(id=1234, head_sha="a" * 40)) for role, value in lock["artifacts"].items()]
    docs = dict(run=run, commit=dict(sha="a" * 40, parents=[dict(sha="e" * 40)],
        files=[dict(filename=reader.WORKFLOW)], commit=dict(tree=dict(sha="f" * 40))),
        tree=dict(sha="f" * 40, truncated=False, tree=[]), artifacts=dict(total_count=4, artifacts=artifacts),
        runs=dict(total_count=1, workflow_runs=[copy.deepcopy(run)]),
        markers=dict(total_count=1, artifacts=[copy.deepcopy(next(a for a in artifacts if a["name"] == reader.MARKER))]),
        pipeline_commit=dict(sha=reader.PIPELINE_COMMIT, commit=dict(tree=dict(sha="9" * 40))),
        pipeline_tree=dict(sha="9" * 40, truncated=False, tree=[]))
    return lock, docs


def write_api(directory, lock, docs):
    target = directory / "api"; target.mkdir(exist_ok=True)
    for role, value in docs.items():
        raw = encoded(value); (target / (role + ".json")).write_bytes(raw)
        lock["api"][role] = pin(raw)


def array_fixture():
    rows = [dict(id="invented" + str(i)) for i in range(14)]
    waves = {row["id"]: struct.pack("<h", i) * 992 for i, row in enumerate(rows)}
    files, records = {}, []
    for row in rows:
        for condition in reader.CONDITIONS:
            original = waves[row["id"]]
            signal = (b"\0" * 32000 if condition == reader.CONDITIONS[1] else b"") + original
            frames = 1 + (len(signal) // 2 - 512) // 160
            labels = [i % 4 for i in range(frames)]
            raw = b"".join(struct.pack("<4f", *(1.0 if index == label else -1.0 for index in range(4))) for label in labels)
            events = engineering.ctc(labels)
            record = dict(id=row["id"], condition=condition, source_pcm_sha256=reader.sha(original),
                derived_pcm_sha256=reader.sha(signal), frames=frames, pytorch_sha256=reader.sha(raw), **events, comparisons={})
            for backend in reader.BACKENDS:
                key = row["id"] + "__" + condition + "__" + backend
                files["raw_outputs/" + key + ".f32"] = raw
                files["attempts/" + key + ".json"] = encoded(dict(key=key, backend=backend, status="consumed_unknown"))
                if backend == "pytorch":
                    continue
                comparison = dict(sha256=reader.sha(raw), **engineering.compare_logits(raw, raw, .001, .0001))
                if backend in reader.BACKENDS[2:]:
                    csv = "".join(f"{i},{512 + 160 * i},{int(i in events['event_frames'])}\n" for i in range(frames)).encode()
                    files["raw_outputs/" + key + ".csv"] = csv
                    files["logs/" + key + ".txt"] = f"MERO43_C_PCM_STREAM_OK samples={len(signal)//2} frames={frames} events={len(events['event_frames'])}\n".encode()
                    comparison["event_csv_sha256"] = reader.sha(csv)
                record["comparisons"][backend] = comparison
            records.append(record)
    return files, dict(records=records), rows, waves


def npy(shape, values):
    header = repr(dict(descr="<f4", fortran_order=False, shape=shape)).encode()
    header += b" " * ((16 - (10 + len(header) + 1) % 16) % 16) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + struct.pack("<" + "f" * len(values), *values)


def pack_fixture():
    geometry = [("input.weight", (96, 40)), ("input.bias", (96,))]
    for layer in range(4):
        geometry += [(f"layers.{layer}." + name, shape) for name, shape in
                     (("linear.weight", (96, 96)), ("linear.bias", (96,)), ("memory.weight", (96, 1, 5)), ("memory.bias", (96,)))]
    geometry += [("output.weight", (4, 96)), ("output.bias", (4,)), ("mel_filters", (257, 40)), ("train_mean", (40,)), ("train_scale", (40,))]
    arrays, bodies = {}, []
    for name, shape in geometry:
        count = 1
        for size in shape:
            count *= size
        values = [1.0 if name == "train_scale" else .5 if name == "mel_filters" else .125] * count
        raw = npy(shape, values); arrays[name + ".npy"] = raw
        bodies.append(struct.pack("<" + "f" * count, *values))
    return struct.pack("<8sII", b"MERO43V1", 1, 54236) + b"".join(bodies), arrays, geometry


def producer_fixture():
    lock, docs = api_fixture()
    identity = dict(run_id=1234, run_head="a" * 40, run_attempt=1, prepared_parent="e" * 40)
    native = {"invented/" + str(i) + ".c": b"invented frozen source " + str(i).encode() for i in range(13)}
    descriptor = dict(repository=reader.PIPELINE_REPO, commit=reader.PIPELINE_COMMIT,
        files={name: dict(bytes=len(raw), git_blob_sha1=reader.blob(raw)) for name, raw in native.items()})
    repository = {"automation/english_v6_b24r1_engineering.py": (HERE / "english_v6_b24r1_engineering.py").read_bytes(),
        "automation/english_v6_b24r1_quality_readback.py": (HERE / "english_v6_b24r1_quality_readback.py").read_bytes(),
        "automation/english_v6_b24r1_engineering_inputs.json": b"{}",
        "automation/english_v6_b24_engineering_pipeline.json": encoded(descriptor)}
    producer = dict(schema=engineering.SCHEMA + "-producer", **identity, git_tree="f" * 40,
        pipeline_commit=reader.PIPELINE_COMMIT, include_hidden_files=True, new_model_forwards=0, packs_created=0,
        closure_rule="complete_tracked_automation_and_workflows_plus_both_original_quality_producers",
        files={name: dict(mode="100644", git_blob_sha1=reader.blob(raw), **pin(raw)) for name, raw in repository.items()},
        pipeline_files={name: dict(git_blob_sha1=reader.blob(raw), **pin(raw)) for name, raw in native.items()})
    files = {**{"repository/" + name: raw for name, raw in repository.items()},
             **{"pipeline/" + name: raw for name, raw in native.items()}, "PRODUCER.json": encoded(producer)}
    tree = lambda mapping: {name: dict(path=name, type="blob", mode="100644", sha=reader.blob(raw), size=len(raw)) for name, raw in mapping.items()}
    lock.update(producer_manifest_sha256=reader.sha(files["PRODUCER.json"]), inputs_sha256=reader.sha(b"{}"))
    return files, producer, lock, identity, docs, tree(repository), tree(native), reader.sha(encoded(descriptor))


class Contracts(unittest.TestCase):
    def test_frozen_helper_digests_equal_reviewed_code(self):
        self.assertEqual(reader.sha((HERE / "english_v6_b24r1_engineering.py").read_bytes()), reader.ENGINEERING_SHA)
        self.assertEqual(reader.sha((HERE / "english_v6_b24r1_quality_readback.py").read_bytes()), reader.QUALITY_READER_SHA)
        self.assertEqual(reader.sha((HERE / "english_v6_b24_engineering_pipeline.json").read_bytes()), reader.PIPELINE_DESCRIPTOR_SHA)

    def test_closed_or_partial_lock_stops_before_evidence_or_child_process(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root / "lock.json"; path.write_text("{}")
            with patch.object(reader.subprocess, "check_output", side_effect=AssertionError("child forbidden")):
                with self.assertRaisesRegex(ValueError, "INERT"):
                    reader.verify(root / "missing", path, root / "output")
            self.assertFalse((root / "output").exists())
        lock = lock_fixture(); reader.validate_lock(lock)
        for change in (lambda x: x["artifacts"].pop("input_proof"), lambda x: x["api"].pop("tree"),
                       lambda x: x.update(evaluator_sha256="f" * 64), lambda x: x.update(quality_reader_sha256="f" * 64),
                       lambda x: x["artifacts"]["result"].update(bytes=128000001),
                       lambda x: x["artifacts"]["result"].update(run_id=True),
                       lambda x: x["artifacts"]["result"].update(run_head="c" * 40),
                       lambda x: x["artifacts"]["result"].update(id=x["artifacts"]["producer"]["id"])):
            broken = copy.deepcopy(lock); change(broken)
            with self.assertRaises(ValueError):
                reader.validate_lock(broken)

    def test_complete_api_provenance_and_every_required_mutation(self):
        lock, docs = api_fixture()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); write_api(directory, lock, docs)
            _, identity, _, _ = reader.api_metadata(directory, lock)
            self.assertEqual(identity, dict(run_id=1234, run_head="a" * 40, run_attempt=1, prepared_parent="e" * 40))
            changes = (
                lambda x: x["run"].update(run_attempt=2), lambda x: x["run"].update(conclusion="failure"),
                lambda x: x["run"].update(event="workflow_dispatch"), lambda x: x["run"]["repository"].update(private=True),
                lambda x: x["run"].update(head_branch="main"), lambda x: x["commit"]["parents"].append(dict(sha="7" * 40)),
                lambda x: x["commit"]["files"].append(dict(filename="automation/model.py")),
                lambda x: x["artifacts"]["artifacts"][0].update(digest="sha256:" + "1" * 64),
                lambda x: x["artifacts"]["artifacts"][0].update(name="foreign-output"),
                lambda x: x["artifacts"]["artifacts"][0].update(expired=True),
                lambda x: x["artifacts"].update(total_count=5), lambda x: x["runs"].update(total_count=2),
                lambda x: x["markers"]["artifacts"][0]["workflow_run"].update(head_sha="2" * 40),
                lambda x: x["tree"].update(truncated=True), lambda x: x["pipeline_commit"].update(sha="3" * 40),
                lambda x: x["pipeline_tree"].update(sha="3" * 40))
            for change in changes:
                with self.subTest(change=changes.index(change)):
                    broken = copy.deepcopy(docs); bound = copy.deepcopy(lock); change(broken); write_api(directory, bound, broken)
                    with self.assertRaises(ValueError):
                        reader.api_metadata(directory, bound)

    def test_saved_api_byte_tampering_fails_without_a_new_reviewed_lock(self):
        lock, docs = api_fixture()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); write_api(directory, lock, docs)
            (directory / "api/run.json").write_bytes(encoded({**docs["run"], "display_title": "changed"}))
            with self.assertRaisesRegex(ValueError, "byte identity"):
                reader.api_metadata(directory, lock)

    def test_run_and_marker_api_copies_must_equal_original_input_proof_bytes(self):
        lock, docs = api_fixture()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); write_api(directory, lock, docs)
            files = {"FETCH.json": b"{}", "CONSUMPTION.json": b"{}",
                     **{"api/" + role + ".json": (directory / "api" / (role + ".json")).read_bytes()
                        for role in ("runs", "markers")}}
            with reader.opened(zipped(files)) as archive:
                reader.retained_api_metadata(directory, lock, archive)
            for role in ("runs", "markers"):
                name = "api/" + role + ".json"
                missing = dict(files); missing.pop(name)
                with reader.opened(zipped(missing)) as archive, self.assertRaises(ValueError):
                    reader.retained_api_metadata(directory, lock, archive)
                # Same parsed JSON is insufficient: retain the original API response bytes.
                (directory / name).write_bytes(json.dumps(docs[role], indent=2).encode())
                with reader.opened(zipped(files)) as archive, self.assertRaisesRegex(ValueError, "byte-for-byte"):
                    reader.retained_api_metadata(directory, lock, archive)
                (directory / name).write_bytes(files[name])
                changed = dict(files); changed[name] += b"\n"
                with reader.opened(zipped(changed)) as archive, self.assertRaisesRegex(ValueError, "byte identity"):
                    reader.retained_api_metadata(directory, lock, archive)

    def test_same_reviewed_workflow_retains_raw_original_receipts_in_existing_artifact(self):
        template = engineering.TEMPLATE.read_bytes()
        workflow = (HERE.parent / engineering.WORKFLOW).read_bytes()
        parent = re.search(r"PREPARED_PARENT: '([^']+)'", workflow.decode()).group(1)
        self.assertTrue(parent == engineering.PARENT_TOKEN or engineering.valid_hash(parent, 40))
        self.assertEqual(workflow, template.replace(engineering.PARENT_TOKEN.encode(), parent.encode()))
        text = template.decode()
        start = text.index("      - name: Preserve original run inventory and consumed-marker API bytes")
        end = text.index("      - uses:", start)
        step = text[start:end]
        self.assertLess(text.index("engineering.py execute"), start)
        self.assertIn("if: always()", step)
        self.assertIn("env: {GH_TOKEN: '${{ github.token }}'}", step)
        self.assertEqual(step.count("          gh api "), 2)
        self.assertIn("actions/workflows/english-v6-b24r1-engineering-once.yml/runs?branch=" + engineering.BRANCH + "&event=push&per_page=100", step)
        self.assertIn("actions/artifacts?name=" + engineering.MARKER + "&per_page=100", step)
        self.assertEqual(step.count("--allow-escape-sequences >"), 2)
        proof = text[text.index("name: english-v6-b24r1-engineering-input-proof-"):]
        for name in ("FETCH.json", "CONSUMPTION.json", "api/runs.json", "api/markers.json"):
            self.assertIn("${{ runner.temp }}/b24r1-engineering-inputs/" + name, proof)
        self.assertNotIn("workflow_dispatch", text)

    def test_original_producer_uses_saved_git_tree_and_rejects_changed_closure(self):
        files, producer, lock, identity, docs, tree, pipeline, descriptor_sha = producer_fixture()
        with tempfile.TemporaryDirectory() as temp, patch.object(reader, "PIPELINE_DESCRIPTOR_SHA", descriptor_sha):
            directory = Path(temp)
            with reader.opened(zipped(files)) as archive:
                value, _, _ = reader.producer_files(archive, lock, identity, docs, tree, pipeline, directory / "good")
            self.assertEqual(len(value["pipeline_files"]), 13)
            cases = []
            changed_tree = copy.deepcopy(tree); changed_tree["automation/english_v6_b24r1_engineering.py"]["sha"] = "0" * 40
            cases.append((files, changed_tree, lock))
            extra = {**files, "repository/automation/unlisted.py": b"raise RuntimeError('must not import')"}
            cases.append((extra, tree, lock))
            altered = dict(files); name = "repository/automation/english_v6_b24r1_engineering.py"
            altered[name] += b"\nraise RuntimeError('must not import')\n"
            altered_producer = copy.deepcopy(producer)
            altered_producer["files"][name.removeprefix("repository/")].update(**pin(altered[name]), git_blob_sha1=reader.blob(altered[name]))
            altered["PRODUCER.json"] = encoded(altered_producer)
            altered_lock = {**lock, "producer_manifest_sha256": reader.sha(altered["PRODUCER.json"])}
            altered_tree = copy.deepcopy(tree)
            altered_tree[name.removeprefix("repository/")].update(sha=reader.blob(altered[name]), size=len(altered[name]))
            cases.append((altered, altered_tree, altered_lock))
            for index, (contents, recorded_tree, bound) in enumerate(cases):
                with reader.opened(zipped(contents)) as archive, patch.object(reader, "load", side_effect=AssertionError("unverified source import")):
                    with self.assertRaises(ValueError):
                        reader.producer_files(archive, bound, identity, docs, recorded_tree, pipeline, directory / str(index))

    def test_failed_original_panel_never_reaches_engineering_pcm_or_candidate(self):
        for failed_panel in ("primary", "replication"):
            lock, docs = api_fixture()
            with tempfile.TemporaryDirectory() as temp:
                directory = Path(temp); originals = directory / "originals"; originals.mkdir()
                for role, bound in lock["artifacts"].items():
                    if role == "input_proof":
                        continue
                    raw = zipped({"invented.json": b"{}"}); (originals / (role + ".zip")).write_bytes(raw)
                    bound.update(**pin(raw))
                    artifact = next(v for v in docs["artifacts"]["artifacts"] if v["id"] == bound["id"])
                    artifact.update(size_in_bytes=len(raw), digest="sha256:" + reader.sha(raw))
                docs["markers"]["artifacts"] = [copy.deepcopy(next(v for v in docs["artifacts"]["artifacts"] if v["name"] == reader.MARKER))]
                raw = zipped({"FETCH.json": b"{}", "CONSUMPTION.json": b"{}",
                              **{"api/" + role + ".json": encoded(docs[role]) for role in ("runs", "markers")}})
                (originals / "input_proof.zip").write_bytes(raw)
                lock["artifacts"]["input_proof"].update(**pin(raw))
                artifact = next(v for v in docs["artifacts"]["artifacts"] if v["id"] == lock["artifacts"]["input_proof"]["id"])
                artifact.update(size_in_bytes=len(raw), digest="sha256:" + reader.sha(raw))
                write_api(directory, lock, docs)
                workflow = directory / reader.WORKFLOW; workflow.parent.mkdir(parents=True)
                workflow.write_bytes(engineering.TEMPLATE.read_bytes().replace(engineering.PARENT_TOKEN.encode(), b"e" * 40))
                visits = []
                def verify_panel(path, panel):
                    visits.append(panel)
                    if panel == failed_panel:
                        raise ValueError("original panel rejected")
                    return {"panel_gate": "PASS", "fixture_summary_has_no_authority": True}
                quality = types.SimpleNamespace(offline_only=nullcontext, verify_quality_original=verify_panel)
                with patch.object(reader, "producer_files", return_value=({"files": {"fixture": {}}}, engineering, quality)), \
                     patch.object(engineering, "ROOT", directory), \
                     patch.object(engineering, "producer_paths", return_value={"fixture"}), \
                     patch.object(engineering, "protocol", return_value={"quality_locks": {"primary": "primary", "replication": "replication"}}), \
                     patch.object(engineering, "original_pair", side_effect=AssertionError("candidate read before dual PASS")), \
                     patch.object(engineering, "engineering_sources", side_effect=AssertionError("engineering PCM before dual PASS")):
                    with self.assertRaisesRegex(ValueError, "original panel rejected"):
                        reader.verify_worker(directory, lock)
                self.assertEqual(visits, ["primary"] if failed_panel == "primary" else ["primary", "replication"])

    def test_original_zip_rejects_aliases_symlinks_duplicate_members_and_crc(self):
        for name in ("../outside", "/outside", "a//b", "a/./b", "a\\b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                reader.opened(zipped({name: b"value"}))
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            member = zipfile.ZipInfo("linked"); member.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(member, b"../outside")
        with self.assertRaises(ValueError):
            reader.opened(stream.getvalue())
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("same", b"one")
            with __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore"); archive.writestr("same", b"two")
        with self.assertRaises(ValueError):
            reader.opened(stream.getvalue())
        raw = bytearray(zipped({"one": b"original"})); raw[34] ^= 1
        with self.assertRaises((ValueError, zipfile.BadZipFile, __import__("zlib").error)):
            reader.opened(bytes(raw))

    def test_all_140_arrays_and_84_csvs_reconstruct_original_comparisons(self):
        files, report, rows, waves = array_fixture()
        with reader.opened(zipped(files)) as archive:
            reader.saved_arrays(archive, report, rows, waves, engineering)
        self.assertEqual(sum(name.endswith(".f32") for name in files), 140)
        self.assertEqual(sum(name.endswith(".csv") for name in files), 84)

    def test_any_backend_missing_or_truncated_at_last_case_fails(self):
        files, report, rows, waves = array_fixture()
        for backend in reader.BACKENDS:
            name = "raw_outputs/invented13__one_second_zero_prefix__" + backend + ".f32"
            for action in ("missing", "truncated"):
                with self.subTest(backend=backend, action=action):
                    broken = dict(files)
                    if action == "missing": broken.pop(name)
                    else: broken[name] = broken[name][:-16]
                    with reader.opened(zipped(broken)) as archive, self.assertRaises((ValueError, KeyError)):
                        reader.saved_arrays(archive, report, rows, waves, engineering)

    def test_saved_summary_cannot_hide_numeric_drift_nonfinite_or_event_change(self):
        files, report, rows, waves = array_fixture()
        for backend in reader.BACKENDS:
            name = "raw_outputs/invented13__one_second_zero_prefix__" + backend + ".f32"
            for number in (.125, float("nan")):
                with self.subTest(backend=backend, mutation=number):
                    broken = dict(files); raw = bytearray(broken[name]); struct.pack_into("<f", raw, 0, number); broken[name] = bytes(raw)
                    with reader.opened(zipped(broken)) as archive, self.assertRaises(ValueError):
                        reader.saved_arrays(archive, report, rows, waves, engineering)
        for backend in reader.BACKENDS[2:]:
            name = "raw_outputs/invented13__one_second_zero_prefix__" + backend + ".csv"
            broken = dict(files); broken[name] = broken[name].replace(b"0,512,0", b"0,513,0", 1)
            with reader.opened(zipped(broken)) as archive, self.assertRaises(ValueError):
                reader.saved_arrays(archive, report, rows, waves, engineering)

    def test_extra_attempt_summary_change_and_pcm_origin_change_fail(self):
        files, report, rows, waves = array_fixture()
        bad_files = dict(files); bad_files["attempts/retry.json"] = b"{}"
        bad_report = copy.deepcopy(report); bad_report["records"][-1]["comparisons"]["gcc"]["max_abs"] = .00001
        bad_waves = dict(waves); bad_waves["invented13"] = b"\x01\x00" + waves["invented13"][2:]
        for archive_files, records, pcm in ((bad_files, report, waves), (files, bad_report, waves), (files, report, bad_waves)):
            with reader.opened(zipped(archive_files)) as archive, self.assertRaises(ValueError):
                reader.saved_arrays(archive, records, rows, pcm, engineering)

    def test_each_original_npz_tensor_equals_candidate_pack_bytes(self):
        pack, arrays, geometry = pack_fixture()
        reader.pack_bytes(pack, zipped(arrays), zipped(arrays), geometry)
        for changed in (pack[:-1], pack + b"\0", pack[:20] + bytes([pack[20] ^ 1]) + pack[21:]):
            with self.assertRaises(ValueError):
                reader.pack_bytes(changed, zipped(arrays), zipped(arrays), geometry)
        missing = dict(arrays); missing.pop("layers.3.memory.weight.npy")
        with self.assertRaises(ValueError):
            reader.pack_bytes(pack, zipped(missing), zipped(arrays), geometry)

    def test_pack_frontend_must_keep_original_b0_normalization(self):
        pack, arrays, geometry = pack_fixture()
        changed = dict(arrays); changed["train_mean.npy"] = npy((40,), [.25] * 40)
        # Both candidate NPZ and pack are internally consistent, but the baseline differs.
        offset = 16 + 4 * (54236 - 80)
        changed_pack = pack[:offset] + struct.pack("<40f", *([.25] * 40)) + pack[offset + 160:]
        with self.assertRaisesRegex(ValueError, "original B0 frontend"):
            reader.pack_bytes(changed_pack, zipped(changed), zipped(arrays), geometry)
        malformed = arrays["output.bias.npy"].replace(b"'<f4'", b"'<i4'")
        with self.assertRaises(ValueError):
            reader.npy_body(malformed, (4,))

    def test_entire_original_result_manifest_is_required(self):
        lock = lock_fixture()
        files = {"RESULT.json": b"{}", "empty-success-log.txt": b"", "original-output.f32": b"bytes"}
        manifest = dict(schema=engineering.SCHEMA, status="PASS_B24R1_DUAL_QUALITY_SOFTWARE_SDK_CANDIDATE",
            files={name: pin(raw) for name, raw in files.items()}, physical_ssc305=False, product_promotion_allowed=False)
        files["MANIFEST.json"] = encoded(manifest)
        lock.update(result_manifest_sha256=reader.sha(files["MANIFEST.json"]), original_result_sha256=reader.sha(files["RESULT.json"]))
        with reader.opened(zipped(files)) as archive:
            reader.result_manifest(archive, lock)
        for broken in ({**files, "unlisted.txt": b"x"}, {**files, "original-output.f32": b"changed"}):
            with reader.opened(zipped(broken)) as archive, self.assertRaises(ValueError):
                reader.result_manifest(archive, lock)

    def test_report_worker_failure_never_writes_a_completion_or_verifier(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); lock = directory / "lock.json"; lock.write_bytes(encoded(lock_fixture()))
            with patch.object(reader.subprocess, "check_output", side_effect=reader.subprocess.CalledProcessError(1, [])):
                with self.assertRaises(reader.subprocess.CalledProcessError):
                    reader.verify(directory / "unused", lock, directory / "report")
            self.assertFalse((directory / "report").exists())

    def test_json_duplicates_and_nonfinite_numbers_reject(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'[]'):
            with self.assertRaises(ValueError):
                reader.document(raw)


if __name__ == "__main__":
    unittest.main()
