"""B23 sealed independent replica, reusing unchanged PRIMARY scientific helpers.

Only the same immutable four model roles may be assessed. A complete original
PRIMARY PASS is verified before source opening, and again before any replica
forward. Numerical scoring and B14 decisions are unchanged.
"""
from __future__ import annotations
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import english_v6_b23_primary_eval as original
import english_v6_b23_replication_source_ancestry as opening
import english_v6_b23_replication_capacity as capacity
from english_v6_b22_replication_reuse import inherit
inherit(original,globals())

SPEC=HERE/'english_v6_b23_replication_inputs.json'
PRODUCER=HERE/'english_v6_b23_replication_producer.json'
WORKFLOW_TEMPLATE=HERE/'english_v6_b23_replication_workflow_template.yml'
SCHEMA='english-v6-b23-fresh-replication-four-frozen-models-v1'
SOURCE_MODULE='english_v6_b23_replication_source'
SOURCE_SCHEMA='english-v6-b23-fresh-replication-source-v1'
SCREEN_SCHEMA='english-v6-b23-120-original-replication-screen-v1'
BRANCH='refs/heads/research/en-v6-b23-replication-scoring-once-20261124'
WORKFLOW='.github/workflows/english-v6-b23-replication-model-eval-once.yml'
SOURCE_BRANCH='research/en-v6-b23-replication-sources-once-20261124'
SOURCE_WORKFLOW='.github/workflows/english-v6-b23-replication-sources-once.yml'
SOURCE_TEMPLATE='automation/english_v6_b23_replication_source_workflow_template.yml'
SOURCE_PARENT_TOKEN='__B23_REVIEWED_REPLICATION_SOURCE_PARENT__'
PARENT_TOKEN='__B23_REVIEWED_REPLICATION_SCORING_PARENT__'
MARKER='english-v6-b23-replication-consumed'
FLAGS=tuple(k for k in original.FLAGS if k!='replication_opened')
PRODUCER_ROOTS=('english_v6_b23_replication_eval',SOURCE_MODULE,'english_v6_four_model_panel','english_v6_b14_decision')


def original_zip(path, pin, *, role=None):
    need(role is None or role in INPUT_KEYS | {'train_producer', 'source_producer',
         *('native_' + provider for provider in PROVIDERS)}, 'explicit original input role')
    if role != 'source':
        return original.original_zip(path, pin)
    raw = Path(path).read_bytes()
    valid_artifact(pin)
    need(len(raw) == pin['bytes'] and sha(raw) == pin['sha256'], 'exact original replica SOURCE archive bytes')
    archive = capacity.check_archive(raw, 'replication_source')
    try:
        manifest = json.loads(archive.read('MANIFEST.json'))
        need(manifest['schema'] == SOURCE_SCHEMA and manifest['panel'] == 'replication',
             'larger envelope is only the self-contained original replica SOURCE')
    except BaseException:
        archive.close()
        raise
    return archive

def protocol():
    doc = json.loads(SPEC.read_bytes())
    need(doc.get("scoring_allowed") is True, "INERT: real B23 training and independently admitted source pins required")
    need(doc["schema"] == SCHEMA and doc["source_module"] == SOURCE_MODULE
         and doc["source_schema"] == SOURCE_SCHEMA and doc["screen_schema"] == SCREEN_SCHEMA
         and doc["prepared_parent"] is None and doc["parent_authority"] == "reviewed_workflow"
         and doc["pair_schema"] == PAIR_SCHEMA and doc["pair_stems"] == STEMS
         and doc["parent_pt_sha256"] == B0_PT and doc["model_pt_sha256"]["B0"] == B0_PT
         and doc["model_pt_sha256"]["B9"] == B9_PT and doc["model_pt_sha256"]["control"] == CONTROL_PT
         and set(doc["model_pt_sha256"]) == set(ARMS)
         and all(valid_hash(value) for value in doc["model_pt_sha256"].values())
         and len(set(doc["model_pt_sha256"].values())) == 4
         and doc["model_comparison"] == MODEL_COMPARISON
         and doc["same_job_matched_control"] is doc["historical_control_retrained"] is False
         and doc["gains"] == list(GAINS) and doc["source_gate"] == GATE
         and type(doc["maximum_model_forwards"]) is int and doc["maximum_model_forwards"] == MAX_FORWARDS
         and all(doc[key] is False for key in FLAGS) and set(doc["inputs"]) == INPUT_KEYS
         and set(doc["native_inputs"]) == set(PROVIDERS) and set(doc["upstream_producers"]) == {"train", "source"},
         "fixed B23 historical-control comparison and unchanged fresh-source gate")
    for field in ("assessment_sha256", "training_design_sha256", "producer_manifest_sha256"):
        need(valid_hash(doc[field]), "complete frozen registration digest " + field)
    need(sha(ASSESSMENT.read_bytes()) == doc["assessment_sha256"] == TRAINING_ASSESSMENT
         and sha(TRAIN_DESIGN.read_bytes()) == doc["training_design_sha256"] == TRAINING_DESIGN
         and doc["training_assessment_sha256"] == doc["assessment_sha256"],
         "same prospective assessment frozen before source and optimizer")
    pins = archive_pins(doc)
    for pin in pins.values(): valid_artifact(pin)
    need(all(doc["inputs"][role] == pin for role, pin in BASELINES.items()), "unchanged original B0/B9 artifacts")
    source_pins = [doc["inputs"]["source"], doc["inputs"]["screen"], doc["upstream_producers"]["source"], *doc["native_inputs"].values()]
    need(all(all(pin[key] == source_pins[0][key] for key in ("run_id", "run_head")) for pin in source_pins)
         and len({pin["id"] for pin in pins.values()}) == 12
         and doc["inputs"]["pair"]["run_id"] != source_pins[0]["run_id"]
         and all(doc["upstream_producers"]["train"][key] == doc["inputs"]["pair"][key] for key in ("run_id", "run_head")),
         "twelve distinct originals; complete authentic train/source producer archives")
    need(doc['replication_opened'] is True and doc['original_primary_pass_required'] is True,
         'only original PRIMARY PASS opens the fixed replica')
    permit=opening.read_opening()
    need(doc['primary_quality_lock']==permit['primary_quality_lock']
         and doc['quality_readback_sha256']==permit['quality_readback_sha256']
         and doc['inputs']['pair']==permit['pair_pin'] and doc['model_pt_sha256']==permit['model_pt_sha256'],
         'same exact original PRIMARY-passed model bundle and frozen quality reader')
    return doc


def primary_quality(directory,spec,fetch=False):
    permit=opening.read_opening()
    need(spec['primary_quality_lock']==permit['primary_quality_lock'],'same separately frozen original PRIMARY lock')
    reader=opening.quality_reader()
    summary=(reader.fetch_quality_original if fetch else reader.verify_quality_original)(Path(directory),permit['primary_quality_lock'])
    opening.validate_primary(summary)
    need(summary['model_pt_sha256']==spec['model_pt_sha256'] and summary['pair_pin']==spec['inputs']['pair'],
         'same immutable models passed PRIMARY and enter replica')
    return summary

def producer_contract(spec, prepared, root=ROOT):
    raw = (root / "automation" / PRODUCER.name).read_bytes()
    need(sha(raw) == spec["producer_manifest_sha256"], "frozen B23 complete producer manifest")
    doc = json.loads(raw)
    need(doc["schema"] == "english-v6-b23-replication-producer-v1" and doc["frozen"] is True
         and set(doc["upstream_manifests"]) == {"train", "source"}, "complete actual upstream manifests retained")
    required_code = static_closure(root / "automation", PRODUCER_ROOTS)
    upstream = doc["upstream_manifests"]
    workflow_paths = {TRAIN_WORKFLOW, SOURCE_WORKFLOW}
    for role in ("train", "source"):
        required_code |= {name for name in upstream[role]["files"] if name.endswith(".py")}
    need(set(doc["replication"]["files"]) == required_code, "complete static and both original dynamic Python dependencies")
    source_files = {name: pin["sha256"] for name, pin in upstream["source"]["files"].items() if name.endswith(".py")}
    source_resources = {name: pin["sha256"] for name, pin in upstream["source"]["files"].items()
                        if not name.endswith(".py") and name != SOURCE_WORKFLOW}
    need(doc["source"]["files"] == source_files and doc["source"]["resources"] == source_resources,
         "entire actual source producer retained rather than guessed closure")
    for role in ("replication", "source"):
        for name, digest in doc[role]["files"].items():
            need(valid_hash(digest) and sha((root / name).read_bytes()) == digest, "frozen complete scientific code " + name)
    required_resources = {"automation/english_v6_b23_panels/assessment.json",
                          "automation/english_v6_b23_prefix_peak_design.json", "automation/english_v6_b14_panels/assessment.json"}
    for role in ("train", "source"):
        required_resources |= {name for name in upstream[role]["files"] if not name.endswith(".py") and name not in workflow_paths}
    need(required_resources <= set(doc["resources"]), "all original dynamic resources and frozen registrations included")
    for name, digest in doc["resources"].items():
        path = PurePosixPath(name)
        need(path.parts and not path.is_absolute() and ".." not in path.parts and "\\" not in name
             and valid_hash(digest) and sha((root / name).read_bytes()) == digest, "frozen complete resource " + name)
    need(all(doc["resources"].get(path) == digest for path, digest in source_resources.items()), "source resources inherited byte-for-byte")
    template = (root / "automation" / WORKFLOW_TEMPLATE.name).read_bytes()
    need(doc["replication"]["workflow_path"] == WORKFLOW and doc["source"]["workflow_path"] == SOURCE_WORKFLOW
         and sha(template) == doc["replication"]["workflow_template_sha256"] and template.count(PARENT_TOKEN.encode()) == 2
         and b"include-hidden-files: true" in template and valid_hash(doc["source"]["workflow_sha256"]),
         "exact hidden-file-preserving workflows")
    need(valid_hash(prepared, 40), "actual reviewed predecessor")
    workflow = template.replace(PARENT_TOKEN.encode(), prepared.encode())
    need((root / WORKFLOW).read_bytes() == workflow, "two reviewed parent substitutions only")
    return doc, workflow

def producer_files(spec, producer, workflow, role):
    if role == "replication":
        extras = {**producer["resources"], WORKFLOW: sha(workflow),
                  "automation/" + SPEC.name: sha(SPEC.read_bytes()),
                  "automation/" + PRODUCER.name: spec["producer_manifest_sha256"],
                  "automation/" + WORKFLOW_TEMPLATE.name: producer["replication"]["workflow_template_sha256"]}
    else:
        need(role == "source", "explicit frozen source producer role")
        extras = {**producer["source"]["resources"], SOURCE_WORKFLOW: producer["source"]["workflow_sha256"]}
    return {**producer[role]["files"], **extras}

def fetch(inputs, prepared):
    spec = protocol()
    identity = guard(spec, prepared)
    producer, workflow = producer_contract(spec, identity["prepared_parent"])
    need(inputs.is_absolute() and inputs.parent.is_dir() and not inputs.exists(), "fresh immutable evidence directory")
    prior = json.loads(gh("actions/artifacts?name=" + MARKER + "&per_page=1"))
    need(prior["total_count"] == 0, "consumed B23 replica cannot replay")
    run_inventory = unused_workflow_run(identity)
    run, commit = json.loads(gh("actions/runs/" + identity["run_id"])), json.loads(gh("commits/" + identity["run_head"]))
    validate_execution_metadata(identity, run, commit)
    inputs.mkdir()
    primary_quality(inputs / "primary_quality", spec, fetch=True)
    metadata = {}
    for key, pin in archive_pins(spec).items():
        metadata[key] = {"run": json.loads(gh("actions/runs/" + str(pin["run_id"]))),
                         "artifact": json.loads(gh("actions/artifacts/" + str(pin["id"]))),
                         "commit": json.loads(gh("commits/" + pin["run_head"]))}
        validate_provenance(pin, metadata[key])
        raw = gh("actions/artifacts/" + str(pin["id"]) + "/zip")
        need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "whole original input " + key)
        (inputs / (key + ".zip")).write_bytes(raw)
    trees = {"replication": git_tree(commit), "source": git_tree(metadata["source"]["commit"]),
             "train": git_tree(metadata["pair"]["commit"])}
    with original_zip(inputs / "source_producer.zip", spec["upstream_producers"]["source"]) as source_archive:
        for role in ("replication", "source"):
            for path, expected in producer_files(spec, producer, workflow, role).items():
                # The checkout and the original source archive already contain full bytes;
                # verify their actual Git blobs instead of thousands of network blob calls.
                raw = (ROOT / path).read_bytes() if role == "replication" else source_archive.read("repository/" + path)
                need(sha(raw) == expected and git_blob(raw) == tree_member(trees[role], path), "actual complete Git-bound producer member " + path)
                target = inputs / (role + "_producer") / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
    write(inputs / "FETCH.json", dict(schema=SCHEMA, identity=identity, replication_run=run, replication_commit=commit,
        inputs_sha256=sha(SPEC.read_bytes()), producer_manifest_sha256=spec["producer_manifest_sha256"], workflow_sha256=sha(workflow),
        source_metadata=metadata, producer_trees=trees, prior_consumed_markers=prior, original_workflow_run_inventory=run_inventory))
    capacity.producer_members(inputs)

def preflight(inputs, prepared):
    spec = protocol()
    identity = guard(spec, prepared)
    producer, workflow = producer_contract(spec, identity["prepared_parent"])
    capacity.producer_members(inputs)
    fetched = json.loads((inputs / "FETCH.json").read_bytes())
    need(fetched["schema"] == SCHEMA and fetched["identity"] == identity
         and fetched["inputs_sha256"] == sha(SPEC.read_bytes())
         and fetched["producer_manifest_sha256"] == spec["producer_manifest_sha256"]
         and fetched["workflow_sha256"] == sha(workflow) and fetched["prior_consumed_markers"]["total_count"] == 0,
         "exact initial acquisition and original unconsumed experiment")
    validate_execution_metadata(identity, fetched["replication_run"], fetched["replication_commit"])
    prior_runs = fetched["original_workflow_run_inventory"]
    need(prior_runs["total_count"] == len(prior_runs["workflow_runs"]) == 1
         and prior_runs["workflow_runs"][0]["id"] == int(identity["run_id"]), "only original first replica run")
    primary_pass = primary_quality(inputs / "primary_quality", spec)
    pins = archive_pins(spec)
    need(set(fetched["source_metadata"]) == set(pins), "exact twelve original input metadata receipts")
    for role, bound in (("replication", fetched["replication_commit"]), ("source", fetched["source_metadata"]["source"]["commit"])):
        tree = fetched["producer_trees"][role]
        need(tree["sha"] == bound["commit"]["tree"]["sha"] and tree["truncated"] is False,
             "real complete producer tree " + role)
        expected = producer_files(spec, producer, workflow, role)
        directory = inputs / (role + "_producer")
        need(set(expected) == {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()},
             "all and only complete producer closure including hidden metadata " + role)
        for path, digest in expected.items():
            raw = (directory / path).read_bytes()
            need(sha(raw) == digest and git_blob(raw) == tree_member(tree, path), "actual complete producer file " + path)
    for key, pin in pins.items():
        validate_provenance(pin, fetched["source_metadata"][key])
    common, decision, source, plan = helpers(spec)
    with ExitStack() as stack, tempfile.TemporaryDirectory() as temporary:
        archives = {key: stack.enter_context(original_zip(inputs / (key + ".zip"), pin, role=key)) for key, pin in pins.items()}
        verify_upstream_archives(archives, spec, fetched, producer)
        pair = archives["pair"]
        train = validate_training_archive(pair, spec, fetched["source_metadata"]["pair"])
        need(sha(archives["B0"].read("models/mero-fsmn-ctc-expanded256.pt")) == B0_PT
             and sha(archives["B0"].read("models/mero-fsmn-ctc-expanded256.npz")) == B0_NPZ
             and sha(archives["B9"].read(common.b9.MODELS["B9_control"]["path"] + ".pt")) == B9_PT,
             "original B0/B9 baseline identities before Torch")
        directory = Path(temporary) / "source"
        directory.mkdir()
        archives["source"].extractall(directory)
        owner = source.evaluation_preflight(directory)
        rows, originals = owner["rows"], owner["all_original_rows"]
        execution, source_meta = owner["source_execution"], fetched["source_metadata"]["source"]
        need(owner["assessment_sha256"] == spec["assessment_sha256"]
             and str(execution["run_id"]) == str(spec["inputs"]["source"]["run_id"])
             and execution["run_head"] == spec["inputs"]["source"]["run_head"]
             and execution["run_attempt"] == 1 and execution["run_branch"] == "refs/heads/" + SOURCE_BRANCH
             and source_meta["run"]["head_branch"] == SOURCE_BRANCH and source_meta["run"]["path"] == SOURCE_WORKFLOW
             and source_meta["run"]["event"] == "push" and len(source_meta["commit"]["parents"]) == 1
             and source_meta["commit"]["parents"][0]["sha"] == execution["prepared_parent"]
             and [entry["filename"] for entry in source_meta["commit"]["files"]] == [SOURCE_WORKFLOW],
             "complete real first public B23 source manifest and workflow-only opening")
        providers = Path(temporary) / "providers"
        for provider in PROVIDERS:
            target = providers / provider
            target.mkdir(parents=True)
            archives["native_" + provider].extractall(target)
        screen = json.loads(archives["screen"].read("SCREEN.json"))
        need(screen["schema"] == SCREEN_SCHEMA and screen["source_execution"] == execution
             and screen["source_manifest_sha256"] == sha(archives["source"].read("MANIFEST.json"))
             and screen["coverage"] == dict.fromkeys(PROVIDERS, len(rows))
             and screen["selected_original_slots"] == 120, "all original native observations and unchanged SCREEN source identity")
        source.verify_screen_contract(directory, screen, providers)
        selected = exact_admitted_join(rows, screen["rows"])
        need(owner["primary_quality"] == primary_pass, "source and replica scorer verified the identical original PRIMARY PASS")
        prior = primary_pass["source_inventory"]
        need(not set(prior["pcm_sha256"]) & {row["pcm_sha256"] for row in originals}
             and not set(prior["source_group_id"]) & {row["source_group_id"] for row in originals}
             and not set(prior["canonical_people"]) & {person(row) for row in originals},
             "every replica original independent of all120 original PRIMARY sources")
        reject_overlap(originals, train["base_train"] + train["natural_train"], owner["history"], set())
        counts = source_counts(selected)
        need(counts == screen["source_coverage"] and screen["frozen_minima"] == GATE
             and all(counts[key] >= minimum for key, minimum in GATE.items())
             and screen["source_gate"] == "SOURCE_GATE_PASS" and not screen["source_shortfall"],
             "source coverage gate before any KWS work")
        need(12 * len(selected) <= MAX_FORWARDS, "complete source population fits preregistered1440 forward ceiling")
    need("torch" not in sys.modules and "numpy" not in sys.modules,
         "all source and original TRAIN admission completes before model/array runtimes are imported")
    chosen = [(row, row["cohort"]) for row in selected]
    admission = dict(schema=SCHEMA, identity=identity, assessment_sha256=spec["assessment_sha256"],
        training_design_sha256=spec["training_design_sha256"], registered_inputs_sha256=sha(SPEC.read_bytes()),
        producer_manifest_sha256=spec["producer_manifest_sha256"], workflow_sha256=sha(workflow),
        source_gate="SOURCE_GATE_PASS", source_shortfall={}, source_coverage=counts,
        original_source_rows=120, observer_eligible_rows=len(rows),
        admitted_rows=[score_identity(row, cohort) for row, cohort in chosen],
        training_source_sha256=sha(json.dumps(train, sort_keys=True, separators=(",", ":")).encode()),
        history_sha256=owner["history_sha256"], model_calls=0, old_development_reused=False,
        replication_opened=True, promotion_allowed=False, same_job_matched_control=False,
        historical_control_retrained=False, model_comparison=MODEL_COMPARISON)
    admission.update(primary_result_sha256=primary_pass["original_result_sha256"],
                     primary_quality_lock_sha256=sha(json.dumps(spec["primary_quality_lock"],sort_keys=True,separators=(",", ":"),allow_nan=False).encode()))
    return spec, identity, common, decision, chosen, train, admission

def score(inputs, output, prepared):
    spec = protocol()  # Before paths, helper imports and model initialization.
    checked = preflight(inputs, prepared)
    spec, identity, common, decision, chosen, train, admission = checked
    permit = json.loads((inputs / "CONSUMPTION.json").read_bytes())
    need(permit["identity"] == identity and permit["marker_name"] == MARKER
         and permit["admission_sha256"] == sha(json.dumps(admission, sort_keys=True, separators=(",", ":")).encode()),
         "one uploaded consumed marker must bind this exact complete preflight")
    unused_workflow_run(identity)
    remote = json.loads(gh("actions/artifacts?name=" + MARKER + "&per_page=100"))
    need(remote["total_count"] == 1 and len(remote["artifacts"]) == 1
         and remote["artifacts"][0]["workflow_run"]["id"] == int(identity["run_id"])
         and remote["artifacts"][0]["workflow_run"]["head_sha"] == identity["run_head"],
         "exact current-run consumed marker; any older or ambiguous consumption forbids replay")
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(), "new result directory only")
    output.mkdir()
    attempts = output / "attempts"; attempts.mkdir()
    raw_outputs = output / "raw_outputs"; raw_outputs.mkdir()
    ledger = Ledger(attempts)
    report = dict(schema=SCHEMA, **identity, assessment_sha256=spec["assessment_sha256"],
        training_assessment_sha256=spec["assessment_sha256"], training_design_sha256=spec["training_design_sha256"],
        registered_inputs_sha256=sha(SPEC.read_bytes()), implementation_sha256=sha(Path(__file__).read_bytes()),
        producer_manifest_sha256=spec["producer_manifest_sha256"], workflow_sha256=admission["workflow_sha256"],
        source_gate="SOURCE_GATE_PASS", source_coverage=admission["source_coverage"], source_shortfall={},
        panel_gate="STARTED", records=[], actual_forwards=0, saved_array_count=0, raw_output_count=0,
        optimizer_updates=0, model_pt_sha256=spec["model_pt_sha256"], installed_software_baseline="B0",
        replication_open_allowed=False, panel="replication", replication_opened=True, research_replication_passed=False,
        primary_result_sha256=admission["primary_result_sha256"],primary_quality_lock_sha256=admission["primary_quality_lock_sha256"],
        B20_failed_primary_preserved=True, B21_failed_primary_preserved=True, B22_failed_primary_preserved=True,
        same_job_matched_control=False, historical_control_retrained=False, model_comparison=MODEL_COMPARISON,
        original_natural_reused_without_reobservation=False, fresh_source_originals=True, ssc305_physical=False,
        natural_field_far_per_hour=None, field_3_5m_frr=None, **dict.fromkeys(FLAGS, False))
    write(output / "ADMISSION.json", admission)
    write(output / "STARTED.json", report)
    hooks, arrays = [], None
    try:
        import numpy as np
        import torch
        need(np.__version__ == "1.26.4" and torch.__version__ == "2.9.1+cpu", "unchanged pinned CPU scoring runtime")
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True); torch.backends.mkldnn.enabled = False
        models, loaded_train = snapshots(inputs, spec, torch, np, common)
        need(loaded_train == train and {k: v[-1] for k, v in models.items()} == spec["model_pt_sha256"],
             "same preflight TRAIN and all four original models/frontends")
        for role, values in models.items():
            def entered(_module, _args, role=role):
                ledger.entered(role)
            hooks.append(values[0].register_forward_pre_hook(entered))
        arrays = zipfile.ZipFile(output / "ALL_FRAME_LOGITS.npz", "x", compression=zipfile.ZIP_DEFLATED)
        with ExitStack() as stack:
            archive = stack.enter_context(original_zip(inputs / "source.zip", spec["inputs"]["source"], role='source'))
            for row, cohort in chosen:
                record = score_identity(row, cohort); record["gains"] = {}; report["records"].append(record)
                raw = archive.read("wav/" + row["id"] + ".wav")
                with wave.open(io.BytesIO(raw)) as audio:
                    need(audio.getparams()[:3] == (1, 2, 16000) and audio.getnframes() == row["samples"], "whole admitted PCM")
                    pcm = audio.readframes(audio.getnframes())
                need(sha(raw) == row["wav_sha256"] and sha(pcm) == row["pcm_sha256"], "same preflight original PCM")
                for gain in GAINS:
                    signal = common.b9.ref.gain_pcm(pcm, float(gain), np)
                    record["gains"][gain] = {}
                    for role in ARMS:
                        net, mel, mean, scale, _ = models[role]
                        feature = common.b9.v5.features(signal, row, torch, np, mel)
                        x = torch.from_numpy(((feature - mean) / scale).astype(np.float32)).unsqueeze(0)
                        key = row["id"] + "__" + gain + "__" + role
                        need(re.fullmatch(r"[A-Za-z0-9_.-]+", key), "safe immutable array key")
                        ledger.current = dict(key=key, role=role, source_id=row["id"], gain=gain)
                        with torch.inference_mode():
                            y = net(x)
                        raw_y = y.detach().cpu().numpy()
                        with (raw_outputs / (key + ".npy")).open("xb") as handle:
                            np.save(handle, raw_y, allow_pickle=False)
                            handle.flush(); os.fsync(handle.fileno())
                        report["raw_output_count"] += 1
                        need(raw_y.shape == (1, len(feature), 4) and raw_y.dtype == np.dtype("float32") and np.isfinite(raw_y).all(),
                             "complete actual finite FP32 frame output")
                        matrix = np.ascontiguousarray(raw_y[0], dtype="<f4")
                        encoded = io.BytesIO(); np.save(encoded, matrix, allow_pickle=False)
                        arrays.writestr(key + ".npy", encoded.getvalue())
                        arrays.fp.flush(); os.fsync(arrays.fp.fileno())
                        report["saved_array_count"] += 1
                        ledger.saved(key)
                        labels = matrix.argmax(-1).tolist()
                        record["gains"][gain][role] = dict(array=key, sha256=sha(matrix.tobytes()), frames=len(matrix),
                            **ctc_events(labels))
        validate_scored_join(report["records"], admission["admitted_rows"])
        need(sum(ledger.counts.values()) == report["saved_array_count"] == report["raw_output_count"] == 12 * len(chosen) <= MAX_FORWARDS
             and all(v == 3 * len(chosen) for v in ledger.counts.values()), "all and only the exact admitted original/gain/model calls completed")
        decision_result = decision.decide(report["records"])
        report.update(decision_result, replication_open_allowed=False,
                      research_replication_passed=decision_result["panel_gate"] == "PASS",
                      candidate_engineering_validation_allowed=decision_result["panel_gate"] == "PASS")
    except BaseException as exc:
        report.update(panel_gate="STOP", error_type=type(exc).__name__, error=str(exc)[:2000], replication_open_allowed=False)
        raise
    finally:
        for hook in hooks:
            hook.remove()
        if arrays is not None:
            arrays.close()
            report["logits_sha256"] = sha((output / "ALL_FRAME_LOGITS.npz").read_bytes())
        report.update(actual_forwards=sum(ledger.counts.values()), model_forward_calls=ledger.counts,
                      consumed_unknown=[key for key, value in ledger.consumed.items() if value != "saved_array"],
                      replay_allowed=False)
        write(output / "RESULT.json", report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "fetch", "preflight", "score"))
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepared-parent")
    args = parser.parse_args()
    if args.mode == "check":
        print(json.dumps(source_check()))
    else:
        protocol()
        need(args.inputs is not None, "original authenticated input directory required")
        if args.mode == "fetch":
            fetch(args.inputs, args.prepared_parent)
        elif args.mode == "preflight":
            checked = preflight(args.inputs, args.prepared_parent)
            admission = checked[-1]
            write(args.inputs / "CONSUMPTION.json", dict(identity=checked[1], marker_name=MARKER,
                  admission_sha256=sha(json.dumps(admission, sort_keys=True, separators=(",", ":")).encode())))
            print(json.dumps(admission))
        else:
            need(args.output is not None, "new output directory required")
            score(args.inputs, args.output, args.prepared_parent)
