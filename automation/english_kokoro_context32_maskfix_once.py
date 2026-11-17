"""One native-int32 first-forward recovery; immutable existing inputs only."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
import english_kokoro_context32_maskfix_source as source
import english_kokoro_context32_mask_observe as observer

SCHEMA=source.SCHEMA
CODE_PATHS={
    "worker_sha256":"automation/english_kokoro_context32_maskfix_once.py",
    "source_code_sha256":"automation/english_kokoro_context32_maskfix_source.py",
    "observer_code_sha256":"automation/english_kokoro_context32_mask_observe.py",
    "processor_check_code_sha256":"automation/english_kokoro_context32_processor_check.py",
    "comparator_code_sha256":"automation/english_kokoro_context32_mask_compare.py"}
need,sha,canonical,regular,loads,once,put,within=(source.need,source.sha,source.canonical,source.regular,source.loads,source.once,source.put,source.within)


def read(path):
    return loads(regular(path))


def validate_plan(plan,check_sources=True):
    need(type(plan) is dict and plan.get("schema")==SCHEMA+"/plan" and type(plan.get("enabled")) is bool,"NEW_PLAN_SCHEMA")
    need(plan.get("campaign_id")==source.CAMPAIGN_ID and plan.get("recovery")==source.RECOVERY
         and plan.get("prior_run_identity")==source.PRIOR_RUN and plan.get("input_manifest_sha256")==source.INPUT_MANIFEST_SHA256
         and plan.get("prior_source_plan_sha256")==source.SOURCE_PLAN_SHA256
         and plan.get("prepared_source_code_sha256")==source.PREPARED_SOURCE_PIN,"FIXED_ACTUAL_ZERO_FORWARD_RECOVERY")
    need(plan.get("main_before")== (source.MAIN_BEFORE if plan["enabled"] else None),"ONE_FIXED_NEW_PREDECESSOR")
    budgets={"declared_inputs":16,"maximum_calls_per_observer":16,"maximum_total_model_calls":32,"maximum_new_model_constructions":2,
        "prior_processor_attempts":32,"prior_model_constructions":2,"prior_forward_attempts":0,
        "retry_count":0,"forward_replays":0,"new_derivations":0,"new_tts_calls":0,"new_text_asr_calls":0,
        "new_embedding_calls":0,"new_training_calls":0,"old_observed_crop_forwards":0,
        "preparation_deadline_seconds":600,"observation_deadline_seconds":1800,"combine_deadline_seconds":900}
    need(all(type(plan.get(k)) is int and plan[k]==v for k,v in budgets.items()),"FINITE_FIRST_FORWARD_BUDGET")
    need(plan.get("diagnostic_only") is True and plan.get("source_material_reused") is True
         and plan.get("recovery_of_failed_preprocessing") is True and plan.get("training_allowed") is False
         and plan.get("training_snapshot_allowed") is False and plan.get("automatic_expansion") is False
         and plan.get("old_labels_modified") is False,"EXACT_RECOVERY_SCOPE")
    need(plan.get("observers")==observer.OBSERVERS and plan.get("observer_files")==observer.OBSERVER_FILES,"UNCHANGED_TWO_MODELS")
    rows=plan.get("input_roster")
    need(type(rows) is list and len(rows)==16 and sha(canonical(rows))==source.ROWS_SHA256
         and len({r["pcm_sha256"] for r in rows})==16,"FIXED_ACTUAL_EXISTING_PCM_ROWS")
    old=plan.get("old_observed_pcm_sha256")
    need(type(old) is list and len(old)==len(set(old))==48
         and sha(canonical(old))=="61d85c6161469682aabc49bb230b66454ead1f1d96ee135112e1780d1375b8a1"
         and not set(old)&{r["pcm_sha256"] for r in rows},"ORIGINAL48_EXCLUSION_RETAINED")
    need(plan.get("prior_input_campaign_id")=="kokoro-v1_0-original-carrier-full-context32-v1","OLD_INPUT_CAMPAIGN_NOT_REWRITTEN")
    if check_sources:
        for key,path in CODE_PATHS.items():
            need(plan.get(key)==sha(regular(ROOT/path,256*1024)),"SOURCE_PIN: "+key)
        source.immutable_readers(plan)
    return plan


def identity(env=None):
    env=os.environ if env is None else env
    return {"repository":env.get("GITHUB_REPOSITORY"),"commit":env.get("GITHUB_SHA"),
        "run_id":env.get("GITHUB_RUN_ID"),"run_attempt":env.get("GITHUB_RUN_ATTEMPT"),"event":env.get("GITHUB_EVENT_NAME")}


def execution_guard(plan,env=None,event=None):
    env=os.environ if env is None else env
    validate_plan(plan)
    event=read(Path(env["GITHUB_EVENT_PATH"])) if event is None else event
    run=identity(env)
    need(plan["enabled"] is True and run["repository"]=="jiying2007/kws-data" and run["event"]=="push"
         and run["run_attempt"]=="1" and env.get("GITHUB_REF")=="refs/heads/main"
         and type(run["run_id"]) is str and re.fullmatch(r"[1-9][0-9]*",run["run_id"])
         and type(run["commit"]) is str and re.fullmatch(r"[0-9a-f]{40}",run["commit"]),"FIRST_PUBLIC_MAIN_ONLY")
    need(event.get("before")==plan["main_before"]==source.MAIN_BEFORE and event.get("after")==run["commit"]
         and event["after"]!=event["before"] and event.get("repository",{}).get("full_name")==run["repository"]
         and event["repository"].get("private") is False,"EXACT_INDEPENDENT_RECOVERY_SUCCESSOR")
    return run


class Deadline(BaseException):
    pass


def set_deadline(seconds):
    def stop(signum,_frame):raise Deadline("FINITE_STAGE_DEADLINE_OR_TERM: "+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(seconds)


def file_manifest(root):
    root=Path(root);rows=[]
    for path in sorted(root.rglob("*")):
        need(not path.is_symlink(),"NO_OUTPUT_SYMLINK")
        if path.is_file() and path.relative_to(root).as_posix()!="FILES.json":
            name=path.relative_to(root).as_posix();need("private-model" not in name.split("/"),"NO_WEIGHTS_IN_DELIVERY")
            raw=regular(path,allow_empty=True);rows.append({"path":name,"bytes":len(raw),"sha256":sha(raw)})
    need(len(rows)<=10000 and sum(r["bytes"] for r in rows)<=source.MAX_EXPANDED,"FINITE_PACKET_BOUNDS")
    return {"schema":SCHEMA+"/files","files":rows}


def closed_readback(root):
    need(read(Path(root)/"FILES.json")==file_manifest(root),"EXACT_CLOSED_PACKET_FILES")


def retain_sources(output,plan):
    for name in sorted(set(CODE_PATHS.values())|set(plan["helper_sha256"])):
        target=within(output,"code/"+name);target.parent.mkdir(parents=True,exist_ok=True)
        put(target,regular(ROOT/name,256*1024))


def packet_readback(path,plan):
    closed_readback(path)
    for name in set(CODE_PATHS.values())|set(plan["helper_sha256"]):
        need(regular(within(path,"code/"+name))==regular(ROOT/name),"EXECUTED_SOURCE_BYTE_HANDOFF")


def prepare(plan,inputs,output):
    run=execution_guard(plan);set_deadline(plan["preparation_deadline_seconds"])
    try:
        result=source.package_inputs(plan,inputs,output,run)
        retain_sources(Path(output),plan)
        with source.readback_package(output,plan):pass
        once(Path(output)/"FILES.json",file_manifest(output))
    finally:signal.alarm(0)
    return result


def observe(plan,provider,source_root,output,assets):
    run=execution_guard(plan);packet_readback(source_root,plan)
    need(read(Path(source_root)/"RESULT.json").get("run")==run,"SAME_NEW_RUN_REUSE_PACKET")
    output,assets=Path(output).absolute(),Path(assets).absolute()
    need(output.parent.is_dir() and assets.parent.is_dir() and not output.exists() and not assets.exists()
         and output!=assets and output not in assets.parents and assets not in output.parents,"DISTINCT_NEW_OUTPUT_AND_PRIVATE_MODEL")
    set_deadline(plan["observation_deadline_seconds"])
    try:
        with source.readback_package(source_root,plan) as material:
            result=observer.run(provider=provider,source_root=material["prepared_root"],output_root=output,
                asset_root=assets,plan=plan,run_identity=run,input_rows=material["manifest"]["rows"])
    finally:
        signal.alarm(0)
        if assets.exists():
            need(assets.is_dir() and not assets.is_symlink(),"PRIVATE_MODEL_CLEANUP_ROOT");shutil.rmtree(assets)
    once(output/"FILES.json",file_manifest(output))
    return result


def copy_tree(old,new):
    old,new=Path(old),Path(new);new.mkdir(parents=True,exist_ok=False)
    for path in sorted(old.rglob("*")):
        need(not path.is_symlink(),"NO_HANDOFF_SYMLINK")
        if path.is_file():
            target=within(new,path.relative_to(old).as_posix());target.parent.mkdir(parents=True,exist_ok=True)
            put(target,regular(path,allow_empty=True))


def combine(plan,source_root,observers_root,output):
    run=execution_guard(plan);packet_readback(source_root,plan)
    need(read(Path(source_root)/"RESULT.json").get("run")==run,"SAME_NEW_COMBINE_REUSE_PACKET")
    output=source.new_directory(output);copy_tree(source_root,output/"recovery")
    roots={}
    for provider in observer.PROVIDERS:
        old=Path(observers_root)/("kokoro-context32-maskfix-observer-"+provider)
        new=output/"observers"/provider
        if old.is_dir():
            if (old/"FILES.json").is_file():closed_readback(old)
            copy_tree(old,new)
        else:new.mkdir(parents=True,exist_ok=False)
        roots[provider]=new
    spec=importlib.util.spec_from_file_location("context32_maskfix_pure_compare",HERE/"english_kokoro_context32_mask_compare.py")
    comparator=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparator)
    with source.readback_package(source_root,plan) as material:
        comparison=comparator.compare(prepared_root=material["prepared_root"],observer_roots=roots,
            source_plan=material["source_plan"],plan=plan,run_identity=run)
    need(comparison.get("training_allowed") is False and comparison.get("automatic_expansion") is False
         and comparison.get("old_labels_modified") is False,"DIAGNOSTIC_COMPARISON_ONLY")
    comparison_pin=once(output/"COMPARISON.json",comparison)
    counts=comparison["providers"]
    known=all(p["actual_forward_attempts"] is not None for p in counts.values())
    attempts=sum(p["actual_forward_attempts"] for p in counts.values()) if known else None
    returned=sum(p["native_forward_returns"] for p in counts.values()) if known else None
    completed=sum(p["completed"] for p in counts.values()) if known else None
    lower=sum(p["forward_attempts_lower_bound"] for p in counts.values());upper=sum(p["forward_attempts_upper_bound"] for p in counts.values())
    need(0<=lower<=upper<=32 and (not known or 0<=completed<=returned<=attempts<=32),"TOTAL32_FIRST_FORWARD_BOUND")
    result={"schema":SCHEMA,"stage":"combine","run":run,"source_run":source.PRIOR_RUN,
        "status":"DIAGNOSTIC_RECOVERY_PACKET_RETAINED","declared_inputs":16,"declared_observation_slots":32,
        "observer_counts":counts,"actual_forward_attempts":attempts,"native_forward_returns":returned,"completed":completed,
        "counter_scope":"complete_terminal_receipts" if known else "unknown_actual_count_retained_bounds",
        "retained_forward_attempts_lower_bound":lower,"forward_attempts_upper_bound":upper,
        "maximum_total_model_calls":32,"retry_count":0,"recovery_of_failed_preprocessing":True,
        "prior_processor_attempts":32,"prior_model_constructions":2,"prior_forward_attempts":0,
        "reused_existing_full_inputs":16,"new_derivations":0,"new_tts_calls":0,"new_text_asr_calls":0,
        "new_embedding_calls":0,"new_training_calls":0,"forward_replays":0,"old_observed_crop_forwards":0,
        "source_material_reused":True,"phone_first_forwards_on_reused_material":attempts,"new_model_calls_during_join":0,
        "diagnostic_only":True,"training_allowed":False,"training_snapshot_allowed":False,
        "automatic_expansion":False,"old_labels_modified":False,"original_probe_decision":"STOP_DIAGNOSTIC_NO_EXPANSION",
        "high_quality_model_passed":False,"comparison_sha256":comparison_pin["sha256"]}
    once(output/"EXECUTION_PLAN.json",plan);once(output/"RESULT.json",result);once(output/"FILES.json",file_manifest(output))
    return result


def source_check(plan):
    validate_plan(plan);observer.source_preflight()
    spec=importlib.util.spec_from_file_location("context32_maskfix_processor_source_check",HERE/"english_kokoro_context32_processor_check.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.source_preflight()
    return {"schema":SCHEMA,"source_check":"PASS","enabled":plan["enabled"],"existing_full_inputs":16,
        "maximum_first_forwards":32,"prior_processor_attempts":32,"prior_model_constructions":2,
        "new_derivations":0,"new_model_calls":0,"new_network_calls":0,"training_allowed":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--plan",required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    for name in ("source-check","guard","prepare","source-readback","combine"):group.add_argument("--"+name,action="store_true")
    group.add_argument("--observe",choices=observer.PROVIDERS)
    for name in ("inputs","source","observers","output","assets"):parser.add_argument("--"+name)
    args=parser.parse_args(argv);plan=validate_plan(read(args.plan))
    if args.source_check:result=source_check(plan)
    elif args.guard:result={"schema":SCHEMA,"run":execution_guard(plan),"allowed":True,"maximum_first_forwards":32}
    elif args.prepare:
        need(args.inputs and args.output,"PREPARE_ARGS");result=prepare(plan,args.inputs,args.output)
    elif args.source_readback:
        need(args.source,"READBACK_ARGS");packet_readback(args.source,plan)
        with source.readback_package(args.source,plan) as material:
            result={"schema":SCHEMA,"source_ready":True,"existing_full_inputs":len(material["manifest"]["rows"]),
                "new_derivations":0,"new_model_calls":0,"prior_forward_attempts":0}
    elif args.observe:
        need(args.source and args.output and args.assets,"OBSERVE_ARGS");result=observe(plan,args.observe,args.source,args.output,args.assets)
    else:
        need(args.source and args.observers and args.output,"COMBINE_ARGS");set_deadline(plan["combine_deadline_seconds"])
        try:result=combine(plan,args.source,args.observers,args.output)
        finally:signal.alarm(0)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
    return (0 if result["status"]=="COMPLETE_DIAGNOSTIC" else 1) if args.observe else 0


if __name__=="__main__":raise SystemExit(main())
