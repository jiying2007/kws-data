"""One-command bounded TTS + blinded three-ASR + optional synthetic-clone R&D.

No private human samples, human-gold fabrication, prod CTC optimization or
shipping status change. Resume skips every already started cell/recognizer.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import campaign as c
import runner
import screen
import asr_dispatch
import receipt_contract as receipts


def _subset_lock(lock,plan):
    used={(v["model_id"],v["model_revision"]) for v in plan["voices"]}
    rows=[r for r in lock["models"] if (r["model_id"],r["revision"]) in used]
    return {"schema":runner.MODEL_LOCK_SCHEMA,"models":rows}


def _record_once(path,value):
    if path.exists():
        existing=c.read_json(path,2*1024*1024)
        c.require(c.canonical(existing)==c.canonical(value),"EXISTING_REPORT_DRIFT")
    else:
        runner._once(path,value)


def _started_count(workspace,kind):
    total=0
    for folder in ("r0","r1","clone"):
        root=Path(workspace)/folder/kind
        if kind=="tts":
            total+=sum(path.name=="started.json" for path in root.glob("cells/*/started.json"))
        else:
            total+=sum(path.name=="started.json" for path in root.glob("*/*/started.json"))
    return total


def execute(plan,workspace,model_lock,model_root,adapter_lock,*,max_tts,
            max_asr,rounds,clone_revision=None,device="cpu",tts_backend_factory=None,
            asr_executor=None,writer=None):
    c.validate_plan(plan)
    adapter_binding=receipts.binding(plan,adapter_lock)
    c.require(type(max_tts) is int and 0<=max_tts<=64 and
              type(max_asr) is int and 0<=max_asr<=192 and
              type(rounds) is int and rounds in (1,2),"EXPLICIT_BOUNDED_BUDGET")
    if clone_revision is not None:
        c.require(type(clone_revision) is str and
                  c._REV.fullmatch(clone_revision),"PINNED_CLONE_REVISION_REQUIRED")
    root=Path(workspace)
    if not root.exists():
        root.mkdir(parents=True)
    c.require(root.is_dir() and not root.is_symlink(),"WORKSPACE_REQUIRED")
    _record_once(root/"campaign-lock.json",{
        "schema":"kws-autonomous-tts-asr-controller-v1",
        **adapter_binding,
        "max_tts":max_tts,"max_asr":max_asr,"rounds":rounds,
        "clone_revision":clone_revision,
        "training_admitted":False,"shipping_approved":False})
    runner_assets={}
    def model_for(subplan):
        key=tuple(sorted({(v["model_id"],v["model_revision"]) for v in subplan["voices"]}))
        if key not in runner_assets:
            runner_assets[key]=runner.validate_models(subplan,
                _subset_lock(model_lock,subplan),model_root)
        if tts_backend_factory is None:
            return runner.QwenOfflineBackend(runner_assets[key],root/"references",device)
        return tts_backend_factory(subplan)
    def one_stage(stage,subplan):
        c.validate_plan(subplan)
        work=root/stage
        receipts.verify_root(work/"asr",receipts.binding(subplan,adapter_lock),
                             "asr-autonomous-lock-v1")
        work.mkdir(exist_ok=True)
        _record_once(work/"plan.json",subplan)
        already_tts=_started_count(root,"tts")
        remaining_tts=max_tts-already_tts
        c.require(remaining_tts>=0,"TTS_BUDGET_ALREADY_EXCEEDED")
        if remaining_tts>0:
            stage_limit=min(remaining_tts,len(subplan["cells"]))
            if writer is None:
                runner.run_batch(subplan,work/"tts",model_for(subplan),
                                 max_calls=stage_limit)
            else:
                runner.run_batch(subplan,work/"tts",model_for(subplan),
                                 max_calls=stage_limit,writer=writer)
        already_asr=_started_count(root,"asr")
        remaining_asr=max_asr-already_asr
        c.require(remaining_asr>=0,"ASR_BUDGET_ALREADY_EXCEEDED")
        if remaining_asr>0:
            kwargs={"max_calls":min(remaining_asr,3*len(subplan["cells"])),
                    "execute":True}
            if asr_executor is not None:
                kwargs["executor"]=asr_executor
            asr_dispatch.dispatch(subplan,work/"tts",work/"asr",adapter_lock,**kwargs)
        report=screen.screen_campaign(subplan,work/"tts",work/"asr",adapter_lock)
        _record_once(work/"screen.json",report)
        return report

    reports={}
    reports["r0"]=one_stage("r0",plan)
    if rounds==2:
        next_plan=screen.plan_refinement(plan,reports["r0"])
        if next_plan is not None:
            reports["r1"]=one_stage("r1",next_plan)

    if clone_revision is not None:
        screened={r["cell_id"]:r for r in reports["r0"]["rows"]}
        selected=[]
        families=set()
        byvoice={v["voice_id"]:v for v in plan["voices"]}
        for cell in plan["cells"]:
            row=screened[cell["cell_id"]]
            if not (cell["kind"]=="wake" and
                    cell["mode"] in ("design","preset") and
                    row["status"]=="MACHINE_SCREENED_POSITIVE_CANDIDATE" and
                    cell["source_family_id"] not in families):
                continue
            families.add(cell["source_family_id"])
            source=byvoice[cell["voice_id"]]
            existing=root/"references"/(cell["cell_id"]+".reference.json")
            if existing.exists():
                reference=c.read_json(existing)
                pcm,_=runner.validate_reference(reference,root/"references")
                c.require(reference["audio_id"]==cell["cell_id"] and
                          reference["source_family_id"]==cell["source_family_id"] and
                          reference["reference_text"]==cell["target_normalized"],
                          "RESUMED_REFERENCE_IDENTITY_DRIFT")
            else:
                reference=screen.create_clone_reference(cell,row,
                    root/"r0/tts",root/"references")
            selected.append({"voice_id":"clone-"+source["voice_id"],
                "mode":"clone","preset":None,"description":None,
                "reference":reference,
                "source_family_id":source["source_family_id"],
                "role":source["role"],
                "model_id":c.MODEL_FAMILIES["clone"],
                "model_revision":clone_revision,
                "model_license":"Apache-2.0"})
            if len(selected)>=3:break
        if selected:
            clone_plan=c.build_plan(selected,batch_id=plan["batch_id"]+"-cl")
            reports["clone"]=one_stage("clone",clone_plan)
    return {"schema":"autonomous-kws-synthetic-controller-summary-v1",
        "stages":{name:report["counts"] for name,report in reports.items()},
        "ttl_tts_started":_started_count(root,"tts"),
        "ttl_asr_started":_started_count(root,"asr"),
        "max_tts":max_tts,"max_asr":max_asr,
        "real_human_labels":0,"new_product_training_runs":0,
        "training_admitted":False,"qualification_allowed":False,
        "shipping_approved":False,
        "status":"SYNTHETIC_RND_AUTOMATED_NOT_PRODUCT_RELEASE"}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",required=True)
    p.add_argument("--workspace",required=True)
    p.add_argument("--model-lock",required=True)
    p.add_argument("--model-root",required=True)
    p.add_argument("--adapter-lock",required=True)
    p.add_argument("--max-tts",type=int,default=36)
    p.add_argument("--max-asr",type=int,default=108)
    p.add_argument("--rounds",type=int,choices=(1,2),default=2)
    p.add_argument("--clone-revision")
    p.add_argument("--device",choices=("cpu","cuda:0"),default="cpu")
    p.add_argument("--execute",action="store_true")
    args=p.parse_args(argv)
    try:
        plan=c.validate_plan(c.read_json(args.plan))
        models=c.read_json(args.model_lock)
        adapters=c.read_json(args.adapter_lock)
        asr_dispatch.check_adapters(adapters)
        if not args.execute:
            print(json.dumps({"prospective_tts":len(plan["cells"]),
                "max_tts":args.max_tts,"max_asr":args.max_asr,
                "new_model_calls":0,"training_admitted":False}))
            return 0
        c.require(os.environ.get("HF_HUB_OFFLINE")=="1" and
                  os.environ.get("TRANSFORMERS_OFFLINE")=="1",
                  "OFFLINE_RUNTIME_REQUIRED")
        result=execute(plan,args.workspace,models,args.model_root,adapters,
            max_tts=args.max_tts,max_asr=args.max_asr,rounds=args.rounds,
            clone_revision=args.clone_revision,device=args.device)
    except (ValueError,OSError) as err:
        print("AUTONOMOUS_SYNTHETIC_NOT_ADMITTED: "+type(err).__name__,
              file=sys.stderr)
        return 2
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
