"""Verify saved B16/B15 primary model evidence; never train/infer or select thresholds."""
from hashlib import sha256
import io,json,sys,zipfile
from pathlib import Path

RECEIPT=Path(__file__).with_name("RECEIPT.json")

def require(cond,msg):
    if not cond:raise ValueError(msg)

def verify(artifact):
    import numpy as np
    doc=json.loads(RECEIPT.read_bytes())
    raw=Path(artifact).read_bytes()
    locked=doc["run"]
    require(len(raw)==locked["artifact_bytes"] and sha256(raw).hexdigest()==locked["artifact_sha256"],
            "exact completed B16 original archive")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        require(z.testzip() is None and set(z.namelist())=={"RESULT.json","ALL_FRAME_LOGITS.npz"},
                "exact two original score members with valid CRC")
        result=z.read("RESULT.json"); logits=z.read("ALL_FRAME_LOGITS.npz")
    require(len(result)==locked["result_json_bytes"] and
            sha256(result).hexdigest()==locked["result_json_sha256"] and
            sha256(logits).hexdigest()==locked["npz_sha256"],"complete saved result and frame-array SHA")
    r=json.loads(result)
    need=doc["source"]
    require(r["schema"]=="english-v6-b16-b15-unseen-primary-four-frozen-models-v1" and
            r["panel_gate"]=="NO_PROMOTION" and r["source_gate"]=="SOURCE_GATE_PASS"
            and not r["product_released"] and not r["promotion_allowed"] and
            r["actual_forwards"]==need["total_frame_logits_arrays"]==912 and r["optimizer_updates"]==0,
            "unmodified failed acoustic primary and no promotion")
    require(len(r["records"])==need["original_total"]==76 and
            sum(v["positive"] for v in r["records"])==need["synthetic_admitted_positive"] and
            len({v["pcm_sha256"] for v in r["records"]})==76,
            "unique 76 actual source originals")
    for m,model in doc["models"].items():
        require(r["model_pt_sha256"][m]==model["pt_sha256"],"same frozen actual model "+m)
    for cohort,arms in doc["cohort_events"].items():
        for arm,expected in arms.items():
            actual=[r["summary"][cohort][arm][g]["events" if cohort!="synthetic_positive_hits" else "with_event"]
                    for g in ("1.0","0.75","0.5")] if cohort!="synthetic_positive_hits" else [
                        r["summary"]["synthetic_positive"][arm][g]["with_event"] for g in ("1.0","0.75","0.5")]
                    ]
            require(actual==expected,"cohort original/gain score "+cohort+"/"+arm)
    require(len(r["added_ordinary_events"])==4 and
            not (r["lost_positives"] or r["extra_positive_events"]) and
            r["natural_above_B0_gains"]==["1.0","0.75","0.5"] and
            not r["replication_open_allowed"],"frozen negative regression blocks model selection")
    with np.load(io.BytesIO(logits),allow_pickle=False) as npz:
        require(len(npz.files)==912 and all(npz[name].ndim==2
                   and npz[name].shape[1]==4 and np.isfinite(npz[name]).all()
                   for name in npz.files),"all finite genuine four-class logit arrays")
    return {"status":"PASS_SAVED_NO_PROMOTION","forwards":912,
            "new_originals":76,"added_ordinary_event_conditions":4,
            "source_gate":"SOURCE_GATE_PASS","model_gate":"NO_PROMOTION",
            "field_frr":"NOT_MEASURED","field_far":"NOT_MEASURED","ssc305":"NOT_RUN"}

if __name__=="__main__":
    if len(sys.argv)!=2:raise SystemExit("usage: verify_saved.py EXACT_ORIGINAL_ARTIFACT_ZIP")
    print(json.dumps(verify(sys.argv[1]),sort_keys=True))
