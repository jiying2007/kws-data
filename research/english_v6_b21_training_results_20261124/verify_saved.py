"""Audit actual B21 saved artifacts with NumPy only; never import Torch or infer.

Requires untouched weights/producer ZIPs, their actual API receipts, the run's
recursive Git tree and the original byte-pinned B17 training reference.
"""
import argparse
from collections import Counter, OrderedDict, defaultdict
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import pickle
import re
import sys
import zipfile
import numpy as np

B0 = "1ae8a2b20c8b6991f925dd4c626af2f36b308adee8733a3a7215818716f565d7"
B9 = "7dba86a3fcda6e5f05cb80b414dc372c70a5a03b3aee783f168441a4871e7655"
ASSESSMENT = "65ad9c7204755f1cd0b639added163387af39604818f9bd26b9da9fe8bd098fc"
DESIGN = "76a1fd1353de00f712023a2b20256a8e622557551823cacd8b808d3f0549b0a1"
TRAINER = "205654f5afa0f8e55e10016354fec5b4c57057600df66daf13e584fea95ad3e3"
PRODUCER = "2409ca5201302f1e4c5fa50560f68693be3322ee0bf4521a753567bffac8b637"
B17 = "1e31bd9c801657c7850fa6fb05c5468cbe8b4ab5bb459b0146c07c4c905ff86d"
MECHANISM = "d666112703dec3c3bdd2635a8259cbef108511e49f9f9ba4646c5108cb348e8d"
BRANCH = "research/en-v6-b21-blank-retention-train-once-20261124"
WORKFLOW = ".github/workflows/english-v6-b21-blank-retention-train-once.yml"
PREFIXES = (6400, 12800, 19200)


def require(ok, reason):
    if not ok: raise ValueError(reason)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def blob(raw): return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def readj(path): return json.loads(Path(path).read_bytes())


def verified_zip(raw):
    archive = zipfile.ZipFile(io.BytesIO(raw))
    names = archive.namelist()
    require(archive.testzip() is None and len(names) == len(set(names)), "complete unique ZIP members")
    require(all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts and "\\" not in n for n in names), "safe ZIP names")
    return archive


def artifact(raw, meta, run, head):
    require(type(meta["id"]) is int and meta["id"] > 0 and meta["expired"] is False
        and meta["workflow_run"]["id"] == run and meta["workflow_run"]["head_sha"] == head
        and meta["size_in_bytes"] == len(raw) and meta["digest"] == "sha256:" + sha(raw),
        "actual original artifact bytes and server run identity")
    return verified_zip(raw)


def checkpoint_tensors(raw):
    """Restricted data-only decoding; permit two known Torch state-dict constructors."""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        require(archive.testzip() is None, "checkpoint ZIP CRC")
        pkl = [n for n in archive.namelist() if n.endswith("/data.pkl")]
        require(len(pkl) == 1, "one tensor pickle")
        base = pkl[0].rsplit("/", 1)[0]
        require(archive.read(base + "/byteorder") == b"little", "checkpoint byte order")

        def tensor(storage, offset, shape, stride, requires_grad, hooks, *extra):
            require(not extra and isinstance(storage, tuple) and len(storage) == 3,
                    "only expected state tensor representation")
            key, dtype, count = storage
            raw_values = archive.read(base + "/data/" + key)
            require(len(raw_values) == count * np.dtype(dtype).itemsize,
                    "storage byte size")
            return np.ndarray(tuple(shape), dtype=dtype, buffer=raw_values,
                              offset=offset * np.dtype(dtype).itemsize,
                              strides=tuple(s * np.dtype(dtype).itemsize for s in stride)).copy()

        class DataOnly(pickle.Unpickler):
            def find_class(self, module, name):
                if (module, name) == ("collections", "OrderedDict"):
                    return OrderedDict
                if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                    return tensor
                if (module, name) == ("torch", "FloatStorage"):
                    return "<f4"
                raise ValueError("unexpected pickle class " + module + "." + name)

            def persistent_load(self, pid):
                require(isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage"
                        and pid[1] == "<f4" and pid[3] == "cpu"
                        and isinstance(pid[4], int) and pid[4] >= 0
                        and re.fullmatch(r"[0-9]+", pid[2]) is not None,
                        "bounded float CPU storage reference")
                return pid[2], pid[1], pid[4]

        result = DataOnly(io.BytesIO(archive.read(pkl[0]))).load()
        require(isinstance(result, OrderedDict) and all(isinstance(k, str) and isinstance(v, np.ndarray)
                                                      for k, v in result.items()), "plain tensor dictionary")
        return result

def audit(weights_raw, producer_raw, receipt, tree, b17_raw):
    metadata, producer_meta = receipt['artifact_metadata'], receipt['producer_artifact_metadata']
    run, commit = receipt['run_metadata'], receipt['commit_metadata']
    run_id, head = run['id'], run['head_sha']
    require(run['status'] == 'completed' and run['conclusion'] == 'success' and run['run_attempt'] == 1
        and run['event'] == 'push' and run['head_branch'] == BRANCH
        and run['repository']['full_name'] == 'jiying2007/kws-data' and run['repository']['private'] is False,
        'actual completed first public B21 training run')
    require(commit['sha'] == head and len(commit['parents']) == 1, 'actual single training predecessor')
    parent = commit['parents'][0]['sha']
    require(re.fullmatch(r'[0-9a-f]{40}', receipt['prepared_parent']) is not None
        and parent == receipt['prepared_parent'], 'registered reviewed main is actual single parent')
    require(tree['sha'] == commit['tree']['sha'] and tree['truncated'] is False, 'complete actual run commit tree')
    git_files = {item['path']:item for item in tree['tree']}
    require(len(git_files) == len(tree['tree']), 'unique actual Git tree paths')
    proof_zip = artifact(producer_raw, producer_meta, run_id, head)
    proof = json.loads(proof_zip.read('PRODUCER.json'))
    require(proof['schema'] == 'english-v6-b21-original-git-producer-snapshot-v1'
        and proof['role'] == 'train' and proof['run_id'] == str(run_id) and proof['run_head'] == head
        and proof['prepared_parent'] == parent and proof['run_attempt'] == 1 and proof['run_branch'] == BRANCH
        and proof['git_tree'] == tree['sha'] and proof['runtime_workflow'] == WORKFLOW
        and proof['runtime_workflow_preserved_hidden'] is proof['source_code_reads_only'] is True
        and proof['network_calls'] == proof['new_source_calls'] == proof['new_model_forwards'] == proof['optimizer_updates'] == 0
        and proof['promotion_allowed'] is False, 'actual Git-bound original scientific producer')
    expected_names = {'PRODUCER.json'} | {'repository/'+name for name in proof['files']}
    require({i.filename for i in proof_zip.infolist() if not i.is_dir()} == expected_names,
        'complete producer ZIP including hidden workflow and no extra files')
    require(set(proof['files']) == set(proof['python_files']) | set(proof['resource_files']) | {WORKFLOW},
        'explicit scientific code/resource closure')
    producer_files = {}
    for name,pin in proof['files'].items():
        raw = proof_zip.read('repository/'+name)
        entry = git_files[name]
        require(entry['type'] == 'blob' and entry['mode'] == pin['mode'] in ('100644','100755')
            and entry['sha'] == pin['git_blob_sha1'] == blob(raw)
            and pin['bytes'] == len(raw) and pin['sha256'] == sha(raw), 'actual Git blob and file bytes '+name)
        producer_files[name] = raw
    required_code = {
        'automation/english_v6_b21_competing_margin_pair.py': TRAINER,
        'automation/english_v6_b21_competing_margin_design.json': DESIGN,
        'automation/english_v6_b21_panels/assessment.json': ASSESSMENT,
        'automation/english_v6_b21_producer_snapshot.py': PRODUCER}
    require(all(sha(producer_files[k]) == v for k,v in required_code.items()), 'reviewed actual training implementation and protocols')
    require(b'include-hidden-files: true' in producer_files[WORKFLOW], 'producer upload preserves actual hidden workflow')
    require(sha(b17_raw) == B17 and len(b17_raw) == 2506252, 'byte-authenticated original B17 TRAIN reference')
    old = verified_zip(b17_raw)
    old_source = json.loads(old.read('SOURCE.json'))
    old_draw = json.loads(old.read('COMMON_DRAW.json'))
    rows = {slot[kind]['id']:slot[kind] for slot in old_draw for kind in ('ordinary','positive','natural')}
    require(len(rows) == 88, 'all original prior TRAIN source identities')
    z = artifact(weights_raw, metadata, run_id, head)
    members = {i.filename:dict(bytes=len(z.read(i)),sha256=sha(z.read(i))) for i in z.infolist() if not i.is_dir()}
    doc = lambda name: json.loads(z.read(name))
    result,started,source = doc('RESULT.json'),doc('STARTED.json'),doc('SOURCE.json')
    for value in (result,started):
        require(value['schema'] == 'english-v6-b21-positive-competing-margin-pair-v1'
            and value['run_id'] == str(run_id) and value['head'] == head and value['parent'] == parent
            and value['design_sha256'] == DESIGN and value['assessment_sha256'] == ASSESSMENT
            and value['dev_cal_locked_forwards'] == value['new_tts_asr_calls'] == 0 and value['promotion_allowed'] is False,
            'actual saved run identity and frozen execution boundaries')
    require(result['status'] == 'TRAINED_PAIR_REQUIRES_FRESH_INDEPENDENT_DEV'
        and result['parent_pt_sha256'] == source['parent_pt_sha256'] == B0 and source['positive_teacher_pt_sha256'] == B9
        and result['actual_optimizer_updates'] == started['maximum_optimizer_updates'] == 384
        and result['trained_model_parameters'] == 43876 and result['seed'] == 20261120 and result['common_learning_rate'] == .00006
        and source['design_sha256'] == DESIGN and source['independent_assessment_sha256'] == result['independent_assessment_sha256'] == ASSESSMENT
        and source['forbidden_sources_opened'] is source['numerical_B17_replay_claimed'] is result['numerical_B17_replay_claimed'] is False
        and source['shared_schedule_repaired'] is True and result['actual_new_diagnostic_forwards'] == 0
        and result['reused_saved_diagnostic_arrays'] == 528, 'fixed prospective pair without new DEV/diagnostic inference')
    require(source['input_zip_sha256'] == {**old_source['input_zip_sha256'],'b17.zip':B17}, 'all16 original input archive hash receipts')
    for field in ('base_train','natural_train','positive_retention'):
        require(source[field] == old_source[field], 'exact unchanged original '+field)
    require(len(source['base_train']) == result['baseline_originals'] == 79
        and len(source['natural_train']) == result['natural_originals'] == 9
        and len({r['source_group_id'] for r in source['natural_train']}) == result['natural_families'] == 8,
        '79 synthetic and nine original owner-reviewed natural TRAIN windows')
    positive = source['positive_retention']
    require(len(positive) == result['positive_path_combinations'] == 120
        and len({(r['id'],r['gain']) for r in positive}) == 120
        and len({r['id'] for r in positive}) == result['positive_originals'] == 43
        and len({r['group'] for r in positive}) == result['positive_families'] == 34
        and Counter(r['gain'] for r in positive) == {'1.0':42,'0.75':42,'0.5':36}, 'same120 admitted positive teacher paths')
    for filename,field in [('COMMON_DRAW.json','common_draw_sha256'),('DRAW_COVERAGE.json','draw_coverage_sha256'),('TRAIN_MECHANISM.json','train_mechanism_sha256')]:
        require(sha(z.read(filename)) == result[field], 'bound saved evidence '+filename)
    require(sha(z.read('TRAIN_MECHANISM.json')) == MECHANISM,
        'exact preregistration saved-TRAIN mechanism independently recomputed before optimizer')
    mechanism=doc('TRAIN_MECHANISM.json')
    require(mechanism['status'] == 'TRAIN_SIGNAL' and mechanism['authenticated_arrays'] == 528
        and len(mechanism['qualifying_groups']) == 10 and mechanism['qualifying_conditions'] == 13
        and mechanism['blank_only_frame_count'] == 37 and mechanism['new_model_forwards'] == mechanism['optimizer_updates'] == 0,
        'zero-new-forward saved TRAIN blank-suppression mechanism')
    groups={kind:defaultdict(list) for kind in ('ordinary','positive')}
    natural={r['id']:r for r in source['natural_train']}
    for uid,row in rows.items():
        if uid not in natural: groups['positive' if row['mono_class'] else 'ordinary'][row['source_group_id']].append(row)
    gids=sorted(groups['ordinary'])
    require(gids == sorted(groups['positive']) and len(gids) == 35, 'same35 paired source groups')
    widths={g:max(len(groups[k][g]) for k in groups) for g in gids}
    require(Counter(widths.values()) == {1:26,2:9}, 'same structural source group widths')
    order=gids*5+sorted(gids,key=lambda g:(-widths[g],g))[:17]
    draw=doc('COMMON_DRAW.json');visits=Counter();exposed=defaultdict(set);nat_exposed=defaultdict(set)
    require(len(draw)==192,'complete common192-step draw')
    for step,(slot,gid) in enumerate(zip(draw,order)):
        visit=visits[gid];prefix=PREFIXES[(gids.index(gid)+visit//widths[gid])%3]
        require(slot['step']==step+1 and slot['source_group_id']==gid and slot['group_visit']==visit
            and slot['group_width']==widths[gid] and slot['prefix']==prefix, 'structural finite draw and prefix')
        for kind in groups:
            candidates=sorted(groups[kind][gid],key=lambda r:r['id'])
            expected=candidates[visit%len(candidates)]
            require(slot[kind]==expected and expected['role']=='train','actual original source per paired step')
            exposed[expected['id']].add(prefix)
        require(slot['natural']==old_draw[step]['natural'] and slot['natural']==natural[slot['natural']['id']],
            'unchanged owner-reviewed natural draw')
        nat_exposed[slot['natural']['id']].add(prefix);visits[gid]+=1
    require(len(exposed)==79 and len(nat_exposed)==9
        and all(v==set(PREFIXES) for v in [*exposed.values(),*nat_exposed.values()]), 'all88 original sources see allthree prefixes')
    coverage=dict(synthetic_originals=79,natural_originals=9,synthetic_original_prefix_pairs=237,natural_original_prefix_pairs=27,
        synthetic_prefixes={k:sorted(v) for k,v in sorted(exposed.items())},natural_prefixes={k:sorted(v) for k,v in sorted(nat_exposed.items())},
        all_originals_all_three_prefixes=True,score_driven_sampling=False)
    require(doc('DRAW_COVERAGE.json')==coverage,'independently recomputed full original coverage')
    teacher=doc('B0_ORDINARY_TEACHER.json')
    require(teacher['parent_pt_sha256']==B0 and teacher['logits_sha256']==sha(z.read('B0_ORDINARY_TEACHER_LOGITS.npz'))
        and teacher['positive_rows']==teacher['dev_cal_locked_forwards']==0 and teacher['reused_for_both_arms'] is True,
        'one unchanged ordinary B0 teacher shared across arms')
    expected_teacher={(s[k]['id'],gain,s['prefix']) for s in draw for k in ('ordinary','natural') for gain in (1.,.5)}
    with np.load(io.BytesIO(z.read('B0_ORDINARY_TEACHER_LOGITS.npz')),allow_pickle=False) as arrays:
        observed,keys=set(),set()
        for r in teacher['rows']:
            key=r['id'],r['gain'],r['prefix_samples'];src=rows[r['id']];m=arrays[r['array']]
            require(key not in observed and r['array'] not in keys and r['role']==src['role']=='train'
                and r['mono_class']==src['mono_class']==0 and r['source_group_id']==src['source_group_id']
                and r['pcm_sha256']==src['source_pcm_sha256'] and np.isfinite(m).all()
                and m.shape==((src['samples']+r['prefix_samples']-512)//160+1,4)
                and r['frames']==len(m) and r['sha256']==sha(m.tobytes()),'complete true TRAIN teacher native frame grid')
            observed.add(key);keys.add(r['array'])
        require(observed==expected_teacher and keys==set(arrays.files)
            and len(keys)==teacher['actual_model_forwards']==result['actual_ordinary_teacher_forwards']==264,
            'exact264 unique required ordinary teacher forwards')
    retained=defaultdict(list)
    for r in positive:retained[r['group']].append(r)
    eligible_groups=sorted(retained)
    expected_forwards=192*6+sum(len(retained[eligible_groups[s%34]]) for s in range(192))
    tensors,frontend,model_pins,loss_summary={},{},{},{}
    with np.load(io.BytesIO(old.read('control/models/mero43-b17-control192.npz')),allow_pickle=False) as old_npz:
        for arm in ('control','challenger'):
            a=result[arm];mode='other_two_nonblank' if arm=='control' else 'other_three_including_blank'
            stem=arm+'/models/mero43-b21-'+arm+'192'
            require(a['stem']==stem and a['optimizer_updates']==192 and a['parent_pt_sha256']==B0
                and a['natural_loss_weight']==a['ordinary_ceiling_weight']==a['positive_retention_weight']==.08
                and a['positive_competitors']==mode
                and a['forward_counts']==dict(optimization=expected_forwards,checkpoint_reload_parity=2,onnx_export_trace=1),
                'paired common losses and sole positive competitor contrast')
            trace=doc(arm+'/OPTIMIZER_TRACE.json');retain=doc(arm+'/POSITIVE_RETENTION_TRACE.json')
            require(len(trace)==len(retain)==192,'all192 actual arm update records')
            for step,(tr,pr,slot) in enumerate(zip(trace,retain,draw)):
                group=eligible_groups[step%34]
                require(tr['step']==pr['step']==step+1 and tr['prefix']==slot['prefix']
                    and all(tr[k+'_id']==slot[k]['id'] for k in ('ordinary','positive','natural'))
                    and tr['ceiling_coefficient']==.08 and tr['positive_competitors']==mode
                    and pr['group']==group and pr['eligible_originals']==len({r['id'] for r in retained[group]})
                    and all(np.isfinite(tr[k]) and tr[k]>=0 for k in ('loss','positive_retention','ordinary_ceiling'))
                    and np.isfinite(pr['unweighted_loss']) and pr['unweighted_loss']>=0
                    and np.isclose(tr['positive_retention'],pr['unweighted_loss']*.08,atol=1e-7,rtol=1e-6),
                    'all384 actual step/source/prefix/loss records correspond to common draw')
            require(set(a['files'])=={Path(stem+'.'+ext).name for ext in ('pt','npz','onnx')},'exact three saved model files')
            for ext in ('pt','npz','onnx'):
                name=stem+'.'+ext
                require(a['files'][Path(name).name]==members[name],'actual saved model file digest '+name)
            state=checkpoint_tensors(z.read(stem+'.pt'))
            require(len(state)==20 and sum(v.size for v in state.values())==43876
                and all(v.dtype==np.dtype('<f4') and np.isfinite(v).all() for v in state.values()),'complete finite43876-parameter state')
            with np.load(io.BytesIO(z.read(stem+'.npz')),allow_pickle=False) as encoded:
                require(set(encoded.files)==set(state)|{'mel_filters','train_mean','train_scale'}
                    and all(np.array_equal(encoded[k],v) and old_npz[k].shape==v.shape for k,v in state.items()),
                    'PT and NPZ exact tensors, same frozen architecture')
                frontend[arm]={k:encoded[k].copy() for k in ('mel_filters','train_mean','train_scale')}
                require(all(np.array_equal(v,old_npz[k]) for k,v in frontend[arm].items()),'frozen original B0/B17 acoustic frontend')
            tensors[arm]=state;model_pins[arm]=dict(pt_sha256=sha(z.read(stem+'.pt')),files=a['files'])
            loss_summary[arm]=dict(first=trace[0]['loss'],last=trace[-1]['loss'],maximum_ordinary_ceiling=max(t['ordinary_ceiling'] for t in trace))
    require(tensors['control'].keys()==tensors['challenger'].keys(),'same paired state keys')
    differing=int(sum(np.count_nonzero(tensors['control'][k]!=v) for k,v in tensors['challenger'].items()))
    require(differing==result['different_trained_float_values'] and differing>0
        and result['actual_student_forwards']==2*(expected_forwards+3)==3670,'actual different trained arm tensors and complete forward counts')
    require('torch' not in sys.modules,'saved-data audit must not import model runtime')
    return dict(schema='english-v6-b21-saved-training-evidence-audit-v1',status='PASS_TRAINING_EVIDENCE_ONLY',
        run_id=run_id,run_head=head,parent=parent,git_tree=tree['sha'],artifact_id=metadata['id'],artifact_bytes=len(weights_raw),
        artifact_sha256=sha(weights_raw),producer_artifact_id=producer_meta['id'],producer_artifact_sha256=sha(producer_raw),
        producer_manifest_sha256=sha(proof_zip.read('PRODUCER.json')),producer_files_verified=len(proof['files']),
        trainer_sha256=TRAINER,design_sha256=DESIGN,assessment_sha256=ASSESSMENT,parent_pt_sha256=B0,positive_teacher_pt_sha256=B9,
        optimizer_updates=384,ordinary_teacher_arrays=264,student_forwards=3670,new_diagnostic_forwards=0,reused_saved_diagnostic_arrays=528,
        original_source_zip_hash_receipts=16,synthetic_originals=79,owner_reviewed_natural_windows=9,
        positive_retention_combinations=120,positive_retention_originals=43,positive_retention_groups=34,
        synthetic_original_prefix_pairs=237,natural_original_prefix_pairs=27,all_originals_all_three_prefixes=True,
        finite_model_parameters_per_arm=43876,differing_trained_float_values=differing,checkpoint_npz_tensor_identity=True,
        model_pins=model_pins,saved_trace_loss_summary=loss_summary,actual_new_local_model_forwards=0,actual_new_optimizer_updates=0,
        new_TTS_ASR_calls=0,dev_cal_locked_forwards=0,promotion_allowed=False,member_sha256=members,
        limits='Saved original artifacts and exact Git source verified only; no KWS replay, independent DEV quality, C11/ARM or board validation is claimed.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('weights',type=Path);p.add_argument('--producer-zip',required=True,type=Path)
    p.add_argument('--receipt',required=True,type=Path);p.add_argument('--git-tree',required=True,type=Path)
    p.add_argument('--b17-archive',required=True,type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    result=audit(a.weights.read_bytes(),a.producer_zip.read_bytes(),readj(a.receipt),readj(a.git_tree),a.b17_archive.read_bytes())
    raw=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if a.output:
        with a.output.open('x',encoding='utf-8') as f:f.write(raw)
    print(raw)


if __name__=='__main__':main()
