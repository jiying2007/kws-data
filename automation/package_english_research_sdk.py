"""Assemble the retained coverage research SDK; no model, compiler or external process.

Only the exact already verified coverage-fullPCM artifact is accepted. Rejected
teacher weights are preserved as evidence and cannot be selected by this entry.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import zipfile


HERE = Path(__file__).resolve().parent

PREVIOUS = {'run_id': 44049954828, 'artifact_id': 14313603148, 'name': 'english-coverage-fullpcm-44049954828', 'head_sha': '95aec73b0ac8b58fa74e62df35c3e443107619b1', 'bytes': 13172135, 'sha256': 'ad5494d3a7456d9c7e1fcd51fec727fa9d9a377a508459db6f88af5256170ba6'}
PREVIOUS_MANIFEST = '48879888a6690d080e0109f2ddc27a2c3d8a36208562e9b6baa8b0826e38e640'
CONSUMERS = {'native': {'bytes': 2452080, 'sha256': 'ae5cdef5223be6274bb61b8ed6ab4ce5fd067cdb8f1c95faeb8f38fab6bbf9d5'}, 'arm': {'bytes': 2862184, 'sha256': 'cdee7e5a2a918315abde4953894e04a8ff8136879df5088d8d81d8a7f9dc9dd3'}}
HELPERS = {'automation/english_coverage_fullpcm_once.py': {'bytes': 44356, 'sha256': '70a9ae5c36191a75f58e8819ac4e6e12f3f7e464e5891d1b1e5fdc648e9ae422'}, 'automation/english_hardneg_increment_once.py': {'bytes': 69681, 'sha256': 'ac89f4564bb3f93661beb02f01f7766616322a39096c94a4bdac84560f5a9ec1'}}


def frozen_helpers():
    modules=[]
    for path,name in [('automation/english_coverage_fullpcm_once.py','coverage_sdk_parser'),
                      ('automation/english_hardneg_increment_once.py','coverage_sdk_head_parser')]:
        local=HERE/Path(path).name
        if not local.is_file() or local.is_symlink():raise ValueError('regular frozen helper required: '+path)
        raw=local.read_bytes();pin={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        if pin!=HELPERS[path]:raise ValueError('frozen helper identity changed: '+path)
        modules.append(module(local,name))
    return modules


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def audit_fullpcm(archive, source, cases, expected_head, h, *, teacher=False):
    manifest = archive.manifest('MANIFEST.json', exact=True)
    result, receipt = archive.json('evidence/RESULT.json'), archive.json('evidence/SUPERVISOR_EXIT.json')
    expected_status = 'VERIFIED_TEACHER_RETENTION_HEAD_FULLPCM_NATIVE_ARM' if teacher else 'VERIFIED_NEW_COVERAGE_INSTALLED_C_FULLPCM'
    h.need(manifest['supervised_success'] is True and result.get('status') == expected_status
        and result.get('passed') is True and result.get('candidate_head') == expected_head
        and str(result.get('run_id')) == str(source['run_id']) and result.get('head_sha') == source['head_sha'], 'ACTUAL_SAME_HEAD_FULLPCM_SUCCESS')
    h.need(receipt.get('passed') is True and receipt.get('returncode') == 0 and receipt.get('timed_out') is False
        and receipt.get('error') is None and not receipt.get('cleanup', {}).get('descendants_found')
        and str(receipt.get('run_id')) == str(source['run_id']) and receipt.get('head_sha') == source['head_sha']
        and receipt.get('run_attempt') == 1, 'SUPERVISED_FULLPCM_SUCCESS')
    reports=[]
    for backend in ('native','arm'):
        for index,item in enumerate(cases):
            case=item['case']; rid=case['id']
            saved=archive.json('inputs/'+rid+'.reference.json')
            h.need(saved == {'case':case,'scores':item['reference'],'embeddings':item['embeddings']}, 'NEW_MODEL_SAVED_REFERENCE_BINDING')
            h.need(archive.read('inputs/'+rid+'.pcm') == item['pcm'], 'ORIGINAL_FULLPCM_BYTES')
            parsed=h.parse_trace(archive.read('traces/'+backend+'-'+rid+'.trace'),case)
            metadata=archive.json('evidence/'+backend+'-case'+str(index)+'.stdout')
            report=h.compare_trace(item,parsed,metadata);report['backend']=backend
            h.need(report['passed'] and archive.json('evidence/'+backend+'-case'+str(index)+'.comparison.json')==report,
                'ACTUAL_SAVED_TRACE_PARITY')
            reports.append(report)
    h.need(result.get('completed_comparisons')==len(reports)==10 and result['comparisons']==reports, 'EXACT_TEN_SAVED_COMPARISONS')
    return result


def readme(card):
    rid=card['cases'][0]['id']
    return f'''# 英文 KWS 研究 SDK

所选权重：`{card['selection']}`，BIN SHA256 `{card['head']['sha256']}`。

本包提供实际训练权重、native / ARM 静态库、公共 C 头文件、CMake 导入配置、命令行消费者、固定 PCM 样例和可重新核对的完整数值证据。当前模型为冻结声学 embedding 加 `1536→32→3` 分类头。强预训练 FSMN-CTC 仍是主线；这个研究模型没有被改称 FSMN-CTC。

## 已完成的软件范围

同一所选权重的 5 条固定已暴露回归 PCM，在 native 与 ARM/QEMU 上合计 10 次完整前端→embedding→分类头→事件运行。每个后端 168 行 embedding、93 个 ready 输出，共 18.04 秒不重复 PCM（两后端累计消费 36.08 秒）。检查全部维度、概率、时钟、ready、事件及 END 记录。新候选只使用其自身保存的参考输出；旧模型参考不替代新权重验证。

五条样例分别覆盖 Mero EOF 事件、Vosen 事件、历史 Vosen 漏检样例、背景和此前修复的 10dB Mero 案例。所选权重的实际事件见 MODEL_CARD.json；它们是软件回归数据，不能用来报告独立 FAR/FRR 或真实远场覆盖。

## 先核对文件

```sh
python3 -I -S -B verify_package.py
sha256sum -c SHA256SUMS
```

第一个命令只读保存字节，重新比较 10 条完整 trace；不运行模型。第二个命令检查清单。所选 BIN 在 `models/selected-head.bin` 与两套 SDK 的 `models/causal-head.bin` 中完全一致。NPZ 在 `models/selected-head.npz`，与 BIN 六个数组逐字节一致。

## 运行接口

输入是 **16kHz、单声道、signed PCM16 little-endian 原始 PCM**，不带 WAV 头。class 0 为背景，1 为 Hey Vosen，2 为 Hey Mero。阈值 0.8、连续 3 个 embedding 确认、13 个 embedding 冷却；这些参数与保存证据一致。

以下命令会实际执行推理，需要在准备好的主机上单独运行：

```sh
sdk-native/bin/ekws_pcm_stream sdk-native/models/causal-head.bin smoke < tests/{rid}.pcm
qemu-arm sdk-arm/bin/ekws_pcm_stream sdk-arm/models/causal-head.bin smoke < tests/{rid}.pcm
```

完整数值 trace 使用 `diagnostics/trace-native` 或 `diagnostics/trace-arm`，参数为 `HEAD.bin INPUT.pcm OUTPUT.trace`。该诊断消费者在已安装静态库上包装原有 head 调用；本包复核的是它的保存运行证据。原 SDK stdin 消费者是否另行重新执行，以 `MODEL_CARD.json` 字段为准。

C 集成使用对应 `sdk-*/include/ekws_research_stream.h` 和 `sdk-*/lib/libekws_research_stream.a`。先调用 bytes/alignment 查询，为每条流独立配置 arena 与 workspace；鉴别模型哈希后初始化。PCM 时钟须连续，丢帧/路由切换走 discontinuity，新的录音才 reset。回调失败会 poison 流，已消费样本和已发事件不得重放。回调时间是样本可用时钟，不是词尾标注。

## ARM 与 SSC305 边界

ARM 文件来自 Ubuntu 24.04 的 `arm-linux-gnueabihf-gcc 13.3.0` 研究构建，并通过 ARM/QEMU 数值回归。它们没有在 SSC305 实板运行，不能据 ARM ELF 格式就视为 SSC305 BSP 可直接部署。接入前需要按实际 BSP 工具链/sysroot、hard/soft-float ABI、libc/内核与 CPU 指令集核对并重建；随后在真实音频路径测 CPU、峰值 RAM、词尾回调延迟、温升和长稳。

目前尚无独立 FAR/FRR、词尾延迟、3–5 米声学验收或 SSC305 资源合格结论。Hey Mero 最终发音仍待实际美国英语证据标定；历史 TTS 请求与三路 ASR 均不能自动成为发音 gold。所有这些边界在 `MODEL_CARD.json` 保持显式未完成。

## 来源与复现

查看 `PROVENANCE.json` 中的原始公开 run / artifact / commit / SHA256；查看 `evidence/fullpcm/` 中的监督退出、实际 trace 与逐案例比较。库和头文件复用原已保存 SDK；仅所选 `causal-head.bin` 发生明确替换，旧 SDK 清单另存为 `evidence/original-sdk-MANIFEST.json`，不伪装为原包仍字节相同。

许可证保存在各 SDK 的 `share/KwsEnglishResearch/licenses/`。本次组包不解码音频、不训练、不编译、不执行模型，也不授予新的模型或数据准入。
'''


def assemble(args):
    h,inc=frozen_helpers()
    previous=h.Archive(args.previous,PREVIOUS)
    candidate=fullpcm=prepared=None
    try:
        previous.manifest('MANIFEST.json',exact=True)
        h.need(h.sha(previous.read('MANIFEST.json'))==PREVIOUS_MANIFEST,'FIXED_PRIOR_MANIFEST')
        teacher=False
        fullpcm_source=PREVIOUS;fullpcm=previous
        head,npz=previous.read('models/coverage-head.bin'),previous.read('models/coverage-head.npz')
        h.need(h.identity(head)==h.HEAD and h.identity(npz)==h.NPZ,'KNOWN_RETAINED_COVERAGE_WEIGHTS')
        candidate_source=h.SOURCES['quality'];selection='coverage-retained-research'
        cases=[]
        for case in h.CASES:
            reference=previous.json('inputs/'+case['id']+'.reference.json')
            h.need(reference['case']==case,'FIXED_OLD_CASE')
            cases.append({'case':case,'pcm':previous.read('inputs/'+case['id']+'.pcm'),
                'reference':reference['scores'],'embeddings':reference['embeddings']})
        inc.verify_saved_npz_bin(npz,head)
        result=audit_fullpcm(fullpcm,fullpcm_source,cases,h.identity(head),h,teacher=teacher)
        out=args.output.absolute();h.need(not out.exists() and out.parent.is_dir(),'NEW_DELIVERY_OUTPUT_ONLY')
        out.mkdir()
        # Copy installed SDK members exactly; head replacement is explicit.
        for name in sorted(previous.z.namelist()):
            if not name.startswith(('installed-base/sdk-native/','installed-base/sdk-arm/')): continue
            relative=name.removeprefix('installed-base/')
            raw=head if relative.endswith('/models/causal-head.bin') else previous.read(name)
            h.put(out/relative,raw)
            if '/bin/' in relative:(out/relative).chmod(0o500)
        for backend in ('native','arm'):
            h.need(h.identity((out/('sdk-'+backend)/'lib/libekws_research_stream.a').read_bytes())==h.SDK_LIBRARIES[backend],'PUBLISHED_LIBRARY_IDENTITY')
            raw=previous.read('bin/trace-'+backend);h.need(h.identity(raw)==CONSUMERS[backend],'PUBLISHED_TRACE_CONSUMER')
            h.validate_elf(raw,backend);h.put(out/'diagnostics'/('trace-'+backend),raw);(out/'diagnostics'/('trace-'+backend)).chmod(0o500)
        for name in ('ekws_stream_trace.c','ekws_stream_trace.h'):
            h.put(out/'diagnostics/source'/name,previous.read('source/'+name))
        h.put(out/'diagnostics/source/english_causal_head.h',previous.read('installed-base/source/research/english_final_embedding/english_causal_head.h'))
        h.put(out/'models/selected-head.bin',head);h.put(out/'models/selected-head.npz',npz)
        for item in cases:
            rid=item['case']['id'];h.put(out/'tests'/(rid+'.pcm'),item['pcm'])
            h.put(out/'tests'/(rid+'.reference.json'),h.canonical({'case':item['case'],'scores':item['reference'],'embeddings':item['embeddings']}))
        for name in ('RESULT.json','SUPERVISOR_EXIT.json'):
            h.put(out/'evidence/fullpcm'/name,fullpcm.read('evidence/'+name))
        for backend in ('native','arm'):
            for index,item in enumerate(cases):
                rid=item['case']['id']
                for suffix in ('stdout','comparison.json','started.json','exit.json'):
                    member=backend+'-case'+str(index)+'.'+suffix
                    h.put(out/'evidence/fullpcm'/member,fullpcm.read('evidence/'+member))
                h.put(out/'evidence/fullpcm/traces'/(backend+'-'+rid+'.trace'),fullpcm.read('traces/'+backend+'-'+rid+'.trace'))
        h.put(out/'evidence/fullpcm/ORIGINAL_MANIFEST.json',fullpcm.read('MANIFEST.json'))
        h.put(out/'evidence/original-sdk-MANIFEST.json',previous.read('installed-base/ORIGINAL_MANIFEST.json'))
        card={'schema':'english-research-sdk-model-card-v1','selection':selection,'head':h.identity(head),'npz':h.identity(npz),
            'model_kind':'frozen_acoustic_embedding_plus_1536_to_32_to_3_research_head','strong_english_fsmn_ctc':False,
            'primary_track':'strong_pretrained_causal_fsmn_ctc','head_parameters':49283,'sample_rate':16000,
            'classes':{'0':'background','1':'Hey Vosen','2':'Hey Mero'},'threshold':0.8,'confirm_embeddings':3,'cooldown_embeddings':13,
            'fullpcm_source':fullpcm_source,'training_source':candidate_source,'cases':[x['case'] for x in cases],
            'software_scope':'FIVE_SAVED_EXPOSED_CASES_FULL_PCM_NATIVE_AND_ARM_QEMU',
            'installed_original_stdin_runner_reexecuted':False,'ssc305_qualified':False,
            'formal_model_promotion_allowed':False,'final_mero_pronunciation_frozen':False,
            'independent_far':'NOT_MEASURED','independent_frr':'NOT_MEASURED','word_end_latency':'NOT_MEASURED',
            'new_model_calls_during_packaging':0,
            'historical_vosen_miss_fixture_now_matches':any(e[0]==1 for x in cases if x['case']['id']=='encov-new8-28b17376eefb7d7a62334bab' for e in x['case']['events']),
            'known_limitation':'exposed weak research and software regression cases are not an independent acoustic test'}
        h.put(out/'MODEL_CARD.json',h.canonical(card))
        h.put(out/'PROVENANCE.json',h.canonical({'sdk_source':h.SOURCES['sdk'],'previous_fullpcm_source':PREVIOUS,
            'selected_training_source':candidate_source,'selected_fullpcm_source':fullpcm_source,
            'libraries':h.SDK_LIBRARIES,'public_header_sha256':h.PUBLIC_HEADER_SHA,'trace_consumers':CONSUMERS,
            'replacement':'sdk-native/models/causal-head.bin and sdk-arm/models/causal-head.bin replaced with selected-head.bin',
            'assembly_execution':'OFFLINE_SAVED_BYTES_ONLY','pipeline_source':'aec192dd7c5955a7fa2b709564cdacb185a590be'}))
        h.put(out/'README_zh.md',readme(card).encode())
        for name in ('english_coverage_fullpcm_once.py','english_hardneg_increment_once.py'):
            h.put(out/'verification'/name,(HERE/name).read_bytes())
        h.put(out/'verify_package.py',(HERE/'verify_english_research_delivery.py').read_bytes())
        files={p.relative_to(out).as_posix():h.identity(p.read_bytes()) for p in sorted(out.rglob('*')) if p.is_file()}
        h.put(out/'PACKAGE_MANIFEST.json',h.canonical({'schema':'english-research-sdk-delivery-v1','files':files,'selected_model':h.identity(head),'ssc305_qualified':False}))
        names=sorted(set(files)|{'PACKAGE_MANIFEST.json'})
        h.put(out/'SHA256SUMS',''.join(h.sha((out/name).read_bytes())+'  '+name+'\n' for name in names).encode())
        verifier=module(HERE/'verify_english_research_delivery.py','verify_new_delivery')
        verified=verifier.verify(out)
        zip_path=out.with_suffix('.zip');h.need(not zip_path.exists(),'NEW_ZIP_ONLY')
        with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for path in sorted(out.rglob('*')):
                if not path.is_file():continue
                info=zipfile.ZipInfo(path.relative_to(out).as_posix(),(2026,11,17,0,0,0));info.create_system=3
                info.external_attr=(stat.S_IFREG|stat.S_IMODE(path.stat().st_mode))<<16;info.compress_type=zipfile.ZIP_DEFLATED
                z.writestr(info,path.read_bytes())
        return {'output':str(out),'zip':str(zip_path),'zip_identity':h.identity(zip_path.read_bytes()),'verification':verified}
    finally:
        previous.z.close()
        if fullpcm is not None and fullpcm is not previous:fullpcm.z.close()
        if prepared:
            prepared['previous'].z.close();prepared['candidate'].z.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    print(json.dumps(assemble(args),ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    sys.dont_write_bytecode=True
    raise SystemExit(main())
