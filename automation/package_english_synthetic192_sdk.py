"""Assemble a retained synthetic192 research SDK using saved artifact bytes only.

The default command validates disabled source configuration and creates no output.
Binding requires actual successful training/fullPCM artifacts and exact raw GitHub
API responses. Assembly never performs model execution, compilation or networking.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import zipfile

HERE = Path(__file__).resolve().parent
VERIFIER = HERE / 'verify_english_synthetic192_sdk.py'
PLAN = HERE / 'english_synthetic192_sdk_plan.json'
spec = importlib.util.spec_from_file_location('synthetic192_sdk_verifier', VERIFIER)
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)


def fixed_plan(*, api=None, head=None, npz=None):
    return {'schema': v.SCHEMA + '/assembly-plan', 'repository': v.REPO,
        'enabled': api is not None, 'assembler': v.file_pin(Path(__file__).resolve()),
        'verifier': v.file_pin(VERIFIER), 'base_sdk': v.BASE_SDK,
        'api': api, 'candidate_head': head, 'candidate_npz': npz,
        'required_training_gates': list(v.GATES), 'fullpcm_cases': 14,
        'fullpcm_processes': 28, 'new_model_calls': 0, 'new_compilations': 0,
        'new_audio_decodes': 0, 'network_requests': 0,
        'original_stdin_cli_reexecuted_for_new_head': False,
        'physical_ssc305_prerequisite': False, 'human_recording_prerequisite': False,
        'physical_ssc305_tested': False, 'source_only_until_actual_artifacts_pass': True}


def source_preflight(plan):
    expected = fixed_plan(api=plan.get('api'), head=plan.get('candidate_head'), npz=plan.get('candidate_npz'))
    v.need(plan == expected, 'EXACT_ASSEMBLER_SOURCE_AND_PLAN')
    if plan['enabled']:
        v.need(type(plan['api']) is dict and set(plan['api']) == {'coverage', 'training', 'fullpcm'}, 'ALL_THREE_ACTUAL_API_BINDINGS')
        for kind, binding in plan['api'].items():
            v.need(type(binding) is dict and set(binding) == {'source', 'run_json', 'artifact_json'}, 'PINNED_API_FIELDS')
            v.source_pin(binding['source'], kind)
            v.need(v.valid_pin(binding['run_json']) and v.valid_pin(binding['artifact_json']), 'PINNED_RAW_API_BYTES')
        for key, size in (('candidate_head', 197908), ('candidate_npz', 199356)):
            v.need(v.valid_pin(plan[key]) and plan[key]['bytes'] == size, 'EXACT_HEAD_FORMAT_BYTES')
        v.need(plan['api']['training']['source']['run_id'] != plan['api']['fullpcm']['source']['run_id'], 'SEPARATE_TRAINING_AND_FULLPCM_RUNS')
    else:
        v.need(plan['api'] is plan['candidate_head'] is plan['candidate_npz'] is None, 'NO_INVENTED_FUTURE_ARTIFACT_PINS')


def api_bytes(directory):
    directory = Path(directory)
    result = {}
    for kind in ('coverage', 'training', 'fullpcm'):
        for name in ('run', 'artifact'):
            path = directory / (kind + '-' + name + '.json')
            v.file_pin(path, v.MAX_MEMBER)
            result[kind + '-' + name + '.json'] = path.read_bytes()
    return result


def derive_api(raw):
    bindings = {}
    for kind in ('coverage', 'training', 'fullpcm'):
        run_raw, artifact_raw = raw[kind + '-run.json'], raw[kind + '-artifact.json']
        run, artifact = v.loads(run_raw), v.loads(artifact_raw)
        digest = artifact.get('digest')
        v.need(type(digest) is str and digest.startswith('sha256:'), 'ACTUAL_ARTIFACT_SHA256_DIGEST')
        source = {'run_id': run.get('id'), 'head_sha': run.get('head_sha'),
            'artifact_id': artifact.get('id'), 'name': artifact.get('name'),
            'bytes': artifact.get('size_in_bytes'), 'sha256': digest[7:]}
        binding = {'source': source, 'run_json': v.pin(run_raw), 'artifact_json': v.pin(artifact_raw)}
        v.audit_api(run_raw, artifact_raw, binding, kind); bindings[kind] = binding
    return bindings


def check_base(reader):
    for name, expected in v.BASE_RECORDS.items():
        v.need(v.pin(reader.read(name)) == expected, 'EXACT_ACCEPTED_BASE_RECORD')
    manifest = reader.json('PACKAGE_MANIFEST.json'); files = manifest.get('files')
    v.need(manifest.get('schema') == 'english-research-sdk-delivery-v1' and type(files) is dict
        and reader.names == set(files) | {'PACKAGE_MANIFEST.json', 'SHA256SUMS'}, 'ACCEPTED_BASE_SDK_EXACT_INVENTORY')
    for name, expected in files.items():
        v.need(v.safe(name) and v.valid_pin(expected) and v.pin(reader.read(name)) == expected, 'ACCEPTED_BASE_SDK_MEMBER')
    checksums = ''.join(v.pin(reader.read(name))['sha256'] + '  ' + name + '\n'
        for name in sorted(set(files) | {'PACKAGE_MANIFEST.json'}))
    v.need(reader.read('SHA256SUMS').decode() == checksums, 'ACCEPTED_BASE_SDK_CHECKSUMS')
    provenance = reader.json('PROVENANCE.json')
    v.need(provenance.get('selected_fullpcm_source') == v.COVERAGE, 'ACCEPTED_BASE_SDK_LINEAGE')
    return manifest, provenance


def inspect_inputs(args, plan, h, inc, raw_api):
    source_preflight(plan); v.need(plan['enabled'] is True, 'DISABLED_NO_PACKAGE_OUTPUT')
    for kind in ('coverage', 'training', 'fullpcm'):
        v.audit_api(raw_api[kind + '-run.json'], raw_api[kind + '-artifact.json'], plan['api'][kind], kind)
    base = v.Archive(args.coverage_sdk, v.BASE_SDK)
    training = v.Archive(args.training, plan['api']['training']['source'])
    fullpcm = v.Archive(args.fullpcm, plan['api']['fullpcm']['source'])
    try:
        base_manifest, base_provenance = check_base(base)
        v.check_manifest(training, v.TRAIN_SCHEMA + '/artifact')
        v.check_manifest(fullpcm, v.PCM_SCHEMA + '/artifact')
        head, npz = plan['candidate_head'], plan['candidate_npz']
        result, metrics = v.audit_training(training, plan['api']['training']['source'], head, npz, h, inc)
        arrays = inc.verify_saved_npz_bin(training.read('models/synthetic-head.npz'), training.read('models/synthetic-head.bin'))
        base_arrays = inc.verify_saved_npz_bin(base.read('models/selected-head.npz'), base.read('models/selected-head.bin'))
        v.need(v.pin(base.read('models/selected-head.bin')) == h.HEAD and v.pin(base.read('models/selected-head.npz')) == h.NPZ
            and all(arrays[k] == base_arrays[k] for k in ('mean96', 'scale96')), 'ORIGINAL_NORMALIZATION_BYTE_IDENTICAL')
        cases = v.audit_fullpcm(fullpcm, plan['api']['fullpcm']['source'], plan['api']['training']['source'], training, head, npz, h)
        for index, original in enumerate(h.CASES):
            old_ref = base.json('tests/' + original['id'] + '.reference.json')
            new_ref = fullpcm.json('inputs/' + original['id'] + '.reference.json')
            v.need(old_ref['case'] == original and base.read('tests/' + original['id'] + '.pcm') == fullpcm.read('inputs/' + original['id'] + '.pcm')
                and len(old_ref['embeddings']) == len(new_ref['embeddings']) and all(
                    h.struct.pack('<96f', *a) == h.struct.pack('<96f', *b) for a, b in zip(old_ref['embeddings'], new_ref['embeddings'])),
                'FIVE_ORIGINAL_PCM_AND_EMBEDDING_IDENTITIES')
        return {'base': base, 'training': training, 'fullpcm': fullpcm, 'base_manifest': base_manifest,
            'base_provenance': base_provenance, 'result': result, 'metrics': metrics, 'cases': cases}
    except BaseException:
        base.close(); training.close(); fullpcm.close()
        raise


def close_inputs(inputs):
    for name in ('base', 'training', 'fullpcm'): inputs[name].close()


def put(root, name, raw, mode=0o644):
    v.need(v.safe(name), 'SAFE_OUTPUT_MEMBER')
    target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream: stream.write(raw)
    target.chmod(mode)


def training_members(reader, cases):
    wanted = {'MANIFEST.json', 'models/synthetic-head.bin', 'models/synthetic-head.npz',
        'runtime/CASE_MAP.json', 'runtime/reference-frames.jsonl', 'runtime/candidate-head-stream.bin',
        'fresh/CASE_ROSTER.json'}
    wanted |= {name for name in reader.names if (name.startswith('evidence/') and name.endswith('.json')) or name.startswith('source/')}
    wanted |= {'fresh/references/' + row['id'] + '.json' for row in cases[5:]}
    for name in ('CORPUS_MANIFEST.json', 'MACHINE_EVIDENCE.json', 'ADMISSION.json', 'EXECUTION_PLAN.json',
        'FILES.json', 'policy/IDENTITY.json', 'policy/profile.json', 'policy/english_synthetic_silver.py'):
        if 'inputs/fresh/' + name in reader.names: wanted.add('inputs/fresh/' + name)
    v.need(wanted <= reader.names, 'SAVED_TRAINING_EVIDENCE_COMPLETENESS')
    return sorted(wanted)


def readme(card):
    head = card['head']['sha256']; first = card['cases'][0]['id']
    return f'''# 英文双唤醒词合成研究 SDK

所选新分类头 BIN SHA256：`{head}`。本包只会在实际新训练通过全部 10 个固定质量门、该权重完成 14 条 PCM × native/ARM 两后端的 28 次完整回放后生成。

## 模型与自动合成路线

模型使用冻结声学 embedding 和 `1536→32→3` 分类头。新数据来自预先隔离的合成音色，三路盲 ASR 与独立 XLSR 音素观察形成明确分开的 silver/weak 研究标签，UNKNOWN 不进入监督训练。训练、development、calibration 和 locked-test 按原始音色来源分组；同一来源的不同文本与噪声条件不能跨组。新音色仍共享 Piper 的预训练来源，不能声称是独立真实场景验收。

Hey Mero 的 `/ɛ/`、历史 `/i/` 与 Vosen 的 `/s/`、`/z/` 是保留原始证据的研究发音假设，未声明用户或美国英语发音 gold。当前软件研究路线不以新增人工录音或 SSC305 实板作为训练、SDK 交付前提。真实环境 FAR/FRR、真实词尾延迟未在本包测量。

权重导出与固定事件策略保持一致。完整实际统计在 `MODEL_CARD.json` 的 `fresh_evaluation`；按词、来源、条件和标签层级的结果保留在 `evidence/training/evidence/`。旧 291 条回归、32 条自然背景回归均通过；锁定合成测试上每词不退化且总正确匹配严格增加。较低训练损失本身不是上述质量门的替代物。

## 包含的安装资产

`sdk-native/` 和 `sdk-arm/` 分别含公共 C 头文件、静态库、可执行 stdin 消费者、CMake 包和原许可证。两处 `models/causal-head.bin` 与 `models/selected-head.bin` 是同一实际新权重。NPZ 在 `models/selected-head.npz`，六个数组与 BIN 逐字节对应，原归一化数组保持不变。

**197,908 字节仅为分类头 BIN 大小**，不代表完整声学前端、embedding、静态库或运行时 RSS。arena/workspace 大小与对齐由公开查询函数给出，目标平台的峰值 RAM 与 CPU 应在其实际运行环境测量。

## 先做离线字节验证

```sh
python3 -I -S -B verify_package.py
sha256sum -c SHA256SUMS
```

第一条命令仅重读保存字节：核对原始 API JSON 的 bytes/SHA256 与 run/artifact 身份、全部 10 个训练门和监督退出、所选权重、14 条 PCM、28 条已保存 trace、每维数值和事件。它不运行模型、音频解码、训练、编译或网络请求。

## 持续 PCM 流接口

输入为 **16 kHz、单声道、signed PCM16 little-endian 原始 PCM**，不带 WAV 头。class 0 为背景，1 为 Hey Vosen，2 为 Hey Mero；背景不触发唤醒回调。固定阈值 0.8，连续 3 个 embedding 确认，13 个 embedding 冷却。

建议每次提供 320 样本，即 20 ms；公开 C API 支持其他连续块长。`first_sample` 从 0 开始，始终等于已经接受的样本总数。冷启动到首个 ready 需 35,200 样本，即 2.2 秒；这不是词尾检测延迟。EOF 不补零，不重放已消费样本。

以下是用户在匹配环境自行执行推理的命令，组包过程没有运行它们：

```sh
sdk-native/bin/ekws_pcm_stream sdk-native/models/causal-head.bin smoke < evidence/fullpcm/inputs/{first}.pcm
# 连续采集时让采集程序持续输出相同格式的 PCM：
capture_pcm16_16k_mono | sdk-native/bin/ekws_pcm_stream sdk-native/models/causal-head.bin live
# ARM 研究二进制可在安装 qemu-user 的主机验证接口：
qemu-arm sdk-arm/bin/ekws_pcm_stream sdk-arm/models/causal-head.bin smoke < evidence/fullpcm/inputs/{first}.pcm
```

`capture_pcm16_16k_mono` 是应用端采集程序占位名，不是包内工具。C 接入需为每条流独立申请 arena/workspace，按查询函数对齐，初始化时使用经 SHA256 鉴别的 head。持续调用 `ekws_stream_accept_pcm16` 并检查返回值及 `consumed`；xrun/路由切换调用 discontinuity，只有新录音才 reset。回调失败会 poison 流，已发事件与已消费样本不可再次发送。CMake 项目可把 `sdk-native` 或 `sdk-arm` 设为 `CMAKE_PREFIX_PATH`，使用保留的 `KwsEnglishResearch` 包配置。

## 已实测完整链与原 stdin CLI 的区别

实际新 fullPCM 运行使用 `diagnostics/trace-native` / `diagnostics/trace-arm` 的 trace-linked 消费者，它们连接同一套安装静态库并包装原 head 调用。原 SDK 的 stdin CLI 文件被完整保留，但未对本次新 head 另行执行；`MODEL_CARD.json` 将此标为 false。

实际测量包含五条旧回归 PCM 和九条新 locked-test 合成 PCM：后者按每类 × clean/10 dB/5 dB、ID 字典序选择，选择不使用候选得分。两后端共 28 个进程，完整前端→embedding→分类头→事件链逐维核对，并比较时钟、ready、事件与 END。旧 coverage trace 不被当作新权重验收；新参考来自冻结新候选自身保存的输出。

保存证据位于 `evidence/fullpcm/`，输入为 `inputs/`，28 条 trace 为 `traces/`，逐进程 start/exit/comparison 和监督退出为 `evidence/`。可选诊断命令形式为 `diagnostics/trace-native HEAD.bin INPUT.pcm OUTPUT.trace`，执行该命令会运行模型。

## ARM 与物理设备范围

ARM 资产为已验证的 Linux ARM hard-float 研究构建，完成 ARM/QEMU 数值回归。它们尚未在 SSC305 BSP/实板运行，ARM ELF 本身不能证明兼容特定 SSC305 的工具链、sysroot、libc、内核或指令集。物理设备接入时应按实际 BSP 适配，并测实际音频路径、资源与稳定性；这项设备工程不阻断当前合成数据训练和软件 SDK 研究交付。

## 原始来源和可复核性

`ASSEMBLY_PLAN.json` 固定三个输入档案/来源及六份原始 run/artifact API 响应的字节身份。`PROVENANCE.json` 保存实际 run、artifact、commit、ZIP SHA256 与原安装资产的来源；`evidence/api/` 保留原响应。新 fullPCM 档案完整保留原清单，训练证据保留原清单和必要报告、源计划、最终 roster、完整保存 reference stream；192 条训练音频仍由原公开 corpus/training artifact 的精确 pin 指向。

`PACKAGE_MANIFEST.json` 和 `SHA256SUMS` 覆盖交付包；原 accepted SDK 清单在 `evidence/base-sdk/`。本包只复制已有安装资产并替换实际新 head，不重训、不编译、不运行模型，也不把被拒绝的旧 teacher 或新候选包装成通过。许可证保留在每套 SDK 的 `share/KwsEnglishResearch/licenses/`。
'''


def assemble(args, plan, inputs, raw_api):
    out = args.output.absolute(); archive_path = args.zip.absolute()
    v.need(out.parent.is_dir() and archive_path.parent.is_dir() and not out.exists() and not archive_path.exists()
        and out != archive_path and out not in archive_path.parents
        and not any(p.is_symlink() for p in (*out.parents, *archive_path.parents)), 'NEW_SEPARATE_OUTPUTS_ONLY')
    base, training, fullpcm = inputs['base'], inputs['training'], inputs['fullpcm']
    out.mkdir(mode=0o755)
    try:
        for name in sorted(base.names):
            if name.startswith(('sdk-native/', 'sdk-arm/', 'diagnostics/')):
                raw = base.read(name)
                if name.endswith('/models/causal-head.bin'): raw = training.read('models/synthetic-head.bin')
                mode = 0o755 if '/bin/' in name or name in ('diagnostics/trace-native', 'diagnostics/trace-arm') else 0o644
                put(out, name, raw, mode)
        for name in v.HELPERS:
            put(out, 'verification/' + name, base.read('verification/' + name))
        put(out, 'verify_package.py', VERIFIER.read_bytes())
        put(out, 'models/selected-head.bin', training.read('models/synthetic-head.bin'))
        put(out, 'models/selected-head.npz', training.read('models/synthetic-head.npz'))
        for name in ('PACKAGE_MANIFEST.json', 'PROVENANCE.json'):
            put(out, 'evidence/base-sdk/' + name, base.read(name))
        for suffix in ('bin', 'npz'):
            put(out, 'evidence/base-sdk/selected-head.' + suffix, base.read('models/selected-head.' + suffix))
        for case in inputs['cases'][:5]:
            name = 'tests/' + case['id'] + '.reference.json'
            put(out, 'evidence/base-sdk/' + name, base.read(name))
        for name in training_members(training, inputs['cases']): put(out, 'evidence/training/' + name, training.read(name))
        for name in sorted(fullpcm.names): put(out, 'evidence/fullpcm/' + name, fullpcm.read(name))
        for name, raw in sorted(raw_api.items()): put(out, 'evidence/api/' + name, raw)
        put(out, 'ASSEMBLY_PLAN.json', v.canonical(plan))
        card = {'schema': v.SCHEMA + '/model-card', 'selection': 'retained-synthetic192-research',
            'head': plan['candidate_head'], 'npz': plan['candidate_npz'],
            'training_source': plan['api']['training']['source'], 'fullpcm_source': plan['api']['fullpcm']['source'],
            'cases': inputs['cases'], 'working_point': v.WORKING_POINT,
            'quality_gates': inputs['result']['gates'], 'fresh_evaluation': inputs['metrics'],
            'model_kind': 'frozen_acoustic_embedding_plus_1536_to_32_to_3_research_head',
            'primary_track': 'frozen_embedding_plus_head_synthetic_research',
            'classes': {'0': 'background', '1': 'Hey Vosen', '2': 'Hey Mero'},
            'sample_rate': 16000, 'recommended_chunk_samples': 320, 'first_ready_samples': 35200,
            'head_blob_bytes_only': 197908, 'full_model_peak_rss': 'NOT_MEASURED',
            'software_research_delivery_verified': True, 'strong_english_fsmn_ctc': False,
            'ssc305_qualified': False, 'physical_ssc305_prerequisite': False, 'human_recording_prerequisite': False,
            'final_mero_pronunciation_frozen': False,
            'installed_original_stdin_runner_reexecuted': False,
            'independent_real_far': 'NOT_MEASURED', 'independent_real_frr': 'NOT_MEASURED',
            'human_word_end_latency': 'NOT_MEASURED', 'new_model_calls_during_packaging': 0}
        put(out, 'MODEL_CARD.json', v.canonical(card)); put(out, 'README_zh.md', readme(card).encode())
        put(out, 'PROVENANCE.json', v.canonical({'schema': v.SCHEMA + '/provenance', 'base_sdk': v.BASE_SDK,
            'api': plan['api'], 'original_sdk_source': inputs['base_provenance']['sdk_source'],
            'assembly_execution': 'OFFLINE_SAVED_BYTES_ONLY', 'new_model_calls': 0,
            'replacement': ['sdk-native/models/causal-head.bin', 'sdk-arm/models/causal-head.bin'],
            'original_stdin_cli_reexecuted_for_new_head': False}))
        files = {p.relative_to(out).as_posix(): v.file_pin(p, v.MAX_MEMBER) for p in sorted(out.rglob('*')) if p.is_file()}
        put(out, 'PACKAGE_MANIFEST.json', v.canonical({'schema': v.SCHEMA, 'selected_model': plan['candidate_head'], 'files': files}))
        checksum_names = sorted(set(files) | {'PACKAGE_MANIFEST.json'})
        put(out, 'SHA256SUMS', ''.join(v.file_pin(out/name, v.MAX_MEMBER)['sha256'] + '  ' + name + '\n' for name in checksum_names).encode())
        report = v.verify(out)
        with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(out.rglob('*')):
                if not path.is_file(): continue
                entry = zipfile.ZipInfo(path.relative_to(out).as_posix(), (2026, 11, 17, 0, 0, 0))
                entry.create_system = 3; entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = (stat.S_IFREG | (path.stat().st_mode & 0o777)) << 16
                archive.writestr(entry, path.read_bytes())
        report['archive'] = v.file_pin(archive_path)
        report['directory'] = str(out); report['zip'] = str(archive_path)
        return report
    except BaseException:
        # These paths were exclusively created by this invocation, never existing user files.
        shutil.rmtree(out)
        if archive_path.exists(): archive_path.unlink()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--bind-plan', action='store_true'); mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--assemble', action='store_true')
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--coverage-sdk', type=Path); parser.add_argument('--training', type=Path); parser.add_argument('--fullpcm', type=Path)
    parser.add_argument('--metadata-directory', type=Path); parser.add_argument('--helpers', type=Path)
    parser.add_argument('--write-plan', type=Path); parser.add_argument('--output', type=Path); parser.add_argument('--zip', type=Path)
    args = parser.parse_args(argv); plan = v.loads(args.plan.read_bytes()); source_preflight(plan)
    if not (args.bind_plan or args.preflight or args.assemble):
        v.need(all(getattr(args, key) is None for key in ('coverage_sdk', 'training', 'fullpcm', 'metadata_directory', 'helpers', 'write_plan', 'output', 'zip')), 'SOURCE_ONLY_NO_OUTPUT')
        print(v.canonical({'status': 'SOURCE_CONTRACT_ONLY', 'enabled': plan['enabled'], 'new_model_calls': 0,
            'new_output_files': 0, 'training_gates': 10, 'fullpcm_cases': 14, 'fullpcm_processes': 28}).decode(), end='')
        return 0
    v.need(all(getattr(args, key) is not None for key in ('coverage_sdk', 'training', 'fullpcm', 'metadata_directory', 'helpers')), 'EXACT_LOCAL_INPUTS_REQUIRED')
    raw_api = api_bytes(args.metadata_directory); h, inc = v.load_helpers(args.helpers)
    if args.bind_plan:
        v.need(args.write_plan is not None and args.output is args.zip is None and not args.write_plan.exists()
            and args.write_plan.parent.is_dir(), 'NEW_PLAN_FILE_ONLY')
        bindings = derive_api(raw_api)
        archive = v.Archive(args.training, bindings['training']['source'])
        try:
            result_raw = archive.read('evidence/RESULT.json')
        finally:
            archive.close()
        candidate = v.loads(result_raw).get('candidate', {})
        plan = fixed_plan(api=bindings, head=candidate.get('bin'), npz=candidate.get('npz'))
    inputs = inspect_inputs(args, plan, h, inc, raw_api)
    try:
        if args.bind_plan:
            with args.write_plan.open('xb') as stream: stream.write(v.canonical(plan))
            report = {'status': 'ACTUAL_SUCCESSFUL_ARTIFACTS_BOUND_OFFLINE', 'plan': str(args.write_plan),
                'identity': v.file_pin(args.write_plan), 'new_model_calls': 0}
        elif args.preflight:
            v.need(args.output is args.zip is args.write_plan is None, 'PREFLIGHT_HAS_NO_OUTPUT')
            report = {'status': 'AUTHENTICATED_NEW_RETAINED_SYNTHETIC192_SDK_INPUTS', 'new_model_calls': 0,
                'candidate_head': plan['candidate_head'], 'cases': len(inputs['cases']), 'fullpcm_comparisons': 28}
        else:
            v.need(args.output is not None and args.zip is not None and args.write_plan is None, 'NEW_PACKAGE_OUTPUT_PATHS_REQUIRED')
            report = assemble(args, plan, inputs, raw_api)
    finally:
        close_inputs(inputs)
    print(v.canonical(report).decode(), end=''); return 0


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
