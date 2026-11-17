# English context controls：48 个 parent、64 个 exact-slice child

本流程验证一个前瞻的研究数据合同：三路 ASR 观察短 child，参考 XLSR 观察完整 parent，并在预先固定的 child 区域检查整段音素。这个证据类型称为 `context_supported_weak`，不等于 child 自身通过独立音素验证。本轮所有 parent、child、观察和候选结果永久排除训练；即使控制检验通过，也不自动扩大数据、不生成训练快照，不代表模型质量验收通过。

代码和计划是独立的新路径。两批 Piper192、Kokoro48、context32 及 maskfix 的原代码、数据、UNKNOWN 和 STOP 保持原状。美式 Mero 的两种元音仍是明确的研究假说，没有把历史 IPA 或本轮模型输出写成最终人工发音真值。

## 固定实验单位与调用分母

四个已有 Kokoro preset 是模型音色组件，不是假定的四名独立真人。它们仅保留 train-side 来源身份；本次探测本身仍永久不可训。

| parent 类型 | parent 数 | 主要 child 数 | 额外 outside child 数 |
| --- | ---: | ---: | ---: |
| 两词的四种目标发音请求 | 16 | 16 | 16 |
| 四种预注册音素对照请求 | 16 | 16 | 0 |
| 四种普通句子请求 | 16 | 16 | 0 |
| 合计 | 48 | 48 | 16 |

四个正例假说是 Mero `ɛ`、Mero `i`、Vosen `s`、Vosen `z`。对照请求使用 Mero `æ`、Mero `ɪ`、Vosen 首辅音 `b`、Vosen 最后元音 `ɪ`。请求只是实验意图；实际对照音素必须由参考证据完整支持，不能直接据 TTS 文本或输入 phones 声称音频成功。

| 阶段 | 实际输入单位 | 最大高层调用数 |
| --- | --- | ---: |
| Kokoro TTS | 每个 parent 一次 | 48 |
| Qwen、Whisper、Parakeet | 每路对 64 个 child 各一次 | 192 |
| XLSR、LV60 | 每路对 48 个 parent 各一次 | 96 |
| 总计 | 固定 roster，无重试 | 336 |

这里的高层调用指一次 TTS 请求或一次输入的原生识别调用，不是自回归解码器内部所有神经网络 step 的计数。另有最多 48 次完整 parent 重采样；child 的派生只做字节切片，不额外调用 TTS、重采样器或模型，其 ASR 观测计入上表 192 次。新训练、embedding、KWS 推理均为零。

## 一次完整派生与 exact-slice

每个 parent 保存原生 24 kHz float32、原生 WAV、完整 duration 和真实直接 phoneme IDs。固定 NumPy/SciPy 运行环境只对完整 native float 调用一次 `resample_poly(..., 2, 3)`，保存完整 16 kHz float32，然后以固定 clip、乘 32768、ties-to-even 舍入规则量化一次 int16。

child 直接从已保存 parent int16 的 `[start, end)` 区间取字节，满足：

```python
child_pcm == parent_pcm[2 * child_start_parent16k : 2 * child_end_parent16k]
```

child 不重新重采样、归一化、调整增益、量化或补静音。所有边界由预注册文本位置和原生 duration 计算，不使用观察结果反馈，不重新寻找通过的裁剪位置。

duration 位置公式为 `600 * (max(d[0] - 3, 0) + sum(d[1:a+1]))`。目标 child 在目标起始预测位置前保留 3600 个 native samples，结束于原生 EOF。outside child 使用 carrier 内固定字符区间 `[78,101)` 的预测位置，左右各保留 3600 个 native samples，并要求 child 结束不晚于 parent 最终目标的预测起始。所有 native 边界必须可被 3 整除，按 `2 * native / 3` 精确映射到 parent16k 坐标。

这些位置是模型预测，不是人工声学词端。纯字节读回核验 native/derived float 与各 WAV/PCM 的哈希、长度和量化关系，核验 duration/IDs/边界及 child 子串；不会再次执行重采样，也不会把未重新计算的 FIR 数值声称为独立数值重现。

## 来源、碰撞与完整窗口

计划固定已实际暴露的同 family 集合：旧 Kokoro48 的 48 个 child PCM、context32 的 16 个完整 PCM、旧 48 个 native float hash 及原请求身份。maskfix 复用同一 16 个输入，不重复增加来源数。此集合不冒称项目所有历史音频的全集。

生成端和独立源读回都执行这些排除，并要求新批所有成功 parent/child PCM 跨整个批次唯一。碰撞保留原 48/64 分母和实际失败字节，禁止观察阶段继续；不自动去重、更换种子或重新生成。

parent 与它的全部 child 共用来源组件和角色。改变 hash、裁剪边界或条件不会变成独立测试来源。旧四个已暴露 preset 不会进入未来未见来源测试。后续模型计划的 6/2/2/2 来源门槛与原有十个模型回归门不在本流程中修改。

正例候选仍需满足原有 KWS 整数几何：38400 samples 前缀、16000 samples 后缀、16000 首端点、1280 步长、35200 ready、左 31840/右 128 支持边界，至少三个完整支持窗口。这一步只算整数，不生成上下文音频或调用 embedding。几何失败不会触发重新裁剪。

## 两种观察输入严格分开

三路 ASR 只接收 child WAV 与其输出目录；不传请求文本、角色或目标提示。每份 capture 保留 child ID、parent ID、child PCM/WAV SHA、采样数、原始文字、固定模型和 decode 身份。缺失或失败的识别不能变成普通负例。

两个 phone observer 只接收 parent 的完整 16 kHz 输入。保存完整归一化输入、原生 int32 attention mask、float32 logits、逐帧 argmax IDs 和 CTC tokens。固定区域仅保留满足 `320*t >= child_start` 且 `320*t + 400 <= child_end` 的帧，随后按原 CTC 规则解码。

参考仍为 XLSR，LV60 仅供并列诊断。固定区域仍来自完整 parent 的 Transformer 前向，不是一次独立 child 前向；400-sample 名义卷积支持范围也不是完整 Transformer 感受野。即使 raw PCM 子串相同，整段归一化和全局上下文仍可能改变识别。因此本流程不会把 full-context 命中升级成 child 独立音素 silver。

冻结的 v2 模块按原字节保留在 `english_context64_policy_v2_reference.py`，仅使用其文本别名、音素规范和矛盾规则，不调用其 Piper 专用准入入口。没有 `ɛ/æ`、`oʊ/ɔ` 等事后音素放宽，也不从两个观察器中择优补票。

## 控制检验

候选分类器先只看盲观察，再对照实验 roster。它不接收预期类别来决定候选是否为正。候选正例需三路 child ASR 全部命中同类冻结别名、无矛盾 veto、参考固定区域唯一 whole-match、与 parent 终端 suffix 同一假说，并具备完整 KWS 支持窗口。

固定检验要求：

- 48 次生成、192 个 child 文本 capture、96 个 parent phone capture 全部完整。
- 16 个音素对照全部由参考整段/终端模式实际支持；未知对照不能算成功负例。
- 16 个普通对照和 16 个 outside 对照全部具备三路一致普通文本、固定区域无目标；outside child 位于 parent 最终目标之前。
- 全部 48 个对照的盲候选正例数为零。
- 每个目标假说至少两个 preset 组件具备候选支持；outside-control 对应 parent 的各目标假说也至少两个组件有实际支持。

输出可能为 `PASS_CONTEXT_CONTRACT_FEASIBILITY_ONLY`、`STOP_CONTEXT_CONTRACT_VALIDATION_FAILED` 或 `INCOMPLETE_DIAGNOSTIC_NO_EXPANSION`。全部状态都保留 `training_allowed=false`、`training_weight=0.0`、`automatic_expansion=false`、`high_quality_model_passed=false`。成功只说明这项合同在本次固定对照设计下可行，不证明真实场景 FAR、FRR、发音真值或延迟目标。

## 执行与证据保存

默认计划禁用。启用只能绑定受审的唯一 main 前驱 `2ca611210126e2f78939e6e50f7cd0157c76d5dd`；执行限制为公开同仓 main push、run attempt 1 和该新 workflow。PR 只运行 stdlib/mock 源码检查，不运行模型或新的处理器 fixture。

main 使用已经实际证明的固定原生 runtime 组件。生成内部预算为 1800 秒，phone 各 1800 秒，ASR 各 2400 秒，combine 900 秒；工作流外层另留退出、收据和上传裕量。每项输入只有一次尝试，无自动重试。

生成保留 `source/RESULT.json` 的 48 parent/64 child 完整账目；root `INPUT_MANIFEST.json` 与原行和 run 绑定。文本和 phone 各自保留 STARTED、尝试、返回、capture 或失败记录。返回后格式/收据写入失败不抹去已发生的返回。缺终态时实际次数为 `null`，另列上下界；不把缺证据写成实际零调用。

缺少 source RESULT、manifest 或闭合 FILES 时，读回会保存 `SOURCE_READBACK_FAILED_COUNTS_UNKNOWN`，将 `source_ready` 和 `observer_ready` 设为 false，然后非零退出。原 source/runtime 仍 always 上传。任何部分失败都不能绕过源验证去观察或训练。

主要新 artifact 为 `kokoro-context64-source`、五份 `kokoro-context64-observer-<provider>`、相应 runtime，以及 `english-kokoro-context64-probe-<实际 run ID>`。最终 packet 保存源、观察、执行代码字节、固定计划、评估和完整 FILES 哈希清单。combine 只读保存证据，不再调用模型。

这项研究与软件交付路线不要求补录新真人音频或先完成 SSC305 实板，仍须如实区分探测合同、合成范围模型表现和最终质量指标。
