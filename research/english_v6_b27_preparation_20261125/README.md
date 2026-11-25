# B27：一个有界自然负例权重候选，条件满足后完成软件落地

状态：**已在本地完成唯一一次前瞻 metadata 冻结；所有实际科学执行仍关闭。** 提案 JSON 保留编制时的零调用状态；当前冻结结果见下述记录。 本文配套 `PROPOSAL.json` 固定一个候选、系数、预算和条件链。假设来自在任何 B24R1 KWS 输出前已保存的 TRAIN-only 提案；Q1 数值结果没有用于选择系数或训练样本。Q1 的真实失败只决定是否具备开展另一项候选的资格，不支持本假设。

## 独立有界实验与原始条件

本候选属于当前推进至软件落地的独立有界项目任务；不改变 Q1 的一次评估/零新训练预算，不修改旧 opener，也不声称用户逐项批准了 0.16。B27 登记发生在原 Q1 完成后、任何新 optimizer/source/native/KWS 之前；0.16 的先验依据是下述已公开冻结的旧 B26 TRAIN-only 提案。技术审阅和完整原始证据条件始终必需。

**唯一拟定进入条件：完整原始 Q1 PRIMARY 被独立认证为 NO_PROMOTION。** 必须以实际 run 45571993561 的完整原始 lock、producer、12 项输入、3 项产物、逐数组与原 B14 重算为依据。PASS 则本提案永不使用；SOURCE_STOP、runtime STOP、缺失、未知消耗或摘要声称均不成立。

旧 B26 始终只接受“原 B24R1 完整 PRIMARY NO_PROMOTION”；Q1 的任何结果都不能满足它。B27 使用新 ID 和新 registration，原 B24/R1 SOURCE STOP、所有旧失败、Q1 和旧实验的封存 reserve 均保留。ID 的只读检索见 `ID_AVAILABILITY.json`；正式注册前须重查并发占用。

## 已有证据与唯一干预

先验定义：[原 B26 冻结 TRAIN-only 提案](https://github.com/jiying2007/kws-data/blob/d5b68dff01abcb66781496369458f7ee006baa87/research/english_v6_b26_preparation_20261125/FROZEN_TRAIN_ONLY_PROPOSAL.json)，SHA-256 `4f2df013204098f1014e366def6e8b0b6b983fcaba73805db1204a415f116b8c`，原记录时间 2026-11-25 02:37:28.394587 UTC。本提案只引用该证据，不重复生成 TRAIN 诊断。

唯一算法变化是 **NATURAL_WEIGHT 0.08 → 0.16**。这是一个预定的 2 倍对比；它同时缩放原有自然负例 suffix 空目标 CTC 和已知静音 prefix 项，不引入新 loss。仍从原 B24 所用的 **B23 checkpoint**（`784def…297f`）开始，fresh AdamW。原 **B24 checkpoint**（`f6fce…2bb4`）是不可变历史对照。不能从 B24 再续训或把质量 control 偷换回 B23。

架构、43,876 参数、192 个 COMMON_DRAW、88 条 TRAIN 原音、122 项 positive retention、264 份原 teacher、标签、gain/prefix、其他系数、LR `6e-5`、weight decay `0.001`、clip `5` 和 decoder 全不变。只使用第 192 次更新后的唯一最终 checkpoint；不选最佳 epoch、补阶段或生成新 TRAIN 音频。

原始 TRAIN pair 是 artifact **14947936847**，run **45449190562**，ZIP SHA-256 `6adc488ceb760747f09c44bb07c591e4a505f7ce727eee8bfbf24330e420f95b`。其保存轨迹报告 202/384 次真实 positive suffix 事件，同时有 35/384 次自然事件观测、47 个自然事件。配对自然条件里 7 组从无事件变有事件，其中 6 组 blank CTC 仍下降；因此加大该损失未必修复事件顺序。标量 loss 与整体梯度范数没有证明梯度冲突，0.16 也不是已知最优值。这些均是变化权重下的中间 TRAIN 观察，不能充当最终准确率或晋级证据。

可用已有预算保存的 logits 检查原预测：自然事件少于 47、含事件观测少于 35，真实正例 suffix 事件不少于 202且没有静音捷径回潮。预测不成立就如实报告；不改系数、补训练或替代 B14，也不增加诊断前向/反向。

## 一次注册覆盖的完整有界路线

| 阶段 | 开启条件 | 最大新增模型/后端调用 | 必须交付 |
|---|---|---:|---|
| TRAIN | 真实 Q1 NO_PROMOTION；B27 producer、算法、完整排除和新 primary/replica assessment 在 optimizer 前冻结并审阅 | **1,847** | 192 updates，1,844 优化输出＋2 reload＋1 export；完整唯一最终 PT/NPZ/ONNX |
| PRIMARY | 唯一 TRAIN 成功且原始最终字节已认证；新 SOURCE 通过原准入门槛 | **1,440** | 全部准入原音×3 gain×4 权重；完整原始证据独立复验 |
| 封存 replica | 只有 B27 PRIMARY 完整数值 PASS 且独立认证 | **1,440** | 同四份权重、同原门槛；完整独立 PASS 复验 |
| 软件工程 | 两项真实质量 PASS 且各自原始证据已独立认证 | **140** | 原 14×2×5 矩阵、140 数组、84 CSV、13 SDK/DSP 文件和完整独立 readback |
| 包装、安装和 loader | 真实工程 PASS、独立 readback、不可变 selection/manifest 均齐全 | **0** | 原 candidate pack＋B0 rollback；实际新目录安装和两份 pack 的 clean GCC load receipts |

**总上限：1,847＋1,440＋1,440＋140＝4,867 次 KWS 模型/后端调用。** ASR/phone/TTS 的调用另列，不能把未计量的内部 TTS 模型前向混称为该总数。TRAIN 为最多 192 次 backward 和 192 optimizer updates；其他阶段零更新。一个候选、一个 PRIMARY、仅在 PASS 后一个 replica、一个工程矩阵；未使用预算不转给重试或下一候选。

双质量 PASS 之后的完整软件落地已包含在此同一范围；不设置新的用户批准障碍。每个阶段仍须满足真实科学门槛、完整 producer 与独立复验，以及审阅后的精确一次性技术激活。

## 新 assessment 与 SOURCE 的边界

实际失败独立认证后，先从原 Q1 SOURCE 做完整不可变 exposure metadata reexport，合并全部旧 PCM/group/person/保守历史名单及封存 reserve。**Q1 120 条原音、36 位 primary 人员和另 36 位从未开启的 replica 人员全部排除**；跨合成/自然的人物别名保持一致；88 条 TRAIN 原音及原工程原音/读者也不能进入新的独立质量人群。

只做一次、在新 optimizer 与任何新音频前冻结的 metadata-only 排名：每 panel 24 位合成声音＋12 位自然读者，共 72 位互不重叠的新人。固定 salts 已写入 JSON；池不够就关闭，不能换 seed、重抽或收回旧 reserve。此提案尚未做这次选择，也没有打开旧 replica。

每 panel 保持 96 次 Piper public synthesize、24 个完整自然原音解码、120 条原始暴露人口。预观察资格 N 为 96–120，五个固定 provider 各调用 N 次，**恰好 5N、最多 600**；两 panel 最多 192 次 public TTS、48 个自然解码、1,200 native 调用。所有未计划的预拒绝原音单列，不补满 600。

复用已批准的 Q1 政策：只有严格证明成功返回的空文本才整条原音隔离、五路均无票、零 KWS；其余计划调用按原计划各一次。其他缺失、损坏、失败、超时、未知消耗或 phone 非有限/空矩阵仍整 panel STOP。原 B14、非空 native validity、数值准入、prompt/model/runtime、全原音格式、长度尺度和 gate 不变；不裁剪、不 padding、不替换、不加第三语速、不 rescue。

SOURCE minima 仍为 12 合成正例、24 合成普通、8 个成对 family、12 自然原音、8 位自然读者。所有 120 原音及各 provider/cohort/class/cell/person attrition 必须完整披露。只对全部五路有效且合格的原音计算原 B14，不能用手选子集或推断隔离原音结果。

## 原 B14 与工程、软件完成标准

B0/B9 字节原封不动；control 固定为原 B24，challenger 为这一次 B27 最终产物；gains 为 1.0/0.75/0.5。正例 family 恢复条件仍要求相同原音/gain 上 B9 命中、control 缺失、candidate 命中；同时保持 B0/control 原命中，不增加 B9/control 的重复正例或任何普通事件、改善至少两名自然读者、各 gain 的自然事件总量不高于 B0。两 panel 分别完整通过，不能合并平均救活。

工程继承 `automation/english_v6_b24r1_engineering.py`（SHA `6e6c5528…bc1e1`）的数值实现：14 个已暴露工程原音、原始/一秒零前缀两个条件，PyTorch/ONNX Runtime/GCC/Clang/Cortex-A32 QEMU 五后端，共 140 输出；三个 C 后端对应 84 CSV。ONNX max/mean 误差不超过 0.001/0.0001，C 为 0.005/0.0005；每帧 argmax、折叠 token、全部事件 frame 和 `512+160*frame` sample clock 必须一致。13 份 SDK/DSP 文件保持 [pipeline 固定 commit](https://github.com/jiying2007/kws-pipeline/tree/88a043c6c00bb3a1a62f45098a023c7601042ded) 的原字节。实现与独立 reader 的 B27 身份桥接仍须技术审阅。

工程独立 readback 必须重开两份完整质量证据、4 项原始工程产物、全部140数组/84CSV/producer/SDK字节。随后绑定实际 report 和 verifier、不可变 manifest 及 selection；只复制原工程 candidate pack。B0 rollback pack 必须为 216,960 字节、SHA-256 `ff5d0868c035bf5b1c696115eb62f42c36ddbd3a62781207b1340f5e7d1f9d9b`。包装、fresh-prefix 安装、installed-byte verification 和 candidate＋B0 clean GCC loader 检查不读取 PCM、不初始化 stream、不重新建 SDK、零模型前向。

只有真实安装和两份 pack 的 loader receipts 全部完成，才可报告软件落地。物理 SSC305、真实 AFE/BSP、现场 3–5m FRR、连续 FAR/hour、物理延迟、cal-locked 与产品 decoder 参数仍为 NOT_RUN/未设置；QEMU 或软件加载不能补足这些证据。

## 失败处理

Q1 PASS 让本提案闲置；Q1 不完整/STOP 不给 B27 资格。B27 训练失败、源失败、PRIMARY/replica NO_PROMOTION、工程 parity/readback 失败均关闭相应后续阶段，保留真实消耗和原始证据，不重跑科学预算、不放宽门槛、不变更样本或系数。PRIMARY 失败不得打开其 replica；replica 失败不得工程化；工程失败不得选入软件交付。零前向交付失败保留 STOP，不宣称已落地，不自动重放已消耗的一次性 activation。

原批准、旧 opener 和现有规则文件均未修改。本目录是本候选的具体、可审阅设计；它不是新的用户全局规则，也不是科学执行凭证。


本地初稿 SHA-256 为 `d72d3020fee3c257c5fb3d3ac3419321a2c0401485b7d6c409906fcc9aff879e`；公开版仅删除私人会话背景并澄清时间范围，科学定义和全部预算保持一致。

## 已完成的唯一 metadata 冻结

在原 Q1 完整 NO_PROMOTION 已由独立原始证据 reader 认证、B27 科学设计已审阅后，使用新设计 SHA `956f0695e6b4e3ac450bc3881c5ef13985d775210cf43cf4da98947a5a1846bf` 完成一次确定性选人。完整 Q1 SOURCE ZIP 和原 producer 中捕获的 Q1 assessment 已重新认证；只解析 MANIFEST/HISTORY/原 assessment 元数据，不解释任何音频、native 回执或 Q1 数值指标。

完整历史保留 2,943 条 PCM，再加入全部120条 Q1 原音，得到3,063条排除身份。Q1 全部36名 primary 与36名封存 reserve 人员均排除。剩余112个合成声音、42名有至少两条合格 metadata 的自然读者（自然 census 62）；一次既定 SHA256 ranking 选择72名互不重叠的新人员，primary/replica 各24合成＋12自然。没有换盐、换种子、补人或回收旧 reserve。

生成六份原始计划文档，并保留两份五字段原 Q1 exposure/origin 白名单。`automation/english_v6_b27_assessment_binding.json` 的80文件哈希闭包与 `METADATA_VALIDATION.json` 一致，`frozen=true`、`optimizer_allowed=false`。新的 assessment SHA-256 为 `8c7b4571d2ac08546fd00789fc2809df8c4290ad53a38c1f6b33a993331ae2ee`；ancestry 为 `79e1181eb36fcb9b4a70438e0ceae9ca1fd34649e1db91c269abcd4722834a5a`。原 Q1 文件未修改，所有 SOURCE/quality/replica/engineering/晋级执行开关仍 false。

`automation/test_english_v6_b27_source_metadata.py` 检查完整历史与reserve、72人跨类型独立、原B14/Q1政策与全原音格式、preoptimizer与B24对照、原始身份误分类、metadata读取白名单、篡改拒绝、禁止第二次freeze及池不足关闭。测试不进行音频或模型调用。
