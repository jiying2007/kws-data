# 英文唤醒词 dense-CE 最后一项对照：实际结果与停止决定

2026-11-17（北京时间）。[源码 PR #177](https://github.com/jiying2007/kws-data/pull/177) 在 32 项检查通过后合并；[公开实际运行 44033745273](https://github.com/jiying2007/kws-data/actions/runs/44033745273) 成功完成，attempt=1，源码 commit 为 `3b356d3e6ed14cd4c408f94ae44411967f49a2e5`。计算、留证和独立审计均已完成，预注册质量门判定为失败。

## 最终决定

**拒绝 dense-CE 候选，继续保留 aug96 作为英文研究对照；停止在同一数据池继续搜索 loss、seed、threshold 或 epochs。**

拒绝原因只有一项：没有修复任何既有 10 dB development 漏检。相对主对照 aug96，全部 264 个配对条件中，修复 0 个、回退 0 个、出现新增错误事件的 case 为 0。预注册门同时要求“不回退”和“至少修复一个原有 10 dB development miss”；保持原成绩不足以通过该门。判定按[原协议](PROTOCOL_zh.md)执行，运行后没有修改条件。

本次 dense 权重 SHA256 为 `353d15bf230680f294a67dc13461012cbe63becada9fc274aee13038be559464`，仅保存为被拒绝的诊断候选。保留的 aug96 权重 SHA256 为 `cdceab93cd6d3d36bff62980e8c7d2a95896ad32ecc88cae04fa232393e3e9e7`。保留研究对照不构成正式模型晋级、产品 SDK 替换或 SSC305 资格授予。

完整原始运行结论逐字保存在 [RESULT.json](RESULT.json)；独立审计摘要与完整报告身份见 [AUDIT_SUMMARY.json](AUDIT_SUMMARY.json)，全部差异和开发集描述统计见 [PAIRED_REVIEW.json](PAIRED_REVIEW.json)。

## 等预算对照实际做了什么

本次仅把正例 bag 的损失从 `-log(mean(p_target))` 改为 `-mean(log(p_target))`，即 `-logp[:, label].mean()`。每个 bag 先取均值，再对 clean、10 dB × 3 类的六个 bag loss 等权平均。负例仍为 `-logp[:, 0].mean()`。

两者使用相同原 baseline 初始化、相同 train-role 输入、原有 3/4 个完整 clip 支持窗口、同一 96×6 抽样序列、固定归一化、fresh AdamW、学习率 0.001、weight decay 0.0001、梯度范数上限 1.0、seed 20261117 和单线程确定性计算。dense 没有接着失败 temporal 权重训练。推理阈值 0.8、连续确认 3、冷却 13、80 ms 输出步长均保持不变；类别固定为 0 background、1 Hey Vosen、2 Hey Mero。

clean、10 dB、5 dB 的 10,354、2,804、2,649 个 embedding 全部复用已认证缓存。没有新混音、新前端、ONNX、TTS、ASR 或 phone 调用。5 dB 仅用于候选冻结后的评估，已明确标记为此前暴露过的研究条件。

实际训练完成 96 updates、576 training forwards；候选执行 264 次 head case pass，得到 11,847 个 ready 输出。监督执行耗时 11.379 秒，包含源下载及子进程执行与清理，不包含依赖安装；重试为 0。六阶段记录显示，候选在第 3 阶段冻结，随后才执行全部候选评估。

## 事件质量没有超过 aug96

| 条件与 development 唤醒词 | aug96 主对照 | dense-CE |
|---|---:|---:|
| clean / Hey Vosen | 3/3 | 3/3 |
| clean / Hey Mero | 8/8 | 8/8 |
| 10 dB / Hey Vosen | 2/3 | 2/3 |
| 10 dB / Hey Mero | 6/8 | 6/8 |
| 5 dB / Hey Vosen | 1/3 | 1/3 |
| 5 dB / Hey Mero | 5/8 | 5/8 |

上述均为已观察有限流内的弱意图匹配事件计数，不是独立自然语音 FRR。

| 全部角色合计 | 有效 case | 弱正意图匹配数：aug96 → dense | 修复 | 回退 | 新增错误事件 case |
|---|---:|---:|---:|---:|---:|
| clean | 117 | 29/29 → 29/29 | 0 | 0 | 0 |
| 10 dB | 76 | 25/28 → 25/28 | 0 | 0 | 0 |
| 5 dB | 71 | 17/23 → 17/23 | 0 | 0 | 0 |

clean 中 48 条普通语句弱负意图和 40 条历史背景朗读均无事件；48 条普通语句在 10 dB、5 dB 下也均无事件。所有条件均没有新增 prefix、错误关键词或额外同词事件。普通语句和背景朗读仍然是 `trusted_negative=false` 的研究材料，不能将这些零计数换算为自然环境 FAR。

5 dB 的计划分母仍为原 77 条，6 条固定 headroom 排除全部来自 train 正意图；有效正意图为 23 条，development 27 条完整保留。排除项未作为模型成败计数，未补样、缩放修补或重选噪音。264 是 117＋76＋71 个配对条件数，重复条件不增加独立音频或说话人数。

事件时刻并非完全相同：clean 的 `DEV-MERO-02` 从样本 53,120 移至 54,400，晚一个 80 ms 步长，仍为同词匹配；其余 28 个 clean 正例匹配时刻相同。10 dB 的 25 个和 5 dB 的 17 个匹配事件时刻均与 aug96 相同。这里比较的是相同构造流的事件时钟，没有人工词尾标签，不能称为 word-end latency 测量。

## 三条原有 10 dB miss 仍未解决

以下分数是在完整已观察 ready 输出中，对目标概率取“连续三帧最小值，再取最佳三帧”。它只描述保存的概率，不代替包含冷却状态的完整事件规则，也不推断尚未观察的后续帧。

| 原有开发 miss | aug96 最佳三帧最低概率 | dense 最佳三帧最低概率 | dense 峰值 | dense 达门总帧 / 最长连续帧 |
|---|---:|---:|---:|---:|
| DEV-VOSEN-02 | 0.756200 | 0.756851 | 0.797694 | 0 / 0 |
| DEV-MERO-05 | 0.698232 | 0.692858 | 0.837461 | 2 / 1 |
| DEV-MERO-07 | 0.108989 | 0.109547 | 0.557608 | 0 / 0 |

三条均未形成实际唤醒事件。Vosen-02 若仅取原 `FEATURE_EVIDENCE` 的完整 clip 支持窗口，对应三帧分数为 0.693661→0.695284；该范围比完整 ready 输出窄。两种范围在配对摘要中分别保留，没有混用。

11 条 development 正意图的完整 ready 三帧分数，在 clean、10 dB 下均为 5 条升高、6 条降低，在 5 dB 下为 4 条升高、7 条降低。它们没有转化为新的匹配事件。训练自身 loss 的前 16 步均值为 0.031230，后 16 步为 0.002897；训练目标降低不代表开发事件质量提高，也不能跨不同 objective 直接比较 loss 大小。

## 消除 temporal 新误触，只是恢复主对照已有表现

[前一项 temporal 实验](../english_temporal_quality_20261117/RESULTS_zh.md) 在 development 背景 `libri-speaker-5338`（`DEV-READ-SPEECH-06`）上新增一次 Mero 事件。dense 在同一记录上没有该事件，但 aug96 原本也没有该事件。

| 流内时刻 | aug96 | 已拒绝 temporal | dense-CE |
|---|---:|---:|---:|
| 14.28 s | 0.897864 | 0.909804 | 0.897950 |
| 14.36 s | 0.779322 | 0.838983 | 0.781593 |
| 14.44 s | 0.821808 | 0.844730 | 0.819424 |

dense 的中间帧低于 0.8，三连被打断。相对失败 temporal，dense 有一个背景 case 恢复；相对预注册主对照 aug96，修复仍为 0。没有把次要对照的恢复计入保留门收益。

## 独立核验与证据身份

独立标准库审计通过：三份输入 ZIP、37 项逐字复制绑定、完整缓存与时间格、同一 96×6 draws、零 development/5 dB 训练抽样、固定归一化、候选二进制导出、保存 logits 的 softmax、三连事件重放、六阶段计数及逐例质量门均一致。新三组 trace 的 softmax 复算最大偏差均为 0。

审计没有重新执行 head 矩阵乘法或梯度，也没有将源码与阶段 receipt 的一致性称为独立运行时证明。`head_logit_weight_correspondence_independently_recomputed` 与 `model_forward_numerics_recomputed` 均为 false。

| 完整证据 | 字节数 | SHA256 |
|---|---:|---|
| 公开运行 artifact | 8,556,668 | `9b2a6cfde7c12ccf31bcc018cf7df76e0b3722cdbe0609efb67195edb2e83747` |
| 独立完整审计 | 210,695 | `561b55f7f84074a14847102e983c12c6244d430f4ebba0066ab053d744e8c994` |
| 完整逐例比较 | 2,189,694 | `81d502bd6aa9313a8fc99b86ec4b4d4101cd530d25788a58720ce802dd4b96fc` |
| 原始 RESULT member | 16,762 | `6f2d83384810cae6e91e17a9d359c6564c206f9d43d022d8d2abece334a36e81` |

公开摘要保留完整报告的身份、所有主对照错误变化（空集）、唯一主对照事件时刻变化、全部 11 条 development 正意图摘要、6 条排除 receipt，以及 temporal 背景误触的旁证。完整逐帧数据仍在相应运行 artifact 中。

## 停止本轮搜索与后续边界

temporal 和 dense-CE 两项新候选均未通过原研究门。本轮因此闭合：**保留 aug96，停止同池 loss、seed、threshold、epochs 的进一步搜索；不继续加步、不自动重试、不把开发失败样本移入训练。**未来若继续质量研究，需要先引入新的可信训练覆盖或独立评估证据，并重新预注册实验。

这两项结果说明，单独改变正例损失聚合没有在本次固定条件下带来可保留的事件收益；它们不证明所有密集监督或时序监督方法都无效，也没有确定某个音素、发音、口音或梯度密度是根因。

本次证据范围为 kws-pipeline 中英文项目的英文分支。既有原来源、development 模式和 5 dB 结果均已暴露，不能重新命名为独立 holdout；11 个开发正意图也不因三个条件而变成 33 个独立样本。人工发音金标、可信近音负例与长时自然语音 FAR/FRR、真实词尾延迟、新 head 的 C 数值一致性和 SSC305 实机验证均未完成。

`formal_training_admitted`、`training_admitted`、`human_gold`、`trusted_negative`、`pronunciation_verified`、`model_promotion_allowed`、`qualification_allowed` 全部保持 false。
