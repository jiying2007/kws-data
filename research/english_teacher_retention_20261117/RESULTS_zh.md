# 英文 KWS 教师保持实验：真实训练完成，候选拒绝保留

日期：2026-11-17（北京时间）。本报告记录一次实际执行及其失败门槛，不作为新的执行入口。

## 结论与交付选择

公开 GitHub Actions [运行 44078712196](https://github.com/jiying2007/kws-data/actions/runs/44078712196) 已完成 96 次优化更新、全部 323 条候选缓存流评估和一次 native 分类头回放。候选因新增一个开发集弱负例事件，未通过既定逐案例回归门槛：`REJECTED_RESEARCH_REGRESSION_GATE`。原有 80 个正例意图匹配均保留，新增修复为 0；291 条回归中正确案例从 280 降至 279。

保留本次实际权重和拒绝证据用于研究复现，`research_retention_allowed=false`、`model_promotion_allowed=false`；它们不替换交付模型。交付继续采用已通过 native / ARM 完整 PCM 软件回归的 coverage 研究权重，其 BIN SHA256 为 `d3e39adf72495034de0405abb997b76744885f91868eccfa97ba09b929bc63e4`。这些权重仍属于冻结 embedding 加小型分类头，强预训练 FSMN-CTC 主线并未因此完成。

## 本次实际机制与数据范围

从原 coverage 暖启动，保持原学习率、seed、96 步预算、归一化字节和原六个训练 bag；每步另加一个人工审核负例来源项。新机制为固定教师保持损失：温度 2、权重 10，以六个原训练 bag 的教师到学生 KL 平均值乘 `10×2²`，加到原七项监督损失均值上。

教师只读取原有 225 个 train bag、8,370 个完整窗口，且在第一步学生更新前保存教师结果。条件仅为 `clean / snr10 / new5 / new15`；development、`snr5`、`new8` 和新增审核负例的教师调用均为 0。学生训练为 672 个 bag forward、96 次优化更新；教师无参数更新，学生蒸馏复用已有 logits。

人工负例范围仍是用户审核的 9 个原始窗口、8 条来源、30.555 秒。仅使用完整 `[s−35200,s)` 支持区间落在审核范围内的 134 个输入，按来源均衡。未扩展到整句标签，未引入新的独立说话人，也未把历史 TTS 请求改成发音 gold。

## 实测结果

| 项目 | 实际结果 | 解释 |
|---|---|---|
| 优化更新 / 学生训练 bag forward | 96 / 672 | 数值训练完整执行 |
| 教师 train bag / 窗口 | 225 / 8,370 | 无开发集或审核负例教师输入 |
| 候选评估 | 323 流、18,982 行、14,137 个 ready 输出 | 291 条旧回归 + 32 条自然原句 |
| 原有正例意图匹配 | 80 → 80 | 没有原有匹配丢失 |
| 新修复案例 | 0 | 不报告新召回收益 |
| 291 条回归正确案例 | 280 → 279 | 新增一个弱负例事件，拒绝候选 |
| 已审核 8 来源背景交叉熵 | 0.27059315879515733 → 0.048414184048506156 | 下降 82.1081%；八来源均降低，属于训练拟合 |
| natural32 实际事件 | 0 → 0 | 未增加已观察事件，不构成 FAR 改善 |
| native logit 最大绝对差 | 8.110633545221901e-06 | 满足固定 1e-4 容差 |
| native 概率最大绝对差 | 1.1327272644301445e-06 | 满足固定 2e-6 容差 |
| native 时钟 / ready / 事件 | 完全一致 | 缓存 embedding 的头部数值一致性 |
| 监督窗口 / 退出码 | 32.539098774999985 秒 / 1 | 门槛拒绝退出；无超时、异常或残留进程 |

以上条件重复使用已暴露源，不能视为 291 个独立自然语音样本。交叉熵降低与缓存数值一致性均不能抵消逐案例新增事件。

## 唯一回归案例

- 条件：`clean`；角色：`development`；弱标签：背景；ID：`libri-speaker-2078`。
- coverage：无事件；新候选：Hey Mero（keyword 2），样本时钟 `200320`，即该流开始后 `12.52` 秒。
- `prior_correct_case_regressed=true`，增加的错误类别为 `weak_negative_intent_events`。
- 该时刻为消费样本时钟，未提供词尾定位或词尾延迟测量。

由此能观察到：教师保持在当前固定方案下保留了原有正例意图匹配，但仍使一条已暴露开发背景流越过原事件判定边界。不能从单次实验推断教师保持机制普遍无效，也不能据本轮拟合下降声称独立误唤醒率改善。

## 验证与保存

独立标准库审计核对了原 ZIP 的全部 418 个 manifest payload、来源与执行身份、225 个教师缓存 bag、训练计数及损失算术、全部 323 条候选流、18,982 行 native 记录、NPZ/BIN 六个数组和拒绝判定。审计执行模型与网络调用均为 0。它没有重新计算教师网络输出、训练梯度或候选矩阵乘法。

本目录内 `RESULT.json`、`PAIRED_REGRESSION.json`、`REVIEWED_FIT.json`、`HEAD_NATIVE_PARITY.json`、`TRAINING.json`、`SUPERVISOR_EXIT.json`、`NATURAL32_PAIRED.json` 和 `models/retention-head.*` 均逐字节来自原 artifact。`api/` 保存实际 GitHub run/artifact 响应，`audit/` 保存独立审计结果与审计器。`ARTIFACT_MANIFEST.json` 保留原 ZIP 清单；本目录是结果与权重子集，不冒称原 419 成员 ZIP 的完整复制。

## 保持未完成的范围

新候选没有运行 ARM/QEMU、完整 PCM 前端或 SSC305 实板，且拒绝结果不触发这些后续执行。独立 FAR/FRR、实际美国英语发音 gold、3–5 米远场、词尾延迟、SSC305 CPU/RAM/温升/长稳均无合格结论。Hey Mero 目标继续以实际美国英语证据为准，旧图示与历史 IPA 不自动成为新准入标准。

本次固定机制已经终止，不换 seed、步数、学习率、蒸馏权重或阈值在同一池内自动重试。新的研究应依赖新的具体证据和明确假设，不能将本次拒绝记录改写为成功。

来源：[源码 PR #187](https://github.com/jiying2007/kws-data/pull/187)、[实际运行](https://github.com/jiying2007/kws-data/actions/runs/44078712196)、[原 coverage 训练](https://github.com/jiying2007/kws-data/actions/runs/44042856615)、[coverage 完整 PCM 回归](https://github.com/jiying2007/kws-data/actions/runs/44049954828)。精确 artifact、权重和文件摘要见 `EXECUTION_RECEIPT.json`。
