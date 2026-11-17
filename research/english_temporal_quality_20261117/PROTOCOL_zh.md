# 英文唤醒词连续确认质量实验：预注册协议

日期：2026-11-17（北京时间）。本协议在新候选训练和 5 dB 模型评估前冻结。

## 目标与已有证据

对 Hey Vosen `/heɪ ˈvoʊzən/`、Hey Mero `/heɪ ˈmiːroʊ/` 改善连续确认稳定性。Hey 轻读，名字首音节重读。固定类别：0 background，1 Hey Vosen，2 Hey Mero。当前材料尚未经人工发音金标审核，文字与合成意图不代替音频真值。

上一轮公开执行为 [44021084918](https://github.com/jiying2007/kws-data/actions/runs/44021084918)，实际完成 96 次参数更新。10 dB development 的 Vosen 为 2/3、Mero 为 6/8；其中两条剩余漏检能短暂达到阈值，但没有连续三个达到阈值的输出。当前推理规则固定为阈值 0.8、连续 3 个 embedding 输出确认、13 个输出冷却；步长为 80 ms。

原正例损失为 `-log(mean(p_target))`，不是最大池化。平均目标概率上升并不保证三个连续输出都达到阈值。本轮只改这一正例聚合，检验该差异是否真正影响完整流式事件。

## 唯一训练改动

对原正例 bag 中连续 N 个目标 log probability，令 `q_i = min(log p_i, log p_(i+1), log p_(i+2))`，使用：

`L_positive = log(N - 2) - logsumexp(q)`

实现固定为 `torch.amin(logp[:, label].unfold(0, 3, 1), dim=1)`，包括 amin 的并列梯度语义。它等于连续三帧瓶颈概率均值的负对数；不取最佳 triplet，不加入温度或额外混合权重。负例继续使用 `-logp[:, 0].mean()`。

所有原正例 bag 都只有 3 或 4 个合法输出，各相邻差为 1,280 samples。每个输出必须保持原 FEATURE_EVIDENCE 中的 head_instance_ends，拥有完整、连续的 16 行 embedding 历史，而且真实 mel 支持 `[t-31840, t-128]` 必须覆盖原始 clip 全长。不得添加尾帧、复制输出、补零或扩大正例时间范围。不会把包含唤醒词历史的后缀改标背景。

## 等预算对照与复用

| 项目 | 冻结设置 |
|---|---|
| 新 arm 数量 | 1 |
| 架构 | 16×96 embedding → 1536→32 ReLU→3 |
| 初始化 | 原 baseline，SHA256 `6d7677b674f4a5e324aa916c5b777792b641fdfd61af8edc2f4e490275d600c4` |
| 等预算对照 | 上一轮 96 步候选，SHA256 `cdceab93cd6d3d36bff62980e8c7d2a95896ad32ecc88cae04fa232393e3e9e7` |
| 训练输入 | 已认证 clean、10 dB 的原 train-role embedding |
| normalization | 原 mean96/scale96 字节一致 |
| 优化器 | fresh AdamW，lr=0.001，weight_decay=0.0001 |
| 固定训练量 | 96 updates，每步 6 个 bag，共 576 training forwards |
| draws | 与旧 TRAINING.json 的 96×6 draw ID 逐项相等 |
| 聚合 | clean、snr10 × 3 类的六个 bag loss 等权均值 |
| seed / CPU threads | 20261117 / 1 |
| gradient clipping | 1.0 |
| checkpoint | 只保留第 96 步，不按 development 或 5 dB 选择 |

旧模型不重新训练。117 条 clean、76 条 10 dB 特征直接复用；原 baseline 与上一轮候选在这些条件上的已保存预测直接复用。训练前端新增调用为零。训练 draw 的 development 隔离保持不变；本轮动机已经受既有 development 失效模式启发，不能宣称整个研究决策未接触 development。

## 固定 5 dB 新压力条件

只使用原 77 个合成意图 clip，名单、role、原始未裁剪 PCM、2.20 秒 prefix、0.32 秒 context 均保持不变。沿用上一轮逐 ID 绑定的语音背景 track、offset 和散列选择域 `english-quality-snr10-v1`，仅将 clip 内信噪比固定为 5.0 dB。

源语音不缩放；依据原 clip 全长的 RMS 求背景增益，不引入 VAD 或活动语音重切分。只在原 clip 内叠加干扰，prefix/context 沿用原流。16-bit PCM 溢出则整条排除、记录原因，不 clipping、不缩放修补、不补样、不换背景。三种 head 在完全相同的可用子集上评估。条件可用性在预测前固定；不根据 head 得分选择混音。

5 dB 不用于训练、是否训练的触发、checkpoint 或阈值选择。新 head 权重完成并记录身份后，才执行 5 dB 前端和三个 head 的预测。5 dB 是同源音频的新增条件，不增加独立说话人或原始语音数。

## 预先固定的研究保留门

以下各项同时满足，才能把新 head 标记为“通过本轮研究保留门”；正式模型晋级权限仍为 false。

1. 完整性：输入身份、class map、缓存、原时间几何、draw、normalization、数值环境、计数与预算全部一致。
2. clean：相对上一轮候选，每个原来事件正确的 case 均不回退。
3. 10 dB：每个原来事件正确的 case 均不回退，且至少修复一个既有 development miss。
4. 所有比较条件：不增加 prefix 事件、错误关键词事件、额外同词事件或弱负例事件；逐 case 比较，不以总分抵消回退。
5. 5 dB：相对上一轮候选，每个原来事件正确的 case 均不回退；Vosen 和 Mero 各至少一条 eligible development 正例。

不满足则保存为未通过的研究诊断候选，不改门、不重选 SNR、不继续加步或自动重试。所有条件都保存完整逐帧 logits、事件、逐例指标和配对比较。原 clip EOF 截断视图标记为 censored；没有人工词尾，不能称为 word-end latency。

## 执行预算与留证

实际数值计算仅在公开 GitHub Actions 的唯一 main successor、run_attempt=1 执行。固定 predecessor 为 `c4f20fcb9f8b92aa77125c85de393df678ac812f`。本地仅源码审查、编译检查、纯标准库证据解析及不执行模型的契约夹具。

- 96 optimizer updates；576 training bag forwards。
- 5 dB 最多 77 条条件、2,834 个 embedding；硬上限 4,000 embedding／8,000 ONNX forwards。
- 新 head 旧条件 193 case passes；5 dB 三个 head 最多 231；新增 head case passes 总上限 424。
- 共同执行时限 900 秒，包含下载和子进程清理；不自动重试。
- 新增 TTS、ASR、phone 调用均为零；不用下载或重解码原背景语料。
- 保留权重、输入身份、原缓存、5 dB PCM/特征、混音 receipt、固定 draws、逐帧输出、事件、失败部分进度与进程清理结果。

## 解释边界

原 159 条有声源的意图中，仅 77 条当前研究意图进入已有缓存；其余没有新增可信准入，不能补充为已验证正例或近音负例。当前普通语句负例不代表发音相近词的覆盖。已有声源家族曾在其他批次出现，部分角色存在跨批暴露；本轮不把这些材料重新命名为独立 holdout。

所有 formal_training_admitted、training_admitted、human_gold、trusted_negative、pronunciation_verified、model_promotion_allowed、qualification_allowed 均维持 false。研究门通过不代表自然语音 FAR/FRR、C-head 数值一致性或 SSC305 实机已通过。当前任务没有新的独立人工金标、长时真实环境负例或板级接入。

## 方法参考

- [State Sequence Pooling Training of Acoustic Models for Keyword Spotting](https://www.isca-archive.org/interspeech_2020/opatka20_interspeech.pdf)：支持将损失与检测序列得分对齐的研究方向；不是本项目三帧公式有效性的既成证明。
- [openWakeWord 官方说明](https://github.com/dscripka/openWakeWord)：80 ms 流式输出、5–10 dB 噪声压力条件，以及长时真实负例评估的背景。
- [Generalization in Adaptive Data Analysis and Holdout Reuse](https://arxiv.org/abs/1506.02629)：开发集重复自适应使用的解释限制。
