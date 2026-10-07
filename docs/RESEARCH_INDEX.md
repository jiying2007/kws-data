# 研究证据索引与当前判断

本索引把已公开分支中的可复用数据、源码记录、失败和修正放在同一主线入口。
**归档完整性 PASS、科学实验 PASS 和产品资格是不同结论。** 这里没有默认模型替换、catalog 准入、追加训练或新的声学评测。

## 先看当前结论

- **研究基线 / 资格待验证**：原 A20 可复用为固定研究参照，但不能可靠排除“小屋”等近邻，也未证明真人、真实房间、连续 FAR/FRR、声学词尾延迟或嵌入式板端表现
- **已停止 / 不采用**：fixed300、Cosy49 step300、first-prefix marginal 候选；保留其权重或计算结果只为追溯失败，不能当作推荐产品
- **上下文敏感性已观察**：固定 1,500 ms 前导零 + 300 ms 尾零下，Melo M1/M2 命中、M3“你好小屋”误触发仍在；不能归因于单一缓存原因，也不是泛化改善
- **仍无自动训练准入**：Qwen6 panel 为 REJECTED；Melo 双 ASR 的原计划文本门为 FAIL。人工词句复核不补造声学边界或独立未见样本

## 可直接使用的证据入口

| 主题与目录 | 当前可陈述结论 | 固定来源与不可越过的边界 |
| --- | --- | --- |
| [16 条合成校准音频](../research/2026-10-01-asr-calibration/README.md) + [双 ASR 与 019 修正](../research/2026-10-02-dual-asr-evidence/README.md) | 32/32 主解码完成；v1 两模型均 0/26 covered。修正后 v2：Qwen 14/26 covered（13 TN、1 FN），SenseVoice 13/26（13 TN）；共同覆盖 12 bit、共同错误 0 | [577f273](https://github.com/jiying2007/kws-data/tree/577f27383985b15ed2223467f471ff33d51ccf1e/research/2026-10-02-dual-asr-evidence)。019 原 shared-FN 结论撤回；当前 8 positive / 18 negative / 6 unknown。v2 为 post-hoc，非 ASR 排名或自动准入 |
| [A20 完整包](../research/2026-10-03-native-a20-packages/README.md) | 历史完整包：6 packages、2,243 原公开文件、341 logical assets；恢复完整性与数值/声学资格分开 | [数据 PR10](https://github.com/jiying2007/kws-data/pull/10)，[7af8f85](https://github.com/jiying2007/kws-data/tree/7af8f8597b0b7fbe8761c9a2400f2e4028395551/research/2026-10-03-native-a20-packages)。旧 strict-FP32 FAIL 与原生输出遗失声明保留；A20/F20/第三方 donor 是研究档案 |
| [Qwen20 原始描述记录](../research/2026-10-04-qwen20-descriptive/README.md) | 原单次运行 FAILED_NO_RETRY，supervision FAIL；保存证据完整性可 PASS | [617b66b](https://github.com/jiying2007/kws-data/tree/617b66bb30148a675805be2f17c61703d31e6e79/research/2026-10-04-qwen20-descriptive)。最终 I/O/children UNKNOWN；原概率和 beam NOT_CAPTURED，不能后补成功 |
| [N0 构造输入](../research/2026-10-04-n0-deterministic-controls/README.md) | 3 × 300 s 固定非语音构造流、0 events；原 strict probe FAILED_NO_RETRY | [06cb1df](https://github.com/jiying2007/kws-data/tree/06cb1dffda0da624a16bce707d7729393d9bf2a8/research/2026-10-04-n0-deterministic-controls)。非 15 分钟墙钟 soak；children NOT_AVAILABLE，lifetime absence NOT_PROVEN，terminal I/O UNKNOWN |
| [域与解码诊断](../research/2026-10-04-domain-decoder-diagnostics/README.md) | DEMAND 300.004 s / 0 events / UNKNOWN；FLEURS20 247.56 s / 2 UNADJUDICATED events；qualified-negative exposure = 0 | [41e9019](https://github.com/jiying2007/kws-data/tree/41e90197c28c9853fc376d342ceb114104cd37fa/research/2026-10-04-domain-decoder-diagnostics)。只保留原公开数值投影、标识/哈希及许可；不增加自然录音或特征，不推导 FAR/FRR |
| [A20 prepared inputs](../research/2026-10-04-a20-prepared-inputs/README.md) + [Cosy17 PRE-CMVN](../research/2026-10-04-cosy17-precmvn/README.md) | 32 native + 20 official reconstruction + 1 原 A20 control；另保留 17 份公开合成来源特征。完整性是可复用输入证据 | [d0d54cf](https://github.com/jiying2007/kws-data/tree/d0d54cf635189bdc522c9d0f6135082e81250fd8/research/2026-10-04-a20-prepared-inputs)，[a3f38af](https://github.com/jiying2007/kws-data/tree/a3f38afed7c2fc9bc214bce8c22928c1a2cb675b/research/2026-10-04-cosy17-precmvn)。PRE-CMVN 下游只应用一次固定 CMVN；特征可能保留可重建语音内容，不是匿名数据 |
| [Cosy49 历史候选](../research/2026-10-05-cosy49-candidate/README.md) + [6 录音负结果](../research/2026-10-05-cosy49-six-recording-followup/README.md) | 数值 fixture 1,334 checks PASS；声学候选 REJECTED/停止。原 A20 1/2 wake、1/4 nonwake 触发；Cosy49 0/2、2/4 | [c81a217](https://github.com/jiying2007/kws-data/tree/c81a217e4dff8eee8858cdb0772b37932a55bcd4/research/2026-10-05-cosy49-six-recording-followup)。暴露 98 样本收益不能覆盖后续负结果；6 片永久排除未来训练、开发、调参和模型选择 |
| [fixed30 双 ASR](../research/2026-10-05-fixed30-exposed-asr-regression/README.md) | 原 setup FAILED_NO_RETRY/0 calls；恢复 60 calls。原计划词句规则 17/30，人类实际词句支持另为 18/30 | [13e458c](https://github.com/jiying2007/kws-data/tree/13e458ce4fc88718ffddcd2e7bc9b8a0792eda94/research/2026-10-05-fixed30-exposed-asr-regression)。17 中 1 条有显式尾部不确定，其余也无完整性认证；非独立准确率或训练准入 |
| [Qwen6 来源筛选](../research/2026-10-05-qwen6-source-screen/README.md) | 执行完成 6 TTS / 12 ASR；仅 Aiden K2 弱支持、5 片 quarantine；panel REJECTED，training rows = 0 | [35ab76b](https://github.com/jiying2007/kws-data/tree/35ab76b5f483195dcea8bb6a58bfcca456ce33c8/research/2026-10-05-qwen6-source-screen)。没有 preset 同时支持 K1/K2；Sohee 没有生成/筛选；旧两个 setup 失败保留 |
| [Melo6 人工增补](../research/2026-10-06-melo6-human-adjudication/README.md) + [固定前后零上下文](../research/2026-10-06-melo5-leading-context/README.md) | 原机器文本门 FAIL 2/6；M1–M5 人工词句增补、M6 首字仍 UNKNOWN。前导条件 2/2 positive 命中、1/3 nonwake 误触发 | [dc90c3a](https://github.com/jiying2007/kws-data/tree/dc90c3aa325700fdf17da449452437c90373818e/research/2026-10-06-melo5-leading-context)。M2 分数 0.2192 小于 M3 误触发 0.8363，统一分数门不能两全；非词尾延迟或声线泛化 |
| [N1 whole-clip](../research/2026-10-07-n1-wholeclip-leading/README.md) | 一次整片 + 固定前后零诊断：1 K1、0 duplicate/K2/other；score 0.9687314846116618 | [aeb7bb8](https://github.com/jiying2007/kws-data/tree/aeb7bb86b5d2886ddca42a0692c5447b625031b6/research/2026-10-07-n1-wholeclip-leading)。CONTEXT_UNVERIFIED；只有 1 positive、无 negative；0.900 s 是输入可用时刻，不是词边界 |
| [first-prefix marginal 失败](../research/2026-10-07-k1-prefix-marginal/README.md) | 固定 M3-reject/N1-accept gate FAIL；M3 A=0.7411511568874126 误接受、N1 A=0.872928816141001 接受；停止、无 rescue fitting | [aeb7bb8](https://github.com/jiying2007/kws-data/tree/aeb7bb86b5d2886ddca42a0692c5447b625031b6/research/2026-10-07-k1-prefix-marginal)。保留最初 underflow 失败、独立 conversion-only 诊断与受限 saved-row 恢复；数值证书只针对 normalized rounded binary64，不覆盖普通浮点舍入误差 |

## 历史主线与分支保留

2026-09-30 的 9 个已合 PR 已在原主线。它们的旧分支显示提交分叉，是 squash 历史；逐路径比较没有遗漏的新资料。勿用旧 README/CI 覆盖新版本。

本次收纳的两个 stack 是：`asr-calibration → dual-asr` 和数据 PR `10 → 11 → 12 → 13 → 14`；其余 PR15–20 独立。PR19/20 同一测试修正只保留一次。所有原研究路径、字节和模式保持固定来源身份，42 条 catalog 及生产导出逻辑不变。仅主 README 导航和该测试 fixture 改动；逐文件证据见 [保留清单](BRANCH_RETENTION_20261007.json)。

本清单覆盖全部 23 个远端分支和本次 1,180 个新增保留路径（包含五个原有校验 workflow），合并视图在新增索引/校验文件前为 1,647 个文件。它保存公开文件的来源，不包含未公开的私人研究工件或权重。既有公开 FLEURS/DEMAND 数值投影仍遵循各自明确的旧公开范围，不能扩展为自然原录音授权。

## 校验与使用

1. 固定最终完整 Git commit，再读 `catalog.json` 与具体档案清单；不要把可变 main 当作训练输入身份
2. `python3 -B tools/verify_branch_retention.py` 读取完整 Git tree，并逐字节检查保留文件的 Git blob/大小/模式及闭合集合
3. 新 `research-retention.yml` 只明确运行保留校验、其 invented-byte 测试、双 ASR public-byte 校验、fixed30 保存记录校验和 Qwen6 保存结果/拒绝样例。Qwen6 只重算原保存结果与 panel，绝不调用模型；没有通用历史测试发现或实验启动
4. 原 catalog/2026-09-30 检查及 A20/Qwen20/N0/prepared-inputs 原有专门 workflow 保留。未列出的新档案受到逐文件身份校验，但其科学语义/恢复验证不因此自动被 CI 全覆盖；source companion 的独立校验仍需按原固定输入单独执行

所有 integrity PASS 都只说明记录/恢复/约束校验通过。人审是有边界的技术标签，非身份或商业许可；合成音频/特征不自动具有商业输出许可，旧许可、来源和隐私投影不变。候选失败、未知字段、未捕获信息和原始 CI 失败记录都保留。

此保留检查使用闭合的 tracked-file allowlist。后续新增文件或有意修改研究记录时，须明确复核并更新清单；不能为让 CI 通过而忽略身份不符或自动吸收未登记文件。
