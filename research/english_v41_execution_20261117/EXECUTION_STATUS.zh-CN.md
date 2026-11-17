# 英文 V4.1 执行状态：资源基线已测，优化 A/B 待执行

本记录冻结于 2026-11-17 的 A/B 执行前。已完成原始数据证据联合复核、旧资产处置审计，以及现有研究 SDK 的主机资源测量。**当前可正式准入训练/开发的数据仍为 0；SSC305 实板未运行；没有最终模型或性能验收通过声明。** 后续 A/B 实测结果应另行入库，不能把本记录中的条件推算改称已测收益。

## 已完成与证据入口

| 工作 | 已完成的实际内容 | 当前边界 |
|---|---|---|
| Piper v4 联合复核 | 16 条原音频、48 条已保存 ASR、16 条已保存音素观察，纯 JSON 复算已入库 | 每词 8 条，正式 train/dev 可用数均为 0；未自动设置人工 gold 或发音已核验 |
| 旧 Qwen / snapshot160 审计 | 77 条已选弱监督逐条溯源、159 个原 WAV/PCM 身份、63 条旧 Qwen 联合观察复用 | 77 条保留研究/开发诊断用途；可信正例、可信负例与本次正式训练准入均为 0 |
| 定点人工复核材料 | 原 16 条 Piper 加 4 条现有 Qwen 原字节，共 20 条、14.170875 秒 | 原始音频已单独交付，未裁切或重生成；人工结果尚未录入 |
| 主机资源基线 | 2 条不同输入各 16 秒；再对第一条诊断重放 16 秒 | 不同输入 32 秒、总模型输入重放 48 秒；独立评估输入 0 秒 |
| 优化准备 | 9 个 unpadded 3x1 节点被定位；有序内核、严格替换器和逐步 trace 源码已提交 pipeline PR #832 | 本记录不含候选运行、54-case 数值结果、A/B 提速或目标板结果 |

Piper 的保存结果见 [既有联合复核](../english_piper_v4_joint_review_20261117/joint-review.json)。旧资产完整处置及逐条身份见 [ASSET_DISPOSITION.md](legacy/ASSET_DISPOSITION.md) 与 [ASSET_DISPOSITION.json](legacy/ASSET_DISPOSITION.json)。最终目标保持 Hey Mero `/heɪ ˈmiːroʊ/`、Hey Vosen `/heɪ ˈvoʊzən/`，Hey 轻读。Piper 观察器不输出元音长度或重音标记；`iː → i`、`r → ɹ` 的表示映射已明确，不能据表示差异直接判定音频错误。ASR 拼写不一致、未识别目标或旧 near-miss 生成意图，均不能替代音频负例证据。

旧 snapshot 内原 train 50 / dev 27 的声线族无交集，但与旧 Qwen63 跨批共享 Dylan、Eric、Serena、Vivian 四族，涉及 47 条已选样本。Serena 曾为 development-only，之后有 8 条进入历史 train；Vivian 曾为 train-candidate，之后 13 条进入 dev。旧 train-candidate 不证明已执行过旧批训练，但此前候选和诊断暴露必须保留。当前 dev 不能声称是独立、从未暴露的评估集。实际音频/最终读音标签、跨批来源族与暴露角色、词尾和环境覆盖，是正式训练仍缺少的证据。

## 主机资源测量的实际结果

原始 [run 44000006165](https://github.com/jiying2007/kws-data/actions/runs/44000006165) / [artifact 14292027025](https://github.com/jiying2007/kws-data/actions/runs/44000006165/artifacts/14292027025) 的 ZIP 为 4,547,376 B，SHA-256 `ce4100bc444cfe99ef18c62e8a1bb17be655c7b57340949049e3abc5867a065f`。执行使用 kws-data `00f3a81524308d8e7f5fc43015a26486de593760` 和 pipeline `eaf4042cf77e18695b35c61c3e6ead4b804b1605` 的固定资源测量源码；现有 SDK 身份保留在 [RESULT](resource_results/evidence/RESULT.json)。

主机为 x86_64 Intel Xeon 6975P-C，4 个可用逻辑 CPU，GCC 13.3.0。两条输入均来自已暴露的 LibriSpeech 朗读拼接池，原角色为 train，`trusted_negative=false`。完整输入分段与 PCM SHA 见 [PLAN](resource_results/evidence/PLAN.json) 和 [inputs](resource_results/inputs/inputs.json)，主机与源码身份分别见 [HOST](resource_results/evidence/HOST.json) 和 [SOURCE](resource_results/evidence/SOURCE.json)。

| 基线输入 | 全流 RTF | 稳态 RTF | 稳态喂入调用 p99 | 结束时进程 PSS |
|---|---:|---:|---:|---:|
| libri-speaker-1272 | 1.636129244 | 1.735694311 | 142.124942 ms | 3,868 KiB |
| libri-speaker-1462 | 1.636807847 | 1.736862769 | 140.283846 ms | 3,878 KiB |

两条基线的加权稳态 RTF 为 **1.736278540**；当前主机实现的串行 accept 吞吐低于实时音频到达速率。每条 16 秒流包含 2.2 秒 warm-up 和 13.8 秒 steady，输入分块为 320 样本/20 ms；整流均产生 188 个 embedding、173 个 ready 输出、0 个事件。稳态 CPU 秒/音频秒约 1.736 是处理成本；CPU/accept 墙钟约 0.9999，约相当于一个核的运行时间。

服务时间 p99 不是词尾唤醒延迟。每条稳态 172/690 次调用超过 20 ms，但未做实时节拍或采集队列实验，不能换算为现场丢帧率。未保存逐调用时延数组，因此没有合并两条输入的总体 p99。PSS 包含 runner、预读输入和计时数组；20,360 KiB lifetime peak RSS 在分配 stream 前已存在，不能当作独立 KWS 峰值。stream state 为 236,328 B，workspace 为 528,384 B，合计 764,712 B；该显式分配量与进程 PSS 口径不同。

完整数值见 [baseline/RESULT](resource_results/baseline/RESULT.json)、[1272 原 stdout](resource_results/baseline/libri-speaker-1272.stdout.json)、[1462 原 stdout](resource_results/baseline/libri-speaker-1462.stdout.json)。[独立结果复核](resource_results/independent-result-review.json) 核对了已保存原 ZIP 的 35 个成员、输入/产物身份、计数、RTF 和完整 profile；本次只读复核没有再次执行模型。原始 JSON、stdout 和命令退出回执保持逐字节一致；其中 Actions 运行目录仅表示当次执行位置，完整二进制/音频可按原 artifact 身份回读，本目录只保存所列文本证据。

## 优化范围与下一次执行

[完整 graph profile](resource_results/diagnostic/graph-profile.json) 的 81 个 graph/node 行与静态图一一对应，每行 188 次调用。九个候选 embedding 节点 6、13、19、26、32、39、45、52、58 占这次诊断节点 CPU 总时间的 **49.20923%**；单输出位置的 node 62 保留原 fallback。完整 join 和静态 MAC 明细见 [profile_static_join](profile/profile_static_join.json)、[static_graph_cost](profile/static_graph_cost.json)，解释见 [profile_summary](profile/profile_summary.md)。

此占比来自包含 warm-up、finite 检查和计时开销的诊断运行，不能混入未插桩基线。假设候选组整体提速 2 倍，图内总时间的条件提速约 1.326 倍；这是 Amdahl 推算，没有实测候选收益。诊断仅核对计数与空事件序列一致；[VERIFIED](resource_results/diagnostic/VERIFIED.json) 和[诊断原 stdout](resource_results/evidence/diagnostic.stdout) 没有证明逐 embedding/logit 或非空事件等价。

优化源码固定于 [pipeline PR #832](https://github.com/jiying2007/kws-pipeline/pull/832) 的 head `342e73655be56d867f7833e6076e16100c429dd0`。下一次受限 A/B 使用相同源输入与既有 head，先验证内核数值及逐步 embedding/score/event trace，再比较独立未插桩基线与候选资源结果。PR 的源码检查不执行模型。候选实际运行、提速是否成立和数值等价结论留给后续结果 PR。

SSC305 实板、目标 ABI/编译器、麦克风链路、板端 CPU/内存、独立 FAR/FRR、词尾延迟均未测量。`performance_budget_evaluated=false`、`formal_training_admitted=false`、`model_promotion_allowed=false` 保持不变。
