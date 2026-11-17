# 英文 V4.1 最终执行状态：主机优化已实测

截至本次结果发布，已有研究 SDK 的资源测量、九个有序 3x1 节点的受限优化、真实 A/B、原始 trace 数值比较和独立只读复核均已完成。**同一次主机运行的加权稳态 CPU 成本由 1.737582741 降至 0.938337066 秒/音频秒，对应 1.851768202 倍吞吐，CPU 耗时减少 45.99756%。两条输入的基线与候选 trace 全部字节一致。** 这是主机工程阶段的实测结果；正式数据训练准入仍为 0，SSC305 实板仍为 NOT_RUN。

原 [A/B 前执行状态](EXECUTION_STATUS.zh-CN.md) 与 [profile 分析](profile/profile_summary.md) 保持原样，作为当时的事实和决策记录。本页是新增的结果入口。机器实测不替代人工发音标签，也没有把研究模型标记为最终交付或放行。

## 1. 实际执行与来源

执行为 [kws-data run 44006082197](https://github.com/jiying2007/kws-data/actions/runs/44006082197)，[artifact 14294576842](https://github.com/jiying2007/kws-data/actions/runs/44006082197/artifacts/14294576842)。原始 ZIP 有 **144 个文件成员、16,161,392 B**，SHA-256 为 `f36222805244d4c724196d3faf9265f776ff020e5478f1a7845ebfd16dda9fe5`。完整日志、二进制、原 PCM 和四份二进制 trace 保留在原 ZIP，并另行交付；本结果目录只收录关键原始文本和独立复核。

执行的 kws-data commit 为 `b1c8d31bb7c79bca68cca5fd520fd3d05e07f7cb`，pipeline 优化源码为 `342e73655be56d867f7833e6076e16100c429dd0`。原 SDK 来自 run 43984579132 / artifact 14285590605，模型、head、输入来源以及完整 SHA 绑定见 [AB RESULT](ordered3x1_ab/evidence/RESULT.json)、[PLAN](ordered3x1_ab/evidence/PLAN.json)、[SOURCE](ordered3x1_ab/evidence/SOURCE.json)。

| 核对项 | 实际结果 |
|---|---|
| 全模型执行预算 | 计划 8 次，启动 8 次，完成 8 次，重试 0 次 |
| 性能运行 | 4 次：每条输入各一次 baseline 和 candidate，共 64 秒重放 |
| 数值 trace 运行 | 4 次：每条输入各一次 baseline 和 candidate，共 64 秒重放 |
| 总模型输入时长 | 128 秒重放；只有 32 秒不同且此前已暴露的输入 |
| 合成内核数值检查 | 54/54 pairs 通过，903,168 个数值比较，失败 0 |
| 本次新生成与训练 | TTS/ASR/音素调用 0；optimizer updates 0 |
| 结果状态 | BOUNDED_AB_COMPLETED；独立复核 INDEPENDENT_AB_REVIEW_PASSED |

输入沿用原先两条 LibriSpeech 16 秒朗读拼接，PCM 字节未变。完整分段来源和身份见 [inputs.json](ordered3x1_ab/inputs/inputs.json)。这些输入原角色为 train、此前已暴露，`trusted_negative=false`；没有新增独立 FAR 评估时长。

## 2. 同次 A/B 的性能结果

本次 [HOST](ordered3x1_ab/evidence/HOST.json) 是 **AMD EPYC 9V74 80-Core Processor** 型号，runner 可用 4 个逻辑 CPU、affinity `[0,1,2,3]`，编译器 GCC 13.3.0。型号中的 80-Core 不能当作此次 runner 使用了 80 个核。此前资源/profile run 使用 Intel Xeon 6975P-C；这里的提速只比较本次同一 runner 新测的 baseline 与 candidate，没有跨主机计算收益。

| 输入 / 版本 | 稳态 CPU 秒/音频秒 | 稳态 wall RTF | 喂入调用 p95 | 喂入调用 p99 | 结束时进程 PSS |
|---|---:|---:|---:|---:|---:|
| 1272 / baseline | 1.737226361 | 1.737368252 | 139.315732 ms | 140.993520 ms | 3,881 KiB |
| 1272 / candidate | 0.940122415 | 0.940235925 | 75.305715 ms | 76.646054 ms | 3,871 KiB |
| 1462 / baseline | 1.737939121 | 1.738094145 | 139.473921 ms | 140.710629 ms | 3,885 KiB |
| 1462 / candidate | 0.936551717 | 0.936631686 | 75.201090 ms | 75.810280 ms | 3,870 KiB |

每个版本合计 27.6 秒 steady 音频。按总成本除以总音频时长，加权 CPU 成本为 `47.957283653 / 27.6 = 1.737582741` 与 `25.898103014 / 27.6 = 0.938337066`；CPU 吞吐比为 **1.851768202**。加权 wall RTF 为 **1.737731199 → 0.938433806**，对应 wall 吞吐比 **1.851735507**。候选两条离线重放的平均吞吐均达到快于实时音频到达速率的水平。

实际顺序是 1272 的 baseline → candidate，再执行 1462 的 candidate → baseline。每条输入只有一个 A/B pair，反转顺序不能替代重复实验、置信区间或长期稳定性验证。逐输入完整原始结果分别为 [1272 baseline](ordered3x1_ab/performance/libri-speaker-1272-baseline/RESULT.json)、[1272 candidate](ordered3x1_ab/performance/libri-speaker-1272-candidate/RESULT.json)、[1462 baseline](ordered3x1_ab/performance/libri-speaker-1462-baseline/RESULT.json)、[1462 candidate](ordered3x1_ab/performance/libri-speaker-1462-candidate/RESULT.json)；每个目录还保留相应的原始 `stdout.json`。

这些分位数是 320 样本/20 ms 喂入调用的服务时间。候选稳态 p99 仍约 75.8–76.6 ms，每条稳态仍有 172/690 次调用超过 20 ms。运行未按真实音频节拍驱动，不能据此宣称麦克风链路无积压或给出现场丢帧率；也没有词尾至事件的延迟测量。逐调用时延数组未保留，因此没有合并两条输入的总体 p95/p99。

stream state **236,328 B**、workspace **528,384 B** 均未变化。表中 PSS 包含预读 PCM、计时数组、runner 和代码；10/15 KiB 的单次差异不构成内存优化证据。进程 lifetime peak RSS 也不是独立 KWS 峰值。本次没有目标板 CPU、内存或应用预算验收。

## 3. 数值一致性与实际优化范围

四个静态库成员中，只有 `fixed_frontend_mel.c.o` 重新编译；stream wrapper、causal head、generated model 三个原 object 保持字节一致，原 head blob 与 public header 也保持原身份。[OBJECT_IDENTITIES](ordered3x1_ab/candidate/OBJECT_IDENTITIES.json) 和 [SOURCE_COPY](ordered3x1_ab/candidate/SOURCE_COPY.json) 保存改动前后身份、九个 eligible 节点及严格 dispatch 条件。节点 62 继续走原 fallback，原 finite 检查保持。

编译继续使用 `-O2 -fno-fast-math -ffp-contract=off`。原 [vectorization.txt](ordered3x1_ab/candidate/vectorization.txt) 实际记录四输出位置核心循环被 16-byte vectors 向量化；这是本次编译日志。原许可证字节身份见 [LICENSE_IDENTITIES](ordered3x1_ab/candidate/LICENSE_IDENTITIES.json)，保留身份不扩大其允许用途。

[kernel-check.stdout](ordered3x1_ab/evidence/kernel-check.stdout) 记录 54 个合成数值 pair。有限值、signed zero 与 infinity 按位比较，NaN 只比较分类；144 个 fallback matcher rejection 检查不等于重新执行 fallback 内核。该检查补充了本次实际流数据的比较范围，没有声称覆盖所有可能输入。

| trace 项目 | 每条输入的 baseline / candidate |
|---|---|
| 输入与调用 | 256,000 个 PCM 样本，800 次 feed |
| embedding | 188 个 × 96 维，逐浮点字节比较 |
| head 输出 | 188 个 × 6 个 float，包含 15 个尚未 ready 的输出 |
| ready 输出 | 173 个 |
| trace | 各 86,197 B、377 records，成对完整字节相同 |
| 实际事件 | 全部为 0，非空事件序列未被覆盖 |

1272 的一对完整 trace SHA-256 同为 `463eb64c5720f5a1322150667d70ff660a641a1892fa366d92be84e7617f5746`；1462 的一对同为 `0a34c5cb4a6f1d0daa804f8e774844306dda31eed3084fd3e047f80350d6b8ef`。四份原始 trace 元数据 stdout 与执行 exit 回执都已收录于 `ordered3x1_ab/evidence/`。

独立复核从原 ZIP、ELF/静态库成员和 trace 字节重新检查身份、计数、float bytes、吞吐计算及执行预算，**没有再次执行模型或内核**。完整复核见 [independent-result-review.json](ordered3x1_ab/independent/independent-result-review.json)。两条输入的全 trace 一致是实质数值证据，但未比较所有中间图张量，也没有验证非空事件行为、所有声学输入或新的目标板编译结果。

## 4. 数据和设备阶段的当前状态

Piper v4 的 16 条原例已联合消费 48 条已有 ASR 与 16 条已有音素观察；旧 Qwen/snapshot 的 77 条选择已逐条处置；20 条、14.170875 秒的原字节复核材料已单独准备。结果与来源入口见 [旧资产处置](legacy/ASSET_DISPOSITION.md) 和 [Piper 联合复核](../english_piper_v4_joint_review_20261117/joint-review.json)。本次性能提升没有改变这些数据的标签或角色。

最终读音仍为 Hey Mero `/heɪ ˈmiːroʊ/`、Hey Vosen `/heɪ ˈvoʊzən/`，Hey 轻读。人工发音/负例核验尚未完成，正式 train/dev 可用计数仍为 **0**。ASR 拼写、未识别目标、机器一致性和旧 near-miss 意图均没有被升级为可信声学标签。跨批声线族暴露、独立划分、词尾及环境覆盖仍需落实，原弱研究模型保留软件诊断身份。

SSC305 实板、目标编译器/ABI、BSP、麦克风采集链路、实板 CPU/内存、独立 FAR/FRR、词尾延迟均未验证。源码入口和主机候选不是板端成功回执；后续实板结果需要绑定实际工具链、设备、输入与测量条件。原结果继续明确 `ssc305=NOT_RUN`、`far_frr=NOT_MEASURED`、`word_end_latency=NOT_MEASURED`、`performance_budget_evaluated=false`、`formal_training_admitted=false`、`model_promotion_allowed=false`、`kws_model_delivered=false`。

本次结果发布只新增保存的证据和本入口，不改 worker、workflow、旧报告、旧模型或准入状态，也不触发第二次 A/B。下一阶段能够改变上述状态的证据分别是原音频的人工/声学标签核验，以及目标设备上的实际测量。
