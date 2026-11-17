# Coverage 新权重的已安装 C 完整音频链验证

## 目的与冻结点

质量训练已经完成，见 [coverage 实测结果](../english_coverage_quality_20261117/RESULTS_zh.md)。
本次只验证已经冻结的新 head 在已有 installed native/ARM C 库上消费真实保存 PCM 时，
是否保持前端 embedding、逐帧分数、样本时钟及事件的一致性。
已完成的 291-case cached-embedding 比对不能代替这一步。

来源为 `kws-data` 的真实提交 `4f4d3f15214613c0c3a8f92c871da0b419b3f7f7`，
新 head binary SHA256 固定为
`d3e39adf72495034de0405abb997b76744885f91868eccfa97ba09b929bc63e4`。
不改前端库、权重、normalization、阈值、确认/冷却规则或音频；不训练、不运行 Python/ONNX
推理、不生成 TTS、不调用 ASR、不重做混音。

## 原始输入与参考

| 已有产物 | Run / artifact | 字节 | SHA256 |
| --- | --- | ---: | --- |
| coverage 实际质量包 | 44042856615 / 14310138100 | 50040751 | `56b7e9181cc0e8662b00dd1344ef414a1466a668428fec8bc709b600ba8e97e1` |
| 已安装 native/ARM SDK | 43984579132 / 14285590605 | 10810227 | `b6411b7b859fa59147c7cd6091113ce6d83e544b75bc55bafa5726626b2ab511` |
| 原 10 dB 保存 PCM | 44021084918 / 14300818291 | 32878537 | `acc6def807b1dd5728111beba4801352d8de9cb453e530f2270005254f44fa15` |

独立质量审计采用本次登记的 `INDEPENDENT_AUDIT.json`，314019 字节，SHA256
`9c1f1769a41552ec21a07d5cc02a645c73ca689b7f901037b28a53766454057d`。
下载前核验实际 run、首次 attempt、head、成功状态；下载后验证 ZIP 身份、内部 manifest，
并绑定独立审计、质量保留门与新 head。

已保存的 Python reference 是本轮唯一数值参考。embedding 由质量包 `runtime/head-stream.bin`
和 `runtime/CASE_MAP.json` 定位，逐帧 logits/probabilities 和事件来自该包对应预测记录。
每个 case 独立从 sample 0 开始；全局 head replay 时钟须还原为 `local_sample`，不得把
case offset 当成真实输入音频。五个 case 不拼接，避免改变上下文与冷却状态。

## 预先固定的五个回归夹具

| 用途 | Case ID | Samples | Embedding / ready | 新 head 预期事件 |
| --- | --- | ---: | ---: | --- |
| Mero 恰在最后样本触发 | `encov-new8-7ec79719666821c08c694dde` | 54400 | 31 / 16 | class 2, sample 54400 |
| Vosen 正常触发 | `encov-new8-69e0f996e6049f0d58d78d65` | 56960 | 33 / 18 | class 1, sample 55680 |
| 保留 Vosen 已知漏检 | `encov-new8-28b17376eefb7d7a62334bab` | 58240 | 34 / 19 | 无 |
| 背景意图无触发 | `encov-new8-1556744d0babcec74d18c76c` | 63360 | 38 / 23 | 无 |
| 验证本轮修复的 Mero 10 dB 漏检 | `enfinal144-5602dd573b0afb64e36b51b2` | 55680 | 32 / 17 | class 2, sample 54400 |

前四段 PCM 已保存在 coverage 包；第五段直接从原 aug96 包读取，其 aug96 保存结果没有事件。
各 PCM SHA256、输入路径和预期输出由 `automation/english_coverage_fullpcm_plan.json` 固定。
该选择利用已观察诊断结果构造软件回归测试，明确不属于未暴露声学测试。

## 数值和状态判据

沿用先前 fullchain 检查的容差作为这次新语音测试的预设上限，执行后不放宽：

| 检查 | 固定要求 |
| --- | --- |
| 96 维 embedding 每一元素 | `abs(actual-ref) <= 0.001 + 0.0002*abs(ref)` |
| 三类 logit 每一元素 | `abs(actual-ref) <= 0.003 + 0.0002*abs(ref)` |
| 三类 probability 每一元素 | 绝对误差 `<= 0.0005` |
| 样本时钟、ready、类别、事件时刻 | 完全相同 |
| 输入消耗、embedding、ready、event 总量 | 与冻结 case 一致，必须有合法 END |
| 非有限值、缺行、重行、尾部垃圾 | 拒绝 |

保持 16 kHz mono PCM16，320-sample 输入块，固定 0.8 / 3 / 13 事件规则，无 EOF 补零或 flush。
复用 `ekws_stream_trace.c` 的原始字节，以 `--wrap=ehead_step` 观察实际 embedding，并继续
调用原安装库 head 一次。没有额外参考模型，也不改变数值运算顺序。

## 执行预算与保留

公开 GitHub Actions，仅在精确 main 前驱 `4f4d3f15214613c0c3a8f92c871da0b419b3f7f7`
的新 push、首次 attempt 执行；PR 只运行源码/虚构字节测试。
每后端五个独立进程，总共十次 full-PCM pass，单线程，整次监督时限 1200 秒，重试为零。
每后端 18.04 秒输入、168 embeddings、93 ready 输出、三个预期事件；两后端共重放 36.08 秒，
独特已观察输入只有 18.04 秒。第一次真实进程启动即记账，任何失败或超时保留部分证据。

完整原始 trace、stdout、编译器/模拟器版本、输入/库/head before-after 身份、实际进程退出
记录与最终比较结果随 artifact 保存。通过后不追加模型调用。任何失败不自动调整容差、
换 case 或重跑实验。

## 结论边界

成功证明的是：新 coverage head 在已安装 native 与 generic ARMhf/QEMU C 库及公开 ABI
上完成五段保存 PCM 到分数和非空事件的完整链一致性验证。实际消费者是 trace consumer，
不是原 `bin/ekws_pcm_stream --verify-cached` 的写死旧参考模式。
本轮不重新测试显式 reset/discontinuity，也不把带数值 observer 的时长作为性能基准。

原 coverage 包按生成时点保留 `new_full_pcm_frontend_parity=NOT_RUN`；本次结果独立绑定
同一 head 与实际执行 run，作为补充证据交付，不改写历史。真实 SSC305、独立 FAR/FRR、
发音 gold、声学词尾延迟、设备资源与 24/72 小时稳定性均需各自实测，继续未完成。
