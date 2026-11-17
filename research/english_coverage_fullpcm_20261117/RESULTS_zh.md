# Coverage 英文改进模型：安装版完整 PCM 链结果

## 实际结论

改进 coverage head 已经在原安装版 native 与 ARM/QEMU C 库上通过完整 PCM 链回放。
五个固定案例在两个后端共执行十次，全部保存 trace 的 embedding、logits、概率、
样本时钟、ready 状态、关键词事件和 END 记录通过独立只读复算；五对 native/ARM
完整二进制 trace 逐字节一致。本轮修复的 10 dB Hey Mero 在两个后端均保留 class 2、
sample 54400 的事件。此结论限定于已有的五段软件回归输入。

- [实际运行 44049954828](https://github.com/jiying2007/kws-data/actions/runs/44049954828)，attempt 1，公开 Actions，success。
- 运行源码与主线提交：`95aec73b0ac8b58fa74e62df35c3e443107619b1`，由 [PR #180](https://github.com/jiying2007/kws-data/pull/180) 落库。
- 原始 artifact：`14313603148` / `english-coverage-fullpcm-44049954828`，13172135 bytes。
- 原始 ZIP SHA256：`ad5494d3a7456d9c7e1fcd51fec727fa9d9a377a508459db6f88af5256170ba6`。
- supervisor 657.927118804 秒正常结束，零重试、无残留子进程。此时间不是 SSC305 性能。

## 五例的实际事件

类别固定为 background=0、Hey Vosen=1、Hey Mero=2；阈值 0.8、连续三个 ready 确认、
13-hop 冷却未改变。每次消费 320 个样本，EOF 不补零。

| 预注册案例 | 输入样本数 | 每后端 embedding / ready | Native 与 ARM 实际事件 |
| --- | ---: | ---: | --- |
| 新 8 dB Mero，事件在 EOF | 54400 | 31 / 16 | class 2，sample 54400 |
| 新 8 dB Vosen | 56960 | 33 / 18 | class 1，sample 55680 |
| 已知新 8 dB Vosen 漏检 | 58240 | 34 / 19 | 无事件，原漏检保留 |
| 已观察背景无事件 | 63360 | 38 / 23 | 无事件 |
| 本轮唯一修复的 10 dB Mero | 55680 | 32 / 17 | class 2，sample 54400 |

每个后端共 168 个 embedding、93 个 ready 输出、3 个事件。独特输入18.04秒，
两个后端合计回放36.08秒。案例ID、输入摘要、原始reference抽取位置均在冻结 plan、
`RESULT.json` 和 `INDEPENDENT_SAVED_TRACE_AUDIT.json` 中。

## 独立保存证据审计

独立审计不 import 执行 worker，不运行模型或编译器。它重新读取原质量与 incumbent ZIP，
抽取同一 PCM、embedding 与 Python 参考输出，并直接解析十份二进制 C trace。
artifact MANIFEST 的173个声明文件全部匹配；十组模型进程 start/exit 和两组 consumer
build start/exit 完整。所有模型、源文件和安装库身份保持冻结。

| 数值项 | 两后端总比较元素 | 最大绝对误差 | 预注册容差 | 越界 |
| --- | ---: | ---: | --- | ---: |
| embedding | 32256 | 9.918212890625e-05 | 0.001 + 0.0002 × abs(reference) | 0 |
| logits | 1008 | 7.271766662597656e-06 | 0.003 + 0.0002 × abs(reference) | 0 |
| probability | 1008 | 1.7285346984863281e-06 | 0.0005 | 0 |

全部事件、时钟、ready、关键词和 END 精确匹配，五对后端 trace bit-exact。
审计脚本在 `audit/audit_saved_fullpcm.py`，其 SHA256 为
`7ad69f90bbc0746b67f953c417fde5623d4b80acf8eb58e8dcafadfa7bde0627`。
审计报告 SHA256 为 `44a1710af6b47fbd238d3af64c0eec3bcc147a6ac283442caad55cffa2bdcb36`。

## 实际质量改进与交付范围

[原质量结果](../english_coverage_quality_20261117/RESULTS_zh.md) 中，10 dB Mero 从6/8到7/8，
原正确案例退步和逐案例事件错误增加均为零；其他已观察条件不变。当前完整 PCM 回放
把这一固定研究增益连接到了真实 C frontend、安装库及新模型头。

交付包的 native/ARM 两个活动安装根均使用197908-byte新头，SHA256：
`d3e39adf72495034de0405abb997b76744885f91868eccfa97ba09b929bc63e4`。
原库、header、CMake配置和stdin consumer保持字节；静态审计核对49项原文件，47项
未变，只有两份模型头按预期更换。CMake15项冻结输入均匹配。
实际新运行的是链接安装库的 trace consumer；原 stdin consumer 未在本轮重跑。
原安装包与 Python 基础包的历史manifest保持原样，新增结果由外层交付状态和证据补充。

本轮没有训练更新、Python/ONNX推理、新混音、TTS、ASR或phone调用。未知发音数据
未改变准入，同池 loss/seed/threshold/epoch 搜索未重开。新增独立原始录音与说话人数为零。
本轮不重复 reset/discontinuity 回归，不把运行墙钟当作性能基准。

独立真人 FAR/FRR、人工发音gold、真实词尾延迟、SSC305实机与稳定性仍未建立。
`formal_training_admitted`、`human_gold`、`pronunciation_verified`、`model_promotion_allowed`、
`qualification_allowed`、`kws_model_delivered` 和 SSC305 qualification 均保持 false。
可核验的研究 SDK 软件交付与项目最终声学/硬件验收分别记录。
