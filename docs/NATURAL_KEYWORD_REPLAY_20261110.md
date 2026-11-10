# Natural73 原生证据与逐关键词视图回放（2026-11-10）

已从原封存材料恢复完整 Natural73 视图，并在规范 CPython 3.12.16 环境实际运行原验证器和已合并的逐关键词 v2 adapter。两个关键词的结果均为 **present 0、absent 0、unknown 73**。该批数据仍没有可评分自然正例，也没有可计入 FAR 的自然负例。

本次记录补足 [PR #81](https://github.com/jiying2007/kws-data/pull/81) 适配器的真实全资产回放证据；不更改该适配器、历史 ASR 输出、数据准入规则或原有模型。

## 实际恢复与验证

- 36 个已存在 capsule，合计 204,171,647 字节，均与固定 SHA256 一致。
- 原 Natural73 目录的 1,288 个文件、309,159,645 字节逐项与恢复 catalog 一致，原 manifest 和完整成员集合检查通过。
- 缺失源码按 catalog 指定的仓库、commit 和路径读取，校验文件长度、SHA256 与 Git blob SHA1；没有拼造本地 Git 提交冒充原来源。
- 原 native verifier 成功回放 73 个窗口、8 个父源、219 条已保存 ASR 观察，其中 216 条质量可用；73 个窗口均保持 development，训练角色为 0，全文共识为 0。
- v2 `create` 和后续只读 `verify` 均正常退出，输出 manifest 与 report 身份固定。

验证范围包括已声明窗口 WAV/PCM 的字节、PCM 派生关系及封存 native ASR 证据。此次没有重新加载完整父录音和上游原始语料压缩包，因此不扩大原始来源验证或授权范围。

## 逐关键词结果

| 关键词 | present | absent | unknown |
| --- | ---: | ---: | ---: |
| 你好小窝 | 0 | 0 | 73 |
| 小窝小窝 | 0 | 0 | 73 |

这不是 73 个负例或零误唤醒结果。`unknown` 不进入正负例分母，不转换成训练监督，也不转换成自然负流时长。本批全 unknown，因此没有再执行无法评分的 KWS 模型前向。

已有 v2 输出包含按固定 salt 抽取的 32 条人工审计计划，状态为 `planned_not_annotated`。本次没有人工标注，没有完成协议声学校验，不把审计抽样计划当作标签或验收结果。

## 执行次数与边界

本次新增 TTS、ASR、模型调用、优化更新、训练准入和自然负例秒数均为 **0**。执行进程禁止网络、子进程和模型模块，写入范围限定为新的派生目录；验证阶段为只读。原 manifest、记录和 producer 源码不改写。

最初 `create` 在写入新报告时被新增的本地保护器误拦截：保护器未识别已获准文件的 `os.fdopen` 描述符。原 native verifier 在此之前已经通过。失败意图、stderr、未完成目录和失败回执全部保留。随后只修正独立保护器的描述符处理，在另一新目录完成同一固定政策/salt 的派生及只读验证；未修改数据或 adapter，也未重复 ASR。

最终计数为 **1 个完成派生包、1 次保留的本地保护器失败尝试**。没有隐去失败，也没有用纯摘要或虚构 fixture 代替本次真实 native 证据回放。

## 精确身份

| 对象 | SHA256 |
| --- | --- |
| 原 Natural73 manifest | `5dcf033a1c1897e8dfb387b5a3c520aeb8a4df3b30d9a7d24e55c6e16bdbaf03` |
| 原 report 文件 | `d3a108c7cc8c0335cb5700ff2d617252cba6083a8df9c92fa4e04bf476a0d05d` |
| recovery ZIP | `7db87506a42d1af3c93f3eb2628fd8f41d666fe2dc862e91f3f75f44c780d64c` |
| recovery catalog | `25b8784546c68fb6440dd8e7a01ea6047272f4d3ef452653a3d46584a76aaf97` |
| v2 adapter 源码 | `62c93cdbcce62de430de18c18b936e9e70e26fe55dfa3522e7a8880289af5c31` |
| 新 v2 manifest | `5ba2c970ff24e64473f2f591d2f3c1a7f4e2aa00b8d4463aca86f76e3bcb1a17` |
| 新 v2 report 文件 | `e9be99757cbb1ce96fd298cbc6255d328407e746364e99c243fa7ed039527086` |

v2 adapter 来自固定提交 `ec2ff456d08b4142233b61f1071402c41cadf60b`，使用 `three-clean-positive-or-inherited-fulltext-v2` 政策。完整公开聚合见 [NATURAL_KEYWORD_REPLAY_20261110.json](NATURAL_KEYWORD_REPLAY_20261110.json)。

## 对模型落地的影响

已确认缺口不是“尚未跑过完整证据链”，而是当前封存数据在既定规则下没有可评分关键词标签。继续增加同一批录音的模型运行不能填补标签缺口。

独立人声 FRR、连续自然 FAR、人工词尾、最终 AFE 与 SSC305 实板验证仍需各自真实输入和测量。不能用本次 native 完整性通过，或机器观察质量可用数，代替这些结果。

所有者交付保留新派生报告、人工审计计划、原/新保护器、执行及恢复脚本、完整失败和成功回执、固定源码以及上游 capsule 精确引用。原音频/native 证据继续保留在原封存包中；本公开增量不上传原音频、逐条 ASR 文本、私有审计映射或权重，也不创建新的工作流或定时任务。
