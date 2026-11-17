# 完整上下文 32 次诊断：原生 int32 mask 修复

本任务修复实际运行发现的输入接口错误，继续尚未发生的模型前向。它复用已经完成的 16 条完整 PCM，不再派生、不合成、不重新运行旧裁剪输入。原任务和所有旧产物保持终态，旧发音标签与 `STOP_DIAGNOSTIC_NO_EXPANSION` 不变。

## 真实失败边界

公开 run [44133547091](https://github.com/jiying2007/kws-data/actions/runs/44133547091)、head `965bb0cc7150d244ac4f06f512f58a3824afbab2` 已完成：

| 已发生的阶段 | 实际计数 |
|---|---:|
| 完整输入派生 | 16，全部完成 |
| 音素模型构造并严格加载 | 2，各模型一次 |
| 原输入预处理尝试 | 32，各模型 16 条 |
| 预处理失败 | 32，均为 `ACTUAL_UNPADDED_ATTENTION_MASK` |
| 模型前向、返回、音素捕获 | 0 / 0 / 0 |

原保护代码错误地要求 mask 为 `torch.int64`。固定版本 Transformers 4.46.3 的原生 feature extractor 在该路径返回 `torch.int32`。模型已成功加载，错误发生在 `model(**batch)` 之前；因此这不是“整个任务从未开始”，也不是“没有模型构造”。两份 MODEL_LOCK、全部 32 份 STARTED/FAILED、两份终态 ledger、实际 job 记录和闭合文件清单共同证明没有前向，不能单靠缺少 capture 作推断。

原 final artifact 固定为：

- ID `14350388434`，名称 `english-kokoro-context32-44133547091`。
- 129,737,354 B；SHA256 `e62a5233353efd4d07974213bef0a4025b056ea4cac7d6d1035ebc96907be378`。
- 原完整输入 manifest SHA256 `f98935a551ea7d78d9118820e9b1c43d1f5cad840ba962df778d813881813cf3`。
- 原 source plan SHA256 `f5bf386a956bce3d1e1bdeee2ebf243026a284ba211d12bc780ca5a3d2a34d9a`。

这个 final ZIP 本身已保留完整 prepared、原更早 source/probe ZIP 和零前向记录。新任务只读取这一个 GitHub 原始 ZIP及其实际 artifact/run/jobs API；不会重新打包后冒充原 artifact，也不把旧失败改成成功。

## 修复范围

新观察器保留同一输入、权重、模型版本、归一化运算、实际 `batch` 对象、CTC 解码、全帧 logits 与计数顺序。修复只接受已核对的原生 `int32` mask，并按原生值保存 `attention-mask.i32`，每样本 4 字节。不会 `.long()`、cast 或改写真正送入模型的 batch。

生产与公开 processor 检查调用同一个 prepare 函数。三份实际安装的 Transformers 前端源码按官方字节锁定；模型仍严格验证完整资产和 state dict。任何真正的 shape、设备、mask 内容、版本或输入身份错误继续失败，不能通过扩大 dtype 集合掩盖。

新 campaign 为 `kokoro-full-context32-native-int32-first-forward-v1`，观察 schema 为 `english-kokoro-context32-observer-v2`。原 `INPUT_MANIFEST`、16 个 ID、PCM/WAV SHA、采样数、旧 run 和原 source plan 全部保留原字节。新 run 身份只写入新的执行与观察收据，比较器显式分别接收 `source_plan`、新 `plan`、新 `run_identity`。

## 公开 processor 检查与执行限制

新工作流 `.github/workflows/english-kokoro-context32-maskfix-once.yml` 的 `processor-native-smoke` 使用与生产相同的固定 CPU runtime，只构造真正的 feature extractor，处理固定 640 样本整数 fixture。该 fixture 不来自这 16 条音频，检查不下载权重、不构造语音模型、不做模型前向。

同仓公开 PR 和启用后的首次 main push 各运行一次检查。PR 的真实 processor 成功证据是合并条件；main 的检查是后继 reuse/observe 的 workflow dependency。两次 fixture 调用与原来的 32 次预处理、新任务至多 32 次原输入预处理和首次前向分别计数，不混称为音素观察。

没有实际 processor 结果时，源码与 mock fixtures 通过不等于原生接口已经通过。源码默认 `enabled=false`；启用时只允许真实 main 前驱 `965bb0cc7150d244ac4f06f512f58a3824afbab2` 的唯一后继，公开 main push、attempt 1。不能只换前驱继续消费相同恢复身份。

| 新任务项目 | 固定上限或限制 |
|---|---|
| 原失败状态保留 | 2 次构造、32 次预处理失败、0 次前向 |
| 复用现有完整输入 | 全 16 条；manifest、ID、真实 PCM/WAV 哈希不变 |
| 新音素模型构造 | 最多 2，每个 provider 一次 |
| 首次模型前向 | XLSR 最多 16，LV60 最多 16 |
| 已消费前向重放 | 0 |
| 新派生、TTS、文本 ASR、embedding、训练 | 0 |
| 同一新任务自动重试或重新选输入 | 0 |

入口将重新验证真实 prior ZIP、全部闭合文件、先前两 MODEL_LOCK、32 个精确错误、完整终态计数与原 16 条有效 PCM。只要发现任意先前前向、返回、capture、错误身份变化或不完整零前向证据，本恢复就拒绝执行。

## 保存方式与诊断口径

复用 source artifact 名为 `kokoro-context32-maskfix-source`，包含一次 `inputs/prior-final.zip`、三份实际 API 响应、`PRIOR_ZERO_FORWARD_PROOF.json`、`INPUT_BINDING.json`、新计划、源码与闭合清单。原 prepared 只临时解包给只读观察器与比较器，不作为第二份完整来源链上传。

最终 `english-kokoro-context32-maskfix-<run>` 只保留 `recovery/` 内的一份 raw prior ZIP，加两个 provider 的新收据和完整数值捕获。上传使用普通 ZIP 压缩以减少存储；嵌入的原 ZIP 文件字节不变。单文件上限 256 MiB、整个展开包上限 1 GiB，明确覆盖原 129 MB ZIP 与新增数值记录。

原生 mask、normalized input、完整 logits、全部 frame IDs 和 CTC 均保留并离线重算核对。缺失最终收据时实际前向计数为 `null`，另给已保存 marker 下界与最大上界；不能把丢失证据当作 0 次。

这一修复的目的仍是完整上下文诊断。原完整输入与旧裁剪对齐后，所有 16 条在裁剪起点后第 8 个样本至 EOF 已独立核实逐字相同；但整个输入归一化、卷积格点和 transformer 上下文仍不同，因此观察变化不能独自证明某一个因果机制，也不能直接更新旧准入。

最终始终保持 `training_allowed=false`、`training_snapshot_allowed=false`、`automatic_expansion=false`、`old_labels_modified=false`。它不证明 FAR、FRR、词尾延迟或高质量模型目标已经达成。新真人录音、同中文的模型架构和物理硬件仍不是继续研究的前置条件。
