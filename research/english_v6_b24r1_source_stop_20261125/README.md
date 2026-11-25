# B24R1 SOURCE：已认证的终止记录

**当前结论：SOURCE_STOP_EMPTY_NATIVE_TRANSCRIPT；没有模型质量判决，没有晋级模型。**

[实际运行 45534020313](https://github.com/jiying2007/kws-data/actions/runs/45534020313) 在 **2026-11-25 11:12:06（Asia/Shanghai，UTC+8）**完成，结论为 failure。运行 head 为 `b65bfe4974432b3d922738d7931d1c2d6d6f367c`，唯一父提交为 `ea9f9e3b467231efe2746cbe64f867481177d09e`，执行次数为 1。SOURCE 生成完成；Parakeet 作业及最终来源准入读回失败。没有 SCREEN artifact，也没有进入 PRIMARY KWS 评分。

这次使用原始 B24 的固定权重。SOURCE 失败既不是 B24 数值质量不达标的判决，也不能视为一次完整 PRIMARY `NO_PROMOTION`。

## 真实消耗与失败位置

来源保留了全部 120 条原始音频：96 次合成与 24 条完整自然语音解码。全部 120 条均符合本次预观察资格。五路的计划上限为 600 次，实际执行如下：

| 观察器 | 尝试 | 原生返回 / raw 保存 | 有效载荷 | 空返回 | 未进入 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen | 120 | 120 / 120 | 120 | 0 | 0 |
| Whisper | 120 | 120 / 120 | 120 | 0 | 0 |
| Parakeet | 9 | 9 / 9 | 8 | 1 | 111 |
| XLSR | 120 | 120 / 120 | 120 | 0 | 0 |
| LV60 | 120 | 120 / 120 | 120 | 0 | 0 |
| **合计** | **489** | **489 / 489** | **488** | **1** | **111** |

Parakeet 的第 9 次调用对应 `b24r1-primary-84-mero-fast`。进程实际启动并正常退出，exit code 为 0，无超时或记录异常；stdout 原始字节精确为一个 `0A` 换行符。原始适配器在任何文字处理前保存二进制 stdout/stderr，随后以原始解析规则拒绝该返回：`EMPTY_NATIVE_TRANSCRIPT`。该条没有有效载荷或成功 capture 回执。

最终准入作业再次以 `ALL_DISTINCT_ATTEMPTS_RETURNS_VALID_PAYLOADS_AND_CAPTURES` 拒绝不完整账本，并记录没有 SCREEN 可上传。111 个未进入的调用保持不存在；没有补跑、重试、替换或重新标注。其他四路已经发生的调用与返回全部如实保留。

这些计数仅描述执行账本，不是 ASR 准确率或总体空返回率。

## 完整原始证据复核

[READBACK.json](READBACK.json) 是独立 stdlib-only 读回器的实际规范化输出，原样保留为 **12,097 字节**，SHA-256 为：

`e3ca7baa272609fc8a4b304fb56b52bd98c8a85a504dd2f4ab0fab298b777ffd`

[ORIGINAL_LOCK.json](ORIGINAL_LOCK.json) 为 **5,435 字节**，SHA-256 为：

`56c10ef6a669650050fd1d821871489c030f13f224fc85d1c9dc537c69ef37f8`

使用的 [读回器](../../automation/english_v6_b24r1_source_stop_readback.py) SHA-256 为：

`ac3e12810f72f68c3bdd4328dc5933b71c3e05da00524b508f91ce870b83d83d`

它验证了全部 14 个原始 ZIP 的 API 身份、摘要、CRC、成员边界和消耗标记，五份最终 API 及两个失败作业日志，实际运行捕获的全部 1,460 个代码和资源 Git blob，以及各观察器完整 FILES 闭包和全部计划账本槽位。原始源码依赖闭包及 19 个动态 helper 也已重新认证。

全部 488 份有效 native 载荷通过捕获的原始纯解析规则重读。两路音素结果的 18,344 帧逐值检查有限数值、模型和词表身份，并重算完整 greedy CTC tokens；不是仅检查文件哈希或形状。真实空返回再次由未更改的原始解析器拒绝。全部 120 条原始 PCM、合成 native chunks、自然来源身份和排除链、原始 TRAIN 证据以及完整嵌套 B24 STOP 均已重开认证；旧 STOP helper 实际运行两次并复现原始报告。

上述复核只读取保存的字节，不产生 TTS、ASR、音素模型、KWS 或优化器调用。报告中的 `new_*_calls=0` 描述这次离线读回，不会抹去上表真实 SOURCE 运行的消耗。

读回器的 17 项契约检查通过普通、`-O` 与 `-OO` 模式，独立代码审查未发现剩余实质阻塞。初稿曾将仅有哈希与形状核对的音素结果描述为完整语义复核，该初稿未发布；最终报告使用补齐数值和 token 检查后的上述 reader。

## 对两次空返回的限定取证

[EMPTY_RETURN_COMPARISON.json](EMPTY_RETURN_COMPARISON.json) 比较本次行与原 B24 的 `b24-primary-284-mero-fast`。两条保存的 stdout 都是单字节换行，203 字节 stderr 相同；进程均正常退出。九个共享捕获源文件相同，捕获层没有在原始记录前移除文字的路径，未发现包装或文字规范化丢字的证据。

原生内部为何返回空文本仍无法确定。完整 WAV/PCM 和命令身份绑定可以认证，但不能凭 stderr 的 full-context 标签独立证明原生内部逐帧处理的情况。时长、RMS、格式检查也不能证明请求词的真实发音或静音。此次没有新增试听、人类金标、观察器运行或质量判断。

## 曝光保留与当前权限

[EXPOSURE.json](EXPOSURE.json) 保留全部 120 条原始 PCM 身份和人群映射，包括失败行及从未被 Parakeet 调用的行，并保留 36 名 primary 人群与 36 名从未打开的 replica 人群。任何未来经明确注册的新选择都必须排除全部旧曝光和保留人群，不能回收未观察行或封存 reserve。

[CURRENT_STATE.json](CURRENT_STATE.json) 是可供后续工作读取的当前终态：

- 原始 B24 和本次 B24R1 的 SOURCE STOP 永久保留；不继续剩余 111 次，不重试，不自动注册 R2。
- 来源没有完成准入，没有数值质量判决，独立 PRIMARY KWS forwards 为 0。
- PRIMARY、封存 replica、工程验证、软件候选安装、晋级及产品发布均未打开；候选选择和候选 manifest 保持为空，B0 保留为基线参考。
- 已准备的 B26 仍要求完整真实 B24R1 PRIMARY `NO_PROMOTION`；本次 SOURCE STOP 不满足它。旧 B25 条件也不改变。
- 新 SOURCE 协议仅能作为另行审阅的未来提案，当前没有批准、注册、选人或激活。不能把提案用于重新解释本次失败。
- SSC305 实机与现场 FRR/FAR 证据均为 `NOT_RUN`。交付源码与 CI 完成不等于候选模型落地。

[ORIGIN.json](ORIGIN.json) 提供公开的最小来源身份及完整摘要。该目录不复制原始 API 正文、作业日志、非空转录、音频、完整 source manifest 或原始 ZIP；完整原件仍由所指 GitHub Actions artifacts 保存。离线复核接口与原始目录布局见 [SOURCE_STOP_READBACK.md](../english_v6_b24r1_preparation_20261124/SOURCE_STOP_READBACK.md)。
