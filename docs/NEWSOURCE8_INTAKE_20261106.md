# 新真实来源的有界输入准备

本阶段为现有 KWS 研究准备最多 8 个新的语料说话人参考音。来源沿用已固定的 AISHELL-3 发布版，排除原 registry 中全部 32 个来源，并保存原音、转换、来源身份和实际命令回执。该入口不执行 TTS、ASR 或训练。

## 冻结范围

| 项目 | 本批约束 |
| --- | --- |
| 来源身份 | 固定发布版、强 ETag、归档长度以及每段实际 Content-Range |
| 前缀正文 | 最多 134,217,728 字节，即 128 MiB |
| 请求粒度 | 一个 fetch 调用，完整获取时为 128 个 1 MiB 范围 |
| 有界基础设施重试 | 同一范围仅对 HTTP 502、503、504 最多尝试两次 |
| 整轮重试、重定向、换源、扩量 | 均不允许 |
| 新来源 | 最多 8 个不同语料说话人，每人一份有效参考音 |
| 选择依据 | 复用既有选择器：有效单声道 PCM16、1–15 秒，距离 5 秒最近，再按内容哈希确定顺序 |
| 时间 | fetch 最多 600 秒；整个准备阶段最多 1,200 秒，并为终态记录预留时间 |
| 模型调用、优化器更新、研究队列 | 全部为 0 |

请求、源码、原 32 份身份清单、解释器和 ffmpeg 均在执行前固定。解释器 pin 必须指向实际普通文件；符号链接会在任何执行 claim 和 HTTP 请求之前被拒绝。通过版本相同的解释器真实路径修正配置时，应另存新请求和实际失败记录，不改写原声明。

## 两个新增入口

- [`automation/run_newsource8_intake.py`](../automation/run_newsource8_intake.py) 编排一次 fetch、选择、转换、registry 追加和研究输入导入。执行 claim 使用独占创建；已存在的失败或未知 claim 不会自动恢复。
- [`automation/recorded_aishell_fetch.py`](../automation/recorded_aishell_fetch.py) 调用原有 fetcher，逐次保留真实 URL-open 请求与响应头。响应头返回与正文校验分别记录；正文长度、哈希和范围接受以原 fetcher 的范围收据为准。

执行器为每个子进程保存原 argv、工作目录、实际 stdout、stderr、返回码、时间和超时状态。超时先向整个进程组发送 TERM，再补发 KILL；即使直接子进程已经退出，也不会省略后代进程组的终止。剩余总预算限制每个阶段。这是有界同步入口，不是常驻监督进程。

[`automation/recover_newsource8_partial.py`](../automation/recover_newsource8_partial.py) 是独立的离线恢复入口。它要求原 fetch 已有真实失败终态和外层返回码，只接受从零连续覆盖、逐段哈希匹配且具有真实 206 响应的完整范围。它在新目录创建精确前缀副本，保留原文件的所有字节与未知请求；不创建原 fetch 的成功 retrieval 收据、不补字节、不重新发出 HTTP。恢复有独立的 300 秒外层截止时间，并继续沿用原 32 来源排除、最多 8 人及原文本状态规则。

## 原身份与参考文本

原 registry 作为独立输入保留；工作副本只能追加真实新来源。旧 32 条记录的身份和 train/development 角色必须逐条不变。新增来源还要通过原始 WAV、归一化 WAV 和 PCM 哈希去重，不能用不同文件名把旧参考音计作新增。

仅当全部实际选择的参考音来自同一个原始分区，并且该分区原版文本确实存在时，才导入其 dataset-provided 文本。混合分区或缺少对应文本时，参考文本保持 unknown，不拼接出一份冒充原版的文本。dataset-provided 仍不表示已经听音核验、原生 ASR 核验或 human gold。

获取或后续步骤失败时保留真实部分文件和命令回执，并停止后续步骤。零个或不足 8 个新来源也是可报告的有界终态；不会补入旧说话人或自动下载更大前缀。实际新增拆分沿用既有的三 train、一 development 追加策略，不会自动产生独立 calibration/test 集。

## 检查与后续消费

[`automation/test_newsource8_intake.py`](../automation/test_newsource8_intake.py) 的 11 项测试用无网络、无模型的合成子进程和范围响应验证失败停链、独占 claim、超时后代终止、来源保护、符号链接输入拒绝及真实分区文本选择。[`automation/test_newsource8_offline_recovery.py`](../automation/test_newsource8_offline_recovery.py) 的 12 项测试覆盖完整范围、哈希、连续覆盖、缺失响应、原 fetch 成功拒绝、未知尾部保留及恢复后的来源约束。两组均已在普通、`-O` 和 `-OO` 三种模式下实际通过；既有 fetch/转换测试继续保留。

原始前缀正文、实际选中的原音及派生输入应与来源和转换回执一起保存。后续参考音 ASR 或声音克隆需要独立固定新的批次身份、输入、模型来源与预算；本次输入准备本身不证明参考文本正确、生成成功、训练准入、自然 FAR 或物理设备表现，也不重发历史 D2 未知调用。

## 本次实际执行记录

2026-11-06 的实际状态如下。首次 launcher 在执行 claim 前拒绝符号链接解释器，返回 rc1，HTTP 与模型调用均为 0。随后另存新请求，仅改为同一 CPython 3.12.16 解释器的实际普通文件路径；旧失败回执保留。

| 阶段 | 实际结果 |
| --- | --- |
| 唯一网络 fetch | 600.004 秒超时，子进程 rc−15，外层 rc1；原 128 MiB 请求保持失败 |
| 完整返回部分 | 81 MiB，即 84,934,656 B；81 个完整范围的正文、哈希和 206 响应均匹配 |
| 未确认请求 | 共有 82 个 URL-open 请求记录，最后一个无响应记录，保持未知；无已观察到的 HTTP 错误或重试 |
| 独立离线恢复 | worker 3.183 秒，外层 3.274 秒、rc0；无新增网络或模型调用 |
| 新来源 | 实际 8 个，6 train-candidate、2 development-only；registry 从 32 追加至 40 |
| 原身份保护 | 旧 32 条身份及角色全部不变；新旧 source family、speaker key、原 WAV、normalized WAV 与 PCM 哈希交集均为 0 |
| 参考文本 | 实际使用原 test 分区对应文本；8 条均仅为 dataset-provided，尚未 content-checked 或 human gold |

已取得前缀的 SHA-256 为 `ba987cf393db589cbe86ca0a584138ebdf0ed0897a41c0e8ff2b77e3ed88b47c`。该值校验已取得的 81 MiB，不是完整约 19.1 GB 归档的哈希。原失败 terminal 的 SHA-256 为 `1c8b4c92964c97c0f6c847e12647636eaecb6621124e551747f613a3f8f8cc3e`；独立恢复 terminal 为 `fd351aa05a0aad6878eb967c80ccc04dde964d09d9cb7cc4f9748d9e705c6563`，新研究输入 manifest 为 `621117a8a37b31a37b5df302e45c90503a0356cc0278346df76d8b8b687fd2ec`。

独立终态审查实际 rc0，核对恢复的 79 个终态文件和 276 项原始绑定；审查记录 SHA-256 为 `d1a20c22d6ac3b42c17d75a2c1b7b78634d9d0bb3416e976b8434e4df12ed87b`。本页记录的是输入获取与保留结果；后续参考音 ASR、克隆或训练需按各自实际运行记录单独报告。
