# Qwen42：冻结 C 模型与外部 Mandarin KWS 开发回读

本目录保存 2026-09-30 的历史研究证据，不是新的数据目录或模型发布。音频仍以 kws-data 原生 catalog/manifest 为唯一来源。没有复制 WAV、模型权重、二进制或虚拟环境。

## 结果与边界

42 条已观察的 Qwen3 合成开发音频，包括 20 条目标正例、22 条混淆负例。三种条件下：

| 条件 | 冻结 C：目标命中 | sherpa Mandarin：目标命中 | 两系统混淆负例事件 |
|---|---:|---:|---:|
| 原始音频 | 0/20 | 19/20 | 均为 0/22 条触发 |
| 末尾加 500 ms 零 PCM | 0/20 | 19/20 | 均为 0/22 条触发 |
| 开头和末尾各加 500 ms 零 PCM | 0/20 | 19/20 | 均为 0/22 条触发 |

外部模型各条件均无错误词或重复目标事件；唯一未命中 `qwen3-kw2-eric`（小窝小窝）。外部模型原生 train/development_a/development_b 角色分别为 14/14、4/4、1/2；人工审核源音频 9/10 正例，ASR 审核源音频 10/10 正例。这些角色是本项目的数据用途，不表示外部模型训练切分。外部 19 个事件均在 EOF flush 之前产生。

这只能确认当前开发素材上的执行差异，并排除上述特定 500 ms 首尾静音作为完整解释。未分离声学模型、前端、解码器和集成问题。两模型的参数量、先验训练数据、解码体系及分数尺度不同，不能据此宣称同资源领先、同误唤醒率领先、产品可用性、真人泛化、3–5 米能力或连续 FAR。未做终点延迟测量。500 ms 派生条件不是新样本，也未经重新人工审听；审核证据仅适用于原始源音频。

## 固定身份

- 数据 commit：`2f9658ffa9568076ef547615c76861abed84f56e`
- catalog SHA256：`27b590cdc325d48d2cf5e1293b8e431c815382be0219bc2e069d4e08896ee391`
- 原生导出收据 SHA256：`8e4d5c13cc5694e813ef6dc58bedbe34d2b893721283f1945d6d944ca38570b5`
- C 模型 SHA256：`ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402`
- C 关键词包 SHA256：`370ee3eeba27b1d62b38f32f53d8302b47c2b762392101e6ca7eb0ccf8dfb723`
- C 原执行源 commit：`0539106167f0bb7a657ce461a0ac5a508ecb3a22`；模型注册元数据及原关键词文本见 `ours/registry/`（仅供本次身份追溯，其中历史 synthetic-qualified 标签不是本次结果或产品许可）；精确编译命令、源文件和二进制哈希见 `ours/manual-build-receipt.json`，与当时 main 的逐文件比较见 `ours/source-main-comparison.json`
- 外部模型：`sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01`，FP32，原发布者 ModelScope revision `3787015f084cb241dfa0e4ba237703a2d4322d50`
- 外部运行时：官方 PyPI `sherpa-onnx==1.13.8` + `sherpa-onnx-core==1.13.8`，NumPy 2.2.6；源 tag 对应 `11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf`

外部 CPU 2 线程、80 维特征、max_active_paths=4、keywords_score=1.0、keywords_threshold=0.25、num_trailing_blanks=1，均在看到结果前冻结。每个原片独立新建 stream，以 20 ms 块输入；保留 API input_finished()，但不附加上游示例的 0.66 秒静音。边界控制仅使用独立给定的相同派生 WAV，参数不变。

## 许可与来源

原模型 README 明确声明 Apache License 2.0，保存在 `external/license/model-README.md`；不是从运行时代码许可推定权重许可。[原发布者模型页](https://www.modelscope.cn/models/pkufool/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/summary)说明 WenetSpeech L 10,000 小时训练。其底层数据的[官方条款](https://wenet-e2e.github.io/WenetSpeech/)另有非商业下载和原音频版权声明；此处仅进行有模型许可依据的研究推理，不宣称商业部署权利已经全部清理。

模型的固定 revision 下载 URL、每文件 SHA256、官方 wheel URL/摘要见 `external/sources/downloads.json`；保留的 PyPI JSON 是原始来源快照。运行时代码采用 Apache-2.0，原始 LICENSE 一并保存。引用的上游代码片段保留原版权头。没有把软件许可扩展为真人声线或训练素材的授权。

## 可携带核验

在任意含 Python 3 标准库的环境运行：

    python research/2026-09-30-qwen42-baselines/verify.py
    python research/2026-09-30-qwen42-baselines/test_verify.py

核验仅检查本历史证据包：文件字节/哈希、42 条源身份、原生角色和审核类型、实际事件与摘要一致性、两系统使用相同派生 PCM 摘要、固定零填充长度。它不下载文件、不运行模型、不重新授权资产，也不取代仓库原生数据核验。`archive-manifest.json` 不是另一个数据 catalog；其自身真实性依赖承载它的 Git commit。

`copied-file-origins.json` 记录每份原始文件的历史绝对路径与复制时哈希；所有复制文件保留原字节，未偷偷改写路径。`external/historical-scripts/` 是实际执行脚本快照，保留历史路径，供审计而非可直接运行的便携入口。真正重跑应另建隔离目录，从固定 data commit 导出同内容 ID，依下载清单验证权重/运行时，明确映射历史路径，重新生成独立 run ID。不要覆盖原始证据。C 运行器和通用消费代码由 kws-pipeline 管理，本目录不另建第二套运行框架。

## 内容索引

- `data/`：原生数据导出收据，含三个批次 content ID
- `ours/{raw,tail500ms,head500ms_tail500ms}/`：C 原始报告、输入列表、事件、运行收据
- `external/{raw,tail500ms,head500ms_tail500ms}/`：外部原始报告、事件、运行收据
- `boundary/`：固定探针规范、条件比较及两份精确派生收据
- `external/config.json`、`keywords.txt`：冻结参数及拼音关键词
- `external/sources/`、`external/license/`：官方来源证据和许可
- `archive-manifest.json`、`copied-file-origins.json`：档案完整性与历史来源
