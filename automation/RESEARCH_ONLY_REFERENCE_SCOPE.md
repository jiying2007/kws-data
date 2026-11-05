# KWS 技术研究：唯一声源导入与用途范围

本仓围绕中文“你好小窝”“小窝小窝”和英文研究用唤醒词开展
**离线技术研究**，优化独立测试语音上的 FRR、连续 FAR、实时性与
SSC305 计算/内存开销。研究数据与最终现场验证分开，不把合成音频、
三 ASR 一致或 Github Actions PASS 当作真实 KWS 效果结论。

当前唯一声源导入入口是 `automation/research_only_bulk_ingest.py`，
使用 `kws-dataset-research-tts-scope-v2`（不接受任何历史版 scope 回退）。
严守具体数据集/发行版适用条款，完整保留原始和转换 WAV 的 SHA、
独立数据研究用途审核、说话人谱系、持久私有身份登记以及创建一次的
私有源收据。旧导入器仅留在 Git 历史，没有当前可调用兼容层。

## 数据/模型研究路线

| 候选 | 研究用途与边界 |
| --- | --- |
| AISHELL-3 / OpenSLR 93 | 中文多声源短参考、TTS 和 KWS 研究；冻结下载版本与发布方条款 |
| THCHS-30 / OpenSLR 18 | 中文语音、ASR、语音负例；仅在允许的参考声线用途内克隆 |
| LibriTTS / OpenSLR 60 | 英文多说话人研究，保留 CC BY 4.0 署名和发行版身份 |
| Hi-Fi TTS / OpenSLR 109 | 英文语音研究，遵守实际录音/声音使用条款 |
| VCTK | 英语多口音研究，保留对应发布版来源 |
| WenetSpeech / Common Voice | 按实际媒体来源条款用于 ASR、负例和鲁棒性研究；不因公开下载自动成为可识别真人克隆来源 |
| Qwen3-TTS CustomVoice/VoiceDesign、F5-TTS | 离线研究合成候选；每份实际模型权重和源家族按许可/版本/哈希绑定，未运行的适配器不能算成数据 |

**重要**：版权声明、出版方使用条款、第三方声音/个人信息限制仍然是
研究使用的前置条件。研究 scope 是已审阅材料的带哈希声明，不能
靠填写 JSON 自动制造权限，也不能上传个人语音或以同一原始说话人
的多个克隆当成独立真人。

## 数据研究导入

`prepare_open_reference_pilot.py` 先把经授权取得的原始 WAV 转为
规范 16/24k PCM16 并保存原始/转换收据；
`open_reference_sources.py` 复核源数据和声源家族，生成私有候选索引；
研究导入器根据以下 scope 建立私有 `kws-bulk-clone-sources-v2`；来源别名
保持不变，新增的转录溯源只写入这个新版本，历史清单不改写。

~~~json
{
  "schema": "kws-dataset-research-tts-scope-v2",
  "dataset_id": "aishell3-slr93",
  "dataset_release": "actual-reviewed-release",
  "source_type": "reviewed-open-audio",
  "source_license": "Apache-2.0",
  "source_terms_sha256": "ACTUAL_64_HEX_SHA256",
  "research_review_sha256": "ACTUAL_64_HEX_SHA256",
  "intended_use": "offline-kws-tts-research",
  "research_tts_references_permitted": true,
  "public_voice_export_permitted": false,
  "forbidden_uses": [],
  "review_status": "independently-reviewed"
}
~~~

实际文件必须使用准确的出版方版本和用途审阅，不得填写占位符。

先为准备好的 source index 建立一次私有 speaker registry。新说话人按
三条 train / 一条 development 登记，8 人首批得到 6/2；已有身份和角色
只读复用，不随密钥、参考片段、数据集发行版或输入顺序改变。

~~~sh
python3 -B -m kws_data_contract.source_registry \
  --index /restricted/corpus/prepared/source-index.json \
  --registry /restricted/corpus/speakers.json --initialize --execute
~~~

以后增加说话人复用同一 registry 并去掉 `--initialize`。预览不写文件；
只有 `--execute` 才以 0600 权限原子保存。丢失 registry 不会自动生成
另一套身份。若已经用旧 HMAC 导入过声源，只在首次初始化时增加
`--legacy-manifest /restricted/old-source-pool.json` 和
`--legacy-key-file /restricted/old-hmac-key.bin`，配合旧原始 index 做一次
身份联接；原 speaker/family/role 原样保留，旧清单不重写，密钥不存入
registry。正常研究导入不再接受 HMAC key，公开别名与研究划分分离。

~~~sh
python3 -B automation/research_only_bulk_ingest.py \
  --index /restricted/corpus/prepared/source-index.json \
  --original-root /restricted/corpus/official-original \
  --audio-root /restricted/corpus/prepared/normalized \
  --conversion-root /restricted/corpus/prepared/conversions \
  --license-snapshot /restricted/corpus/publisher-terms.txt \
  --research-scope /restricted/corpus/research-tts-scope.json \
  --research-review-evidence /restricted/corpus/research-terms-review.txt \
  --speaker-registry /restricted/corpus/speakers.json \
  --max-refs 8 --output /restricted/corpus/research-source-pool.json
~~~

默认只校验，追加 `--execute` 才创建一次私有 0600 manifest。该 manifest
不代表任何训练样本已经通过真实语音和标签审查。
同一个 `audio-root` 可逐步补转录、内容检查或声源，每次指定新的 `--output`
清单路径。新收据以完整内容的 SHA-256 命名；已存在且字节相同的收据直接
复用，新增声源或来源条款快照生成新收据。同路径字节冲突会拒绝发布，旧
清单及旧命名收据均不覆盖。收据在完整写入后以 create-only 方式发布。
研究运行链：`bulk_pool.py` → `kws_pilot_preflight.py` →
`supervise_bulk_autopilot.py` → 原生三 ASR/声学复核 →
`kws-pipeline` 受控配对训练 → 独立语音和板端效果测量。

## 保存出版方转录并准备 ICL 参考

导入器和 `open_reference_sources.py` 都接受同一组可选参数：

~~~sh
--transcript-source /restricted/corpus/official-original/train/content.txt \
--transcript-sha256 ACTUAL_64_HEX_SHA256 \
--transcript-format aishell3-content-v1 \
--reference-content-checks /restricted/corpus/reference-content-checks.json
~~~

`aishell3-content-v1` 读取 AISHELL-3 的 WAV 文件名及交替汉字/拼音记录，
按原始文件名和说话人编号联接。其他已整理的出版方转录可使用
`path-text-tsv-v1`，每行是原始相对路径、Tab 和完整文字。输入文件需保留
实际字节与 SHA-256；每个选中片段保留出版方原记录、行号、文字哈希、
原始/规范 WAV 哈希。参考片段改变时，不能沿用另一段的文字或识别结果。

| 参考文字状态 | Qwen 克隆方式 |
| --- | --- |
| `unknown`：出版方未提供对应记录 | x-vector-only，`ref_text=None` |
| `dataset-provided`：保留出版方转录，内容尚无一致观察或存在分歧 | x-vector-only；文字和分歧继续保留 |
| `dataset-provided-content-checked`：独立的 audio-only 识别观察与转录一致 | ICL，传入实际参考转录；仍是机器证据 |

既有 `reference-content-checks.json` 的 `kws-reference-content-checks-v1`
继续原样兼容，
`observations` 中每条保存 `engine`、`model_id`、`model_revision`（40 位）、
`model_assets_sha256`、`wav_sha256`、`pcm_sha256`、`sample_rate_hz`、
`audio_frames`、`actual_text`、`input_mode="audio-only"` 和
`decoder_completed=true`。这些字段必须来自实际完成的识别，不能用生成
意图填充。每个片段最多三路不同引擎；已有观察全部一致才标记内容检查通过。
参考转录检查与最终生成样本的弱标签是两件事；本导入器不会自动产生
`verified` 或 human-gold，也不会执行 ASR。

新证据可使用 `kws-reference-content-checks-v2`，其中每行显式标为
`schema="kws-reference-content-observation-v2"`。音频身份、几何、全文匹配、
audio-only 和完成状态要求完全相同，模型身份区分如下：

| `model_revision_kind` | `model_revision` 的真实含义 |
| --- | --- |
| `model-repository-commit` | 模型仓库的 40 位 Git 提交，例如锁定的 Qwen/SenseVoice 权重发行版 |
| `native-checkpoint-sha256` | 原生检查点的完整 64 位 SHA-256，例如官方 Whisper `small.pt` |

两类都必须保留 `model_assets_sha256`。另一个字段
`implementation_source_commit` 只保存已知实现的 40 位代码提交；未知时为
`null`，不编造版本。代码提交不得冒充原生模型检查点身份。v1/v2 容器与
行版本必须匹配；旧 v1 清单与观察不重标为 v2。

中文比较忽略标点和汉字间空格；英文忽略大小写、常规标点，保留单词
边界和缩写撇号，避免把不同句子拼成同一字符串。Qwen prompt cache 绑定
参考 WAV、文字状态、实际文字和溯源；更新内容检查后不会误用旧提示。
出版方全文、识别原始结果、参考 WAV 与含原始路径的来源清单都继续私有保存。

本改动已覆盖离线合约和 mock prompt 调用。真实参考音频的识别与
ICL/x-vector 成对生成仍需实际执行，不能从导入成功推断克隆质量改善。

### 已规范参考音频的原生 ASR 输入

`reference_asr_inputs.py` 把已验证的参考清单、原始索引和转换收据联接到
现有 `kws-pipeline/tools/active_asr_batch.py` 的输入格式。它保留原始 WAV
及真实采样率/PCM16 哈希，直接将已经规范的 16 kHz PCM16 样本除以
32768，编码为 float32；这一步不重采样、不增益、不补静音或裁剪。
原始 44.1 kHz PCM16 不会被伪装成 D0 的 24 kHz FLOAT WAV。

~~~sh
python3.12 -I -S -B automation/reference_asr_inputs.py \
  --sources /private/research-source-pool.json \
  --sources-sha256 ACTUAL_SOURCE_MANIFEST_BYTE_SHA256 \
  --index /private/prepared/source-index.json \
  --original-root /private/official-original \
  --audio-root /private/prepared/normalized \
  --conversion-root /private/prepared/conversions \
  --terms /private/source-terms-snapshot.txt \
  --output /private/reference-asr-inputs
~~~

`inputs.json` 只含音频身份和变换说明；出版方文字、角色和说话人信息不传给
解码器。独立的 `reference-bindings.json` 将 opaque clip ID 连接到原始参考、
规范 PCM 和既有转换收据。调用 `verify` 可重新计算原始/规范 WAV、PCM、
float 样本的对应关系。部分官方旧 WAV 的外层 RIFF 长度可能陈旧；只有每个
实际 chunk 和完整 PCM 均存在时才能读取，原字节和长度不一致事实完整保留，
不会修写原 WAV 或接受缺失的音频样本。

后续使用已有 ASR worker 的同一 heavy-model lock、create-only 调用 claim
和原生空文本条件；完成结果与缺失/失败记录都保留。只有真实结果经独立核验
后才生成新的内容检查文件，并用新的 `--output` 清单路径运行研究导入器；
不原位改写旧的 dataset-provided 清单。

`reference_asr_checks.py` 在现有 batch auditor 完成后读取原始
claim/native/result、完整文件哈希审计及参考输入绑定。它要求本批计划的
三路观察均完整可用，才把三条原文一起导出到 v2 检查收据；任何分歧都保留。
三路全文经既有正字归一化后全部匹配出版方原文才可启用 ICL。导出不运行
识别器，不做多数票替换，不把错误或缺失的那一路藏掉。

~~~sh
python3.12 -I -S -B automation/reference_asr_checks.py \
  --input-root /private/reference-asr-inputs \
  --sources /private/research-source-pool.json \
  --sources-sha256 ACTUAL_SOURCE_MANIFEST_BYTE_SHA256 \
  --asr-results /private/reference-asr-results \
  --model-locks /checkout/kws-pipeline/research/qwen6_asr/model-locks.json \
  --whisper-source-lock /checkout/kws-pipeline/research/source-screen32-whisper-v1/source-lock.json \
  --output /private/reference-content-review
~~~

结果包括 `reference-content-checks.json` 和带完整分歧的私有
`content-review.json`。把新检查文件传给既有研究导入器并指定新的来源清单
输出路径，原来的来源、角色、出版方文字和历史清单保持原字节。

## 质量和实验边界

继续执行真实语音权限审核、PCM/说话人/派生家族隔离、文本盲筛和
错误音频隔离、长流 FAR/FRR 单侧置信上界、触发词尾延迟与目标板
CPU/RSS/功耗测试。禁止伪造数据效果、私有文件公开或以纯合成 CI
给出新模型有效的结论。

CI 只使用人工 PCM WAV 和虚构授权材料，运行 normal/-O/-OO 的
离线契约检查；真实语料/模型仍需私有环境另行执行。
