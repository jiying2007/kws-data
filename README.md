# kws-data：中文唤醒词研究数据

[中英文 KWS 音频数据源与模型许可研究入口](automation/README.md)：仅无模型的离线任务规划；不改变当前训练/发布准入。


## 2026-11-03 研发目标更新：非商业技术研究

**KWS 当前是 noncommercial technical research，目标是技术效果达到或超过优秀商用方案，不是发布/商用授权。** 按[研究专用数据/模型用途与声源导入说明](automation/RESEARCH_ONLY_REFERENCE_SCOPE.md)，允许在具体开源发布版条款范围内使用 AISHELL-3 等非商业研究语料，以及 F5-TTS 等非商用模型权重作为研究候选；不再要求先证明商业声线克隆许可。`automation/research_only_bulk_ingest.py` 已成为唯一参考声源导入器，直接执行原始 WAV、PCM、匿名身份及研究用途验证。平行商用导入器和兼容层已退役；独立数据审查、目标板性能测试等实验验收要求仍保留。许可快照、研究用途审核和原始声源仍需私有保存；源码验证和人工合成测试不能声称真实语音效果。

## 2026-11-03 数据准备现状与唯一闭环入口

> **数据集准备状态：PARTIAL / BLOCKED_EXTERNAL_SOURCE。** 软件源码、批次计划、保存的合成音频、原始标签、筛选收据和产品资格证据不能相互替代。不能以 CI 成功、声音数量、ASR 多数投票或“holdout”目录名宣称数据集/模型已经具备量产资格。这里记录当前主线可验证的事实；历史研究结论见下节。

| 资产或环节 | 当前可核验范围 | 用途及剩余关口 |
| --- | --- | --- |
| [正式 catalog](catalog.json) + [唯一验证/导出器](tools/codex_assets/__main__.py) | 3 批、42 条历史 Qwen3 合成音频（20 条人工审核开发、18 条 ASR 训练开发、4 条已观察回读） | **仅 observed development**；不允许把原数据改名后充当未暴露封存测试。训练消费须固定 data commit + catalog SHA-256，原始标签和 speaker split 不得重写 |
| [Qwen16 原始证据](research/qwen16-archive-20261030-v1/README.md) + [盲标输入](research/qwen16-label-inputs-20261030-v1/README.md) | 16 条已经生成并归档的原始目标语音（43.2 秒）；12 条 train 候选、4 条 development_a；12 条已准备匿名音频的三 ASR **请求** | **尚无独立审定的实际标签或训练准入**。只有冻结原 WAV、真实三 ASR 完整收据、抽查/争议复核及每条权利检查通过后才可能入库；不能重跑已消耗的 Qwen16 生成 |
| [批量 TTS/克隆/ASR](research/auto-tts-voice-clone-20261102/BULK_AUTOPILOT.md) | 单机有界任务、缓存模型/参考声线、不可变开始/完成收据、独立 pipeline 消费与模拟后端回归测试 | **实现可测，不等于真实音频可用**。仍需合法参考音频、具体声音使用权、模型资产、真实生成/ASR 回执与声线分组的有效正负例率；失败/未知尝试不静默重放 |
| [中英文 Provider 与请求规划](automation/README.md) | 有界、离线、source-first 的模型/许可候选与三家族 ASR 规划；不调用模型 | **仅 planning**。中文词固定为“你好小窝”“小窝小窝”；英文正式唤醒词尚未授权指定，不根据研究示例发明产品配置 |
| [真人采集计划](research/field-collection-pilot-20261102/README.md) | 私有采集请求、身份与证据检查的源码及无模型测试 | **并无经证明的真实物理采集和封存验收**。真实录音/同意/房间/设备元数据应保留在受控存储，不公开上传个人声音 |
| [产品数据闭环跟踪](https://github.com/jiying2007/kws-pipeline/issues/691) | 数据效果/训练准入/比较器/资格门禁在 `kws-pipeline` 跟踪 | 仍缺可证明的训练语音、独立长时负流、真实 FRR/FAR 和最终 AFE/SSC305 证据，禁止自动标记 shipping PASS |

### 唯一可信执行链

1. **合法来源**：精确的来源、模型与参考声音身份及使用权；无独立许可的真人/公开录音禁止声音克隆。原始证据在私有位置按 hash 绑定。
2. **不可变音频**：先保存原始 waveform、PCM hash、采样格式、真实声源 family / session；衍生 RIR、噪声、远场和变声保留父源谱系，不以新 seed/新名字制造独立说话人。
3. **自动筛查**：按中文/英文的准确目标词、近音、残缺前缀、跨句残留、普通谈话、媒体/音乐及真实环境负例分类；盲态三 ASR 的一致只产生 *machine-screened candidate*，分歧/不完整/疑似负例中有唤醒词均 HOLD。
4. **审核与分割**：对争议、同音/不完整词、许可、真实音频进行独立核验。按 speaker + 原始 source + clone/augmentation ancestry + session 防跨 train/development/calibration/test 泄漏；已观察历史音频不得伪造封存资格。
5. **训练准入**：`kws-pipeline` 的既有 `speech_like_corpus_plan` → `training_admission` 是唯一实际授权链。**仅当真实验证的行数大于 0** 才启动有界、一次只改一变量的训练/模型对照；source-only 计划不授予训练权。
6. **产品验收**：独立真人最终 AFE 最低 40 人×2 session、≥1200 正例、≥24 小时有效连续负例；同时验收两完整词的 FRR/FAR（含 95% 单侧置信上界）、触发延迟，再完成物理 SSC305 的 CPU/I/O/RSS/功耗/连续运行，最后冻结完整发布元组。参见 [kws-pipeline #691](https://github.com/jiying2007/kws-pipeline/issues/691) 与 [外部产品资格 #2](https://github.com/jiying2007/kws-pipeline/issues/2)。

### 去兼容、去冗余的可执行边界

- 正式数据目录与导出：**`catalog.json` / `tools.codex_assets` 唯一权威**。研究中的请求计划、来源/ASR 收据和模型排行榜**不是**第二个生产 catalog。
- 历史 research 分支收据、已发表音频/失败证据是不可变审计资产，不纳入默认训练，不对消费链提供历史格式兼容回退。先明确实际消费者并迁移，再删可证明未使用的重复运行器；不得通过改旧 hash、删除固定 evidence 或关闭门禁制造“清理完成”。
- 新的真实数据批次**发布新不可变 dataset ID、manifest / review / split / rights hash**，不得覆盖已消费的 Qwen42/Qwen16 身份。大文件按不可变对象的内容地址和实际字节验证，私有数据与公共索引严格隔离。
- 优先运行现有的原生 batch / 三 ASR / pipeline consumer；**不再新造平行的 TTS/ASR runner、虚假的 English shipping phrase、模拟真实 speaker count 或自动批准资格**。

开发数据的现有验证命令：`python3 -m tools.codex_assets verify`。所有未来训练必须使用经批准且 commit/hash 锁定的源，不能从最新版 `main` 隐式接受新数据。

### 有界常驻 Qwen 数据生成进程监督器（2026-11-03）

现有 `research/auto-tts-voice-clone-20261102/bulk_autopilot.py` 已在单次执行中复用模型和声音提示缓存；其 `--wall-seconds` 只限制**新任务启动**，无法中断已卡住的 TTS。新增 [外层强制超时入口](tools/supervise_bulk_autopilot.py) 复用原控制器、不复制 TTS/ASR 生成链：固定源代码 Git blob、Linux 独立进程组、整次执行硬超时与子进程树终止，保护原有 `started.json` 不自动重放语义。仅在合法来源、模型资产和许可已实际验证的私有离线机器上显式传入 `--execute`：

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -B tools/supervise_bulk_autopilot.py \
  --plan /restricted/plan.json --sources /restricted/sources.json \
  --audio-root /restricted/references --workspace /restricted/autopilot \
  --adapter-lock /restricted/three-asr.json --pipeline-root /restricted/pinned-pipeline \
  --model-folder /restricted/qwen-assets --model-lock /restricted/model-lock.json \
  --max-tts 16 --max-asr 48 --wall-seconds 300 --hard-seconds 600 --execute
```

`--hard-seconds` 达到后返回 124，终止同进程组模型/子进程；已经开始但未完成的任务保留为未知消耗，**不会自动重试**。这是进程级硬上限（包含模型加载），不是单样本性能上限，也不证明训练准入/发布资格。CI 只用假子进程做正常、异常退出和超时测试，不运行真实模型。

### 小批量语音实验执行前检查（2026-11-03）

新增 [源/模型资产/语音类别小批量预检](docs/BOUNDED_PILOT_READINESS.md)（仅只读、不开模型）：会按当前执行器实际的前 N 条任务检查两完整唤醒词、近音负例、普通负例、不同声源家庭的覆盖，重验来源 WAV/权利文件、Qwen 模型文件与三 ASR 模型权重及执行器 SHA。检查失败不得因有限预算误以为模型学习失败或生成够量；成功也只证明技术源一致，仍需真实声源授权、ASR 执行和独立审定。

### 开源语音数据集参考声源候选（2026-11-03）

已增加 [AISHELL-3、THCHS-30、LibriTTS、Hi-Fi TTS、VCTK 等公开来源候选与本地导入](automation/OPEN_SOURCE_REFERENCES.md)：只从官方许可渠道获得音频，私有机器导入原始/规范 WAV 与不可变转换收据，选择每说话人一条约 5 秒参考并生成唯一研究导入器使用的私有候选 catalog。WenetSpeech 等具有第三方来源或特殊许可限制的资源不能自动进入真人克隆候选；在具体版本与非商业研究用途获得审阅后才可继续 Qwen 实验和严格训练准入，不要求或虚构商业许可。此更新没有下载真实音频或生成新语音。

### 公开语料小批量 WAV 自动准备（2026-11-03）

[原始 WAV → 规范 16/24k PCM16 → 哈希及私有转换收据](automation/BOUNDED_CORPUS_PREP.md) 现可离线自动完成每个来源说话人最接近 5 秒参考片段的选择、固定本地 ffmpeg 文件 SHA、拒绝更改源文件和软链接，并直接产生现有开源声源候选导入器接受的私有 source index。不会下载未经许可的音频、执行 TTS/ASR 或自动发放商业声音克隆权限。AISHELL-3 等真实人声须先核实实际版本的商用和声音使用范围；原创合成声线仍可并行研发。

## 历史研究入口（2026-10-07 整理）

- [完整研究索引与当前结论](docs/RESEARCH_INDEX.md)：覆盖 ASR 校准、A20、N0、Cosy49、Qwen6、Melo6、N1 与失败 first-prefix 候选；标明 PASS / FAIL / REJECTED / 待资格验证的实际范围
- [逐文件来源与保留清单](docs/BRANCH_RETENTION_20261007.json)：23 个远端分支的固定提交、1,180 个新增保留文件、原主线逐文件身份；归档保留不等于模型采用或训练准入
- 当前判断：原 A20 仍只是**不足的研究基线**；fixed300、Cosy49 和 first-prefix 候选均已停止。Melo 的上下文结果恢复了部分命中，但“小屋”误触发仍在。真人泛化、连续 FAR/FRR、声学词尾延迟和目标板资格均未建立
- 原 42 条 catalog、split、生产导出逻辑保持原身份。已有公开合成数据与研究权重只保留原权利/用途边界；没有新增私人自然语音、特征或权重，没有启动新采集、训练或推理

以下保留原有数据使用说明；历史报告中的阶段性描述请结合上方当前索引阅读。

本仓整理“小窝小窝”和“你好小窝”的语料、TTS 生成方法、收集方法与筛选证据。首批为 **42 条 Qwen3 合成开发音频**，包含人工确认与机器筛选两种证据。所有录音均为 16 kHz、单声道、PCM16 WAV；每条记录文件/PCM SHA-256、文本、正负标签、声线、seed 与源 ID。

| 数据集 | 条数 | 准入证据 | 原有用途 |
| --- | ---: | --- | --- |
| `datasets/qwen3-reviewed-v1` | 20 | 对固定审听包的逐条人工确认；收据绑定 WAV 哈希 | 12 训练、4 开发 A、4 开发 B |
| `datasets/qwen3-train-asr-v1` | 18 | 精确 ASR 文本匹配；连续正例内部低能量间隔小于 120 ms | 扩展训练 |
| `datasets/qwen3-holdout-asr-v1` | 4 | 精确 ASR 文本匹配；连续性筛选 | 已观察的开发回读 |

这些样本属于研究开发数据。它们均已被观察过，不是封存盲测集；五个预置 TTS 声线不是五名真实受试者。人工接受和 ASR 自动接受分别记录；“优质”在本仓表示达到对应批次的准入规则，不表示已覆盖真人、真实房间或目标设备。

## 研究档案与当前结论

这些入口分别回答“当前模型能否识别”“从零训练的小模型是否泛化”“云端运行消耗多少资源”“紧凑输出头是否保真”；原始证据保留在各自目录，不另建数据 catalog。

| 研究档案 | 已观察结果 | 使用边界 |
| --- | --- | --- |
| [冻结 C 与外部模型回读](research/2026-09-30-qwen42-baselines/README.md) | 42 条原始音频：冻结 C 命中 0/20 正例，外部 sherpa 命中 19/20；均为 0/22 混淆片触发。固定首尾静音探针没有改变命中数 | 只说明这批开发素材的执行差异；不同前端、训练与解码预算，不能当作同资源产品排名或连续误唤醒率 |
| [轻量 CNN/FSMN 四运行负结果](research/2026-09-30-lightweight-clip-learnability/README.md) · [完整结果](research/2026-09-30-lightweight-clip-learnability/RESULTS.md) | 各运行仅命中人工审核开发正例的 1/4 或 2/4；跨声线双词目标未达到，已停止追加训练 | 保存失败证据与自产研究权重，没有提升为产品候选；整片预测与事件指标不能混用 |
| [云端 CPU / 内存 / I/O 测量](reports/host-resource-profiles-20260930/README.md) | 冻结 C、cFSMN、sherpa FP32/INT8 与轻量 A/B 的测量范围、时序和资源证据 | x86 host-only，未测 SSC305；各路径计时范围不同，不直接排名。CPU/I/O 优先，模型/RAM 可调整；单核 5%/10% 只是待确认预算建议 |
| [紧凑输出头蒸馏失败档案](research/2026-09-30-compact-head-distillation/README.md) | 保留开发集 teacher 的 3 条命中，但四个目标类误差均未通过固定保真门，已停止该候选 | 仅保存本次训练的 846 参数 head；冻结 donor 按来源、许可与 SHA 重建，不重复发布整块预训练权重。失败、数值报表异常及缺失统计均保留 |
| [完整 FSMN 数值诊断与固定原生成本](research/2026-09-30-full-fsmn-numeric-alignment/README.md) | 旧 B / 实际 PCM 严格数值门仍失败；非分数事件字段一致，34 条触发分数存在微小差异；后续 x86 固定原生测量 RTF 0.024435（235.68 s） | 全层传播界过宽，不证明保真；保留全部原始标量报告，未发布 donor 权重/大数组；不是 SSC305 或产品验收 |

新增[源标注近邻与单线程配置回读](research/2026-09-30-himia-source-neighbors/README.md)：相同历史7006片三模型均零事件，Qwen正例仍为0/20、19/20、11/20；sherpa API1重复保持全部事件，实测worker采样均1线程。仅12种弱文本、已观察开发素材，非连续家庭FAR；主机CPU成本与部署边界见原报告。

新增[FFT定位与冻结编码器P/R头失败档案](research/2026-09-30-frontend-and-frozen-head/README.md)：两个窗口定位首个差异为FP32 FFT，原数值门未修复；唯一P/R实验虽然P两输出BCE较低，但开发双bit仅4/8 vs1/8、P人审TRAIN0/12，两严格门均FAIL。保留原异常与只读恢复顺序、限定FLEURS配方授权和有限derived-feature/小head资产。

### 数据角色与准入边界

- **训练/开发角色**：上表数据集的原有用途保持不变；具体实验还要固定实际取用范围。四运行小模型只训练 12 条人工审核片，开发 A/B 各 4 条，22 条 ASR 片全部只报告结果；不是把原生训练角色改成盲测
- **已观察开发集**：全部 42 条已经回读；历史目录名 `qwen3-holdout-asr-v1` 不代表新的 holdout 或封存资格集。人工审核描述合成音频的审核方式，不表示真人录音
- **未准入来源**：来源调查、下载或模型回读记录不等于可训练语料；未进入固定 catalog 并通过对应审核/权利规则的材料不自动成为训练输入。见[来源状态](docs/SOURCES.md)、[准入规则](docs/CURATION.md)和[权利记录](docs/RIGHTS.md)
- **产品边界**：现有结果没有证明真人/3–5 米泛化、连续流 FAR、真实词尾延迟或 SSC305 CPU/I/O/功耗。允许继续使用合成和获准开源资产推进软件实验；板端验收与最终产品放行仍需独立证据

## 使用

```bash
git clone git@github.com:jiying2007/kws-data.git
cd kws-data
python3 -m tools.codex_assets verify
```

`catalog.json` v2 是唯一目录，固定各批 manifest、收据和切分文件的 SHA-256，并引用结构化来源/权利、审核与切分规则。核验工具检查全部音频格式、文件/PCM 哈希、标签与审核覆盖、重复 PCM、跨批声线切分一致性及生成脚本身份。工具没有第三方依赖，Python 3.8+ 可运行；TTS 推理环境另见生成配方。

训练加载时读取 `manifest.json` 的 `recordings`，将 `path` 相对于该数据集目录解析。`splits.json` 固定开发切分：Vivian、Uncle_Fu、Dylan 为训练声线；Serena 为开发 A；Eric 为开发 B。同一声线的新增 seed 或增强派生必须沿用切分。音频只有整段文本标签，没有可靠逐词/音素结束时刻，不应把音频末尾直接当作唤醒词结束帧。

长期消费使用 `python3 -m tools.codex_assets export`，必须指定已固定的完整数据 commit、catalog SHA-256 和数据集 ID。只在干净 checkout 导出；保留原生 split/审核类型，不发明 token 或词尾时刻。详见[资产架构与固定消费](docs/ASSET_ARCHITECTURE.md)。

## 内容入口

- [收集和派生流程](docs/COLLECTION.md)：现成数据、真实录音和合成生成的记录要求。
- [筛选与版本规则](docs/CURATION.md)：人工/ASR 证据、近邻负例、连续性、切分与不可变版本。
- [来源登记与本任务结果](docs/SOURCES.md)：Qwen3、Spark、Melo、Kokoro、CosyVoice、VoxCPM2、AISHELL3、HI-MIA 的历史研究状态。
- [Qwen3 生成配方](recipes/qwen3/README.md)：精确权重身份、环境、seed、生成参数与原始执行脚本。
- [权利与用途记录](docs/RIGHTS.md)：公开归档范围与上游模型许可来源。

首批数据小于 3 MiB，每条 WAV 小于 1 MiB，直接使用 Git 保存，clone 后可立即核验。后续大批量版本应使用 Git LFS 或不可变对象存储，并在 catalog 固定下载地址、大小和 SHA-256，确保取回的是音频本体。

上游实验代码：[kws-pipeline](https://github.com/jiying2007/kws-pipeline)，首批归档依据的代码提交为 `c4025ee2e686c0c6cfd1ba078e6d1dabc3f70ee8`。数据仓版本应按自己的 Git commit 固定，不使用可变化的 `main` 作为训练身份。
