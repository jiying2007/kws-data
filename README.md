# kws-data：中文唤醒词研究数据

## 当前入口（2026-10-07 整理）

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
