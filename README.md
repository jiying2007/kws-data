# kws-data：中文唤醒词研究数据

本仓整理“小窝小窝”和“你好小窝”的语料、TTS 生成方法、收集方法与筛选证据。首批为 **42 条 Qwen3 合成开发音频**，包含人工确认与机器筛选两种证据。所有录音均为 16 kHz、单声道、PCM16 WAV；每条记录文件/PCM SHA-256、文本、正负标签、声线、seed 与源 ID。

| 数据集 | 条数 | 准入证据 | 原有用途 |
| --- | ---: | --- | --- |
| `datasets/qwen3-reviewed-v1` | 20 | 对固定审听包的逐条人工确认；收据绑定 WAV 哈希 | 12 训练、4 开发 A、4 开发 B |
| `datasets/qwen3-train-asr-v1` | 18 | 精确 ASR 文本匹配；连续正例内部低能量间隔小于 120 ms | 扩展训练 |
| `datasets/qwen3-holdout-asr-v1` | 4 | 精确 ASR 文本匹配；连续性筛选 | 已观察的开发回读 |

这些样本属于研究开发数据。它们均已被观察过，不是封存盲测集；五个预置 TTS 声线不是五名真实受试者。人工接受和 ASR 自动接受分别记录；“优质”在本仓表示达到对应批次的准入规则，不表示已覆盖真人、真实房间或目标设备。

## 使用

```bash
git clone git@github.com:jiying2007/kws-data.git
cd kws-data
python3 -m tools.codex_assets verify
```

`catalog.json` 固定各批 manifest、收据和切分文件的 SHA-256。核验工具检查全部音频格式、文件/PCM 哈希、标签与审核覆盖、重复 PCM、跨批声线切分一致性及生成脚本身份。工具没有第三方依赖，Python 3.8+ 可运行；TTS 推理环境另见生成配方。

训练加载时读取 `manifest.json` 的 `recordings`，将 `path` 相对于该数据集目录解析。`splits.json` 固定开发切分：Vivian、Uncle_Fu、Dylan 为训练声线；Serena 为开发 A；Eric 为开发 B。同一声线的新增 seed 或增强派生必须沿用切分。音频只有整段文本标签，没有可靠逐词/音素结束时刻，不应把音频末尾直接当作唤醒词结束帧。

## 内容入口

- [收集和派生流程](docs/COLLECTION.md)：现成数据、真实录音和合成生成的记录要求。
- [筛选与版本规则](docs/CURATION.md)：人工/ASR 证据、近邻负例、连续性、切分与不可变版本。
- [来源登记与本任务结果](docs/SOURCES.md)：Qwen3、Spark、Melo、Kokoro、CosyVoice、VoxCPM2、AISHELL3、HI-MIA 的历史研究状态。
- [Qwen3 生成配方](recipes/qwen3/README.md)：精确权重身份、环境、seed、生成参数与原始执行脚本。
- [权利与用途记录](docs/RIGHTS.md)：公开归档范围与上游模型许可来源。

首批数据小于 3 MiB，每条 WAV 小于 1 MiB，直接使用 Git 保存，clone 后可立即核验。后续大批量版本应使用 Git LFS 或不可变对象存储，并在 catalog 固定下载地址、大小和 SHA-256，确保取回的是音频本体。

上游实验代码：[kws-pipeline](https://github.com/jiying2007/kws-pipeline)，首批归档依据的代码提交为 `c4025ee2e686c0c6cfd1ba078e6d1dabc3f70ee8`。数据仓版本应按自己的 Git commit 固定，不使用可变化的 `main` 作为训练身份。
