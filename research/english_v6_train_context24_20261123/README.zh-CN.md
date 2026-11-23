# 英语自然朗读训练来源复核

本批从 LibriSpeech 的 train-clean-100 分区提取了 12 位朗读者的 24 条原始录音，共 122.51 秒。每段保留带来源标识的原始 FLAC 及完整 16 kHz 单声道 PCM16 WAV。ASR 标签与模型触发都用于准备人工核听。训练授权和独立评测资格均未签发。

| 步骤 | 实际结果 |
| --- | --- |
| 锁定来源 | 先检查说话人编号、正式转写和音频在分片中的行号。排除了 976 个已知编号。首个分片有 18 位读者至少各有两条合格文本。从中按预登记的哈希顺序选 12 人、每人 2 条。 |
| 音频取证 | 从已锁定的 469,629,285 字节 Parquet 分片提取 24 个 FLAC。再次用 FFmpeg 解码，与每个封存 WAV 的 PCM 字节逐一相同。 |
| 三方盲转写 | Qwen、Whisper 和 Parakeet 各完成 24 次整条转写。10 条获得三方逐词一致，6 条获得两方逐词一致，8 条因一致票数不足保留为未知。 |
| 送交诊断 | 16 条，涉及 11 位读者，共 82.34 秒。8 条未知录音保留原始来源与转写，未送入四模型诊断。 |

完整上游版本为 `openslr/librispeech_asr@71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`，分片为 `clean/train.100/0000.parquet`。Kokoro 合成音色的真实说话人身份未知。源名单只能核验已知说话人编号的互斥关系。

来源使用 CC BY 4.0。署名为 V. Panayotov、G. Chen、D. Povey 和 S. Khudanpur，*LibriSpeech: An ASR Corpus Based on Public Domain Audio Books*, ICASSP 2015，OpenSLR 12。工作包保留 HF 托管分片中的原始 FLAC，另转封装成等值 PCM16 WAV。供核听的片段按原始采样号截取。上游资料见 [OpenSLR 12](https://www.openslr.org/12/) 和 [OpenSLR 的 HF 数据集](https://huggingface.co/datasets/openslr/librispeech_asr)。

`SOURCE_RUNS.json` 记录注册、音频和三路原生观察包的公共 Actions 编号、大小及 SHA-256。`MACHINE_SCREEN.json` 保存每条完整原始转写、一致票数和来源状态。`SOURCE_MANIFEST.json` 记录从上游标识到 FLAC、WAV 和原始 PCM 的哈希。将对应 ZIP 按 `roster.zip`、`source.zip`、`screen.zip`、`qwen.zip`、`whisper.zip`、`parakeet.zip` 放入同一输入目录，可独立复算以下证据。

```sh
python verify_source.py INPUTS SOURCE_RUNS.json source-readback.json
```

冻结的转写与评分窗口方案、原始音频、逐个标注状态及使用边界分别保留。现场 FAR、3–5 米真人 FRR 和物理设备时延需要相应的真实现场采集。

## 四个模型在原始音量下的结果

| 模型 | 完整事件 | 有事件的原句 | 有事件的读者 |
| --- | ---: | ---: | ---: |
| B0 | 4 | 4 | 4 |
| B9 | 26 | 14 | 10 |
| B13 | 6 | 5 | 5 |
| B14 | 11 | 6 | 6 |

四个模型共运行 64 次，保存 32,760 帧完整输出。在 5 位读者的 5 条原句上，B14 分别比 B13 多触发一次。总共 14 条原句、10 位读者产生至少一个事件。以所有模型的触发区域合并成 23 个可听窗口，22 个窗口有可用的相邻无触发对照片段，其中一段对照由多个窗口共享。全部 16 条送评原句、模型输出、原始采样时钟、听审页面和音频保存在同一复核包中。包内标注初始为空，使用者可保存或导出自己的核听结果。

[四模型评分与离线浏览器测试](https://github.com/jiying2007/kws-data/actions/runs/45114288521) · [数据取得与盲转写](https://github.com/jiying2007/kws-data/actions/runs/45113854481) · [元数据注册](https://github.com/jiying2007/kws-data/actions/runs/45113585559)
