# Hey Mero 训练原件上的双音素观测

对 44 条已准入的唤醒词原音和 13 条能使冻结 B9 在原始音量误唤的普通原音，重放了两个原始未提示观察器保存的 114 份 392 类帧级 logits。核对每条完整原音的 WAV 和 PCM、源组、观察器权重和词表的身份。使用原始声学输出在 16 kHz 下的 20 ms 卷积步长审查时间位置。模型使用双向上下文，帧索引不能证明物理发音边界。

| TRAIN 原音 | 原件 | 至少一套自由解码完整 | 两套自由解码完整，三个元音峰各相差 ≤40 ms |
|---|---:|---:|---:|
| 全部已准入正例 | 44 | 15 | 3 |
| 实测较快的正例 | 14 | 3 | 0 |
| B9 原幅命中、半幅漏检的较快正例 | 2 | 1 | 0 |
| B9 原幅误唤的普通语音 | 13 | 0 | 0 |

各子组可重叠。三条有一致时序支持的正例来自三个 Kokoro 常规读速声源。各自 HEY、MEH、ROH 元音峰在两观察器间分别相差 0/0/0、20/20/0、0/0/0 毫秒。两个困难快读声源没有双观察器一致的整词路径；13 条普通语音的自由解码也都没有出现目标连续音素子串。双模型完整路径蒸馏缺少该快读目标的训练监督。未启动优化，未读取 DEV/CAL/LOCKED 原始音频或 logits，也未做新推理、TTS 或听审。

DESIGN.json 记录固定的发音解释与 40 ms 诊断容差；制定前检查了两个快读原件的 XLSR 转写。RESULT.json 列出全部原件、自由路径、峰值帧和每组原生 logits 的哈希。OBSERVER_LOCK.json 保存 12 个原始公开观察器归档的 SHA、运行和工件编号。下载各文件至同一目录，将每个 ZIP 保留为其工件编号加 .zip；还须从 RESULT.json 的 input_receipts 和 source_archives 下载 2 个源诊断与 6 个源音频 ZIP。安装 NumPy 后在本目录运行：

```
python readback.py --sources /path/to/original_zips --output replay.json
```

原始报告来自已用于标签准入的两套机器观察器。该结论只评估原幅合成训练音频上是否存在明确监督，不衡量新模型、半音量、真人、生产 FAR/FRR 或声源独立泛化。输出禁止自动改写为人审帧标签。原始语料保留其项目授权边界，数据只按经核准的 TRAIN 原件与研究用途处理。观察器权重在各自官方模型卡中标记 Apache-2.0；Piper 原始许可回执登记了引擎 GPL-3.0 及 LibriTTS 数据 CC-BY-4.0，Kokoro 源方案登记 Apache-2.0。两类许可不能替代生产录音、合成声线或产品发布的授权。

[冻结原始训练正例](https://github.com/jiying2007/kws-data/actions/runs/45094053219) · [困难普通语音](https://github.com/jiying2007/kws-data/actions/runs/45094102574) · [历史音素准入规则](https://github.com/jiying2007/kws-data/blob/8709eca5be5256034a09541b5ccad8e137b4c7a9/automation/english_fresh24_ctc_admission.py)

[Meta XLSR 原始模型](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft/tree/2c733782da5604684829819a5eb744c193fe9398) · [Meta LV60 原始模型](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/tree/ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4)
