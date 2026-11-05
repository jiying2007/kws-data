# 公开语音数据集 → KWS 非商业研究参考声源

当前唯一导入器是 automation/research_only_bulk_ingest.py；源目录准备工具
open_reference_sources.py 只产生研究候选 catalog，不允许单凭下载许可
自动宣称可克隆具体真人，也不生成训练或发布资格。

## 数据集与用途

- AISHELL-3、THCHS-30：中文研究优先候选；保留具体数据集版本、
  官方条款、源 WAV / 规范 WAV SHA 与独立研究 TTS 用途审核。
- LibriTTS、Hi-Fi TTS、VCTK：英文研究候选，遵守特定发行版条款和署名。
- WenetSpeech、Common Voice：可按版本条款进行 ASR/负例研究，
  但不自动进入真人声线克隆参考链。

唯一执行链：prepare_open_reference_pilot.py 规范原始 WAV →
open_reference_sources.py 导出私有候选 →
research_only_bulk_ingest.py 验证研究 scope / 匿名化 →
bulk_pool.py / 有界 Qwen/三 ASR / 研究数据评估。

## 第一阶段：源文件和候选检查

保留原始 WAV、转码后的 16/24k mono PCM16、实际许可条款快照和
转换收据；所有路径必须在私有目录，文件 SHA 实际校验。

~~~sh
python3 -B automation/open_reference_sources.py \
  --index /restricted/open-voices/local-index.json \
  --original-root /restricted/open-voices/official-original \
  --normalized-root /restricted/open-voices/normalized \
  --conversion-evidence-root /restricted/open-voices/conversion-receipts \
  --license-snapshot /restricted/open-voices/saved-publisher-terms.txt \
  --max-speakers 8 --output /restricted/open-voices/candidates.json --execute
~~~

输出候选目录不含研究许可结论，绝不上传原声和说话人身份。
候选格式为 `kws-local-open-audio-catalog-v2`。选中参考的出版方转录通过
`--transcript-source`、`--transcript-sha256`、`--transcript-format` 保留；
有实际 audio-only 内容检查时可加 `--reference-content-checks`，详见
[转录与 ICL 规范](RESEARCH_ONLY_REFERENCE_SCOPE.md#保存出版方转录并准备-icl-参考)。
未找到转录的片段仍为 `unknown`，不会从生成请求补造参考文字。

## 第二阶段：唯一研究导入器

在特定官方数据集/版本的非商业 TTS 研究范围已被审阅后，准备
kws-dataset-research-tts-scope-v2 研究 scope 和独立研究审阅材料。
不要求商业授权，不允许伪造已审阅结论或违反版本条款。
先按 [研究导入规范](RESEARCH_ONLY_REFERENCE_SCOPE.md) 初始化或一次性
迁入私有 speaker registry，之后复用同一登记表。

~~~sh
python3 -B automation/research_only_bulk_ingest.py \
  --index /restricted/open-voices/local-index.json \
  --original-root /restricted/open-voices/official-original \
  --audio-root /restricted/open-voices/normalized \
  --conversion-root /restricted/open-voices/conversion-receipts \
  --license-snapshot /restricted/open-voices/saved-publisher-terms.txt \
  --research-scope /restricted/open-voices/research-scope.json \
  --research-review-evidence /restricted/open-voices/research-review.txt \
  --speaker-registry /restricted/open-voices/speakers.json \
  --max-refs 8 --output /restricted/open-voices/research-source-pool.json --execute
~~~

上述命令只生成私有匿名声源清单、不可变数据身份和审阅收据。来源划分
读取持久 registry，不再由可轮换的 HMAC 密钥决定。
历史来源版本仍可从 Git 追溯；后续训练、独立录音评估及
SSC305 目标板实测都必须具备自身真实证据，不会因为源导入成功而被自动标记通过。

## CI 验证

automation/test_open_reference_sources.py 使用人工 PCM16 WAV 和伪许可快照验证：每 speaker 选源、转换字节绑定、来源漂移、跨说话人原始/PCM 重用、拒绝 WenetSpeech/Common Voice 克隆路径、不把公开许可自动当作具体研究范围的证明、私有只创建一次且权限 0600。普通 / -O / -OO 全部无网络、无模型推理、不产生真人声音。
