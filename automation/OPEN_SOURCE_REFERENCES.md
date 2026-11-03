# 公开语音数据集 → 私有 KWS 参考声源池

这是 source-first 的本地候选目录桥接，绝不因为数据集标注 Apache/CC0/CC BY，就自动宣布原说话人的声音克隆、商用训练、声音肖像权或标签准确性获得授权。

公开候选表：[open_reference_datasets.v1.json](open_reference_datasets.v1.json)。其中包含官方来源 URL、发布许可声明、语言、版本风险和用途。这不是独立许可证明或已获得的语音文件。

## 来源与当前判断（2026-11-03）

| Dataset | Language | Release licence claim | Clone reference lane | Reason |
|---|---|---|---|---|
| [AISHELL-3 / OpenSLR 93](https://www.openslr.org/93/) | Mandarin | OpenSLR: Apache 2.0；[发布方](https://www.aishelltech.com/aishell_3)页面还标注 CC BY-NC-SA 4.0 | 候选，独立权利复核后 | 为多说话人 TTS 设计、218 人、85 小时；原始 44.1k PCM 需转 16/24k；商用/声音克隆权利有必要核实 |
| [THCHS-30 / OpenSLR 18](https://www.openslr.org/18/) | Mandarin | Apache 2.0 | 候选，独立复核后 | 普通话 ASR 语音；真人声音克隆许可不能从 Apache 直接推断 |
| [LibriTTS / OpenSLR 60](https://www.openslr.org/60/) | English | CC BY 4.0 | 候选，独立复核后 | 有声书叙述者、署名和声音使用权需独立审查 |
| [Hi-Fi TTS / OpenSLR 109](https://www.openslr.org/109/) | English | CC BY 4.0 | 候选，独立复核后 | 约 292 小时、10 名说话人；需确认叙述者声音使用权 |
| [VCTK / CSTR](https://www.cstr.ed.ac.uk/downloads/) | English | 具体发布版本需核实 | 候选，独立复核后 | 109 位说话人，Voice Clone Toolkit 学术语境不代表商用声线复刻默认获批 |
| [WenetSpeech](https://wenet-e2e.github.io/WenetSpeech/) | Mandarin | 发布方限定 non-commercial，且不持有部分原始媒体版权 | 禁止克隆候选路径 | 仅另行审核后用于允许的 ASR/研究负例 |
| [Common Voice / MDC](https://mozilladatacollective.com/terms/consumers) | Multi | 版本及本地化条款可能不同 | 禁止克隆候选路径 | 不从 CC0 自动推断声音克隆权；需检查实际条款 |

默认不自动下载完整语料（AISHELL-3 原始完整包很大），也不接受未许可爬取/第三方转载替代官方条款。

## 唯一执行链的接入

已经合并的原生 bulk_ingest.py 支持本地 catalog、独立 permit + rights-evidence、HMAC 去标识和每 speaker 只选一条约 5s 参考。新增 [open_reference_sources.py](open_reference_sources.py) 仅作为该 importer 的前置候选导出，绝不替代它，不生成 permit。

在私有机器从官方许可渠道取得语料，并准备已经提取的原始文件和转换好的单声道 16/24k PCM16 WAV。原始 44.1k WAV 需在本地记录工具版本/参数后转换；这里仅核对原始及输出 SHA 和声明的转换收据，并不声称可复现或验证转码过程。

索引示例（实际运行必须将占位符改为真实值）：

~~~json
{
  "schema": "kws-local-open-reference-index-v1",
  "dataset_id": "aishell3-slr93",
  "dataset_release": "SLR93-official-acquired-release",
  "license_snapshot_sha256": "REAL_64_HEX",
  "rows": [
    {
      "speaker_key": "dataset-native-speaker-code",
      "original_path": "raw/SSB0005/clip.wav",
      "original_sha256": "REAL_64_HEX",
      "normalized_path": "normalized/SSB0005/clip.wav",
      "normalized_sha256": "REAL_64_HEX",
      "conversion_receipt_path": "conversion/SSB0005/clip.json"
    }
  ]
}
~~~

转换收据必须与本地原文件/规范 WAV 的 SHA 匹配：

~~~json
{
  "schema": "kws-local-reference-conversion-receipt-v1",
  "original_sha256": "REAL_64_HEX",
  "normalized_sha256": "REAL_64_HEX",
  "conversion": "externally-documented",
  "conversion_recipe_sha256": "REAL_64_HEX"
}
~~~

未经重编码使用 conversion=identity、conversion_recipe_sha256=null，原始和规范 WAV 文件哈希必须相同。外部转换收据自声明不等于执行核验。

### 第一步：仅导出私有候选 catalog

~~~sh
python3 -B automation/open_reference_sources.py \
  --index /restricted/open-voices/local-index.json \
  --original-root /restricted/open-voices/official-original \
  --normalized-root /restricted/open-voices/normalized \
  --conversion-evidence-root /restricted/open-voices/conversion-receipts \
  --license-snapshot /restricted/open-voices/saved-publisher-terms.txt \
  --max-speakers 8 --output /restricted/open-voices/candidates.json
~~~

默认 dry-run 不写文件。加 --execute 后仅创建一次 0600 私有候选 JSON，不下载、不克隆、不训练、不覆盖或创建授权。选取每原始 speaker 最接近 5 秒的一条，拒绝跨 speaker 重复原始音频/PCM、哈希漂移、符号链接、未批准的来源候选。真实语料转录默认 unknown/null，不会把公开转写自动认作“已验证”的 Qwen 提示文本。

### 第二步：真实权利验证后才可调用已有 bulk_ingest

独立审核者必须确认**具体来源版本、每个参考说话人、允许商用合成/声线克隆用途及肖像权/授权**。在私有存储建立原生 kws-external-voice-clone-permit-v1 和独立 rights-evidence（含来源条款及证据哈希），禁止伪造 independently-reviewed 状态。

~~~sh
python3 -B research/auto-tts-voice-clone-20261102/bulk_ingest.py \
  --catalog /restricted/open-voices/candidates.json \
  --permit /restricted/open-voices/actual-verified-permit.json \
  --rights-evidence /restricted/open-voices/actual-rights-evidence.json \
  --audio-root /restricted/open-voices/normalized \
  --pseudonym-key-file /restricted/open-voices/private-hmac-key.bin \
  --max-refs 8 --output /restricted/open-voices/reviewed-source-pool.json --execute
~~~

创建 source pool 后，继续复用原 bulk_pool.py → tools/kws_pilot_preflight.py → tools/supervise_bulk_autopilot.py → 三 ASR 原生审计 → kws-pipeline 训练准入。source pool、hash 一致或三 ASR 一致均不自动赋予 human gold、真人 final AFE/SSC305 或 shipping 权威。

严禁公开原语料、真实姓名与 speaker 对应表、HMAC key、私有同意/许可收据、私有音频、输出 WAV 和权重。

## CI 验证

automation/test_open_reference_sources.py 使用人工 PCM16 WAV 和伪许可快照验证：每 speaker 选源、转换字节绑定、来源漂移、跨说话人原始/PCM 重用、拒绝 WenetSpeech/Common Voice 克隆路径、不把公开许可当 cloned voice consent、私有只创建一次且权限 0600。普通 / -O / -OO 全部无网络、无模型推理、不产生真人声音。
