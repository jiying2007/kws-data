# KWS 技术研究：唯一声源导入与用途范围

本仓围绕中文“你好小窝”“小窝小窝”和英文研究用唤醒词开展
**离线技术研究**，优化独立测试语音上的 FRR、连续 FAR、实时性与
SSC305 计算/内存开销。研究数据与最终现场验证分开，不把合成音频、
三 ASR 一致或 Github Actions PASS 当作真实 KWS 效果结论。

当前唯一声源导入入口是 `automation/research_only_bulk_ingest.py`，
使用 `kws-dataset-research-tts-scope-v2`（不接受任何历史版 scope 回退）。
严守具体数据集/发行版适用条款，完整保留原始和转换 WAV 的 SHA、
独立数据研究用途审核、说话人谱系、HMAC 私有匿名化以及创建一次的
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
研究导入器根据以下 scope 建立匿名 `kws-bulk-clone-sources-v1`。

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

~~~sh
python3 -B automation/research_only_bulk_ingest.py \
  --index /restricted/corpus/prepared/source-index.json \
  --original-root /restricted/corpus/official-original \
  --audio-root /restricted/corpus/prepared/normalized \
  --conversion-root /restricted/corpus/prepared/conversions \
  --license-snapshot /restricted/corpus/publisher-terms.txt \
  --research-scope /restricted/corpus/research-tts-scope.json \
  --research-review-evidence /restricted/corpus/research-terms-review.txt \
  --pseudonym-key-file /restricted/corpus/private-hmac-key.bin \
  --max-refs 8 --output /restricted/corpus/research-source-pool.json
~~~

默认只校验，追加 `--execute` 才创建一次私有 0600 manifest。该 manifest
不代表任何训练样本已经通过真实语音和标签审查。
研究运行链：`bulk_pool.py` → `kws_pilot_preflight.py` →
`supervise_bulk_autopilot.py` → 原生三 ASR/声学复核 →
`kws-pipeline` 受控配对训练 → 独立语音和板端效果测量。

## 质量和实验边界

继续执行真实语音权限审核、PCM/说话人/派生家族隔离、文本盲筛和
错误音频隔离、长流 FAR/FRR 单侧置信上界、触发词尾延迟与目标板
CPU/RSS/功耗测试。禁止伪造数据效果、私有文件公开或以纯合成 CI
给出新模型有效的结论。

CI 只使用人工 PCM WAV 和虚构授权材料，运行 normal/-O/-OO 的
离线契约检查；真实语料/模型仍需私有环境另行执行。
