# 非商业 KWS 技术研究声源路径（独立于量产发布）

当前 kws-data / kws-pipeline 的工作是 **noncommercial research**：追求不逊于商用产品的唤醒率、FAR/FRR、资源/时延和鲁棒性，但不等于商业使用或分发。研究用途允许使用相应研究许可的语料及非商用权重；将来商业发布必须另行审核。既有独立真人 final-AFE、SSC305 和 production 门禁均不降低。

## 研究数据与模型候选

| 来源 | 用途 | 当前条款判断 |
| --- | --- | --- |
| [AISHELL-3 / SLR93](https://www.openslr.org/93/) | 中文多说话人 TTS 研究、8 声源参考、小样本测试 | 218 人/约 85 小时，OpenSLR 标 Apache-2.0，出版方另示 CC BY-NC-SA；按具体已获取版本的研究条款执行，不要求商用授权，但不得把研究输出当商业声线 |
| [THCHS-30 / SLR18](https://www.openslr.org/18/) | 中文 ASR、普通语音/近音负例、声学研究 | OpenSLR Apache-2.0 且介绍学术用户免费；使用具体版本范围核对；从 ASR 语料资格不能直接推断可识别个人声音克隆 |
| [LibriTTS / SLR60](https://www.openslr.org/60/) | 英文多说话人语音和 TTS 研究 | CC BY 4.0 署名保留；研究参考还需核对下载版本及声音使用说明 |
| [Hi-Fi TTS / SLR109](https://www.openslr.org/109/) | 英文 TTS 发音和声源研究 | CC BY 4.0 等实际发布范围须保留来源 |
| [VCTK](https://www.cstr.ed.ac.uk/downloads/) | 英文口音与多音色研究 | 依实际发布版本的研究条款 |
| [WenetSpeech](https://wenet-e2e.github.io/WenetSpeech/) | 中文多场景 ASR、长流负例和研究音频 | 发布方允许非商业用途，但原始媒体版权归第三方。保留研究用途，不自动作为真人克隆来源 |
| [Common Voice / MDC](https://mozilladatacollective.com/) | 多语言 ASR 和真实语音鲁棒性 | 遵守 MDC 版次/使用限制、不可镜像及个人去识别限制；不自动进入真人克隆路线 |
| [F5-TTS](https://github.com/SWivid/F5-TTS) | 中英文非商用 TTS 研究候选 | 代码 MIT、预训练权重 CC BY-NC，研究可用；尚未实现/运行本仓新的 F5 adapter，不声称已生成音频 |

Qwen3-TTS CustomVoice/VoiceDesign 可以并行生成原创研究声线，保留 parent/generator-family 谱系，不以大量 clone 片段冒充独立真实发音人或封存测试。

## 新增研究专用导入器

现有源准备链：automation/prepare_open_reference_pilot.py → automation/open_reference_sources.py，实际检验原始 WAV、规范 16/24k WAV、转换收据 SHA、相同 speaker 原始声音去重。

**automation/research_only_bulk_ingest.py 是唯一声源导入器。** 它直接实现源 catalog 校验、HMAC 匿名 speaker/source family、私有原子写入及 bulk_pool 源清单，要求具体发布版本已审阅的非商用 TTS 研究 scope。平行商用导入器和兼容层已从当前源码树移除；历史实现仅保留在 Git 提交中供审计。研究入口不伪造标签或后续部署批准。

私有 research scope 示例（必须替换真实 HASH/版本及真实审查记录）：

~~~json
{
  "schema": "kws-dataset-research-tts-scope-v1",
  "dataset_id": "aishell3-slr93",
  "dataset_release": "actual-reviewed-official-release",
  "source_type": "reviewed-open-audio",
  "source_license": "Apache-2.0",
  "source_terms_sha256": "ACTUAL_64_HEX_SHA256",
  "research_review_sha256": "ACTUAL_64_HEX_SHA256",
  "intended_use": "noncommercial-offline-kws-tts-research",
  "research_tts_references_permitted": true,
  "commercial_use_permitted": false,
  "public_voice_export_permitted": false,
  "forbidden_uses": [],
  "review_status": "independently-reviewed"
}
~~~

许可条款与研究审阅材料须分别留存于私有机器，SHA 与上述值一致。此记录按数据集/版本审阅一次，不要求额外取得商用许可或逐条素材申办商用权限；但不可无依据填写许可允许克隆或虚构“independently-reviewed”。模型/数据研究权限和真人可识别声音公开传播权限是两回事。

运行，默认只读不写：

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

审查通过可添加 --execute 只创建一次私有 mode 0600 source manifest 和原生 rights sidecars；训练准入、产品资格标志仍为 false。下一步复用原 bulk_pool.py → tools/kws_pilot_preflight.py → tools/supervise_bulk_autopilot.py → 真实三 ASR/效果筛查 → 受控研究训练 A/B。不生成额外 runner，不下载整套 19GB AISHELL-3 到 GitHub Runner，不上传真实录音、声源身份关联表、模型权重或许可文档。

**注意**：本人提供的 JSON 声称 research_tts_references_permitted=true 是用途审核记录，不是密码学的法律授权证明。若语料具体版本禁止克隆，就不能借此记录绕过禁令。商业目的仍须重新审核相应数据/权重和实际目标板资格。

## CI

只用人工 PCM、虚构研究材料和无模型回归验证研究源准入唯一入口与商用断言不能通过、source-family/PCM/源字节身份、独占写入、FAR/FRR/量产资格未升级；正常/-O/-OO 模式不得下载真实数据或调用 TTS/ASR。
