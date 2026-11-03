# 有界开源语料 WAV 准备：仅本地，不运行模型

本工具 automation/prepare_open_reference_pilot.py 复用现有 open_reference_sources.py 和 bulk_ingest.py。它仅从已合法取得的官方语音包中选择少量原始 WAV，为现有 reference-candidate 入口生成规范 WAV 和转换收据，不负责下载原语料、执行语音克隆、训练或自动批准许可。

AISHELL-3 的 OpenSLR Apache-2.0 发布标识与 AISHELL 发布者开放数据页面的学术研究限制需要依据具体语料版本复核。THCHS-30 也强调学术用途。商业使用、可识别人物声音的克隆及公开再分发必须另行审核。代码核验哈希或官方许可文字并不会自动建立声源权利。

## 私有输入格式

selection.json 为 kws-private-open-corpus-wav-selection-v1，结构示例：

~~~json
{
  "schema": "kws-private-open-corpus-wav-selection-v1",
  "dataset_id": "aishell3-slr93",
  "dataset_release": "actual-official-release",
  "license_snapshot_sha256": "ACTUAL_64_CHARACTER_SHA256",
  "clips": [
    {
      "speaker_key": "SSB0005",
      "original_path": "train/wav/SSB0005/example.wav",
      "original_sha256": "ACTUAL_64_CHARACTER_SHA256"
    }
  ]
}
~~~

每条原始 WAV 必须是单声道 PCM16、1–15 秒、16/24/44.1/48 kHz，并且实际字节、相对路径和 SHA 符合 selection。总输入最多 128 条，每个原始 speaker 只选最接近 5 秒的 1 条，输出最多 16 speaker（首批建议 8 speaker）。原始目录和文件拒绝软链接/路径穿越；原文件不被修改。许可证快照按真实 SHA 固定，不得使用示例占位符。

## 只读预览

~~~sh
python3 -B automation/prepare_open_reference_pilot.py \
  --selection /restricted/corpus/selection.json \
  --original-root /restricted/corpus/original \
  --license-snapshot /restricted/corpus/publisher-terms.txt \
  --output-root /restricted/corpus/prepared \
  --max-speakers 8
~~~

预览不写任何文件，不调用模型。已经是 16/24k 的 WAV 会重写为简单 44 字节头的 PCM16 WAV；44.1/48k 输入必须通过本地已 SHA 锁定的 ffmpeg 转成 16k PCM16。执行前验证 ffmpeg 文件哈希，失败则不创建输出：

~~~sh
converter="$(readlink -f "$(command -v ffmpeg)")"
converter_sha="$(sha256sum "$converter" | cut -d ' ' -f 1)"
python3 -B automation/prepare_open_reference_pilot.py \
  --selection /restricted/corpus/selection.json \
  --original-root /restricted/corpus/original \
  --license-snapshot /restricted/corpus/publisher-terms.txt \
  --output-root /restricted/corpus/prepared \
  --max-speakers 8 --ffmpeg "$converter" --ffmpeg-sha256 "$converter_sha" \
  --execute
~~~

私有目录仅创建一次（0700），包含 source-index.json、normalized/*.wav、conversions/*.json；索引及转换收据为 0600。所有输出只代表未许可/未训练的 candidate。转换执行有 30 秒硬超时、无 shell、不传递任何用户 token；失败可能留下不完整目录，不允许自动覆盖、假定成功或重放未知状态。

继续复用原生 open_reference_sources.py 做源 WAV/规范 WAV/转换收据二次验证。只有在独立批准具体来源/说话人/克隆商业范围后，才能使用已有 bulk_ingest.py 的真实私有 permit、rights evidence、HMAC 去标识及后续 KWS 数据审查。公开数据许可证并不自动代表商用声音克隆许可。

本工具不会创建真实 TTS、ASR、CTC 训练数据、human gold、FRR/FAR 或 SSC305 生产资格。即使预检 PASS，也必须在私有实际模型和人工/机器审核证据上继续完成训练准入。

## 测试

automation/test_prepare_open_reference_pilot.py 只生成人工 PCM WAV，用本地 ffmpeg 验证 44.1k 到 16k 的真实规范化、5 秒原始 speaker 选择、转换收据、私有权限、哈希漂移、跨 speaker 原音复用、拒绝非商业克隆候选和失败闭合；正常/-O/-OO 三模式均不调用真实 TTS、ASR 或下载公开音频。
