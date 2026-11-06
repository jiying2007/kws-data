# 可追溯声学素材子集

`acoustic_assets.py` 从固定 OpenSLR SLR28 ZIP 中按 HTTP byte range 取得明确指定的原始成员。该输入收据沿用研究来源登记；不建立第二个正式 catalog。

请求固定官方 URL、归档字节数、强 ETag、成员名、来源组、用途、下载字节与时间预算。服务端忽略 Range、返回不同 ETag、越界或截断均终止，保留已取得证据。完整成员通过 ZIP CRC、WAV 结构与每文件 SHA-256 检查；不宣称整份归档 CRC 已验证。

同一房间或来源组不能分到 train 与 development。保留双声道 RIR 原音，后续选择通道必须显式记录。公开素材中 RIR 的房间名称不等于项目实测距离或 RT60。source-marked noise 不自动等于没有目标词的人工负例，accepted_natural_negative_hours 固定为 0。

执行：

```sh
python3 research/auto-tts-voice-clone-20261102/acoustic_assets.py collect --request REQUEST.json --output NEW_DIRECTORY
python3 research/auto-tts-voice-clone-20261102/acoustic_assets.py verify --root NEW_DIRECTORY
```

2026-11-06 实际子集：4 个 AIR 房间响应原始双声道文件、4 条 Freesound 噪声、5 个原始说明/许可/列表文件。噪声均分配 train；booth、office 为 train，meeting、lecture 为 development。噪声总时长 63.90475 秒，不能计为自然 FAR 分母。采集 42 次精确 HTTP 206，共 9,906,884 字节；99 个清单文件实际回验通过。原始素材留在私有执行证据中，仓库只保存代码和匿名统计。

公式、音频变换与派生标签管理位于 kws-pipeline 的 `training/research_acoustic_augment.py`。衍生波形继承已有来源身份；数字增益不新增真人轻声证据，卷积不新增现场远场证据。
