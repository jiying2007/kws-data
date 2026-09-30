# 来源与研究状态登记

以下是 **2026-09-28 的本项目历史观测**，不是最新模型排行榜，也不能推导某生成器所有音频的发音真值。完整研究记录固定在 [kws-pipeline 的归档提交](https://github.com/jiying2007/kws-pipeline/tree/c4025ee2e686c0c6cfd1ba078e6d1dabc3f70ee8/docs/research)。

| 来源 | 本任务观测 | 本仓状态 |
| --- | --- | --- |
| [Qwen3-TTS 0.6B CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | 用户认为听感最佳；固定 20 条人工确认；后续批次按 ASR/连续性逐条筛选 | 首批归档 42 条、五种中文预置声线 |
| [Spark-TTS](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) | 来源级反馈为可用、逊于 Qwen3；基础/语速探针合计精确 ASR 接受 9/18 | 方法和结果登记，音频未纳入首批公开包 |
| [MeloTTS](https://github.com/myshell-ai/MeloTTS) | 用户认为尚可，发音好于 Kokoro；单条准入不能从来源评价推导 | 方法登记，待独立批次审核 |
| [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) | “窝”容易听成“沃” | 不按目标正例发布；可另建疑难发音探针 |
| [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | SFT 的“你好小窝”出现末字错读；逗号提示读准但停顿明显 | 失败现象登记，逗号版不作连续目标词 |
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | 六条探针仅 1/6 精确 ASR；实听与 ASR 疑点相近 | 暂停扩量，待不同条件重新验证 |
| [AISHELL3](https://www.aishelltech.com/aishell_3) 相关历史 TTS 训练池 | 128 条来源级“基本没问题”；Zipformer 仅 14/128 精确，第一词 0/16；另一 ASR 同样第一词 0/16 | 不把整批标成逐条人工合格，不上传原来源数据 |
| [HI-MIA](https://www.openslr.org/85/) / HI-MIA-CW | 真人近邻负例用于已观察开发回读，需按说话人隔离 | 登记方法；从原始来源核许可、版本和标注后再建立独立包 |

Qwen3 固定 revision 为 `85e237c12c027371202489a0ec509ded67b5e4b5`，权重哈希见每个 manifest。Spark 的历史官方代码 commit 为 `2f1ea9082400547242641f5271b6f941c9f439d1`，模型 revision 为 `642071559bfc6346c2359d19dcb6be3f9dd8a05d`；该权重许可记录为 CC BY-NC-SA 4.0，需保留研究用途边界。其他来源的 exact 资产身份见对应上游研究文档，未取得完整身份/收据的来源不能成为可训练数据版本。

不同速度、音色、风格和年龄描述是受控生成维度，不能自动增加真实说话人数。扩展其他家族前先做双词与缺首/重复近邻的小样本，再抽听和 ASR，确认能完整读出“窝”后才扩量。
