# 英文 10 dB 实验：源获取恢复与首次数值执行

第一轮 [run 44019143948](https://github.com/jiying2007/kws-data/actions/runs/44019143948) 已以失败终态消费。其 source commit 是 `b1b70a8a0ab08cdde87868788e8a18425c7e5cca`，artifact `14300895359` 共 6,346 字节，SHA-256 为 `43ad533ed64eb25421fd24478c69e1cb26f5412f864229f902c0641b498b47e3`。

该运行在原 `www.openslr.org` 地址发生 TLS 连接超时，curl 返回 28；总执行约 32.006 秒。完整 14 成员证据包保留原失败、计划、命令退出和错误输出。包中没有 `model-worker` stdout/stderr/exit，没有 `WORKER_STARTED` 或 `PROGRESS`，失败回执的 `partial_progress` 为 null。与已固定 SHA 的原代码顺序结合，可以确认模型工作进程从未启动：原程序只有在下载命令成功、归档字节数和 SHA 都通过后才调用数值 worker。

**原失败保持 FAILED_NO_RETRY；本次记录 source acquisition attempt 2、numeric attempt 1。** 这是一个新的、有上述完整证据约束的源获取 continuation。没有重跑原 once，没有重训原 96 步实验，也没有把已执行或状态不明的数值阶段重新开放。

## 源恢复边界

新 continuation 先从公开 GitHub 读取并核对第一轮 run 的 failure/attempt/head 和 artifact 身份，下载该固定小 ZIP，验证完整 14 成员闭集和零模型执行证据。任何多出或缺少的成员、出现 worker 日志或非空进度，都会拒绝继续。子进程的 `--worker` 和 `--prepare-noise` 入口也会在导入依赖前重新验证失败 ZIP 与已保存 gate。

随后仅请求一次 [OpenSLR 官网列出的 ELDA 镜像](https://openslr.org/)：`https://openslr.elda.org/resources/12/dev-clean.tar.gz`。没有自动 fallback 或重试。所需归档仍必须是原始 **337,926,286 字节**，SHA-256 **`76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3`**。镜像仅改变传输地址，不改变语料。

## 未改变的数值实验

原 worker、plan、workflow 三个文件保持字节不变。新代码自动核验它们的原 SHA，并检查 `worker` 全文以及混音、来源选择、事件统计、候选触发等十个函数逐文本一致。包括 10 dB、77 个新条件、4,000 次 embedding 上限、训练侧触发、固定 96 次更新、开发集隔离、原归一化、0.8/3/13 事件规则、900 秒总监督期限在内的研究参数全部继承。

原失败没有消耗新混音特征或候选更新，所以本次只允许原来尚未启动的这组数值工作。真实质量结论仍须等待保存的逐帧预测、缓存、训练回执和逐条对照；完成源恢复本身不代表效果改善。

## 40 条历史背景 PCM 的持久保存

源恢复后，独立 source-only 子进程读取已验证归档，按照原训练 artifact 的 FLAC 身份、PCM 身份和片段清单，还原原 40 条 16 秒背景轨。每条 512,000 字节，共 **20,480,000 字节**；全部逐轨 SHA 必须与历史训练相同。该过程只做音频解码和字节拼接，不运行特征模型、ASR、TTS 或优化器。

原数值 worker 全文不改，依旧按原逻辑校验输入。新增保存的 `background_tracks/` 及清单随实际 artifact 保留，可直接用于后续噪声来源、混音算术的独立审计，也能避免未来再依赖完整 337 MB 下载。它们是原有已暴露输入，不增加独立样本、说话人或 FAR 小时。

语料来源为 LibriSpeech，作者 Vassil Panayotov、Guoguo Chen、Daniel Povey 和 Sanjeev Khudanpur，源有声读物来自 LibriVox，许可证 CC BY 4.0；具体原文件、说话人与片段偏移保存在清单中。

正式发音准入、人工金标、独立 FAR/FRR、词尾延迟、模型晋级和 SSC305 资格仍保持未通过。任何新失败也保留原始退出与部分台账，不自动重新执行。
