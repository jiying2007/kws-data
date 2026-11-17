# V4.1 旧 Qwen 与 snapshot160 资产处置审计

审计日期：2026-11-17。结论：**77 条仍只能作为已暴露的弱监督研究与开发诊断资产；本次可确认为可信正例、可信负例或正式训练准入的数量均为 0。** 这里的 0 表示缺少足够的准入证据，不是判定这些音频全部发音错误。没有新生成、ASR、音素推理、训练或远端修改。

## 1. 77 条到底来自哪里

本次审计复用了已保存的训练产物，并读取其中 `SOURCE_SNAPSHOT.zip` 的原始清单、音频和观察器回执。原始训练 ZIP 未纳入本目录，可直接从以下已记录来源回读： [run 43867764366](https://github.com/jiying2007/kws-data/actions/runs/43867764366)，[artifact 14225893574](https://github.com/jiying2007/kws-data/actions/runs/43867764366/artifacts/14225893574)。

来源分母为 terminal144 的 144 个请求，加 schwa16 的 16 个请求。原有 1 条生成失败、159 条原 WAV 可读；77 条既有选择由 **terminal144 的 74 条 + schwa16 的 3 条**构成。83 条未进入选择，其中 82 条原决策 UNKNOWN，1 条源音频不可用。没有把这些未入选条目改作负例。

| 原记录角色 | Mero 弱正意图 | Vosen 弱正意图 | 普通负意图 | 合计 |
|---|---:|---:|---:|---:|
| train | 10 | 8 | 32 | 50 |
| development | 8 | 3 | 16 | 27 |
| 合计 | 18 | 11 | 48 | 77 |

来源家族如下。每一条具体 cell ID、原始 WAV/PCM SHA、源 artifact/member、已有三路 ASR、音素输出和处置均写入 [ASSET_DISPOSITION.json](ASSET_DISPOSITION.json) 的 `selected_rows`。

| Qwen 预设声线 | snapshot 角色 | Mero | Vosen | 普通负意图 | 合计 | 旧 Qwen63 角色 |
|---|---|---:|---:|---:|---:|---|
| aiden | train | 3 | 5 | 8 | 16 | 无该批记录 |
| dylan | train | 1 | 0 | 8 | 9 | train-candidate |
| eric | train | 6 | 3 | 8 | 17 | train-candidate |
| ryan | development | 3 | 3 | 8 | 14 | 无该批记录 |
| serena | train | 0 | 0 | 8 | 8 | development-only |
| vivian | development | 5 | 0 | 8 | 13 | train-candidate |

terminal144 的源生成 [run 43816869386](https://github.com/jiying2007/kws-data/actions/runs/43816869386) 保持 INCOMPLETE_NO_RETRY，已有四路观察来自 [run 43822597212](https://github.com/jiying2007/kws-data/actions/runs/43822597212)；schwa16 的源与四路观察来自 [run 43865217824](https://github.com/jiying2007/kws-data/actions/runs/43865217824)。这批唤醒词请求的最终文字意图已经明确为 Hey Mero `/heɪ ˈmiːroʊ/`、Hey Vosen `/heɪ ˈvoʊzən/`，Hey 轻读。Mero 的合成文字使用 Meero 等形式；明确目标意图不等于已经核验实际发音。

## 2. 为什么机器支持仍不能直接变成可信标签

29 条唤醒词候选达到旧策略的机器核心支持；48 条普通负意图达到两路以上请求文字一致及未观察到目标核心。该策略明确 `length_stress_and_hey_acoustic_verification=false`。它接受 Mero 的 mero/miro/meero 拼写、音素 i/iː、r/ɹ 与 oʊ/oː/o 等候选写法，因此不是完整最终 IPA 判定器。所有原始 `pronunciation_verified`、`human_gold`、`formal_training_admitted`、`trusted_negative` 均为 false。

处置分为：原 train 50 条保留历史弱研究和软件诊断用途；原 development 27 条保留开发诊断用途；未入选 83 条保留 UNKNOWN/不可用原因。普通负意图 48 条、旧 near_miss 或 ASR 未识别目标的音频，都没有被重新授予可信负例标签。正式正例需要原音频的最终读音核验；正式负例需要充分证据确认音频确实不含目标。

snapshot 的音素观察器是 `facebook/wav2vec2-xlsr-53-espeak-cv-ft`，revision `2c733782da5604684829819a5eb744c193fe9398`，与 Piper v4 的 bookbot 观察器不同。保留各自原 token 和模型锁，不将两类观察器的输出表示直接等同。主要待确认项是 /iː/、/oʊ/、/v/、/zən/、Hey 轻读、姓名首音节重音和自然度。

## 3. 与旧 Qwen63 的关系及角色冲突

已将旧 Qwen63 的 63 个 WAV/PCM SHA 与 snapshot 的 159 个有效原例逐一比较：**相同 WAV 为 0，相同 PCM 为 0**。这是不同录音批次；共享的是 Dylan、Eric、Serena、Vivian 四个预设声线家族，覆盖当前 77 条中的 **47 条**。

同一 snapshot 内，train/development 的来源家族及 PCM 都没有交集。但跨批存在明确角色冲突：**Serena 在旧 Qwen63 是 development-only，snapshot 已把其 8 条普通负意图用于 train；Vivian 在旧 Qwen63 是 train-candidate，snapshot 的 13 条属于 development。** 旧 train-candidate 这个称谓本身不证明旧批曾执行训练；它确实记录了此前候选和诊断暴露。不得把当前 27 条开发集当作从未暴露的独立评估集，也不应直接继承当前 50 条清单作为新正式训练集。

原训练回执还确认：50 个合成 train ID 全部出现在已有优化器抽样中；27 个 development ID 在抽样中为 0。历史训练共 96 步，含 228 次合成样本抽样和 60 次朗读背景抽样。已有模型权重保留用于软件数值一致性与资源测量，模型晋级和资格验证仍未获准。

## 4. 旧 Qwen63 独立机器观察已经可直接复用

三份原始观察 ZIP 合计仅 32,053 B：来源关联三路 ASR [run 43267739514](https://github.com/jiying2007/kws-data/actions/runs/43267739514)、DTW [run 43334622072](https://github.com/jiying2007/kws-data/actions/runs/43334622072)、音素观察 [run 43340235981](https://github.com/jiying2007/kws-data/actions/runs/43340235981)。ZIP 与内层 JSON 均匹配既有脚本的固定 SHA。纯 JSON 复建结果与原发布 [run 43346200939](https://github.com/jiying2007/kws-data/actions/runs/43346200939) 的联合报告 SHA 完全相同：`89f20b7ac5a672840013b3cf780422dcf68451b6f5d618e8f494a0e5f5dc6b12`。

可复用内容为 63 行原波形身份、189 次 ASR、63 次音素观察、DTW 对比和 14 个唤醒词复核对。这里的独立是机器观察来源，不是人工 gold。三份原 JSON 均没有明确最终目标 IPA 合同，故旧 Qwen63 的目标 IPA 保持 UNKNOWN；其中 28 条 wake、21 条 near_miss、14 条 ordinary_negative 都只是历史生成意图。尤其 **Hey Miro 的 near_miss 意图不能凭文字变成最终 Mero /miːroʊ/ 的可靠负例**。

本目录实际收录 [63 行完整联合结果](qwen63_evidence/rebuilt/english63-three-evidence-review.json)、[WAV/PCM hash 索引](qwen63_evidence/hash-index.json) 和 [来源 artifact 清单](qwen63_evidence/artifact-inventory.json)。三份原始观察 ZIP 不随 Git 提交，可按清单中的 `api_archive_url` 或 `artifact_url` 回读，完整内外 SHA 均保留；`source_archive_filename` 仅记录来源压缩包名称。没有下载旧 Qwen63 的大型原音频包。

## 5. 四条原 Qwen 定点复核音频

从已经本地保存的 snapshot 提取原 WAV 字节，每词两条、Aiden/Eric 两个声线。Mero 两条均为 soft_audible，Vosen 两条均为 gentle，便于人工比较相同风格下的关键音段；选择依据仅用于安排复核。

| 最终目标 | 声线 | 秒数 | 用户复核 ZIP 内的文件名（不在 Git） |
|---|---|---:|---|
| Hey Mero | aiden | 0.8 | `qwen_hey_mero_aiden.wav` |
| Hey Mero | eric | 1.2 | `qwen_hey_mero_eric.wav` |
| Hey Vosen | aiden | 0.72 | `qwen_hey_vosen_aiden.wav` |
| Hey Vosen | eric | 1.28 | `qwen_hey_vosen_eric.wav` |

总计 **4.000 秒**，没有裁切、变速、拼接或补音频。[原始复核 manifest](review_audio/manifest.json) 以原字节收录，包含每条原 WAV/PCM SHA、完整源成员链、四观察器原始 summary 行及完成回执、模型锁、中文定点疑问。四条 WAV 不纳入 Git，已放入单独交付的 `KWS_English_Pronunciation_Review_20261117.zip`，该 20 条复核包的 SHA-256 为 `68631a2fde685fd4e40e678708bfc072930cf16be87b718150ad70a7caa0ac51`。也可按 manifest 的源 artifact/member 回读原字节。人工核验尚未完成，标签仍 UNKNOWN。此 4 条已与原 16 条 Piper 合并为 20 条复核材料。

## 6. 其余已保存资产与明确缺口

40 条 LibriSpeech 朗读轨不属于这 77 条合成音频：train 30、development 10，每轨 16 秒，共 640 秒，来自已暴露朗读池且显式 trusted_negative=false。它们可作为已保存的背景上下文和软件诊断，不能冒充独立自然连续 FAR 证据。合成候选的已有拼接流使用 2.2 秒前缀与 0.32 秒原剪辑后上下文；尚无词尾标注，不能声称满足词尾后 1 秒上下文或词尾延迟资格验证。

下一步能够改变可信训练计数的证据是逐原音频的最终读音/负例核验、跨批来源族与暴露角色落实、词尾及环境覆盖。新增机器一致性或沿用旧标签不能替代这些缺口。本审计没有改写旧 split、旧模型或任何历史准入标志。

## 校验边界

实际核验了 snapshot manifest 的 36 个成员、159 个原 WAV/PCM、77 条选择及其 308 个同 PCM 观察器行；验证选择语义 SHA 与原训练输入绑定一致。[ASSET_DISPOSITION.json](ASSET_DISPOSITION.json) 保存全部可回读身份和逐条处置。这里不包含新的人工试听结论、模型推理或正式训练通过声明。

## 本目录的文本交付范围

本目录仅包含本说明、处置 JSON、Qwen63 完整联合 JSON、hash 索引、artifact 清单和四条 Qwen 原始复核 manifest，共 6 个文本文件。训练 ZIP、观察 ZIP、模型、WAV 与审计缓存未作为 Git 文件提交；原始 artifact 的 URL 与 hash 提供回读入口，音频另在用户复核包中交付。原始复核 manifest 中的 `local_outer_archive` 是生成记录时的历史本地缓存位置，不是仓库路径或长期下载入口；应使用其中的原始 artifact/member 身份。保留该原始记录的字节用于追溯，不能由缓存路径推断文件已入库。
