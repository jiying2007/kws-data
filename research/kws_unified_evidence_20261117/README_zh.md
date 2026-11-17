# 已保存证据横向比较：实际回读结果

本次完成 **5 个真实配对组**的保存结果回读，训练、模型前向、解码器重放均为 **0**。只比较原有弱意图诊断；`formal_ranking_allowed=false`，没有统一质量分数、冠军或新独立测试集。`READBACK.json` 可由本目录原生 ZIP 证据胶囊与标准库工具逐字节再生。

## 实际可比较组

| 配对 | 完整配对记录 | 原指标单位 | 已暴露 development 结果 |
| --- | ---: | --- | --- |
| 原 English63 GRU → DS-TCN | 63 | 整段 requested-intent 分类 | Vosen 3/4→1/4；Mero 4/4→4/4；普通语句 3/4→1/4 |
| aug96 → coverage，clean | 117 | 有限观察区间弱意图事件 | Vosen 3/3→3/3；Mero 8/8→8/8 |
| aug96 → coverage，10dB | 76 | 同上 | Vosen 2/3→2/3；Mero 6/8→7/8 |
| aug96 → coverage，5dB | 71 | 同上 | Vosen 1/3→1/3；Mero 5/8→5/8 |
| aug96 → coverage，new8 | 27 | 同上 | Vosen 1/3→1/3；Mero 7/8→7/8 |

整段 GRU/DS-TCN 的 train 意图符合数均为29/30，已暴露 development 为10/12与6/12；21条近音隔离记录的目标类 proposal 为17/21与14/21，**不是 false accepts**。coverage 四个条件分别保持本来的精确 roster，不能把噪声变体加作独立样本。clean 的26条 development 弱背景含16条合成普通语句和10条已暴露 LibriSpeech 轨道；噪声条件表的背景分母为16。所有这些背景都不是合格连续自然负流。

工具核验每组的记录集合、原始源/PCM身份、role、目标版本、标签等级、曝光、观察时钟和评分规则；从保存的离散事件重算 matched/prefix/wrong/extra 等计数，再核对原 RESULT。它不从 logits 重放事件，不改变阈值。4份小模型权重的实际保存字节 SHA256 已核验，未构造任何模型。

## 不能混合的模型与指标

中文 R1 FSMN-CTC 的公开结果是96条训练/34条已暴露开发中的完整 greedy 词序列诊断，K1/K2各10/10；80mel、左右上下文、skip3，与这里的英文 clip/event 任务不同。保留的英文 FSMN-R0 是三类整段弱意图头；公开来源没有把它定义为英文 CTC，本包不拥有它的权重或逐条结果。英文因果 FSMN-CTC 在已核查主线中仍列为 challenger，尚无可纳入这次配对的已执行权重/结果。详见 `MODEL_INVENTORY.json` 的来源。

本包明确拒绝原 English63 与 snapshot160/coverage 的跨数据排名：它们的原语料、目标历史、split、评分单位和阈值历史不一致。架构或前端不同本身不构成拒绝条件；将来同音频、同目标/标签、同split与曝光、同事件评分和独立阈值选择证据齐备后，可以报告完整系统比较。当前工具只支持历史弱证据诊断，不把 exposed 数据重命名为 holdout。

历史读音 `/heɪ ˈmɪəroʊ/`、`/heɪ ˈvoʊsən/` 与2026-11-16图片目标 `/heɪ ˈmiːroʊ/`、`/heɪ ˈvoʊzən/` 分别保留；这里没有决定本轮用户冲突后的现行目标，也没有把旧音频、旧权重重标为新读音。原 English63 仅按原请求拼写/意图身份保留，不能从拼写推定实际 IPA。

## 复用范围与保存输入

原有 [`research/model_comparison/compare.py`](https://github.com/jiying2007/kws-pipeline/blob/aec192dd7c5955a7fa2b709564cdacb185a590be/research/model_comparison/compare.py) 专用于中文同训练条件、独立 train/calibration/selection 的开发比较。这里只复用其 SHA/identity/digest 边界及依赖 `tools/source_identity.py` 的冻结原字节；没有伪造三个分组以强行调用它的正式比较入口。

通用事件匹配继续使用 [`eval/score_events.py`](https://github.com/jiying2007/kws-pipeline/blob/aec192dd7c5955a7fa2b709564cdacb185a590be/eval/score_events.py)，Wilson/Poisson 上界继续使用 [`tools/statistical_bounds.py`](https://github.com/jiying2007/kws-pipeline/blob/aec192dd7c5955a7fa2b709564cdacb185a590be/tools/statistical_bounds.py)，分片曝光汇总继续使用 [`eval/aggregate_far.py`](https://github.com/jiying2007/kws-pipeline/blob/aec192dd7c5955a7fa2b709564cdacb185a590be/eval/aggregate_far.py)。本包不复制这些算法，也不为不合格弱背景计算 FAR 置信区间。

`SOURCE_MANIFEST.json` 记录19个胶囊成员的大小/SHA、原路径、Git commit/blob或原 artifact身份。`saved_inputs.zip` 含原 RESULT/PLAN、roster、event JSON和小头权重，**没有音频或 embedding 数组**。复现时核验胶囊整体与全部成员；原50MB完整coverage artifact不在胶囊中，因此回读输出明确标记未重新核验该原 ZIP。所有来源字段仍保留，能够按原 artifact14310138100/run44042856615还原核对。

## 无模型复现

从仓库根运行，输出必须是不存在的新文件：

```sh
python3 -B research/kws_unified_evidence_20261117/compare_saved.py --output /tmp/kws-saved-readback.json
cmp research/kws_unified_evidence_20261117/READBACK.json /tmp/kws-saved-readback.json
python3 -B -m unittest discover -s research/kws_unified_evidence_20261117 -p test_compare_saved.py -v
```

13项标准库测试包括：IPA/数据/评分/阈值历史不一致拒绝、split/PCM/曝光/时钟不一致拒绝、禁止 holdout 冒充、缺记录不取隐式交集、重复ID、原事件与汇总矛盾、不同前端可作系统比较，以及全部保存输入逐字节再生 READBACK。测试和再生均不下载依赖、不训练、不推理、不运行 C 二进制。
