# 英文唤醒词质量工程本轮结果

**已完成固定 coverage 头的32条原始英文语句扫描，并通过独立保存证据审计：0个唤醒事件，9个待审核窗口。**本轮交付了挖掘、统一比较及执行恢复实现；没有训练新权重，候选尚未获得可信负例资格，不能据本次扫描宣布新增模型质量提升或晋级。

## 实现与数据边界

[PR182][pr182] 合入固定来源准备器、保存分数矿工及统一比较；[PR183][pr183] 增加下载恢复和隐藏源码上传修复；[PR184][pr184] 显式准备 FFmpeg/GCC/binutils，复用第二次产物中的原始音频。旧入口、权重、配置及历史回执保留原字节，见[矿工][miner]与[执行入口][worker]。

32句来自既有517句中两套 train 划分的24人交集，排除当前背景轨消费过的全部114个原句，按与分数无关的规则固定选择。它们仍属历史已暴露来源，新增独立说话人数为0。每个自然原句单独启动流；204.38秒不作为连续 FAR 暴露量。

矿工消费保存分数，以事件或任一关键词概率≥0.5提出前2.4秒、后1秒窗口；重叠合并，每源最多2窗、总计最多32窗，并绑定来源、PCM、trace、模型、配置及样本边界。空队列也是有效结果，不生成可训练空标签。

## 五组相容的保存结果比较

下表来自 [READBACK.json][readback] 和[比较说明][compare-doc]；匹配数均为已暴露 development 的弱意图诊断。

| 配对 | 完整配对记录数 | Vosen | Mero |
| --- | ---: | ---: | ---: |
| English63 GRU → DS-TCN | 63 | 3/4 → 1/4 | 4/4 → 4/4 |
| aug96 → coverage，clean | 117 | 3/3 → 3/3 | 8/8 → 8/8 |
| 同上，10 dB | 76 | 2/3 → 2/3 | 6/8 → 7/8 |
| 同上，5 dB | 71 | 1/3 → 1/3 | 5/8 → 5/8 |
| 同上，new8 | 27 | 1/3 → 1/3 | 7/8 → 7/8 |

第一组为整段意图分类，后四组为有限观察区间事件匹配；本次比较回读没有训练、模型前向或解码器重放。English63缺逐条PCM摘要，核验限于原包及ID；后四组有逐条PCM身份。clean弱背景含十条已暴露朗读轨，噪声条件不含这些轨。近音隔离 proposal 不算误唤醒，噪声派生不增加独立样本；跨语料排名、独立 FRR/FAR 和架构因果结论均不成立。

## 三次执行与独立核验

| 执行 | 实际结果 | 解码 / 构建 / 模型 |
| --- | --- | ---: |
| [44057968882][run1] | 语料下载300秒超时，返回28；监督307.017秒 | 0 / 0 / 0 |
| [44060185229][run2] | 下载517.931秒后保存32个FLAC；FFmpeg版本探测因程序缺失失败；监督537.495秒 | 0 / 0 / 0 |
| [44063710088][run3] | 成功，监督441.570564166秒，无遗留进程 | 32 / 1 / 32 |

[第一次审计][audit1] 验证84个保留 payload；唯一缺失的固定 workflow 另有来源补证，原包仍为非闭合集。[第二次审计][audit2] 验证291个 payload 严格闭合、32个FLAC/32份选中文本/27份章节转写及嵌套旧证据；此时尚未生成实际PCM。

第三次未再获取OpenSLR全包，原训练/质量/SDK三个固定ZIP仍用于严格再生。两次前驱累计计算为0，因此三次执行累计仅32次模型进程。成功包为 artifact **14318639588**、34,035,367字节；身份见 [API_RECEIPT.json](API_RECEIPT.json)。[执行结果](RESULT.json)确认32句、24人、204.38秒、2174个 embedding、1694个 ready 分数及0事件；[监督回执](SUPERVISOR_EXIT.json)记录用时与清理。

[独立审计](ACTUAL_AUDIT.json)已通过：659个 payload 加 MANIFEST 严格闭合，91个保留源文件与第二次原包逐字相同；核对实际PCM摘要、全部trace、固定事件策略及候选窗口。审计只读取保存字节，没有重算embedding/logits、解码或编译。

## 9个候选窗口

来自8个原句，Mero 4窗、Vosen 5窗；均为 `UNKNOWN_REVIEW_REQUIRED / UNREVIEWED`，`trusted_negative=false`、`training_admitted=false`。表中时间为原句内秒数，区间为左闭右开；峰值保留六位小数，完整精度见[候选台账](LEDGER.json)与[摘要](CANDIDATE_SUMMARY.json)。

| 原始源ID | 关键词 | 起止秒 [start, end) | 峰值 |
| --- | --- | --- | ---: |
| 2902-9006-0011 | Mero | [0.360, 3.920) | 0.870470 |
| 1272-135031-0020 | Mero | [0.360, 3.920) | 0.814636 |
| 5536-43363-0014 | Vosen | [6.440, 10.000) | 0.703360 |
| 2035-152373-0004 | Vosen | [3.800, 7.280) | 0.684729 |
| 3170-137482-0023 | Vosen | [2.760, 6.240) | 0.680988 |
| 8297-275156-0002 | Vosen | [0.520, 3.920) | 0.652437 |
| 3000-15664-0039 | Mero | [2.120, 4.755) | 0.571831 |
| 5536-43363-0014 | Vosen | [0.840, 4.320) | 0.562393 |
| 7976-110523-0009 | Mero | [1.640, 5.040) | 0.531215 |

前两条Mero源在sample 44160、45440、46720的保存分数分别为 `[0.81121522, 0.87046963, 0.63574523]`、`[0.80761516, 0.81463629, 0.60734940]`：连续两步达到0.8，第三步跌破，未满足现有三步确认。摘要保留对应原始 `scores/` 成员路径。这解释了高分候选与0事件并存，不能证明降低FAR或没有召回损失。未进行听音审核，转写也未被当作声学负例标签。

## 复核保存证据

在含本报告的 kws-data 仓库根目录，使用已认证的 `gh` 下载原产物，再运行标准库审计器。临时目录保证审计输出为新文件；该审计不执行模型或音频解码。

```bash
KWS_RESULTS_DIR=research/english_hardneg_natural32_results_20261117
KWS_AUDIT_TMP="$(mktemp -d)"
gh api repos/jiying2007/kws-data/actions/artifacts/14318639588/zip \
  > "$KWS_AUDIT_TMP/original.zip"
python3 -B "$KWS_RESULTS_DIR/audit/audit_saved_natural32_retained.py" \
  --artifact "$KWS_AUDIT_TMP/original.zip" \
  --api-receipt "$KWS_RESULTS_DIR/API_RECEIPT.json" \
  --output "$KWS_AUDIT_TMP/AUDIT.json"
```

## 模型身份与尚缺验收

现有 coverage 为冻结 oWW 96维 embedding 加因果三类 MLP 头：0背景、1 Vosen、2 Mero；本轮审阅的主线尚未交付可准入的强英文 FSMN-CTC，见[模型登记][inventory]。历史 `/heɪ ˈmɪəroʊ/`、`/heɪ ˈvoʊsən/` 与该模型 profile `/heɪ ˈmiːroʊ/`、`/heɪ ˈvoʊzən/` 分别保留，未重标音频或权重，目标IPA不作为录音发音真值。

SSC305实机资源与实时性、真人3–5米召回、独立长时自然负流FAR、人工发音gold、词尾延迟及INT8全链精度仍缺实测；本轮软件和证据交付不替代这些验收。

[pr182]: https://github.com/jiying2007/kws-data/pull/182
[pr183]: https://github.com/jiying2007/kws-data/pull/183
[pr184]: https://github.com/jiying2007/kws-data/pull/184
[miner]: https://github.com/jiying2007/kws-data/blob/4f998e89d9f344677b8a97fdb78352e42b8cf6bd/research/english_hardneg_natural32_20261117/mine_saved_scores.py
[worker]: https://github.com/jiying2007/kws-data/blob/4f998e89d9f344677b8a97fdb78352e42b8cf6bd/automation/english_hardneg_natural32_retained_once.py
[readback]: https://github.com/jiying2007/kws-data/blob/4f998e89d9f344677b8a97fdb78352e42b8cf6bd/research/kws_unified_evidence_20261117/READBACK.json
[compare-doc]: https://github.com/jiying2007/kws-data/blob/4f998e89d9f344677b8a97fdb78352e42b8cf6bd/research/kws_unified_evidence_20261117/README_zh.md
[inventory]: https://github.com/jiying2007/kws-data/blob/4f998e89d9f344677b8a97fdb78352e42b8cf6bd/research/kws_unified_evidence_20261117/MODEL_INVENTORY.json
[run1]: https://github.com/jiying2007/kws-data/actions/runs/44057968882
[run2]: https://github.com/jiying2007/kws-data/actions/runs/44060185229
[run3]: https://github.com/jiying2007/kws-data/actions/runs/44063710088
[audit1]: https://github.com/jiying2007/kws-data/blob/4f998e89d9f344677b8a97fdb78352e42b8cf6bd/research/english_hardneg_natural32_recovery_20261117/FAILED_ACQUISITION_AUDIT.json
[audit2]: https://github.com/jiying2007/kws-data/blob/4f998e89d9f344677b8a97fdb78352e42b8cf6bd/research/english_hardneg_natural32_retained_20261117/RETAINED_SOURCE_FAILURE_AUDIT.json
