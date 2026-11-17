# 本轮吸收的上游实现与已有能力

本轮以实际源码版本核验贴文建议，并把可用机制接入已有项目；没有导入在线训练服务、
第三方推理引擎或新模型资产。原始源码身份见 `UPSTREAM_SOURCE_MANIFEST.json`。

| 来源 | 固定版本 | 本轮采用的机制 |
| --- | --- | --- |
| openWakeWord | `368c03716d1e92591906a84949bc477f3a834455` | 有界扫描、跳过已消费来源、从逐帧分数回收候选；补齐本项目来源/窗口/审核/去重ledger |
| onnx-wakeword | `741fb6d044eeed3491b829a1e0ddd2c292528479` | 声学分数与后处理分离，保存分数独立回放；不复制多配置sweep或额外门禁 |
| microWakeWord | `4665173cd35f1cff9a61e06fc427f124766c488e` | 流式state和实际stride校准方法，作为后续完整INT8导出的技术依据 |
| WeKws | `6a45aeb994dd81c0969ff877a5a7c46d60ed0c86` | 显式FSMN cache输入/输出与量化实现参考；保留原FSMN路线 |

## 对贴文的两处实质修正

Voicute仓库明确只提供推理源码。自定义训练和自动挖掘服务不等于开源训练管线，
因此本轮可参考的开源挖掘实现实际来自openWakeWord的
[`examples/mine_false_positives.py`](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/examples/mine_false_positives.py)。

Voicute后处理默认值依包装层而异：当前
[`python/src/onnx_wakeword/engine.py`](https://github.com/voicute/onnx-wakeword/blob/741fb6d044eeed3491b829a1e0ddd2c292528479/python/src/onnx_wakeword/engine.py)
关闭L1–L5；底层DetectionLogic构造器又开启各层，顶层README仍写仅L1默认开启。
因此不能把“默认仅L1”当统一契约。L4确为同词爆发抑制，L5还增加后验确认等待。
本轮保持项目已验证的0.8/3/13，不默认引入这些过滤层或特定说话人验证器。

## 已有能力与本轮新增能力

当前英文C链已经共享前端，使用96D×16→32→3的背景/Vosen/Mero分类头，并包含
连续帧确认、冷却和显式流状态。它不是新建议中尚未实现的共享backbone，也不是
英文FSMN-CTC。`eval/mine_hard_negatives.py` 已有完整事件标签与正例重叠排除；
本轮新增的是之前缺失的自然英语扫描→候选来源ledger→待审核结果入口。

共享特征或多输出头不自动构成两个独立关键词的质量通过。已有GRU/DS-TCN clip结果、
中文CTC整段序列结果和coverage事件结果必须按真实口径呈现；本轮统一读取保存证据，
不创建虚假的共同测试集或跨口径排行榜。

## 流式与量化的实现边界

microWakeWord的流式state更新、真实stride代表集校准、state变量量化，以及WeKws的
input/cache→output/r_cache值得继续采用。当前英文模型的feature维度、80ms hop、
FP32 cache与microWakeWord的前端不同，不能直接拼接。后续量化应分别报告权重、
activation与state/cache dtype、实际完整链精度、峰值内存和每hop开销；单个head变小
不能表示整个模型已INT8或SSC305实时性已达标。本轮没有声称完成新的INT8运行。

源码许可与资产许可分别记录：Voicute代码MIT；microWakeWord、WeKws和openWakeWord
代码Apache-2.0；openWakeWord附带的唤醒词模型许可与Google speech embedding另有
各自声明。本轮只吸收工程方法，没有新增或改写模型/数据准入。
