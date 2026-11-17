# Kokoro48：零前向权重初始化恢复与训练侧探针

本提交先保持计划关闭，只补充真实初始化诊断；尚未修复或放宽权重校验错误。公开 PR 保存实际差异后，再依据证据提交具体修复与唯一前驱绑定，重新通过该提交的完整检查才可合并执行。

实际 run `44114592216` 在导入依赖时因 `No module named 'click'` 停止：原始包保留 48 条 NOT_STARTED_NO_RETRY，TTS 前向/返回/完成均为 0；没有逐条 STARTED、模型加载标记或音频，五个观察器全部未开始。原始失败包、UNKNOWN 和旧工作流 guard 原样保留。这是初始化失败，未得到该轮音频质量结论。

后继实际 run `44118893938` 已通过同一生产运行时的安装、`pip check` 和离线导入；生成阶段进入了模型构造，随后在严格 checkpoint 读回时以 `CHECKPOINT_EXACT_KEYS` 拒绝。该次有 `MODEL_LOAD_STARTED.json`，没有已验证的 `MODEL_LOCK.json`；48 条请求仍全部未开始，TTS 前向、返回、完成和五路观察调用均为 0。此阶段不能称为“没有模型加载”，也没有音频质量结果。

本次独立恢复身份为 `kokoro48-zero-forward-checkpoint-recovery-44118893938-v1`。共享 `automation/english_kokoro48_runtime.py` 的安装与导入命令原样保留：固定 `click==8.3.1`、`typer==0.29.1`、`typer-slim==0.24.0` 及全部原运行时版本。公开同仓 PR 的 `runtime-initialize` job 先保存独立导入子收据，再真实构造一次固定模型，逐组件保存 checkpoint keys、缺失/多余项及 shape/dtype/value 比较结果。构造次数、返回次数与前向次数分列，初始化入口不执行 48 条请求、不生成音频。诊断记录完整不等于严格权重验证通过；任何不一致继续拒绝。真实初始化与 exact-head 检查通过、独立证据审阅完成后才合并，实际 PR run/job/artifact/head 收据另存，源码不预填未来标识。

首次合成前必须下载并逐字校验最新 source/runtime/probe 三个真实 artifact（14344442264、14342639905、14344293045）的固定大小及 SHA256，核验真实 run/head、48 个请求/种子/音素/预设不变、TTS 和观察前向均为 0、无逐条 STARTED/音频/捕获/已接受 MODEL_LOCK。旧 `MODEL_LOAD_STARTED` 与已通过的导入子收据必须存在；包内第一次失败的三个原始 ZIP 及恢复收据也逐字保留和核验。不会重新判定两轮旧 UNKNOWN。

源码只允许从实际旧 head `ef128ea00e350bf42392a3c9bc02d15f691528da` 的一个精确主线后继执行，attempt 1；不能只改前驱重复消费恢复身份。默认 plan 关闭，没有自动重试或循环恢复。主线生成器仍只在原 `_load_native` 中构造一次模型，首次 TTS 前向前重新严格核验并保存真实 MODEL_LOCK，不增加一次生产预构造。

以下声学设计、48 个请求身份、两个模式、参考规则、运行时版本和 288 次前向尝试总预算均未变化。PR 的一次模型初始化单独计数，不是 TTS 前向或音频样本。

## 冻结范围

四个美式声音预设 `af_bella`、`af_sarah`、`am_michael`、`am_puck` 均来自固定 Kokoro v1.0 资产。它们是共享模型的合成风格，不是经验证的四名独立真人。全部 48 行都属于训练侧探针；没有开发、校准或锁定测试角色。

每个预设生成四种唤醒假说的 short/carrier 两种条件，共 32 段；另外每个预设生成四条普通句子，共 16 段。所有请求、音素字符、随机种子、速度、载体文本及裁剪公式都在 `automation/english_kokoro48_recipe.py` 中固定。一次原生调用对应一条来源，禁止重试、选择更好的重复样本、混合音色、追加静音及根据观察结果重新裁剪。

Kokoro 原生接口以 `A` 表示 /eɪ/，以 `O` 表示 /oʊ/。四种请求为 `hˈA mˈɛɹO.`、`hˈA mˈiɹO.`、`hˈA vˈOsən.`、`hˈA vˈOzən.`。这些只是前注册研究假说，不是品牌最终美式读音。全部请求使用固定直接音素，不调用运行时 G2P。

carrier 条件使用固定中性前缀，使原生输入达到较长语境。左边界只由原生 `pred_dur` 和目标字符索引计算，按固定公式留 150 ms 左余量；右边界始终为原生音频 EOF。生成器保留完整 float32、完整 PCM16 WAV、时长向量、裁剪映射、精确 float32 子串及派生 16 kHz PCM/WAV。这个边界是模型时长预测，不是已验证的声学词边界。

## 五路观察与固定参考判断

每路最多观察 48 段新音频，各自单次：原 Qwen、Whisper、Parakeet 三 ASR；原 XLSR 音素观察器；新增 LV60 音素观察器。因此全轮上限为 48 + 5 × 48 = 288 次模型尝试，旧音频调用、重试、KWS 前向和训练调用均为零。STARTED 在调用前持久化，失败、中断、未开始均保留在 48 行分母内。

参考判断固定读取 kws-pipeline 提交 `011341c8541705a80da93ae423cfdb3d1ac9a82a` 的 `training/english_phonetic_silver.py`。读取前核对完整 bytes、SHA256 和 Git blob。仅复用其 PCM、盲观察绑定、文本/音素标准化和单行联合规则；不会调用该模块的 Piper 来源设计检查或 `build_admission`。

参考支持仍需完整三 ASR 捕获及原 XLSR：三票为 silver，两票为 weak；第三路相反唤醒核心仍 veto；缺 Hey /h/ 只有三票时才允许。ɛ/i、s/z、整句与尾元音规则均不变。普通句子需要两票一致且所有参考观察未发现唤醒词，缺失识别不能成为负例。原 PCM 时长、能量、削波条件继续适用。

LV60 来自官方固定 revision 的六份资产，首次真实下载逐一核对字节数和 SHA256。它与 XLSR 的 392 token 词表要求字节相同；仍保留独立模型与原始捕获。**LV60 只列诊断结果，绝不补参考票、替换 XLSR 或改变任何旧 UNKNOWN。**它也不能被称作纯英语专用模型。

## 唯一继续条件

对单个 mode 分别计算下列两项，两个计数都必须达到 3/4：

1. 至少三个预设在该 mode 同时有参考规则支持的 Mero、Vosen，以及至少一条参考支持的普通句子。普通句子在两个 mode 共用，不重复生成或观察。
2. 至少三个预设的 `mero-us-eh-v1` 请求本身得到原 XLSR 完整 `mero-us-eh-v1` 音素支持，并同时满足参考联合判断。只靠 /i/ 的 Mero 支持不能通过这一项。

所有正例还必须按现有 frozen core 的整数支持区间取得至少三个窗口：2.4 秒前缀、1 秒后缀、1280 样本帧步长、35200 样本 ready、31840/128 样本声学边界。这里只计算整数几何，不计算 embedding。

两种 mode 都满足时固定优先 short；只有 carrier 满足时选择 carrier；均不满足为 `STOP_DIAGNOSTIC_NO_EXPANSION`。满足时也只输出 `PROPOSE_SEPARATELY_PREREGISTERED_NEW_BATCH`。所有分支都固定 `training_snapshot_allowed=false`、`automatic_expansion=false`、`high_quality_model_passed=false`。

## 执行与产物

`english-kokoro48-once.yml` 只接受 public main 的精确下一次 push、attempt 1。默认 plan 关闭；真实前驱提交和代码/配方 pins 审阅齐全后才可启用。PR 运行 stdlib source/fixture，以及含一次真实模型构造、零前向的 `runtime-initialize` 检查；安装/导入与初始化证据分开上传，失败也保留诊断。主线执行顺序为 source contract → generation → 五路独立 observation → combine，观察失败仍保留可用捕获并在 combine 显式记缺失。

最终 artifact 为 `english-kokoro48-probe-<run_id>`，包含完整来源、`CORPUS_MANIFEST.json`、`MACHINE_EVIDENCE.json`、`PROBE_ANALYSIS.json`、固定参考模块/档案、所有阶段收据与 `FILES.json`。combine 绿色只表示诊断包已保存。任何实际计数、通过预设、时长或声学改善都必须来自此次真实执行，源码不填入未来结果。
