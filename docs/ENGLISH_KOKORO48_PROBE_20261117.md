# Kokoro48：仅训练侧的合成机制探针

这是一轮尚待执行的独立诊断。前两批 Piper 192 的原始录音、机器观察、UNKNOWN 和拒收结果保持原样。本探针只决定是否值得另行预注册一批新来源；它不训练 KWS，不更换当前模型，不产生训练快照，也不证明真实 FAR/FRR 或硬件质量。

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

`english-kokoro48-once.yml` 只接受 public main 的精确下一次 push、attempt 1。默认 plan 关闭；真实前驱提交和代码/配方 pins 审阅齐全后才可启用。PR 只运行 stdlib source/fixture 检查；执行顺序为 source contract → generation → 五路独立 observation → combine，观察失败仍保留可用捕获并在 combine 显式记缺失。

最终 artifact 为 `english-kokoro48-probe-<run_id>`，包含完整来源、`CORPUS_MANIFEST.json`、`MACHINE_EVIDENCE.json`、`PROBE_ANALYSIS.json`、固定参考模块/档案、所有阶段收据与 `FILES.json`。combine 绿色只表示诊断包已保存。任何实际计数、通过预设、时长或声学改善都必须来自此次真实执行，源码不填入未来结果。
