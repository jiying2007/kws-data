# Kokoro 完整上下文 32 次音素诊断

本任务只调查**完整输入与旧裁剪输入的观察差异**。它不重新合成音频，不重放已经消费的裁剪输入，不更改原来的准入标签，不训练模型。默认计划为 `enabled=false`；真正的结果只能由首次公开主分支执行产物填写。

## 已有证据与本次问题

原公开执行 [44125960085](https://github.com/jiying2007/kws-data/actions/runs/44125960085) 的 48 次合成及五路各 48 次观察均已完成。原结果为 `STOP_DIAGNOSTIC_NO_EXPANSION`：Mero 的文本票多数兼容，但原 XLSR 的完整音素串未满足冻结规则。LV60 的输出与 XLSR 有分歧，因此不能直接换观察器、放松音素规则或把原 UNKNOWN 变成训练正例。

这次固定取原批次的**全部 16 条 carrier 完整原生音频**：四个美式合成 preset 各四条，包含 8 条 Mero 和 8 条 Vosen。选择只依据预注册的 `mode=carrier`，与原 ASR、音素通过与否无关。四个 preset 仍是训练侧已暴露的合成风格，不是真人独立说话人，也不是新的未见测试来源。

原始身份固定如下：

| 项目 | 固定值 |
|---|---|
| 原 run / attempt | `44125960085` / `1` |
| 原 head、唯一新任务主分支前驱 | `573e807c330f7b55b239220e9380e761bb95f42e` |
| 原 source artifact | `14347901645`，59,802,210 B |
| 原 source ZIP SHA256 | `eb6c5f6da584e2517915c517ed316cb45d4a207e9f863f949f5a110f730022e9` |
| 原 probe artifact | `14348910717`，60,974,804 B |
| 原 probe ZIP SHA256 | `3a0cf8fb1aeefa658a1418aa5eaa43e9b867caedfc3fbb09004df893b3ed7a66` |
| 新 campaign | `kokoro-v1_0-original-carrier-full-context32-v1` |

## 固定输入与预算

两个旧 ZIP 按 GitHub 原始下载字节验证，不重新打包后冒充原 artifact。保留旧 run 与 artifact 原始 API 响应；核对原闭合文件清单、48 条实际生成收据、模型读回、全音频与裁剪坐标、source 到 probe 的逐文件复制，以及旧 STOP。原 48 个已观察 PCM 的 SHA256 集合由这些实际字节重新计算，不能只信计划中的排除表。

计划的 `input_roster` 只包含已经存在的原生 float 与旧裁剪身份、固定坐标和新路径，不预填尚未产生的完整 16 kHz PCM 哈希。实际派生后才写 `INPUT_MANIFEST.json`。16 个新完整 PCM 必须彼此不同，并且不能与旧 48 个 PCM 中任何一个同哈希。任何缺失、碰撞或失败都保留在 16 条总分母里；不能筛除失败后另选来源。

| 阶段 | 最大新增调用 | 输入或限制 |
|---|---:|---|
| 完整输入派生 | 0 个模型 | 全部 16 个已保存的 `native-full.f32` |
| 原 XLSR | 16 | 每个新完整输入一次 |
| 原 LV60 | 16 | 每个新完整输入一次，只作诊断 |
| TTS、文本 ASR、embedding、训练 | 0 | 不执行 |
| 旧裁剪输入前向 | 0 | 原观察只读 |
| 重试 | 0 | 不允许自动重试、重选或再裁剪 |

准确口径为 `source_material_reused=true`、最多 16 个 `new_derivative_inputs` 和最多 32 次 `phone_forwards_on_reused_material`。它们使用已生成素材的新完整输入视图，不应写成“完全没有使用旧音频”。

## 派生与样本坐标

完整输入读取原始 24 kHz、单声道、little-endian float32 文件，使用原来的 NumPy 1.26.4 和 SciPy 1.15.3：

```python
values = np.frombuffer(native_full_bytes, dtype="<f4")
full = scipy.signal.resample_poly(
    values, 2, 3, window=("kaiser", 5.0), padtype="constant", cval=0.0
)
pcm = np.rint(np.clip(full, -1.0, 32767 / 32768) * 32768).astype("<i2")
```

完整原生区间从 0 到 EOF，全程不依赖 ASR、音素输出或 KWS。输出样本数必须为 `ceil(native_samples * 2 / 3)`。WAV 必须逐字承载同一份 PCM16、16 kHz、单声道数据。

16 个原裁剪起点均能整除 3，因此其对应 16 kHz 偏移为 `2 * crop_start_native_sample / 3`。这不代表新完整输入的后缀与旧单独重采样的裁剪逐样本或逐帧相同。保存的 `DERIVATION.json` 包括：

- 原完整 float、原裁剪 PCM、新完整 PCM 与 WAV 的身份和实际版本。
- 原生与 16 kHz 坐标、原裁剪右边界为 EOF 的事实。
- 对齐后逐样本差异数、完整差异半开区间、首末差异索引、绝对差异最大值和总和。
- 完全相同的前缀、后缀范围及后缀字节哈希，供离线逐样本复核。
- 16 kHz 偏移对 320 的余数。已有静态坐标的分布为 0:1、80:7、160:4、240:4。

这些是采样坐标与字节比较，不是真人词边界标注。FIR 左边界、卷积格点以及按整个输入重新计算的归一化都可能改变观察结果，不能把差异单独归因于某一种机制。

## 两套观察与完整数值收据

沿用原 XLSR 与 LV60 的精确仓库 revision、权重、六个配置/词表文件，以及原 greedy CTC 逻辑。模型保持 CPU、float32、eval、`weights_only=True`、严格 state-dict 加载；两套 392 token 词表字节一致。固定旧 `ctc_tokens` 只通过纯 AST 读取，不运行任何旧任务入口。

每个新输入保存真正的 processor `normalized-input.f32`、`attention-mask.i64`、完整 `[1,T,392]` logits、全部 argmax frame IDs、CTC tokens 和输入 SHA。模型返回后立即计数，随后才访问、格式化或保存 logits，避免把输出保存失败误计成没有模型返回。

独立比较器从保存的 logits 重新计算 argmax 与 CTC，从 PCM 重新核对归一化数值与 mask。名义卷积支持为 `[320*t, 320*t+400)`，只描述卷积输入范围；最终 transformer logits 仍受整个输入上下文影响。用原固定裁剪偏移机械选出的尾部帧只是诊断视图，不是新的模型前向，更不是新的准入标签。

缺少最终 provider 收据时，实际调用数保留为 `null`，同时报告已保存 marker 的下界与最大上界；不能把缺失证据写成真实 0 次。每份最终报告继续保留 16 个来源与 32 个观察槽位。

## 执行与产物

工作流为 `.github/workflows/english-kokoro-context32-once.yml`。PR 只跑 stdlib/mock 源检查，常规、`-O`、`-OO` 都执行；没有模型初始化、权重下载或音频派生。只有预先绑定前驱的首次公开 main push、attempt 1 可以运行数值阶段。准备阶段内限 600 秒，两个观察阶段各 1,800 秒；外层 workflow 保留独立超时兜底与退出记录。

源 artifact `kokoro-context32-source` 保存：

- `inputs/source.zip`、`inputs/probe.zip` 及其原始 API 身份。
- `source/audio/<new-id>/full-16k.{wav,pcm}`、STARTED 和 DERIVATION/FAILED 收据。
- `INPUT_MANIFEST.json`、`EXECUTION_PLAN.json`、RESULT 和闭合 FILES。
- 本次执行的新源码与只读旧 helper 字节。

两个 observer artifact 名称为 `kokoro-context32-observer-phoneme` 和 `kokoro-context32-observer-phoneme_lv60`。最终 `english-kokoro-context32-<run-id>` 包含原 prepared 闭包、两个观察目录、`COMPARISON.json`、实际 RESULT 和全文件清单。比较器始终输出诊断状态，保持 `training_allowed=false`、`training_snapshot_allowed=false`、`automatic_expansion=false`、`old_labels_modified=false`。

当前新任务尚未执行，不预写任何音素改善或模型质量结论。即使完整上下文观察更接近某个研究发音假说，旧 48 条的 UNKNOWN、训练排除及 STOP 仍原样保留；后继研究需要另行明确数据与评价设计。真人采集和物理硬件均不是本次工作的前置条件。
