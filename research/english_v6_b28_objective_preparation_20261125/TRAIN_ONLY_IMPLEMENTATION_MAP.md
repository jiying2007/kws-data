# B28 natural suffix no-event objective：TRAIN-only 静态实施图

状态：DRAFT / NOT REGISTERED / NOT EXECUTED。当前 B27 仍是唯一正式候选。本文件不判定 B27 终态、不定义其终结条件、不授予 B28 SOURCE 资格，也不开放任何新预算。

依据：数学假设 checkpoint 4bc1b66a092be75452e4d092d0f3a71033dac85a；固定目标源码 SHA256 1321eef19cf02d8ce8009e307d4fc384eeac7f59a57e2e07c05dbf79d273350a；纯数学 CI run 45629815249 的 10/10 成功原报告（SHA256 5c9e0c5f09444308bad66638f22abaf9e193313fb88ba2c78588259d8d521bec）。本图只读了源文件与这些数学原件，没有读取 B27 PRIMARY 结果，没有执行 fit、model、reader、Torch 或新增数学测试。

## 1. 唯一机制变化

当前 B27 的自然 ordinary 分支使用空目标 CTC。在未来独立登记的新候选中，只将这个分支的 suffix 项替换为固定 no-complete-event NLL：

- raw = model(x)，形状 [1,T,4]，原 FP32 输出。
- logp = raw.log_softmax(-1)，保留 FP32，用于旧 prefix blank、B0 ordinary teacher、margin trace。
- suffix_start = prefix // 160 - 3。
- prefix_blank_end = prefix // 160 - 8。
- 原 guard：0 < prefix_blank_end < suffix_start < T - 5。
- E = no_event_nll_torch(raw[:, suffix_start:, :], torch)[0]。
- P = max over t < prefix_blank_end of softplus(logsumexp(logp[t,1:]) - logp[t,0])。
- L_natural_gain = tier_weight * (0.6 * E + 0.5 * P)，silver=1.0、其余合法原 tier=0.75。
- 一步自然贡献 = 0.16 * (L_gain1 + L_gain0.5) / 2。

E 是 CTC 全路径分布中“没有任何完整 1,2,3 事件”的负对数概率。原 decoder 先折叠相邻 raw 重复，再去 blank；固定四 live + HIT DFA 正是该事件语义。它不是 greedy argmax 的事件计数、blank occupancy、逐帧伪标签或最长窗内 surrogate。保留 logZ/logH 双质量与 logaddexp(0, logH-logZ)，没有 epsilon、clamp、长度归一化或额外系数。

DFA 在 suffix 边界从 A 开始；不涵盖 prefix/suffix 跨边界事件。这是已冻结 suffix-only 假设的边界，不应扩展措辞为全录音 no-event 概率。

只有新 E 的内部计算转为 float64。其他旧 logp/teacher/prefix/synthetic/positive 接口保持原 FP32；自然项加入总标量时的混合精度提升应在新设计中明写，不应把整张旧图或已记录 raw logits 改成 float64。

## 2. 实际函数链与最小接入点

| 层 | 当前真实函数 | 未来最小动作 |
|---|---|---|
| 初始化 | english_v6_b27_natural_weight.load_initial_state(directory, frontend, torch, np) | 保留从 original_b24 原包内 control/models/mero43-b23-challenger192.pt 取 B23 初始化及 PT/NPZ/frontend 对照；不从 B27 最终 checkpoint 初始化 |
| 私有参数环境 | english_v6_b27_natural_weight.fit_engine() | 目前只给 b24.b23.fit 的私有 old.NATURAL_WEIGHT=0.16；新机制不能只靠这个旧 code object 表达 |
| 数值 loop | english_v6_b23_prefix_peak.fit(initial, slots, base_cache, natural_cache, positive_paths, teachers, frontend, output, torch, np, F) | 建立新的、将来独立冻结的 fit 源文件；只替换 natural loop 的一个 loss 调用，另加不产生 forward 的 trace |
| 共享旧 loss | english_v6_b23_prefix_peak.prefix_peak_suffix_loss(model,x,klass,prefix,tier,torch,F,return_logp=False) | 保持原字节；synthetic_ordinary 与 synthetic_positive 继续调用它 |
| 固定新 adapter | torch_objective_draft.natural_no_event_suffix_loss(model,x,prefix,tier,torch,F,return_logp=False) | 保留 SHA1321… 文件原字节；只在 natural_ordinary 处调用，注意比旧接口少 klass=0 参数 |
| 新 NLL 核 | torch_objective_draft.no_event_nll_torch(logits,torch) | 原版本不变；惰性接收已有 raw suffix，不另作模型 forward |
| export wrapper | english_v6_b24_b23_warm_start.fit(...) | 可继续用私有 FunctionType globals 替换其 b23.fit 指向新的 fit，并将 STEM 指向未来新候选；不改旧源文件 |
| B27 wrapper 模式 | english_v6_b27_natural_weight.fit(...) | 仅作为组织模式参考；未来需要独立新身份/设计与目标 hash，不能把新算法冒充为旧 B27 weight-only recipe |

最重要的接入陷阱：不能将 b23.fit 全局 prefix_peak_suffix_loss 替换成新 adapter。它有 synthetic 和 natural 两个 call sites，synthetic ordinary 与 natural 的 klass 都是 0；只按 klass 分流会同时更改 synthetic ordinary。应在新 fit 源中明确只改 natural_cache 那个 call site：

旧调用：
    prefix_peak_suffix_loss(net, natural_cache[row['id']][gain][prefix],
                            0, prefix, row['tier'], torch, F, True)

新调用：
    natural_no_event_suffix_loss(net, natural_cache[row['id']][gain][prefix],
                                 prefix, row['tier'], torch, F, True)

这份数学 adapter 的完整 fit 从未执行，数学 CI 也没有证明完整 fit 集成。新 fit 必须单独冻结源码/AST 差异；不能继续声称数值 loop 与 B23 fit 原 code object 完全相同。

另一个接口陷阱：现有 B24 export wrapper 要求底层 fit 暂时返回 mero43-b23-challenger192.* 文件名和旧 parent_pt_sha256=old.B0_PT 元数据，再由 wrapper 校正实际 B23 initializer 与新 STEM。若复用该 wrapper，新 fit 要保留这组内部返回约定，不能提前改名导致 guard 不匹配。对外结果仍必须写真实 B23 parent hash，不得把旧临时元数据当初始化证据。

## 3. 必须完整保留的训练不变量

- B23 initializer SHA256：
  784def9df6c2a8bfbecdb7ba2d7b70630337a1cda72561c05392327aa980297f。
- 历史 B24 control SHA256：
  f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4。
  继续只复制其原 pt/npz/onnx，不重训练、不增加 control forward。
- 同一 88 TRAIN originals：79 base + 9 natural；9 natural 属于原 8 readers。此处没有新来源或新音频。
- 同一 COMMON_DRAW.json 原字节 SHA256：
  90b41a650ad5439524bef4bc6d13394fd65034e98c2c5a7862e0a0c1de78a83d。
- 192 draws / 192 fresh AdamW updates；seed=20261120、lr=0.00006、weight_decay=0.001、gradient clip=5。
- gains=(1.0,0.5)，三种原 prefix samples=(6400,12800,19200)；原 tier mapping 不变。
- synthetic 项仍为四次旧损失的平均；原 positive labels=[1,2,3] 和 synthetic ordinary 空 labels=[] 都不变。
- gain consistency 仍只在原 synthetic 两类上，系数0.03、两类平均、原 KL clip=8；不向 natural 新增一致性。
- outer natural=0.16、inner negative=0.6、prefix coefficient=0.5、prefix temporal reduction=amax。
- ordinary ceiling=0.08；positive retention=0.08；所有 group/original/gain/token/frame 聚合层级不变。
- frontend、feature cache、模型结构43876参数、determinism设置、无 Dropout/BatchNorm、checkpoint reload 与 ONNX export流程不变。

总 loss 结构只把原 natural_ctc 的语义换成 natural_no_event：
    sum(synthetic losses)/4
    + 0.03*sum(two synthetic consistency terms)/2
    + 0.16*sum(two natural losses)/2
    + positive_retention
    + 0.08*ordinary_ceiling

预算账本保持：
- synthetic_ordinary：384 forwards。
- synthetic_positive：384 forwards。
- natural_ordinary：384 forwards。
- positive_retention：692 forwards。
- optimization 总计1844；原 checkpoint reload=2；ONNX export=1；新模型总 forward=1847。
- 192 optimizer trace rows、692 positive trace rows、768 ordinary-margin rows。
- 264 saved B0 teacher arrays、原120 positive conditions+同样2 additions=122；无新 teacher/source/observer forwards。

## 4. B0 teacher 与 positive retention 的真实保留链

english_v6_b22_union_retention_continuation.saved_ordinary_teachers(inputs,torch,np) 从 b21_weights.zip 的 B0_ORDINARY_TEACHER.json / B0_ORDINARY_TEACHER_LOGITS.npz 读取原数组，键为 (id,gain,prefix_samples)，不重跑 B0。

natural adapter 仍返回原 FP32 logp；fit 仍把 (row,gain,prefix,logp) 放入 ordinary 列表。于是 english_v6_b17_b0_margin_pair.ceiling_loss(samples,teachers,torch) 保留每步四个 ordinary 样本（synthetic/natural 各两 gain）及两个 source groups 的原聚合。不能把事件 NLL 或已切 suffix 数组替换这个完整 logp 接口。

english_v6_b22_union_retention_continuation.margin_statistics(student,teacher,torch) 继续记录：
    frames, mean, maximum, active_fraction, active_cells, cells
及原 step/id/group/gain/prefix 身份。student 与 teacher 的逐 token-vs-blank margin 差保持原式。

positive retention 是另一条旧函数，不是 prefix_peak_suffix_loss：
- prepare_union_paths(...) 保持原120 B9 references和同样两个新增原teacher paths；
- positive_loss(step,model,by_group,recorder,trace,torch) 保留旧 group轮换、teacher spans、full-competitor margin、gain/token/original/group聚合及0.08权重；
- trace仍为 step/id/group/gain/teacher/newly_protected/unweighted_loss；
- 692个声学返回按原Recorder登记，不触碰新增no-event目标。

## 5. 原 raw logits 已足够；只增加观察 trace

当前 baseline.Recorder.before(**metadata) / hook(module,args,output) / save(output) 已构成完整账本：
- hook 在模型返回时保存 output[0].detach().cpu().numpy().copy()；
- 在后续 log_softmax 和新 float64 DP 之前保存原 FP32 [T,4] raw logits；
- TRAIN_FORWARD_INDEX.json 每行含 step/kind/id/gain/prefix，positive retention另含group/teacher/newly_protected；
- 公共字段为 array=forward_XXXXX、frames、sha256（原float32 payload hash）；
- TRAIN_FORWARD_LOGITS.npz 含1844个对应 .npy；
- 顶层 count=1844、pending_unreturned_forward=null、logits_are_raw=true、arrays_sha256 指向整个NPZ。

新fit不需额外 model(x)，不需为了观察再跑 no_event_nll_torch。调用固定 adapter 后只记录其已返回 scalar loss，并链接刚产生的 recorder.rows[-1]。

建议未来新增一个观察文件 NATURAL_OBJECTIVE_TRACE.json，固定384条，文件schema：
    english-v6-b28-natural-no-event-trace-v1

每条建议精确字段：
    step, id, group, gain, prefix, tier,
    array, raw_logits_sha256, frames,
    suffix_start, prefix_blank_end,
    weighted_loss

weighted_loss 是 float(loss.detach())，已经包含 inner0.6、prefix0.5、tier，不含 outer0.16 和gain平均。不通过减去prefix项反推出稀有event NLL，避免消去；不伪称这个字段是纯suffix NLL。

文件头应包含 objective_schema、objective_sha256、reference_sha256、common_draw_sha256、count、records。constants可由设计读取，reader仍必须与冻结设计逐项匹配，不能信任日志自报。

OPTIMIZER_TRACE.json 应在新schema中将 natural_ctc 改为 natural_no_event，定义为原外系数后的每步自然总贡献。其他字段保留：
    step, prefix, ordinary_id, positive_id, natural_id,
    loss, synthetic_ctc, gain_consistency, positive_retention,
    ordinary_ceiling, ceiling_coefficient,
    preclip_gradient_norm, gradient_clipped

新trace schema 可为 english-v6-b28-optimizer-trace-v1；不得把新数值继续贴旧CTC语义而让旧reader照常接受。原 forward index 的 raw数组、命名、顺序和字段不必改变。

RESULT.saved_training_observations 需要由原六件扩为精确七件，包含新增 NATURAL_OBJECTIVE_TRACE.json 的原 bytes/SHA；model artifact成员闭包和新reader OBSERVATIONS同步冻结。不得临时append到旧B27 artifact。

## 6. 独立stdlib重算的具体路径

新reader只读原件，不导入Torch/NumPy，不执行模型、梯度或optimizer：

1. 沿原 expected_forward_schedule(slots,source) 验证192 draw、1844 index、exact role顺序，frames=(samples+prefix-512)//160+1，拒绝缺件/重复/unknown。
2. 沿原 read_npy(raw,expected_shape) 读 NPY，要求 <f4、C-order、完整(T,4)、全finite，核 raw payload SHA。
3. 对 index 中恰好384个 kind=natural_ordinary，以 step/id/gain/prefix 匹配 COMMON_DRAW、自然原source行及新trace；不能只取argmax labels，必须使用全部4类原logits。
4. 用 struct.iter_unpack("<4f",payload) 得到原FP32值在Python double中的精确表示；只切 logits[suffix_start:]。
5. 调用独立固定 no_event_reference.no_event_nll(raw_suffix)：
   logits_to_log_probabilities → log_probability_masses → stable softplus(logH-logZ)。
   no_event_reference.py SHA为034a4827fb9ff3091a5e23ea25e2eab0bc23ee7263bce644fd52741e6b80f969。
6. 独立重算未变prefix项。数学上可从同一原raw计算
   max softplus(logsumexp(raw[t,1:])-raw[t,0])；
   与原FP32 logp实现等价但不承诺跨实现bitwise相等。
7. 合成 tier*(0.6*E+0.5*P)，与每条实际 weighted_loss 比较；按两个gain组合再乘0.16，核每步 natural_no_event；最后沿旧scalar trace规则核total loss。
8. 比较容差必须前瞻冻结，不能看实际B28 loss后调整。旧reader对总scalar使用 rel_tol=abs_tol=2e-6，可作为保留的总scalar边界；新NLL的float64参考与旧FP32prefix应分别说明数值误差来源。若另加更严格event分量容差，应在新登记前定死，而非从未来训练结果拟合。
9. reader输出应明确 natural_objective_returns_verified=384、natural_no_event_terms_recomputed=true、原观察文件hash、误差统计，以及new_model_calls/new_optimizer_updates=0。
10. 不声称复算了训练梯度或192次optimizer更新；梯度链的数学CI、原执行代码hash、完整forward/trace/最终原件是证据，独立stdio reader不是一次训练重放。

极稀有event低于float64可表示范围的NLL/gradient可能underflow，这已在数学测试中明示；不能引入epsilon、阈值裁剪或从0推断不存在事件。固定 adapter 的T<3 graph-connected zero不是放宽实际suffix边界guard。

## 7. 旧B27 TRAIN reader为何不能直接认证

english_v6_b27_training_readback.saved_training_arrays(models,result,source) 确实核1844个原数组、全部SHA、shape与schedule，也核192/692/768 trace数量。但它对自然项只要求natural_ctc有限，并验证：
    loss ≈ synthetic_ctc + gain_consistency + natural_ctc
           + positive_retention + ceiling_coefficient*ordinary_ceiling
它没有从自然raw logits独立重算空CTC或新no-event NLL。把新标量塞进natural_ctc仍可能满足旧算术，因此不能以此宣称新目标认证完成。

此外旧reader硬编码：
- english-v6-b27-original-training-lock-v1 / english-v6-b27-original-training-readback-v1；
- B27 train schema、design/opening/assessment SHA、branch/workflow、artifact names、candidate stem；
- 原六观察文件集合和完整model ZIP闭包；
- B27的原Q1 failure资格链与源码producer闭包。
这些是既有B27证据身份，不能放宽、覆盖或改名借用。未来新reader应保留原byte/provenance/teacher/source guards，再增加独立no-event重算与新身份，旧B27 reader保持原字节。

## 8. 未来必须独立冻结的模块、schema和hash

以下是建议名称，不代表已经创建或登记：
- automation/english_v6_b28_no_event_objective.py：保留固定Torch数学源码原字节，SHA1321…。
- automation/english_v6_b28_no_event_reference.py：保留固定stdlib参考原字节，SHA034a…。
- automation/english_v6_b28_natural_event_fit.py：新的明确单call-site fit源，完整文件与函数/AST差异单独冻结。
- 新B28 TRAIN orchestrator、design、opening、assessment binding、workflow template、producer、training binder、training readback、对应契约测试。
- design/train/source/producer/lock/readback/optimizer trace/natural trace都用新的B28 schema；禁止接受B27 schema作为新结果。
- 新 RESULT / SOURCE / PRODUCER 至少显式绑定 objective_schema、objective_sha256、reference_sha256、fit_adapter_sha256、design_sha256、common_draw_sha256、initializer/control hashes、loss固定系数与suffix边界、新观察清单。
- 新producer沿现有 paths/capture 规则纳入全部旧338依赖及新objective/reference/fit/reader/新schema metadata，验证Git tree、blob SHA1、bytes/SHA256；动态import/FunctionType依赖必须显式列入，不能只靠静态import闭包。
- 新原件容量应把新增384条trace纳入执行前证据容量核算；不改旧128MB archive cap或1847/192预算，不把大旧proof嵌入新producer。
- future branch/marker/model stem与原B27完全独立；新评估来源、完整前瞻assessment、实际B27终态资格证明及预算仍需未来单独定义与审核。此文件没有把B27 SOURCE_PASS或数学CI_PASS当作这些资格。

当前只读核得的原源码SHA：
| 原文件 | SHA256 |
|---|---|
| english_v6_b27_natural_weight.py | 43257d9f5349b4efb08b20c35575a75b181f277e769d9f7ff405f5778440c401 |
| english_v6_b23_prefix_peak.py | 4c61ffafb00fa75478424179b80d004eb4370e8050b1841caa47c51c669ccbae |
| english_v6_b24_b23_warm_start.py | 6259ae829a32655822350caee2608c9d162fdadc6366a3fa6ecf3fb6ab452cd7 |
| english_v6_b22_union_retention_continuation.py | 5afb83c3c0b840135c1d6e8bb0f97b2b4bb0988c3e4fdb87f1d6c1e347c2001a |
| english_v6_b17_b0_margin_pair.py | 7473e468ce55e96718b6599c7b9187377710258238b330758bec79c9d96e8df3 |
| english_v6_b27_training_readback.py | 9b952dc1787f5b88afd038b29e881b006fde5a3b64c7829074f765f26b829195 |
| english_v6_b27_producer.py | 5580609155e0504f69626d3d3b5a0b11887e0721ad9ac40359e99caed0ccac01 |
| torch_objective_draft.py | 1321eef19cf02d8ce8009e307d4fc384eeac7f59a57e2e07c05dbf79d273350a |
| no_event_reference.py | 034a4827fb9ff3091a5e23ea25e2eab0bc23ee7263bce644fd52741e6b80f969 |

## 9. 可主张与不可主张的范围

固定no-event数学已经通过纯虚构logits的路径/数值/gradient检查；实施图只说明如何保留旧训练条件，把一个自然suffix目标接入可独立回读的账本。

它不证明真实训练后的greedy事件减少、positive retention改善、B28优于B27或可晋级。实际新模型质量仍须其自己的前瞻SOURCE、PRIMARY、REPLICATION与工程链认证。当前没有修改任何science文件、登记、refs，也没有执行新训练或评测。

