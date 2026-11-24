# B22 原 TRAIN 保存数组机制分析（只读）

## 结论

不能把两条新增教师 margin loss 升高等同于“两条保护均失效”，也不能由 192/192 次总梯度裁剪判定分项梯度冲突。

- Santa gain 1.0 的六次已保存访问均无完整 `[1,2,3]` 事件；一直缺 token 2，属于尚未学得完整事件，未观察到先命中再丢失。
- Alice gain 0.75 的六次访问全部仍有完整事件。教师 margin loss 从 0.1996 上升到 3.9991，但 token 2 从教师 `[98,100]` 移到 `[100,101]`，token 3 区间收缩，固定帧 hinge 与事件指标出现明确分离。
- 这些记录均是对应更新前的原 TRAIN forward。最后访问分别为 step 175 和 176；没有对最终 step 192 checkpoint 新做推理，不能据此声称最终训练或独立质量结果。

## 可复核证据

原 models ZIP SHA256 为 `b5b81de69cdd80c102cdfcc8aa83dea3e685a22e99fb5dbff7613ead35fbda3b`。

`B22_TRAIN_ONLY_TRACE_ANALYSIS.json` 保留输入成员 SHA、1,844 个数组逐一 SHA 核验、192 步损失重建、同 original/gain/prefix 的首末配对、margin / event / prefix 统计。`B22_TRAIN_ONLY_ARRAY_READBACK.json.gz` 以无损 gzip 压缩保留每个原数组的派生读数。脚本 `analyze_saved_b22_train.py` 仅导入 NumPy 与标准库；无模型框架、无网络、无模型 forward/backward、无 optimizer update、无 DEV 读取。

使用纯 NumPy float64 CTC 动态规划重建原 objective；与原 float32 trace 的最大绝对差：总 loss 2.67e-5、synthetic CTC 3.14e-5、natural CTC 4.27e-6、positive retention 2.48e-7、ordinary ceiling（实际乘0.08后）2.86e-8。

## 损失量级与裁剪

原 trace 的 synthetic_ctc、natural_ctc、positive_retention 已包含各自权重，ordinary_ceiling 需要再乘 0.08 才能比较。

| 项 | 192步均值 | 总 loss 占比 |
|---|---:|---:|
| synthetic CTC（含原 prefix 项） | 16.33005 | 80.925% |
| natural CTC（含原 prefix 项） | 2.57541 | 12.763% |
| positive retention | 0.96699 | 4.792% |
| ordinary ceiling ×0.08 | 0.25126 | 1.245% |
| gain consistency | 0.05542 | 0.275% |

synthetic CTC 几乎全部来自 positive；合成 ordinary 的空目标 CTC 已近零。单个 `mero24-453-mero-slow` 的10个原 forward 占全部 synthetic-positive CTC 的49.407%；它对应的5次更新恰为 preclip gradient norm 最高的5次（694.72–721.60）。这支持“少数正例产生很强优化压力”的机制假设，但不是参数梯度因果分解。

全部192步被阈值5裁剪；缩放系数范围0.00693–0.51119，中位0.16520。全局裁剪缩放已相加的梯度，不会自动揭示各项方向是否相反。

从原 logits 解析计算 CTC 对输出 logits 的导数，不做任何模型反向传播：加权 CTC L1 每步均值2.0353、范围0.7804–5.4409。现有0.08 retention 的全部选中输出导数 L1 理论上界为0.16，ordinary ceiling 实际相应 L1 均值0.0560。因此存在明确的**输出空间尺度不均**；这并不等于参数梯度支配或冲突，因为没有模型 Jacobian、分项参数梯度、Adam moment 或 cosine。

在同一 ordinary 输出上，空目标 CTC 与 ordinary ceiling 都把 blank 相对非 blank 提高，二者的输出梯度符号相容。不能把这两项彼此冲突当作已证实原因。它们与 positive 的共享参数作用仍未知。

## 覆盖新增带来的稀释

Santa 原来只有 gain0.75 一条保护；加入1.0后，同一 group/original 的旧路径系数从0.08减半到0.04。Alice 原来的两条 gain 各0.04，加入第三条后各变0.026667。两条新增条件各只访问6次。

这是原设计明确允许的均值变化，不是实现错误；但“新增覆盖”等于“保护强度增加”并不成立。更改时必须显式区分覆盖与权重。

全部122个匹配条件的教师损失均值从12.9761降到10.7025，113条降低、9条升高。原贪心完整事件的首末转移为99 hit→hit、9 miss→hit、12 miss→miss、2 hit→miss。两条 hit→miss 的教师损失反而都降低，再次证明该 margin 指标不能代替事件指标。

## 更直接的 TRAIN 缺陷：首帧 token 1 与静音前缀

所有1,844个已保存输出的首帧都为 token1。对已知人工静音前缀的1,152个 base forward：

| 原 forward 类别 | 完整数组有完整事件 | 完整事件从人工静音首帧 token1 开始 | suffix 内有完整事件 |
|---|---:|---:|---:|
| synthetic positive | 280/384 | 277 | 3/384 |
| synthetic ordinary | 0/384 | 0 | 0/384 |
| natural ordinary | 98/384 | 98 | 0/384 |

这些数字采用原 `completed_word` 的规则，仅从原保存数组重新读出。不能改写为独立 DEV 表现，也不改变 B22 冻结 decoder。

旧120条 B9 教师路径没有任何一条把 token1 放在 frame0；新增 Santa 也位于声学区间。新增 Alice 的 B0 教师却把 token1 固定为 `[0,1]`，其后 token2 位于 `[98,100]`。对无前缀原音频，未读取波形因此不声称 frame0 必然无语音；但是它与所有有人工静音前缀输出的共同首帧模式一致，足以反对盲目增强该教师路径。

## 条件性最小 B23 建议

仅在 B22 按原独立晋级流程实际未晋级之后考虑。现在不执行、不改 B22、不新增预算。

**优先一个明确的 TRAIN 单因素：把已知静音 prefix blank loss 的时间 `mean` 改成 `max`，其余 CTC、ordinary ceiling、retention、数据、source/gain/prefix schedule、192步、架构与 decoder 保持原方案。** 当前公式是 `0.5 * softplus(blank_margin).mean()`；候选公式为 `0.5 * softplus(blank_margin).amax()`。继续使用原有 forward，不增加音频或模型调用。其目的只是在已知静音区间中给予错误首帧足够的训练压力，避免32/72/112个blank-prefix帧均值稀释单帧异常。

这是针对已观察 TRAIN 缺陷的可检验假设，不是已经验证有效的修复。它可能减少依赖首帧拼接的名义 positive hits；是否学会声学区间真实 token1，必须由新的预注册独立评估决定。不得把这一单因素与增大 retention λ、改 CTC reduction、剔除困难样本、调整 decoder 或阈值一起打包，否则无法解释结果。

若团队选择研究覆盖权重而非首帧问题，另一个独立单因素才是保持原120条件权重不被新增条件稀释，再给新增条件独立系数。不要同时做两条路线；特别不建议仅凭 margin 数字把全局 retention λ 直接放大10倍，这会强化 Alice 的异常起始教师区间。

新训练如获授权，可在既有 forward 计算图上保存分项参数梯度 norm/cosine 作为诊断，事先登记额外 backward 预算；当前证据不足以声称已经找到参数梯度冲突。不需要现在做新的训练实验来完成本报告。
