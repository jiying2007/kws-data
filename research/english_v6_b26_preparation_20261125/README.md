# B26：自然负例权重单变量准备

本目录保留 2026-11-25 02:37:28.394587 UTC、任何 B24R1 KWS 输出产生前冻结的 TRAIN-only 提案原字节。`FROZEN_TRAIN_ONLY_PROPOSAL.json` 的 SHA-256 为 `4f2df013204098f1014e366def6e8b0b6b983fcaba73805db1204a415f116b8c`；`TRAIN_ONLY_EVIDENCE.json` 保留 1,844 份实际 B24 优化输出的哈希检查、CTC 重算、逐阶段事件和配对结果。原始 TRAIN 运行是 [45449190562](https://github.com/jiying2007/kws-data/actions/runs/45449190562)。

原 B24 轨迹建立了 202/384 次真正 positive suffix 事件，全部已知静音前缀保持 blank。与此同时，自然负例在五个完整 draw pass 中出现事件的观测数为 0、1、7、10、13；全程 35/384 次观测、47 个事件。重复的相同 original/gain/prefix 中，7 组自然负例从无事件变为有事件，其中 6 组总 blank CTC 仍下降。以上是训练期间变化权重下的保存观测，不是最终 checkpoint 的独立准确率。

唯一拟检验的变量是 `NATURAL_WEIGHT` 从 0.08 变为 0.16。候选仍从原 B24 使用的同一个 B23 checkpoint 开始，fresh AdamW、同一 192 draws、88 TRAIN original、122 retention condition、264 teacher array、其他损失和系数、43,876 参数、标签、gain、prefix、LR、clip 和 decoder 均保持原样。原 B24 exports 是不可变历史对照，不重新训练或推理 control/teacher。

科学实现复用原 B23 fit 的同一 code object，并为其建立只改变一个系数的隔离 globals；B24 export wrapper 同样复用。历史模块中的系数始终保持 0.08。B23 初始化字节从实际 B24 模型 artifact 中已经保存的 control exports 读取，B24 challenger exports 直接复制为新的历史对照。

完整原 B24R1 **PRIMARY `NO_PROMOTION`** 是唯一可能的前置失败条件。fail-only reader 保留原 R1 的完整 Git producer/source/每个 raw return 核验，只改变所需 verdict 与下游权限状态；SOURCE STOP、PASS、部分结果、摘要伪造、未知消耗都不成立。原 B14 阈值不变。

当前 opening 与 assessment binding 均惰性。没有 B26 workflow、people roster、PCM 登记、source 生成或预算开启；assessment validator 明确拒绝执行。需要等完整实际 R1 source 暴露名单已知后，另行实现并冻结全量 PCM/person 排除与新独立 assessment，才可能讨论执行。旧条件性 B25 不受此新 trial 名称或提案影响。

TRAIN 机制诊断预期是自然完整事件少于 47、含事件观测少于 35，同时不低于原轨迹的 202 次 genuine positive suffix 命中、无静音捷径回潮。这些诊断不替代独立晋级门槛；失败不自动触发继续训练或系数搜索。保存的 scalar loss 与整体裁剪范数也没有证明参数梯度冲突或 0.16 是最优值。

## 实际 R1 SOURCE 终态与审阅边界（2026-11-25）

原始 B24R1 SOURCE 运行 [45534020313](https://github.com/jiying2007/kws-data/actions/runs/45534020313) 已因 Parakeet 的第 9 次原生调用返回空文本而终止。失败行是 `b24r1-primary-84-mero-fast`：原始进程退出码为 0，保存的 stdout 只有单字节换行符；冻结解析器据此拒绝载荷并触发整批停止。R1 没有进入 SCREEN 或 KWS，也没有形成完整 PRIMARY `NO_PROMOTION`。

因此，上述 B26 唯一开启条件当前不成立，opening 保持 `enabled=false`，assessment binding 保持 `frozen=false`、`optimizer_allowed=false`。不得把 SOURCE STOP 改称 PRIMARY 质量失败，也不得用另一项未来评估的结果替代原 B24R1 条件；剩余调用不恢复，不重试，也不自动开启 R2、B26 或任何新 panel。

本审阅分支同时保留[未来空返回隔离提案](../future_source_empty_quarantine_proposal_20261125/README.md)，仅用于用户审阅完整的依赖和选择。该提案明确未批准、未注册、未激活，所有执行权限均为 false；它没有改变本目录的冻结提案、开启条件或任何原始试验的终态。本分支不选择人群、不生成音频、不执行模型，也不构成新协议获批或晋级完成。
