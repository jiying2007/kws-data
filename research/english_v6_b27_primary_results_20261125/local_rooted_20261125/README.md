# B27 完整原件终态：NO_PROMOTION

本目录保存一次已完成的本地只读认证。它重新打开原 PRIMARY、TRAIN、SOURCE、SCREEN、五路观察器和历史资格证明，未启动新的模型、ASR、来源生成或优化器。

原评分运行：[45627775991](https://github.com/jiying2007/kws-data/actions/runs/45627775991)，head `dce5b169163039091afcc5ab62beed3211d37552`。完整 59 件原始证据共 742,469,141 bytes；认证前后的完整 inventory 相同。独立 reader 验证了 996 个原始评分返回及 178,776 帧，重算全部来源入场和原 B14 决策，结论为 **NO_PROMOTION**。B27 不能进入 replica、工程验证或模型安装。

## 认证身份

- 完整原 reader canonical report：`29c443c060cd80caaba29d4f59e63f976ad6c75f68bf82ae55f403f99c41ba4b`。
- 独立导入根适配器 wrapper：`4bd55367437ffa2680622799b629abe63a5a58492dce1378405fe8a432345585`。
- 原质量 reader 保持 `87bc31725380cbf70d70d6b46929a724dc6bff656d421c5ae63b54c0724e31aa`。
- 新适配器为 `9de0cb1e8619d6d3127c09cc279d2480621663feabf1208535f9099e2c271518`，使用一个完整、已认证的原 producer 模块根，保留原代码、路径、散列和只读守卫。

原独立延续运行 [45643969797](https://github.com/jiying2007/kws-data/actions/runs/45643969797) 仍为失败：两个解包目录的 Python 模块缓存曾混用。本目录不是该运行的重试，也不把它改写为成功。当前认证明确记录为 local-only；另行公开的只读复核应保留自己的运行身份。

## 完整保存内容

| 文件 | 内容 |
|---|---|
| `PRIMARY_LOCK.json` | 原 PRIMARY run、原 producer/result/consumed artifact 及固定 reader 输入身份 |
| `PRIMARY_READBACK.json` | 完整原 reader 的 canonical 报告 |
| `ROOTED_PRIMARY_READBACK.json` | 独立适配器 wrapper 与完整内层报告 |
| `ORIGINAL_RESULT.json` | 原 result ZIP 中结果成员的精确原字节 |
| `ORIGINAL_INVENTORY.json` | 全 41＋18 件原始证明、运输身份、原始 API bytes 和审查文件 pins |
| `ROOTED_TERMINAL.json` | 本地只读过程退出码 0、零新调用、原失败运行身份和终态收据 |

这份失败认证只能作为新候选资格前置；它不提供新候选质量证据。B28 仍必须在优化器之前冻结新名单，使用独立原件完成 PRIMARY 和 replica，并通过原有工程与软件安装验证。
