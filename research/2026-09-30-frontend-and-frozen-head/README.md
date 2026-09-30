# 两份失败研究的追加证据：FFT定位与冻结编码器双输出头

本目录保留两个独立、已经结束的研究，按原历史字节归档；没有重跑前端、声学模型或训练，没有覆盖已有数值/CTC/蒸馏失败。

| 研究 | 已观察结论 | 仍未通过 |
| --- | --- | --- |
| [两个固定窗口FFT诊断](fft/README.md) | 两个400采样窗口的DC/预加重/加窗结果逐bit一致，差异首先出现在FP32 FFT；选定旧log误差精确复现 | 原1e-3门未修复；不是完整前端通过或板端性能 |
| [冻结P/R编码器+282参数头](head/RESULTS.md) | 唯一seed/100步对照中，P在8片已观察开发集的两个输出平均BCE均低于R | 双bit全对仅P4/8、R1/8，两严格门FAIL；P在人审TRAIN仍0/12 |

## 时间顺序必须保留

FFT原`run.py`仅执行这两个窗口，保存trace后在最终JSON序列化处因NumPy float32的TypeError退出1；`execution-status.json`保留失败。随后独立审阅的`summarize_saved.py`显式用Python complex双精度计算误差，仅从已保存数组恢复标量报告，没有再次执行前端。最终preflight还由review补入编译器/运行版本/执行脚本身份；`prepare.py`单独重跑不会重建该最终收据，不应称完全自动复现入口。Uncle_fu单帧Python的mel39相对原批处理差9.536743e-7，选定失败mel9仍精确复现；不能把选定窗口归因扩成全前端逐bit一致，也不能把double DFT解释成新的验收oracle。原6个fbank/11个splice及logits失败仍有效。

P/R保留早期测试的浮点逐bit断言失败、preflight运行时路径AttributeError、未执行旧lock以及最终审阅lock→唯一run→只读结果复核的完整顺序。locked-v1曾因git子进程与单process watchdog竞态被退回，修复并独立审阅后才在2026-09-30T14:04:51Z开始唯一真实run；44 TRAIN池先提取并fit归一化，两臂100步都结束后才推理其余Qwen（含8主开发片）。P为冻结预训练编码器，R为同架构固定随机编码器，R不是端到端从零训练。只有每臂140→2的282参数affine head训练；44 TRAIN为12人审Qwen+32固定FLEURS native TRAIN来源转录absence负例，每臂TRAIN-only归一化。开发A/B共8片、ASR22片均不参与训练/归一化。人审仅指合成音频审核方式；原报告“人声”在此指Qwen预置声线，不是真人受试者。

FLEURS是这一个配方的限定使用授权，原`admitted_for_training=false`收据原样保留；没有扩成全局准入或精确人工转录。P的人审TRAIN0/12、后缀激活、双关键词误激活不能被开发集BCE平均优势掩盖。本次没有重训、改seed/阈值/池化，仍无流式唤醒、连续FAR、盲测或产品放行。

## 保存内容与大小边界

- FFT：仅已审9份文本/标量白名单及原索引，排除原音频、PCM窗口、instrumented.so和大trace NPZ
- P/R：70份原归档逻辑文件，包含原日志/源码/许可/索引、数据身份、step/readout/review以及两个**derived-feature/小head NPZ**。不是全部纯文本，也不含完整donor权重或网络中间大轨迹
- 105,542-byte NPZ保存两臂44×140 TRAIN池、42×140 Qwen池、归一化和各282参数head；8,194-byte NPZ仅重复其中小head/归一化用于原独立制品身份。保留有限float32/dtype/shape检查，不含pickle/对象数组或音频
- 小UTF8文件原样保留。大JSON无损gzip；大NPZ原字节分片。`logical-files.json`绑定每个原逻辑路径/长度/SHA、编码及有序物理片段。两个完全相同的180,892-byte dataset历史版本共享同一个压缩对象，原逻辑身份都保留
- 每个物理blob≤48KiB，后续base64≤64KiB；装配/解压最大1MiB，NPZ仅固定成员/shape/有限float32，展开总量≤256,000bytes。没有重复大整包，没有原TAR副本

原P/R tar.gz为301,115 bytes、SHA256 `2e6e54fdab0e258e9e1dbf9046a928fc59a8ee9ce188e093133c25f7346140b6`，只读核验70普通成员均与原local archive一致；其原index SHA在`verify.py`固定。源代码的上游来源/许可证仍由`head/reference-source-index.json`和REFERENCE-NOTICE.md保存，不改称自产参考网络。

## 标准库核验与复现范围

```sh
python3 research/2026-09-30-frontend-and-frozen-head/verify.py
python3 research/2026-09-30-frontend-and-frozen-head/test_verify.py
```

核验物理/逻辑/原白名单哈希链、FFT标量误差与失败状态、锁/批准/运行/结果绑定、数据角色、100步、两严格失败门、NPZ白名单与有限数值，并以Python double对保存向量的小head读出作容差复核。它不运行Torch、donor、FFT或训练，不宣称逐bit重演浮点框架。历史源码/绝对路径供审计，不能直接运行来覆盖一次性执行记录。

原产物不得改写。新研究仍须单独固定协议；归档不放宽任何当前准入/数值门，也不构成商业许可或SSC305结论。
