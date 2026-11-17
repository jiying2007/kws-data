# 英文 coverage 质量提升模型：实际结果与交付身份

## 已完成的质量提升

公开 Actions [44042856615](https://github.com/jiying2007/kws-data/actions/runs/44042856615)
首次执行在 `4f4d3f15214613c0c3a8f92c871da0b419b3f7f7` 完成。
原始 artifact `14310138100` 为 50,040,751 字节，SHA256
`56b7e9181cc0e8662b00dd1344ef414a1466a668428fec8bc709b600ba8e97e1`。

候选通过原先冻结的研究保留门。在相同固定阈值、三次连续确认、13 hop 冷却和逐样本比较下，
修复一个原 aug96 的 10 dB Hey Mero 开发漏检；所有已正确案例无退步，任一案例的
弱负意图、前导、错词及多余事件均未增加。

| 已观察开发条件 | Hey Vosen：aug96 → coverage | Hey Mero：aug96 → coverage |
| --- | ---: | ---: |
| 原干净条件 | 3/3 → 3/3 | 8/8 → 8/8 |
| 原 10 dB 干扰语音 | 2/3 → 2/3 | **6/8 → 7/8** |
| 原 5 dB 干扰语音 | 1/3 → 1/3 | 5/8 → 5/8 |
| 新 8 dB 配对条件 | 1/3 → 1/3 | 7/8 → 7/8 |

这些是已暴露合成意图样本在有限构造上下文内的事件匹配数，分母不是独立真人 FRR，
弱负例零事件也不是自然连续流 FAR。最终目标发音为 Hey Mero `/heɪ ˈmiːroʊ/`、
Hey Vosen `/heɪ ˈvoʊzən/`，Hey 轻读、名字首音节重读；当前意图标签未获得人工发音 gold。

## 实际训练与证据范围

新增训练侧 5/15 dB 噪声配对共有 100 个计划条件，96 个通过 PCM headroom 检查；
四个超幅条件直接排除，没有削波、补位或重试。训练只消费原 train 角色，保持原初始化、
normalization、loss、optimizer 和 96 次更新。开发侧 27 个新 8 dB 条件在候选权重冻结后构造，
全部有效。新条件不增加独立原始录音或真人数量。

实际完成 96 次 optimizer 更新、576 次 training forward、2,577 次新增 embedding、
5,154 次 ONNX forward；复用 1,968 个前导 embedding。新增 TTS/ASR/phone 调用及重试均为零。
这份结果登记与封装没有再次训练或执行模型。

`INDEPENDENT_AUDIT.json` 绑定原训练包、aug96 包、temporal 缓存包和此次质量包，
在外部指定真实 run/head 后通过。审计独立重建 PCM 混音和排除结果，核对缓存、抽样顺序、
冻结顺序、保存的 softmax/事件、模型字节与 C 原始输出。它不重算梯度、head 矩阵乘法或新
suffix embedding；`VERIFIED` 不代表人工标签或独立声学资格。

## 新权重和可复现封装

| 文件 | 字节 | SHA256 |
| --- | ---: | --- |
| `models/coverage-head.npz` | 199356 | `75c1d13203f77fcec9b66013bd6fc8c1979cb3397207931eaf31241eeb396a35` |
| `models/coverage-head.bin` | 197908 | `d3e39adf72495034de0405abb997b76744885f91868eccfa97ba09b929bc63e4` |

六个有限 LE float32 数组与 `EHKWS01` C ABI payload 已独立逐字节比对，见
`HEAD_ABI_REVIEW.json`。类别顺序是 0 background、1 Hey Vosen、2 Hey Mero。

`PACKAGE_INPUTS.json` 是实际可执行输入身份，`PACKAGE_MANIFEST.json` 与
`PACKAGE_VERIFICATION.json` 对应已生成的 51 文件研究包。两条不运行模型的验证入口都通过；
manifest SHA256 为 `97ad6c8e2e22b763bfc660d90292a3605ed8fe5201e2e00ae6f74c32c381cc62`。
原 packager、runtime、frontend、license 和旧 SDK 字节没有修改。
重建命令见 [package/README.md](package/README.md)，将 `--pins` 指向本目录实际
`PACKAGE_INPUTS.json`，并按其中四个输入的精确身份取回原文件。

研究包包含 Python/ONNX 前端、C head 与证据；完整安装 C 前端库沿原 installed SDK 独立绑定。
本次已保存的 Native、ARM/QEMU 头部比对各覆盖 291 cases、16,808 embeddings、
12,443 ready 输出和 80 events，样本时钟、ready 与事件完全一致；最大 logit 误差
`9.541357421483099e-06`、概率误差 `1.4308039245625181e-06`。

新权重的完整 PCM → C 前端 → head → event 验证独立登记在
[fullPCM 协议](../english_coverage_fullpcm_20261117/PROTOCOL_zh.md)。该验证消费冻结的新 head，
包含本轮修复的 Mero 案例；其实际结果不能由这里的 cached-head 比对推定。
原包的历史 `NOT_RUN` 字段保留，后续完整链证据按新 run/head 作为补充交付。

## 剩余漏检与下一步依据

只读保存分数分析见 `SAVED_SCORE_DIAGNOSTICS.json`。原 10 dB 剩余 Vosen 漏检的峰值目标概率
约 0.694124，Mero 约 0.508448，均没有任何帧达到固定 0.8。因此减少确认帧数不能修复
这两个案例。三个噪声条件中的十个开发漏检里，九个峰值低于阈值，一个 Vosen 仅有一帧
达到阈值。当前观察到原片段 EOF 后 0.32 秒且没有真实词尾标注，不能用这组记录声称声学
词尾延迟或完整长流漏检率。

同池 loss、seed、threshold、epoch 搜索保持关闭，不继续用这些开发分数挑选参数。
后续质量增量应来自新原始说话人/背景、最终发音证据与独立评价，优先补 Vosen 的发音及
口音覆盖。新声线、噪声派生和真人来源分别计数；人工 gold 缺失时保留机器弱标签身份。

独立 FAR/FRR、词尾回调 P95、真实 SSC305、设备实时性与 24/72 小时稳定性尚未完成，
全部正式训练准入、发音认证与晋级 flags 仍为 false。完整链软件验证通过也不改变这些边界。
