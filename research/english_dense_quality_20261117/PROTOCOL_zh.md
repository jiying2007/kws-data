# 英文逐窗交叉熵：最后一项固定、仅使用缓存的对照

2026-11-17（北京时间）。本协议在 dense 候选计算前冻结；前一项 temporal 实验已闭合为拒绝结果。

## 要回答的问题

相同 baseline、clean/10 dB train 数据、96×6 抽样和更新预算下，对全部原合法正例窗口使用均值交叉熵，是否比已完成的 aug96 概率均值损失取得可信的逐例事件改善？本项是本轮最后一个新训练 arm，不作 loss、轮数、seed 或阈值搜索。

前一项 hard-min 实验的持续性分数有小幅提升，但没有事件修复且新增背景事件。训练正例已接近饱和，不能宣称发现了梯度实现缺陷或证明 dense CE 会成功。当前研究重点仍是泛化与误触。

## 唯一算法改动

原正例项：`-log(mean(p_target))`。

新正例项：`-logp[:, label].mean()`。

背景项继续为 `-logp[:, 0].mean()`。每 bag 内先 mean，再将 clean/snr10×3 类的原六项等权 mean，不能改为按窗口总数平均或 sum。

对同一组给定概率，Jensen 不等式说明新正例项不小于原项；这不证明训练后事件质量更好。原概率均值损失对各窗口 CE 梯度的相对系数为 `p_i/sum(p)`，新项为 `1/N`，因此所有原合法窗口都参与监督。与 hard min 没有统一的收益优越关系。

## 冻结配置与复用

- 类映射：0 background、1 Hey Vosen `/heɪ ˈvoʊzən/`、2 Hey Mero `/heɪ ˈmiːroʊ/`。Hey 轻读，名字首音节重读；发音验证状态不变。
- 初始化：original baseline `6d7677b674f4a5e324aa916c5b777792b641fdfd61af8edc2f4e490275d600c4`。不从 temporal 或 aug96 继续训练。
- 主对照：aug96 `cdceab93cd6d3d36bff62980e8c7d2a95896ad32ecc88cae04fa232393e3e9e7`；失败 temporal 只作附加诊断。
- 49,283 个参数；1536→32 ReLU→3；原 mean96/scale96 字节一致。
- fresh AdamW，lr 0.001、weight_decay 0.0001，clip norm 1.0、seed 20261117、单 CPU 线程和原固定库版本。
- 原 96×6 draws 逐项相等；96 updates、576 training bag forwards。
- 训练只用原 clean 与 10 dB 的 train；原正例 3/4 个连续输出、完整 mel 支持与 16 行连续历史保持不变。
- 5 dB 只作重复评估，不进入训练。它已被查看，不能再叫独立新验证条件。
- 推理固定 threshold 0.8、confirm 3、cooldown 13。

三个已完成 artifact 分别固定为：原训练 run 43867764366、aug96 run 44021084918、temporal/cache run 44031666420。第三个 artifact 仅复用 5 dB 的 71 个有效条件缓存/roster、控制模型预测及其原始来源记录；不重复混音、前端、控制模型推理或旧训练。

## 零前端预算

| 新增工作 | 固定数量 |
|---|---:|
| embedding / ONNX forward | 0 / 0 |
| optimizer updates / training bag forwards | 96 / 576 |
| 新 head case passes | 264 |
| 新 ready 输出 | 11,847 |
| 新 TTS / ASR / phone / 音频混合 | 0 |
| 自动重试 | 0 |

264 条评估为 clean 117、10 dB 76、5 dB 71。三条件完整流式评估，不能只截正例训练窗评分。5 dB 原 77 条计划、6 条 train 溢出排除和 development 全 27 条的分母继续保持。

数值计算仅在公有 GitHub Actions 的唯一 main successor、attempt=1 执行，固定 predecessor `b4982d7a2f39c4b2647a86cbf96d3701a1123b6a`；共同监督期限 900 秒，部分结果留证，无自动重试。

## 原质量门保持不变

相对 aug96，clean、10 dB、5 dB 的任何原来正确 case 都不得回退；任何 case 不得增加 prefix、wrong-keyword、extra-same-keyword 或 weak-negative 事件；必须至少修复一个旧 10 dB development miss，且 5 dB 两个关键词均有 development 覆盖。不得仅因优于被拒绝 temporal 而通过。

候选训练完成即冻结，后续评估不得改变 checkpoint、阈值、确认或冷却规则。所有完整 logits、事件、权重身份、draws、阶段和计数都保存；独立审计验证缓存复用和状态机，不重跑数值模型。

若未通过上述门，本轮停止继续同池正例聚合/seed/轮数/阈值尝试，保留 aug96。真实录音、发音核验、近音负例、独立说话人及长时真实环境负例的覆盖问题不会因改损失而消失。全部正式训练准入、human gold、trusted negative、pronunciation verified、模型晋级及设备资格标志继续 false；词尾延迟与自然 FAR/FRR 未测。
