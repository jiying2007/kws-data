# 两个轻量因果模型：固定四运行负结果归档

本目录保存已完成的 `causal-two-word-clip-learnability-v1` 研究，**不是新训练任务、数据 catalog 或产品候选**。
两个候选×两个seed，各1000steps，只有12条人工审核片参与训练。
完整结果见 [RESULTS.md](RESULTS.md)：训练可学习性存在，跨声线双词目标没有达到，停止追加训练。

现存数据目录、42条WAV、原始manifest/收据/splits均未修改或复制。
数据权威仍是根 `catalog.json` 和固定的 kws-data `2f9658ffa9568076ef547615c76861abed84f56e`；
本目录的 `archive-manifest.json` 只绑定历史研究证据字节，沿用 baseline archive 的 v1结构和 purpose，
没有建立另一套语料准入系统。

## 内容与完整性

- `spec.json`、`review-approval.json`：正式训练前冻结的研究协议及独立 AI 代码/协议审核；不是人工音频审核
- `revisions/preregistered-v1/`：第一次未训练协议与源码，保留后续身份校验修订的来源
- `historical-source/`：确切原型/训练/8项模型测试/准备/训练后分块检查/描述性特征审查代码
- `runtime/`：官方CPU PyTorch安装来源/hash报告、实际依赖lock、安装日志；没有wheel/venv/模型运行时
- `data/`：原生消费收据和42条数据/特征元数据；没有WAV或完整特征cache
- `feature-fingerprints.json`：42条父WAV/PCM、帧数、逐片feature JSONL和float32字节指纹
- `results/`：4个最后checkpoint的安全权重表示、逐片预测、计数、progress及invocation
- `results-recomputed.json`：标准库从已保存scores重算的分组指标，非重新运行神经网络
- `postcheck.json`、`feature-audit.json`：保留的训练后数值/描述性审查结果；CI核对它们的绑定和基本一致性
- `PIPELINE_ROADMAP_UPDATE.md`：根据用户允许全面优化的要求修订pipeline研究限制的具体指南

本归档的四份 `checkpoint.weights.json` 是**纯JSON有限float32张量**，不含pickle、可执行对象、优化器或第三方权重。
从本次自产 `last.pt` 用 `torch.load(weights_only=True)`读取，逐张量转JSON再转回float32后逐值完全一致；
每张量保留little-endian float32 SHA-256、shape和dtype，保留原始checkpoint SHA以对应原运行summary。
原 `.pt` 容器字节未复制到仓库；**保留的是完全相同的参数，不声称JSON与`.pt`文件哈希相同**。
标准库 verifier 重构float32 bytes并核对hash/shape/参数量，不能证明其重新训练必定逐字节一致。

## 不安装PyTorch的CI

```sh
python3 research/2026-09-30-lightweight-clip-learnability/verify.py
python3 research/2026-09-30-lightweight-clip-learnability/test_verify.py
```

CI验证：mandatory文件覆盖、字节/大小/hash、路径/符号链接边界、spec/code/review/输入身份、
四运行1000steps、42片原生角色/标签、固定logit0预测与逐词/分组计数、权重数值字节和参数量。
CI不下载数据、不安装PyTorch、不训练、不声称重新验证了保留的神经网络数值轨迹。
可变Git提交不是签名；修改研究证据需审阅新提交，不能只更新hash就宣称原始实验未变。

## 可选数值复现

数值复现是另行启动的软件实验，**当前CI和本归档没有重跑训练**。
先准备记录的Python/PyTorch/NumPy环境，固定官方wheel hash；
研究使用Python3.12.14/torch2.13.0+cpu，不等价于正式Python3.12.11及OCI训练镜像。

1. 取回固定数据commit并调用其唯一exporter；不得把当前main当训练数据版本
2. 用已记录的pipeline源码/编译身份重建C `kws_feature_dump`，确认binary identity；重提42片logmel32
3. 核对本目录 feature fingerprints 和逐片有效帧数；不要用文件尾伪造词尾
4. 安全恢复模型参数可用 `checkpoint_conversion.py` 的 `restore()`；只读JSON，不解pickle
5. 将状态装入 `historical-source/models.py` 对应模型，重算预测/分块一致性，与保存的scores和postcheck比较
6. 若重新训练，先把历史脚本的执行路径显式映射到当前独立工作区，保留原源码原文和差异；
   不覆盖此目录，不伪称路径/依赖不同的新运行就是当年的原始制品

历史脚本保留原始执行路径与source hash，因此不是可随意修改后仍保持相同身份的通用CLI。
原 full `.npz` 文件hash保留作历史身份；重建应优先核对稳定的逐张量/逐片字节指纹，
不要把归档容器封装差异直接解释成浮点数值变化。

## 权利与用途

这是用户授权保存的研究证据及自产小模型参数。来源许可/输出使用边界沿用根权利账本；
本次没有给Qwen输出、音频或模型参数新增统一商业许可，也没有授予再分发第三方权重的权利。
这里没有私有真人音频、模型下载token或凭据。`qualification_allowed=false`；
没有事件FAR/FRR、真实词尾延迟、板端CPU/功耗或产品放行声明。

同目录的后续迁移提案只是提案；没有下载17GB Mobvoi归档，也没有额外训练。
