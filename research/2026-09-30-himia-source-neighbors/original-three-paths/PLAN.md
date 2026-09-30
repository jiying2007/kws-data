# 固定方案：HI-MIA-CW 已观察近邻片段回读

状态：仅方案与音频头核对；未执行模型推理。待独立审阅及父线程执行决定。

## 问题与证据范围

对同一历史 15 人 / 7,006 片，比较冻结 C、sherpa INT8 encoder 与完整 cFSMN 在各自固定解码点的**源标注近邻片段事件**，联查 Qwen 正例，判断“高命中是否伴随近邻触发”。不训练、不扫参数、不自动校准阈值，不修改现有文本监督准入门。

这些真人片段已被历史开发观察，不能称新 holdout。官方文本是弱来源标注，未经本次逐片人工确认；近邻片段触发不自动等于转录已证实的错误唤醒。另一个历史 TRAIN 的 40 片中文 ASR 筛选失败，只说明该固定文本监督配方未通过，不证明本语料全部发音错误，也不允许把失败片换进本集合。本次不使用 ASR 输出筛选、替换或重新标注这 7,006 片。

不报告家庭连续 FAR/hour、真实词尾延迟或产品通过率。7,006 个短片重置不能当作连续流；总时长只作输入覆盖和运行成本分母。不同前端/先验训练/解码点不能形成同资源或同 FAR 模型排名。

## 固定输入与校验

源目录：`/workspace/shared/kws-real-negative-preflight`。

- `slr120-data.tgz`：550,623,081 bytes；SHA256 `5de169ac1931a0eab46546c477965edad23069f0a2c0c4eb14e798814f83c91a`
- `slr120-resource.tgz`：SHA256 `8628c75e6ec534a9458229d5ac5267e696b953b5f905c26f889187f6de1ee4e4`
- `transcription.original.txt`：SHA256 `4e2ab8e9bd098e066c0f4dbe86e77325957a63f719eeb74d78124c0bebe8c13c`；16,343 唯一记录；不凭网页常见总数覆盖本包计数
- 复用历史规则：按 `SHA256('kws-himia-hardneg-20260928:'+四位speaker ID)` 排序，前20人 TRAIN、其余15人已观察 development。固定为 0004/0005/0007/0008/0010/0013/0014/0015/0017/0021/0025/0026/0027/0032/0035
- 所选 recording 按字典序连接，每条后加 LF，SHA256 `5011b2faa1eb5bbb3212c65a48a88f384748af1b034a7517cb910d5e010bcc5a`
- 只读头检查已确认 7,006 片均 PCM16/16k/mono：161,688,053 帧，10,105.5033125 秒（2.807 小时），最长3.465375秒；所选 WAV payload 323,684,370 bytes。检查记录：`kws-himia-readback-header-audit.json`，不是逐片 PCM 完整性收据
- 实现时顺序扫描归档，拒绝重复/缺失成员、非普通文件、越界路径、未知角色或格式差异；逐片完整读取校验真实帧数，并记录 WAV/PCM SHA256。禁止静默丢片或重采样。三模型必须消费相同逐片 PCM 身份
- 官方来源 `https://www.openslr.org/120/`，既有权利记录 CC-BY-SA-4.0；本次仅在已有本地源包上研究。公开音频再分发、许可清理与训练准入不是本次动作
- Qwen 固定 data commit `2f9658ffa9568076ef547615c76861abed84f56e`，catalog `27b590cdc325d48d2cf5e1293b8e431c815382be0219bc2e069d4e08896ee391`，现有导出收据 `8e4d5c13cc5694e813ef6dc58bedbe34d2b893721283f1945d6d944ca38570b5`。保留42片原生角色/审核类型；20正例、22混淆片，均已观察

## 三条冻结路径

1. **C**：沿用 `/workspace/shared/kws-data-consumer/build/observed-readback/readback-spec.json`（SHA256 `e2c97b082792ed65b5b5dfeab4266d623b7f05c3793d95633ed8879599505809`）、manual build receipt、`kws_wav`、原模型和关键词包；模型 SHA256 `ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402`。逐片新进程维持冷重置语义，不修改阈值/EOF策略
2. **sherpa**：`/workspace/shared/kws-sherpa-int8/config.json` SHA256 `9850355f89c74dbe6613870355287e5d3f270fc77188d99749b46ac8356d01cf`，downloads清单 `d3e7dfdb46f349fade7d7e136265796f2b7ddde17b6951a317f565e9db9dfffd`；encoder `017af32f2c0138f931d05fbc009ee864295e910aff304f77d2f563815fc834fb`，其余组件保留FP32。固定2线程、320samples、threshold0.25、score1、max_active_paths4、trailing_blanks1、无额外静音，input_finished并记录EOF来源事件；每片新stream，事件后原样reset
3. **完整 cFSMN**：`/workspace/shared/kws-cfsmn-baseline/baseline-spec.json` SHA256 `26016a32d42ff25b77dbf011064e07cc52abe5dfec0c9a2a994c766734300831`，2599输出头，权重 `d02b09c34f4a8bbb06f0dd1bf5eb58db3395eb7f1fd15c3625fe09d3a2492233`。本次单一路径固定发布者声明的 Hamming，不根据开发成绩选窗；Povey不加入。保留300ms feed、threshold0、beam3/20、prune0.05、duration5/250、interval50、无额外EOF flush；缓存/解码器逐片重置。PyTorch1线程，既有CPU环境。保留已披露的上游suffix检查/每chunk首事件后停止解码语义；这是宽松未校准诊断点

执行前重算所有所引配置、权重、token/关键词、源码、二进制和运行时版本身份，与已有收据比对；将实际适配runner源码/hash加入本方案执行收据并通过独立审阅。不得把导入历史脚本变为触发其顶层全量推理；复用已审阅函数时显式控制入口。模型/解码修改使本方案失效，需新方案。

## 简单逐片执行与预算

- 一个薄runner，三条独立顺序执行路径；无需新通用框架。先流式扫描tar一次，将所选片安全写入本次独立目录并形成身份清单，解压上限350MB；之后每次只加载一片，推理后释放，不预加载整个7006集合。不改原包或数据仓
- 每模型先回读原始Qwen42，再按固定recording顺序回读7006，最后重复Qwen42核对状态/漂移。每路径7,090次，共21,270次；两次Qwen只作复现检查，不作为新增样本合并计数
- 先做runner的空片/损坏输入拒绝、计数、跨片reset及事件序列保留测试；Qwen初次事件/命中须与对应已保存原始基线一致才进入7006。不一致时停止定位，不搜索阈值
- 每路径CPU累计上限900秒、墙钟1200秒（含模型启动、读取、子进程与输出）；三路径累计CPU2700秒/墙钟3600秒。单片墙钟上限10秒，RSS保护上限1GiB/路径；以上是本次主机安全上限，不是SSC305或产品RAM限制。解包/身份检查另限墙钟300秒
- C必须累计子进程CPU，不能仅用父Python process_time；记录process+children CPU、wall、峰值RSS、/proc I/O或可用计数。这里只是研究运行开销，含读盘/启动且批块不同，不当纯模型基准。无GPU、下载、安装、后台并行训练

## 输出与停止条件

按路径逐行保存：源ID/弱文本/speaker/原生development角色、文件/PCM身份、输入帧数、完整关键词事件及输入可用位置/EOF标识、耗时、状态。不得虚构人工审核、精确转录或词尾。聚合总事件数、触发片数/7006、按关键词/15人/官方文本/源速度标签分组；没有独立性假设的总体显著性或市场领先结论。Qwen独立列20正例命中、错词/重复事件、22混淆片触发和原生分组。

源包/hash/输入覆盖不符、NaN、未知关键词、状态复现失败、错误片或超预算：停止对应路径，保存原因与已完成分母，不跳片、不自动重试换模型。发现同PCM或缺失记录必须显式报告，不能悄悄减少分母。高触发率本身是要保留的结果，不作为提前停止/调阈值理由。三条全量完成后结束；后续校准、人工复核触发片或新数据扩展需另案，不由本结果自动触发。
