# 长期数据资产：一个目录、不可变版本、一个消费入口

## 现状与本次范围

现存 3 批、42 条、78.56 秒 Qwen3 合成开发音频是长期资产，不是临时 build 输出。
20 条有固定批次人工确认，18 条训练和 4 条开发回读有 ASR/连续性筛选。
它们已经被观察，均不得提升为封存资格数据。五种预置声线不是真人受试者。

首批原始身份由 Git `8e1de6081fe675fbf45afeaf09d7a644cc298f6c` 保存。
本次不修改 `datasets/**` 的 WAV、manifest、review 或 splits 字节；
只把来源、审核、切分约束从一次性导入程序提升为 catalog v2 的可检查配置，
增加消费收据与 CI。没有重新许可数据，也没有下载新语料或生成新录音。

## 单一权威与职责

`catalog.json` 是唯一资产入口：

- `sources`：来源/生成器家族、具体模型 revision/权重身份，以及分层权利记录
- `review_policies`：人工或 ASR 审核类别；ASR 工具/model/tokens 身份与筛选规则
- `split_policies`：原生开发角色和声线归属；已观察标记及资格禁用约束
- `datasets`：不可变批次 ID、manifest/review/splits hash、上述配置引用、存储类别
- `keyword_texts`：当前数据标签的词 ID/文字语义；不是任何训练器的 token vocabulary

批次内部既有 manifest/review/splits 仍是逐条事实和原始证据，不再复制一套 registry。
`tools.codex_assets` 是校验与消费导出的唯一实现；流水线调用它，不再复制数据集解析器或审核器。
文档解释状态与决策，不应被消费代码当成结构化配置。

每个批次 `content_id` 在核验时由 canonical JSON 的 dataset_id、manifest/review/splits
三个 SHA-256 计算，不再存一个容易漂移的重复哈希。消费端同时固定数据 Git commit 和整个
catalog SHA-256：内容身份回答“哪一批字节”，catalog/commit 回答“在什么审核/使用规则下消费”。
这不是数字签名，也不能验证维护者权利声明的真实性；消费者仍需信任并审阅所固定的提交。

## 版本与证据生命周期

1. 采集/生成进入未发布工作目录，记录原始音频、来源身份、生成/采集参数和权利证据
2. 审核按具体批次执行；`generated`、`ASR accepted`、`human accepted` 不互换
3. 发布一个新 dataset ID/版本，并把 manifest、收据、split 哈希加入 catalog
4. 修改音频、标签、切分或审核结论都发布新批次版本，明确取代关系，保留旧版原始身份
5. Git 的历史 catalog 和验证器一起 checkout 才能重放历史版本；当前工具只处理当前 catalog v2

一轮训练/分析看到的开发数据，不能通过改 `observed_development` 或改名 holdout 重新取得盲测身份。
本仓当前只接受已观察、公开获准的合成开发数据。未来的独立真人/资格资产需要单独受控存储和
经过审查的 schema/访问方案；不能只把当前 JSON 布尔字段翻转就接入。
正式产品证据仍由 kws-pipeline 对应治理通道决定，本数据仓不签发 shipping qualification。

## 新来源和派生数据的最小准入

本轮实现支持当前公开 preset-voice 合成资产，不预先实现所有未来格式。
新增来源时先增加具体 source/review/split 配置，再用明确 fixture 扩展必要的 verifier 能力。
不要为尚未到来的数据格式加 optional fallback 或兼容层。

新来源至少记录：精确 URL/版本、原始包/权重 hash、source_kind、generator_family、
代码/模型/参考声音/音频输出各自的权利依据与缺失项。`model_license=Apache-2.0`
不能自动推出音频统一商用授权；当前输出商业许可状态为 `not-established`。
科研开发消费与产品资格分开，不因暂缺真人设备而阻断获准的软件实验。

未来新版本的逐条 lineage 至少需要：

- 原声：source ID、文件和 PCM hash、真实帧数/采样格式、voice 或伪名 speaker ID
- 真实采集：真实 session/source recording/room/device ID；缺失写 unknown，不能编造
- 派生音频：父 PCM/content ID、变换程序 commit/hash、参数/seed、noise/RIR/AFE 身份
- 标签：文字与类别、审核方法/版本/收据、alignment 是否存在；没有时刻就不补伪造时刻

清洁原声作为 immutable parent；混响、混音、远场、AEC 等派生进入新版本，不能覆盖原声。
派生数据沿用父来源切分，但须重新检查文本完整性/削波/标签有效性；原声人工收据不自动批准派生音频。
保留生成失败/排除原因摘要与来源 hash，避免只见成功样本；有权保留的失败原音频才进入适当私有层。

切分按 source recording、speaker/voice、session 与派生祖先分组，不按 WAV 文件随机分。
同家族内同 voice 不能通过新 seed 或重命名绕过隔离；clone reference/embedding 也属于同声源。
当前 verifier 检查全仓 PCM/source 重复和 generator-family+voice 的角色一致性。
它没有做近重复检索、真实人物身份识别或跨生成器底座去重；不能宣称这些已完成。
跨生成器实验须明确留出 family；当前同一 Qwen family 的 A/B 只表示声线隔离开发集。

## 存储分层与保留

- 当前：小于 3 MiB 的 42 条合成 WAV 直接 Git 保存，clone 即可核验，不迁移、不复制
- 增长后：先测量仓库和批次体积/clone 成本，再选择 Git LFS 或不可变对象/Release 资产；
  catalog 固定位置、大小、hash、访问类别，并先实现验证实际下载字节的 materializer
- 私有真人、敏感元数据、未获公开许可的第三方原音频：受控私有存储；公开仓最多脱敏索引/hash/权利状态
- 从 Git 改成 LFS pointer 不是验证了 WAV；对象过期、权限不足或 hash 不符时消费必须失败
- 保留已发布版本和所需配方/依赖清单；撤回/删除请求按具体权限处理，不能以“永久资产”覆盖同意撤回或法律义务

当前 verifier 有意拒绝非 Git/非公开合成 storage，尚未实现远程下载或 LFS 消费。
不把临时签名 URL 当稳定资产 ID，不向公开仓上传凭据、私有音频或大模型权重。

## 可复现取回与重新生成

精确取回以已有 WAV+PCM hash 为准；相同 TTS seed 不能保证跨设备/依赖逐字节再生。
现有 recipe 保留脚本/权重/依赖身份，但历史没有 OCI digest，不能补造完整环境证明。
以后新生成记录 container digest/依赖锁、执行参数和实际产物 hash；新音频重新审核。

一次性 `import-existing` 已从当前命令行退役：它只会迁入历史 42 条并生成 catalog v1，
没有当前消费者需要它，也不能生成新的长期 schema。需要重放这次历史迁入时，
使用旧提交 `8e1de6081fe675fbf45afeaf09d7a644cc298f6c` 的代码和 README；
五个原始生成脚本及 hash 仍原样保存。当前不保留无用旧入口。

## 固定版本消费

在审核过的**干净 checkout** 中运行：

```bash
python3 -m tools.codex_assets verify
python3 -m tools.codex_assets export \
  --dataset qwen3-xiaowo-reviewed-development-20260928 \
  --expected-commit "$PINNED_DATA_COMMIT" \
  --expected-catalog-sha256 "$PINNED_CATALOG_SHA256" \
  > /tmp/kws-data-consumer.json
```

两个 PINNED 值来自实验配置预先固定的提交/目录身份，不是在训练时自动接受最新 main。
export 还核对所有消费依赖实际存在于 pinned Git tree；被 .gitignore 忽略的本地 WAV 即使 hash 正确、git status 干净也不能导出。
输出到仓外，以免收据自身让 checkout 变脏。可重复传 `--dataset` 明确选择批次；
人审和 ASR 批次不会自动合并或把 ASR 升格为人工真值。

收据包含 Git commit、catalog SHA、每批 content ID/三份证据 hash、source/family、
原生文字/类别/声线/seed、数据仓根相对 WAV 路径、真实格式/测量时长、review class、native split 和已观察状态；
每批导出 rows 另有 canonical JSON SHA-256。
它不生成 token、intent、语音活动或词尾时刻；下游训练适配器单独定义这些。
`frames/sample_rate_hz` 只是文件时长，不能作为词尾真值。

流水线应取回固定数据提交，调用该提交的 exporter，绑定收据 hash 到实验 provenance，
按明确 native split 消费。训练 adapter 只做模型专属映射，不能再执行一套不同的语料准入政策。
人审 20 条与 ASR 22 条可独立选择；下游不能因文件名有 holdout 就改变其 observed-development 角色。

## 当前可执行门禁与后续工作

CI 实际执行全仓 verify、反例单元测试及干净提交的 pinned export。
门禁覆盖字节/PCM/格式、来源与审核资产身份、标签/收据、原生切分、已观察/资格禁用、重复及路径边界。
还没有实现：新来源采集、声源跨家族近重复识别、真实事件 alignment、远程存储、封存资格或产品放行。

下一步最小价值是让现有 20 条人审批次进入一次受控软件实验，再决定补哪些真实数据/负流；
不要为了预想的巨大数据平台先增加无消费者的层、数据库或通用工作流。
