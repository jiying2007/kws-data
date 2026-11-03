# 小批量 TTS / 三 ASR 数据准备预检（仅本地字节核验）

入口：python3 -B tools/kws_pilot_preflight.py。此工具只检查原有 bulk_pool.py 生成的计划和 bulk_autopilot.py 将要使用的素材，不运行模型、不生成 WAV、不确认声源同意、不写训练集，也不启动新的生成器。

## 预检解决的具体缺口

现有批量执行器按冻结计划中的 cell_id 顺序执行，--max-tts 16 不保证覆盖第二个声源家庭；更小的预算也不一定同时覆盖“你好小窝”“小窝小窝”、近音负例和普通负例。预检读取实际会执行的前 N 条，按类别报告覆盖。缺口返回退出码 2 和最小需要的计划前缀长度（如果冻结预算容许）。

三 ASR adapter lock 已声明 model_assets_sha256，但声明本身不能证明实际模型权重字节一致。新增私有 inventory 将三模型的真实文件哈希与声明核对，并重新验证 adapter 可执行文件。注意：实际文件字节一致不等于能够证明执行器运行时真的使用这些权重，不能自动授予 ASR 标签权威。

## 私有输入

- --sources：现有 source manifest，包含严格声源 lineage、参考 WAV、权利审核记录，来源音频必须存在于 --audio-root 并匹配 SHA 和 PCM。
- --plan：原生 bulk plan；重新计算规范 JSON 完全比对，不可私自换序。
- --model-root / --model-lock：Qwen Base 模型资源目录及完整锁；调用已有 bulk_runner.asset_lock() 逐文件核实。
- --adapter-lock：三 ASR 的模型 revision、model_assets_sha256、executable_sha256 等声明。
- --asr-assets：仅保存在私有机器的实际模型文件清单。schema 是 kws-private-native-asr-weight-inventory-v1，models 恰有三条：qwen06、sensevoice、whisper。每条包含 engine、model_id、model_revision、root（绝对路径）、files。每个文件包含 path（相对 root）、bytes、sha256。禁止符号链接、共享 root、未列出的文件和路径穿越。

每个 adapter 的 model_assets_sha256 必须等于以下规范对象（使用 bulk_pool.canonical 按键排序、紧凑 UTF-8 编码）的 SHA-256：

~~~json
{
  "schema": "kws-native-asr-model-asset-lock-v1",
  "engine": "qwen06",
  "model_id": "MODEL_ID_FROM_NATIVE_ADAPTER",
  "model_revision": "ACTUAL_40_HEX_COMMIT",
  "files": [
    {"path": "weights.bin", "bytes": 123, "sha256": "ACTUAL_64_HEX_FILE_SHA256"}
  ]
}
~~~

files 按 path 排序。模型权重 digest 不包含私有绝对路径，但执行前会从 root 读取实际文件并验证。单文件最多 8 GiB、单模型 12 GiB、每模型最多 256 个文件，且限制目录扫描数量。使用实际锁定的权重，而非填写示例占位符。

## 只读运行命令（不带 --execute）

~~~sh
python3 -B tools/kws_pilot_preflight.py \
  --sources /restricted/kws/source-manifest.json \
  --audio-root /restricted/kws/reference-wavs \
  --plan /restricted/kws/pilot-plan.json \
  --model-root /restricted/models/qwen-base \
  --model-lock /restricted/models/qwen-base-lock.json \
  --adapter-lock /restricted/models/three-asr-adapters.json \
  --asr-assets /restricted/models/three-asr-private-assets.json \
  --max-tts 32 --max-asr 96 --min-source-families 2
~~~

预检成功仅为 SOURCE_CONSISTENT_PILOT_PLANNED：声源/权重字节一致，预算内同时覆盖两个完整唤醒词、近音负例、普通负例，至少两个不同 source family 且存在 train 来源。所有 rights_externally_authenticated、spoken_labels_authenticated、training_admitted、qualification_allowed、shipping_approved 仍为 false。计划不完整返回 PILOT_COVERAGE_HOLD，或模型/权利数据不一致直接失败。

真实执行继续使用现有 tools/supervise_bulk_autopilot.py 显式运行。预检不能替代源权利授权、真实三 ASR 生成的回执、人审/鉴别标签、独立长流 FAR/FRR、最终 AFE 与 SSC305 物理验收。

## CI 范围

tests/test_kws_pilot_preflight.py 使用短人工 PCM、假模型文件和假 adapter，覆盖正常、-O、-OO 三模式：计划偏置、三 ASR 预算、真实字节/锁/源权利漂移、软链接和未登记文件，严格拒绝异常。不会加载 Qwen、推理、训练、下载权重或公开私人录音。
