# Synthetic192 SDK 离线装配契约

当前没有实际成功的新 synthetic192 training/fullPCM 产物，因此默认计划 `enabled=false`。只交付源码，不生成候选 SDK，也不把已有 coverage 或失败 teacher 包冒充新结果。所有动作仅处理保存的字节。

## 输入

1. 已验 accepted coverage SDK ZIP：13,148,833 bytes，SHA256 `912ed6143a794bcee8cc3b10c53a61eac1240330d4f319ea81daf33981175587`。这是离线交付 ZIP；其原始 GitHub 来源是 coverage fullPCM run `44049954828` / artifact `14313603148`，不是为离线 ZIP 虚构新 artifact ID。
2. 实际 `english-synthetic192-training-<runid>` ZIP：真实公开 main push、attempt=1、成功 run/artifact API；retained 状态、全部 10 gates、成功 supervisor、实际新 NPZ/BIN 与完整保存 references。
3. 实际 `english-synthetic192-fullpcm-<runid>` ZIP：同一候选，14 条 PCM × native/ARM 28 个进程；新 head 自身 refs、成功 supervisor、零新增编译/训练/Python-ONNX/TTS-ASR。旧 coverage 的 10 条 traces 不能代替。
4. 六份原始 API 响应，独立目录内固定文件名：`coverage-run.json`、`coverage-artifact.json`、`training-run.json`、`training-artifact.json`、`fullpcm-run.json`、`fullpcm-artifact.json`。计划保存每份原始 JSON 的 bytes/SHA256，验证内容中的 run/head/attempt/public repository、artifact digest/size/workflow_run。保留原始响应，不规范化后冒称原 bytes。
5. 已固定的两个纯字节解析 helper，使用仓库原 `automation/english_coverage_fullpcm_once.py` 和 `automation/english_hardneg_increment_once.py`；源码 bytes/SHA256 必须匹配，不导入档案中的未知代码。

所有 ZIP 512 MiB 上限，展开 1 GiB，上单文件 128 MiB，拒绝重复项、路径穿越、符号链接、加密成员。首次审阅所有输入时验证闭合 manifest；包内保留新 fullPCM 完整档案内容和训练的必要可核对子集及原 manifest。

## 输出

完整保留两套 `sdk-*/include`、`lib`、CMake、`bin`、许可证及诊断消费者；仅两处 `sdk-*/models/causal-head.bin` 被实际新 head 替换。原 stdin CLI 与 trace-linked consumer 的已执行范围分别说明。CLI/诊断二进制保留可执行 mode，ZIP Unix mode 一致。

保留所选 NPZ/BIN、原归一化身份、10 个质量门与原始报告、完整保存 reference stream、最终 fresh roster、新 fullPCM 的 14 条 PCM/refs、28 traces/进程记录，以及原 API 响应。独立 `verify_package.py` 重新进行字节验证、格式解析和保存 trace 数值比较；不运行模型。

README 中文说明冻结 embedding+head 的合成研究范围，16 kHz mono s16le 连续流、0 背景/1 Vosen/2 Mero、320 样本建议块、首次 ready 2.2 秒、固定阈值确认冷却；197,908 bytes 仅为 head BIN。未宣称独立真实 FAR/FRR 或物理 SSC305 通过，也不以新人工/设备测量阻断当前软件交付。

## 实际产物到齐后的命令

以下路径占位符必须替换为保存的真实文件，不能填入预期 run 或虚构 hash。`--bind-plan` 会先检查全部输入和质量门再写计划；失败不生成计划或 SDK。

```sh
python3 -I -S -B automation/package_english_synthetic192_sdk.py

python3 -I -S -B automation/package_english_synthetic192_sdk.py --bind-plan \
  --coverage-sdk /absolute/accepted-coverage-sdk.zip \
  --training /absolute/actual-synthetic192-training.zip \
  --fullpcm /absolute/actual-synthetic192-fullpcm.zip \
  --metadata-directory /absolute/raw-api \
  --helpers /absolute/kws-data/automation \
  --write-plan /absolute/new-enabled-sdk-plan.json

python3 -I -S -B automation/package_english_synthetic192_sdk.py --preflight \
  --plan /absolute/new-enabled-sdk-plan.json \
  --coverage-sdk /absolute/accepted-coverage-sdk.zip \
  --training /absolute/actual-synthetic192-training.zip \
  --fullpcm /absolute/actual-synthetic192-fullpcm.zip \
  --metadata-directory /absolute/raw-api --helpers /absolute/kws-data/automation

python3 -I -S -B automation/package_english_synthetic192_sdk.py --assemble \
  --plan /absolute/new-enabled-sdk-plan.json \
  --coverage-sdk /absolute/accepted-coverage-sdk.zip \
  --training /absolute/actual-synthetic192-training.zip \
  --fullpcm /absolute/actual-synthetic192-fullpcm.zip \
  --metadata-directory /absolute/raw-api --helpers /absolute/kws-data/automation \
  --output /absolute/new-sdk-directory --zip /absolute/new-sdk.zip
```

新目录和 ZIP 必须尚不存在；验证不通过会移除本次调用创建的未完成输出，已有用户文件不覆盖。最终报告只反映已保存证据；源码测试不被表述为新模型或 SDK 实测。
