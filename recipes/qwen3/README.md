# Qwen3 固定生成配方

## 资产与环境

- 模型：`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`。
- revision：`85e237c12c027371202489a0ec509ded67b5e4b5`。
- `model.safetensors` SHA-256：`bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb`。
- tokenizer 权重 SHA-256：`836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258`。
- `qwen-tts==0.1.1` wheel SHA-256：`11a290d8dabc7ef91a90c54478c8ab19b3edb1d85c0882313721892bdc4af15d`。
- 历史隔离环境：Python 3.12，torch/torchaudio `2.11.0+cpu`，transformers `4.57.3`，accelerate `1.12.0`，scipy `1.18.1`，soundfile `0.13.1`。
- CPU、float32、eager attention、4 线程、逐条 torch/numpy seed；`language=Chinese`、`do_sample=True`、`max_new_tokens=120`，不传 instruct 或参考音频。
- 模型原生输出 24 kHz；对原始浮点输出使用 `scipy.signal.resample_poly` 转 16 kHz，再保存 PCM16。直接取回归档 WAV 才能得到已验证的固定字节；相同 seed 不保证不同软件/硬件版本逐字节重现。

## 冻结文本矩阵

每种声线生成“你好小窝”“小窝小窝”“窝小窝”“你好你好”；前两条为目标正例，后两条为近邻负例。正例的关键词 ID 分别是 1 和 2，负例为 null。

人工批次的脚本顺序：

1. `historical/kws_qwen3_probe_20260928.py`：Vivian 的“小窝小窝”，seed 1337。
2. `historical/kws_qwen3_cohort_20260928.py`：六条，seed 1338–1343，产生 7 条首批清单。
3. `historical/kws_qwen3_stage2_20260928.py`：十三条，seed 1344–1356，补齐 5 声线 × 4 文本，共 20 条。

扩展批：`historical/kws_qwen3_software_closure_batch_20260928.py` 固定 Vivian/Uncle_Fu/Dylan，每种声线两轮四文本，共 24 条，seed 4101–4124；经 ASR 和连续性筛选发布 18 条。开发回读批：`historical/kws_qwen3_software_holdout_20260928.py` 固定 Serena/Eric，共 8 条，seed 6201–6208，发布 4 条。

## 执行方式与边界

历史脚本按原字节保存，SHA-256 固定在 `historical-scripts.json`，使用容器内 `/work/build/...` 路径。复跑时先准备空的独立工作目录，把本仓只读挂载到 `/data`、工作目录挂载到 `/work`，在 `/work/build/qwen-tts-20260928/model` 放 exact revision 的全部模型文件；依次执行前三个脚本或独立的扩展脚本。部分脚本拒绝覆盖既有输出，不应在已归档批次目录执行。

示例，假设 `QWEN_WORK` 和 `QWEN_IMAGE` 指向已准备的工作目录与固定推理镜像：

```bash
docker run --rm --network none \
  -v "$PWD:/data:ro" -v "$QWEN_WORK:/work" \
  "$QWEN_IMAGE" python /data/recipes/qwen3/historical/kws_qwen3_probe_20260928.py
```

历史推理环境没有发布可验证的 OCI 镜像 digest，因此本配方是生成方法与脚本身份归档，不宣称完整容器级可复现。新生成音频必须重新做文件/PCM 哈希、文本筛选与独立审核，不能沿用旧人工收据。

## 迁入已有批次

`python3 -m tools.codex_assets import-existing --source-root /path/to/kws-pipeline --scripts-root /path/to/historical-scripts --dry-run` 可只读验证本次迁入输入。去掉 `--dry-run` 只适用于尚无 catalog 的新仓初始化；工具拒绝覆盖已发布 catalog。通常消费者只需运行 `verify`。
