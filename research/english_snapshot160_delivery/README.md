# 英文 KWS 研究包：解压、运行和读取事件

本说明和 `read_c_trace.py` 是外层交付工具。当前准备源码不包含新的实际训练成功或全链验证成功结论。只有对应真实产物的 `RESULT.json`、权重身份和完整清单均回读通过后，才能把那些产物作为研究候选交付。

Python 推理包与 C 全链产物有各自严格的文件清单。保持两个内层目录原样；把本 README、解析器、虚拟环境和新运行输出放在外层。直接把新文件放入 Python 内层包会触发 `UNLISTED_BUNDLE_FILE`。

## 1. 对应目录和用途

| 原始产物解压位置 | 实际入口 | 用途 |
| --- | --- | --- |
| 训练 artifact 的 `english-snapshot160-embedding-research-bundle/` | `english_final_embedding_infer.py`、`manifest.json`、`MODEL_CARD.md` | 已保存 head + 固定 ONNX 前端的 Python 推理；另含仅 head 的 C 重放源码 |
| 全链 artifact 的 `snapshot160-fullchain-result/` | `frontend-native`、`frontend-arm`、`causal-head.bin`、`RUN_C.md`、`RESULT.json`、`MANIFEST.json` | 固定 C 前端 + 同一已保存 head；ARM 运行环境是 host QEMU |
| 全链 artifact 的 `snapshot160-fullchain-evidence/` | 原始输入 ZIP、API 元数据、工具版本、退出记录 | 执行来源与终态证据 |

最终外层包可以把前两项完整目录分别命名为 `python-bundle/` 和 `c-fullchain/`。以下命令使用该外层布局：本 README 与 `read_c_trace.py` 在外层根目录，`python-bundle/`、`c-fullchain/` 是其子目录。重命名目录不改变目录内相对路径及文件字节。若直接解压原始 Actions artifact，请先进入表中对应的原始目录，并据实调整命令路径。

两种推理入口必须引用同一真实训练结果的权重。Python 的 `models/causal-head.bin` 与 C 的 `causal-head.bin` 应逐字节相同。不要用仅含 backbone 的旧 C 产物冒充带 head 的全链产物。

## 2. Python：16 kHz mono PCM16 WAV

使用研究 worker 的 Python **3.12**。安装依赖发生在准备环境时；推理过程不下载模型。`requirements.txt` 固定 `numpy==1.26.4`、`onnx==1.17.0`、`onnxruntime==1.20.1`，无需 Torch。不要使用 `-S` 运行实际推理，否则虚拟环境的第三方依赖不可见。

从外层根目录执行：

```sh
cd python-bundle
python3.12 -I -S -B english_final_embedding_infer.py --bundle . --verify-only
python3.12 -m venv ../english-kws-venv
../english-kws-venv/bin/pip install -r requirements.txt
../english-kws-venv/bin/python -I -B english_final_embedding_infer.py \
  --bundle . --wav /absolute/path/input-16k-mono.wav \
  --output ../python-result --execute
cd ..
```

`python-result` 必须尚不存在，且位于内层包之外。查看其 `RESULT.json` 中的 `events`、样本时钟及 EOF 状态；`frames.jsonl` 保留逐 embedding 的 logits、probabilities、ready 和 keyword（0 表示无事件）。`--verify-only` 只验证文件身份，不运行模型。

## 3. C：原始 PCM16LE，不是 WAV 容器

输入为无文件头的 **16,000 Hz、单声道、有符号 little-endian int16**。C harness 不读取 WAV 格式信息；把 WAV 整个文件改名成 `.pcm` 会把容器头也当成音频，结果错误。以下命令仅从已经合规的 WAV 原样取出 PCM，不重采样、不裁剪、不归一化：

```sh
python3 -I -S -B - /absolute/path/input-16k-mono.wav input.pcm <<'PY'
import sys, wave
from pathlib import Path
with wave.open(sys.argv[1], 'rb') as source:
    if (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype()) != (1, 2, 16000, 'NONE'):
        raise ValueError('Input must already be mono PCM16 WAV at 16000 Hz')
    frames = source.getnframes()
    raw = source.readframes(frames + 1)
    if len(raw) != frames * 2:
        raise ValueError('Truncated WAV')
with Path(sys.argv[2]).open('xb') as output:
    output.write(raw)
PY
```

预编译 native 程序目标为 **Linux x86-64**，不是 Windows/macOS 可执行文件。当前编译命令动态链接系统 C/math 库；最终实际二进制须以 `readelf -l -d --version-info` 的只读结果为准。之前同工具链的 mel-only 二进制使用 `/lib64/ld-linux-x86-64.so.2`、`libc.so.6`、`libm.so.6`，最高需求版本为 GLIBC_2.34；这不替代新全链二进制的实际依赖核查。

ARM 程序由 `arm-linux-gnueabihf-gcc -static` 构建。它不应依赖 ARM 动态加载器或额外 ARM sysroot；在 Linux host 上需要 `qemu-arm`（Ubuntu 的 `qemu-user` 包）。QEMU 自身使用 host 的运行依赖。该路径不代表 SSC305 板端部署或实时性通过。

从外层根目录运行，使用不同且尚不存在的 trace 文件：

```sh
chmod u+x c-fullchain/frontend-native c-fullchain/frontend-arm
cd c-fullchain
./frontend-native ../input.pcm ../native-trace.bin 1,1279,257,4096 causal-head.bin
qemu-arm ./frontend-arm ../input.pcm ../arm-trace.bin 1,1279,257,4096 causal-head.bin
cd ..
```

`chmod` 只恢复可能被解压工具丢失的执行权限，不改变二进制内容。chunk 参数单位是样本数，循环使用这些块大小；它不添加音频。两个命令每次都会真实运行 C 模型，不属于单纯验证文件或解析 trace。

若 host 的动态库不兼容，可从内层保存的源文件在适配环境中重新编译；这是新的二进制，不能继承原二进制的数值验证结论。完整源文件应包括 `source/{fixed_frontend_mel.c,fixed_frontend.h,frontend_replay.c,english_causal_head.c,english_causal_head.h}` 和 `generated/{fixed_models.c,fixed_models.h,export_manifest.json}`。可重建命令为：

```sh
cd c-fullchain
cc -std=c11 -O2 -fno-fast-math -ffp-contract=off -Wall -Wextra -Werror \
  -DEF_WITH_HEAD -Isource -Igenerated \
  source/fixed_frontend_mel.c source/frontend_replay.c generated/fixed_models.c \
  source/english_causal_head.c -lm -o ../frontend-native-local
cd ..
```

ARM 重建使用 `arm-linux-gnueabihf-gcc` 并增加 `-static`，输出另取新文件名。编译需要 C 工具链；运行已编译程序无需 Python、NumPy、ONNX Runtime 或 ONNX 文件。

## 4. 读取 C trace，查看具体事件

C 程序的标准输出是样本、embedding、ready output 和事件的总数。具体 logits、概率、关键词 ID 和触发时间保存在 `TRACE.bin`，不能直接用文本工具读取。

```sh
python3 -I -S -B read_c_trace.py --trace native-trace.bin --output native-frames.jsonl
python3 -I -S -B read_c_trace.py --trace arm-trace.bin --output arm-frames.json --format json
```

解析器只读取已保存的二进制：**0 新模型调用、0 新 C 回放**。它验证 `EFFTR01\0`、完整顺序 stages 1/2/3/4、各 stage 的 count 3104/2432/96/8、同窗样本时钟、有限浮点数、暖机 ready 状态及固定事件逻辑。整份 trace 完整通过后才发布新的输出文件；目标已存在时拒绝覆盖。它不重新核验原音频→logits 的数值正确性。

JSONL 首行为 metadata，随后每窗一条 `head_frame`，末行为 summary。JSON 格式为 `metadata`、`frames`、`summary`。每个 frame 含 `sample_end`、`seconds`、`ready`、`event`、`label`、`logits[3]`、`probabilities[3]`。只看非零事件可用：

```sh
python3 -I -S -B - native-frames.jsonl <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as source:
    for line in source:
        row = json.loads(line)
        if row.get('type') == 'head_frame' and row['event']:
            print(json.dumps(row, ensure_ascii=False))
PY
```

## 5. 固定类别、暖机和时间含义

- 概率/ logits 顺序：`0=background`、`1=Hey Vosen`、`2=Hey Mero`；事件字段 `0=none`，不表示 background 事件。
- 首个 embedding 在已观察 **16,000 个样本（1.00 s）** 后出现，此后间隔 **1,280 样本（80 ms）**。
- 累积 **16 个 embedding** 后才有 head 输出：第一个 `ready=true` 在 **35,200 样本（2.20 s）**。
- 固定 float32 阈值 **0.8**，同一前景类连续 3 次达到阈值才触发；最早 proposal 在 **37,760 样本（2.36 s）**。触发后的 cooldown 是 **13 个 hop（1.04 s）**，之后仍需重新连续确认。
- 每次进程从空历史开始；没有启动静音、重采样、gain 或 EOF 补样。短片段可能没有任何 head 输出。没有事件不能被报告为实测 FRR。
- `sample_end` 和 `seconds` 是 proposal 可用的输入样本时钟，**不是词尾时间，也不是词尾检测延迟**。Trace 没有记录完整原输入长度，解析器不会用最后一个 embedding 时钟冒充真实 EOF。

这些工具提供离线研究候选的运行入口。独立真人/环境评估、发音准入、FRR/FAR、词尾延迟和物理 SSC305 结论，应分别依据真实对应证据；它们不由包存在或 trace 能解析而成立。
