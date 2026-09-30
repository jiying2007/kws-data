# 源标注近邻片段与单线程配置：固定研究证据

本档案保存一次三模型回读与一次已审阅的 sherpa API 单线程重复，不包含真人音频、第三方权重、运行时二进制或新的训练数据。数据 catalog、审核门和既有42合成音频均未改变。

| 固定路径 | HI-MIA-CW源标注触发片 / 7006 | Qwen正例前/后（各20片） | 主机CPU秒 / 墙钟秒 |
| --- | ---: | ---: | ---: |
| 冻结 C | 0 | 0 / 0 | 19.964 / 22.168 |
| sherpa INT8，API num_threads=2 | 0 | 19 / 19 | 777.232 / 195.320 |
| 完整 cFSMN / Hamming | 0 | 11 / 11 | 143.924 / 145.286 |
| sherpa INT8，API num_threads=1 | 0 | 19 / 19 | 195.725 / 198.227 |

四条路径的Qwen混淆片均前/后0/22；三原路径与各自旧基线的前后完整事件一致，API1重复的全部7090条事件与API2精确一致。API1的3910次worker-only线程采样均为1；旧API2没有实测线程峰，配置2不是全进程最多2线程的证明。

结果支持优先验证单线程部署配置；本次同输入仍有顺序/cache差异、新增线程采样和逐行旧结果比对开销，不是纯算法因果加速或SSC305测量。CPU含框架、读取、哈希、报告及子进程；RSS不是C11固定工作区。生产回调不能照搬本实验逐片写监护文件的I/O方式。

## 证据边界

- 7006片来自历史15人已观察development，2.807小时，但只有12种你好/米亚相关弱文本。不是新盲测或家庭连续流，不报告连续FAR/hour、精确转录或真实词尾延迟
- 冻结C在Qwen20正例零命中，零近邻触发不能作可用性背书；外部系统也未由本次获得产品放行
- 另一批40条TRAIN的固定ASR文本监督准入失败保持原状；本次弱标签回读不修改该门，也不说明全部语音发音错误
- HI-MIA-CW来源：[OpenSLR120](https://www.openslr.org/120/)，CC BY-SA 4.0；保留来源标识/弱文本及运行观察用于研究。本包不再分发源WAV，也不新增统一商业授权。上游模型、训练来源与部署权利仍按各自既有记录审查

## 一个输入清单，原字节压缩与分片保留

- `data/source-input-identities.json.gz`：唯一共享7006输入弱标签、WAV/PCM身份清单；不复制音频
- `runs/{c,sherpa-api2,cfsmn,sherpa-api1}/records.jsonl.gz`：原始逐片JSONL的gzip封装，解压得到精确原字节；各路径原生身份字段保留，未为减体积重写科学结果
- 五个大gzip逻辑文件以`*.gz.parts/0000.bin`起的有序二进制片段保存，每片最多128KiB；不重复保留整包。`gzip-shards.json`绑定顺序、每片长度/SHA、完整gzip与展开原文的长度/SHA。小型线程采样gzip保持原文件。标准库`verify.read(root, logical_name)`装配、`verify.payload(root, logical_name)`解压；每片128KiB、压缩总量16MiB、展开32MiB硬上限，拒绝缺片、重排、路径逃逸与symlink
- `copied-file-origins.json`：各复制文件历史路径、原字节长度/SHA256与压缩方式。`archive-manifest.json`沿用现有历史证据格式，绑定实际归档字节，不是第二个数据catalog
- `original-three-paths/`与`thread1-repeat/`：原方案、源码、规格、测试、结果；两个RESULTS.md保留完整解释
- `runs/*/{complete,supervisor,recomputed}.json`：完成/保护/计数证据；API1另存压缩的worker线程采样

历史源码/规格含原绝对研究路径与独立环境，不是当前可直接运行的通用命令。仅做档案核验时不要运行historical driver或freeze。数值复现需另建工作区，依法取得并验证固定源包/模型/运行时，将路径显式映射并记录差异，不能覆盖历史记录。本次不声称解压证据等于重跑模型。

## 标准库核验

```sh
python3 research/2026-09-30-himia-source-neighbors/verify.py
python3 research/2026-09-30-himia-source-neighbors/test_verify.py
```

核验检查固定文件集/大小/SHA、gzip展开原字节SHA、唯一输入清单、四路径的逐片身份/计数/前后与API间事件一致性、保存的资源上限及线程样本。它不读取源WAV、不安装模型框架、不推理、不重新确证系统计数器真实性，也不签发产品资格。承载清单的Git提交和审查是其信任依据。
