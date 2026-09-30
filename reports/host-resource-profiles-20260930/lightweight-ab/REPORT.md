# 两个轻量原型的主机资源剖析

日期：2026-09-30。只剖析已训练 A/B seed1337，不新增训练，不改已批准研究归档。
本机为 x86_64，**不是 SSC305**；以下 CPU 数字不能换算成目标板占用。

## 测量协议

- 原研究代码与模型/特征 cache SHA 固定；PyTorch2.13.0+cpu，Python3.12.14
- 进程内 PyTorch intra/inter-op 均1线程，`/proc/self/status`测得Threads=1，未固定CPU亲和性（可调度9个CPU）
- batch1，一次输入1个32维特征帧，对应20ms音频hop；状态连续传递
- 200帧预热，随后3000帧，分3段各1000帧报告；输入为预载的既有C logmel特征
- 测量模型 `.chunk()`，包括Python/PyTorch调度、张量拼接/动态分配和算子；**不包括PCM采集、FFT/mel/PCEN、VAD或事件解码**
- 原型每次重新拼接缓存，不是C固定arena/ring-buffer实现；不能当成最终优化后的C开销
- `perf_counter_ns`测wall，`process_time_ns`测进程CPU；空计时器CPU中位开销约0.19–0.27µs，未从结果扣减
- 首次脚本命名`profile.py`遮蔽Python标准库同名模块，在测量前失败；更名后完成固定协议，无模型/数据改动

## CPU时间：实际测量

单位ms/20ms帧。A为depthwise temporal CNN，B为低秩FSMN。

| 原型 | CPU均值 | CPU p50 | CPU p95 | CPU p99 | wall p50/p95/p99 |
| --- | ---: | ---: | ---: | ---: | --- |
| A |1.812|1.580|2.642|2.909|1.582 / 2.644 / 2.912|
| B |0.635|0.558|0.976|1.114|0.559 / 0.977 / 1.116|

最大wall帧为A5.094ms、B8.246ms，尾部仍有抖动；不能只看均值证明实时资格。
按本机均值×50帧/秒计算，**本机模型-only**的单核等效占用约A9.06%、B3.17%。
此算术仅解释本机profile，不是SSC305预测，也不是完整KWS/音频系统CPU。
三个分段都保存在JSON，不挑最快分段；并未固定CPU频率或执行长期热稳态测试。

## 模型与状态：实际字节

| 项目 | A | B |
| --- | ---: | ---: |
| 参数量 |14,882|11,954|
| FP32参数有效载荷 |59,528 B|47,816 B|
| 参数实际storage字节 |59,528 B|47,816 B|
| 原始`.pt`序列化文件 |66,561 B|54,105 B|
| 因果状态logical tensor bytes |23,808 B|11,520 B|
| 当前PyTorch状态实际backing storage |24,768 B|11,904 B|

状态是拼接结果的view，底层storage额外保留一个当前帧，所以实际storage略大于逻辑状态。
以上没有把FFT/特征缓存、临时activation、进程运行时、模型加载器或代码算进“固定KWS RAM”。

全Python/PyTorch进程的观测VmHWM约A330,900KiB（323.1MiB）、B333,116KiB（325.3MiB），
包含解释器、框架、全部加载特征、计时数组、allocator和模块；**不能声称这就是小模型在C端所需内存**。
JSON另保留窗口结束较早时的ru_maxrss（A330,636/B332,936KiB）；采样时点不同，不强行当成完全相同读数。

仅做设计估算的 int8权重+FP32 bias：A16,472B、B12,680B；不含scales/对齐。
本次没有执行量化，不能宣称量化后的质量、速度或RAM已通过。

## I/O与缺页

计时窗口内输入已在内存，循环没有文件/日志/网络操作。
两进程的`/proc/self/io`差量：read_bytes=0、write_bytes=0、syscw=0、wchar=0；
raw rchar=100、syscr=1来自I/O计数器读取本身。
两者rusage input/output blocks=0、major faults=0、minor faults=12。

这支持“本次模型增量推理窗口没有观测到存储读写”，不表示整套应用无需音频采集I/O。
音频ring buffer、驱动中断、AEC/播放、丢帧与后台竞争必须在整机另外测量。

## SSC305预算建议：待确认，不是已达标

用户明确CPU和I/O优先，RAM/模型可以放宽，因此不应继续把极小模型文件当首要优化目标。
先给**总KWS路径（前端+模型+解码）**设一个可讨论的单核CPU预算；AFE是否计入须单独明确。

- 建议目标档5%单核CPU：50fps下平均可用约1ms/帧
- 建议上限档10%单核CPU：50fps下平均可用约2ms/帧
- 这两个数字是预算模板，尚未获产品确认，也不是根据本机profile推算的SSC305能力
- p99与最大处理时间另受20ms block deadline、后台任务和安全余量约束；平均CPU百分比不能替代deadline门槛
- RAM按整机可用余量设明确arena/模型预算，不强制沿用26KB级旧模型；用适量缓存、预计算和ring buffer换取CPU/访存稳定性
- 运行中模型预载，避免每帧文件读取、动态内存、同步日志和重复计算完整2.5秒窗口；需要调试时在计时窗口外限量保存

下一步在真实SSC305上，使用相同编译器/优化、固定频率与热策略、真实后台负载，
测完整C KWS路径的CPU/每hop尾延迟、arena/stack/RSS、I/O/缺页/XRUN及功耗。
本次不因B主机开销更低而忽视其跨声线准确性不足：两个原型均未成为产品候选。

## 证据

- `resource_profile.py`：本次独立profile代码
- `A-profile.json`、`B-profile.json`：完整身份、计时、存储、RSS、I/O和分段统计
- `A-timings.npz`、`B-timings.npz`：3000个原始CPU/wall时长
- `A-console.json`、`B-console.json`：精简结果

已批准的研究归档、原数据及四个训练checkpoint保持不变。
