# B24 条件性 warm-start 新阶段（仅 TRAIN 证据）

准备时间：2026-11-24 16:46:34 UTC。依据原 B22/B23 TRAIN 保存数组；没有读取 B23 后续 KWS DEV 输出。B24 当前不授权执行。完整数字与原报告 SHA 见 `TRAIN_ONLY_EVIDENCE.json`。

## 已确认的作用与瓶颈

B23 只把已知静音 prefix 的 blank loss 从时间 mean 改为 amax。每类384次 base 观测的首帧 token1 从384降到92，且这92次全部位于steps1–46；之后不再出现。prefix输出空间导数L1均值从0.01675升至0.39323。自然ordinary完整事件98→28，支持前缀捷径已被抑制；这不是参数梯度冲突的测量。

真正声学事件仍未建立：synthetic-positive完整事件280→67，B23的67次也全部在前46步；suffix完整事件由B22的3/384变为B23的0/384。相同original/gain/prefix的120组配对中119组CTC下降、中位7.80→3.75，却全部suffix miss→miss。不能把去掉名义命中说成正例恢复。

全部192步均发生norm5裁剪；单个 `mero24-453-mero-slow` 占48.982%正例CTC，但只在20/55/90/125/160出现。最后22步不含此困难原音，其较低均值混入了采样组成变化，不能单独证明收敛或未收敛。全局裁剪和标量loss集中也不能确定参数梯度方向冲突或证明应提高LR、重加权。

## 唯一候选与否证条件

仅将本phase初始化从B0改为实际B23 checkpoint `784def9d…980297f`。建立fresh AdamW；同一192步COMMON_DRAW、88原TRAIN、122保护条件、264已有教师、amax、其他loss聚合/系数、LR6e-5、clip5、43876参数、标签及decoder保持B23原样。原B23仅复制为immutable historical control，不重训。本phase仍为192 updates / 1847新forwards。

这是一项warm-start新实验。B23没有保存Adam状态；不恢复或伪造旧moment，不重算已消耗的B23前192步，也不称为exact continuation、384步连续轨迹或纯更新次数效应。

待另行授权后，预先按原条件配对检查本phase已保存的forward：真正suffix `[1,2,3]` 能否出现且已知静音首帧捷径不回潮。若仍全部suffix miss，或所谓恢复仅来自prefix复发，则否定这个具体warm-start方案，不自动追加新phase。该诊断不替代独立晋级门槛。

## 封闭状态

只有真实完整B23 PRIMARY `NO_PROMOTION`及原始producer/source/全部数组证明才能绑定opening。失败reader的身份须取自实际原PRIMARY证明链，当前不锁临时版本。新面板等本次B23 SOURCE全部完成后，排除全部实际120原PCM与72 primary/reserve人再冻结；现在不造PCM、不生成名单、不进行模型或语音调用。
