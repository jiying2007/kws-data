# Mero 英文主读音规范 v1

日期：2026-11-18。对应声明文件：`automation/english_mero_meh_profile_v1.json`。

## 本轮选择

按用户本轮授权，选定 **Hey Mero /heɪ ˈmɛroʊ/**，提示读法 **HEY MEH-roh**。Mero 两音节，首音节重读。**Hey Vosen /heɪ ˈvoʊsən/** 保持原主读音，其中辅音目标为 `/s/`，不增加 `/z/` 正例变体。

这是品牌发音规范的选择；不声称 Mero 有唯一英语词典读法，也不声称真人统计已证明多数用户会自然这样读。旧 `/heɪ ˈmɪəroʊ/` 不作为本轮权威标准。

`ɛ` 与 Cambridge 用来表示 head 元音的 `e` 可视为本规范中的转写差异。[1] 这不是任意声学近似或识别 token 的自动替换规则。MEER-oh 一组 `/ˈmɪroʊ/、/ˈmiːroʊ/、/ˈmɪəroʊ/` 不混入本轮主正类；“不收为主正类”不自动等于“严格负类”。英式末元音 `/əʊ/` 仅列为后续支持候选，本轮主训练不混入。

## 碰撞词边界

Cambridge 把 marrow 的美式读音列为 `/ˈmer.oʊ/`，其英式读音则为 `/ˈmær.əʊ/`。[1] 北美英语地图集说明，多数北美说话者的 Mary/marry/merry 合流，但不是所有方言都如此。[2]

因此 **Hey Marrow、Hay Marrow** 可以与本轮目标同音；hey 和 hay 均为 `/heɪ/`。[3][4] **Hey Merrow** 等名字或不同拼写也应按实际音频核实，不假定所有名字持有者读法一致。这些碰撞或待核实项不得作为严格负类或计入严格误触发分数，须单列处理。同音且声学相同的片段，无法靠纯声学模型按意图拼写可靠分开。

## 旧探针与后续验证

此前探索性 Mero 探针保持 **excluded**，保留原验收结论，不追溯改判，不用于本轮训练或调阈值。原先要求 Marrow 必须呈现 `/mæ/` 的对照，只能检验那份旧契约；未通过不等于自然发音错误。

旧 Pocket/XLSR 观测也不能证明用户自然读法，不能据 `/ɔ、ʌ、ə/` 等输出反向放宽本轮末元音 `/oʊ/`。后续按已冻结规范使用新样本开展验证。新真人录音不作为启动规范、数据处理和模型优化的前提；合成验证也不冒充真人自然度或真实场景质量结论。

本 profile 控制本轮新生成和新准入的 cohort。训练仍保留既有 **225 个 weak bags** 的能力回放及旧回归检查；这些既有数据不成为新发音规范的权威，也没有追溯重标。“MEER-oh 不混入本轮主正类”针对新 cohort，不要求删除既有回放。

本文件仅确定发音与解释边界。数据准入规则、合成输入及具体质量门槛由对应实现定义。文本匹配使用的拼写等价关系不授权新的正例声学变体。

## 来源

1. [Cambridge：marrow 发音](https://dictionary.cambridge.org/pronunciation/english/marrow)。美英音标及逐音说明，含 head 对应的 `/e/`。
2. [The Atlas of North American English：第 2 章，页 14](https://www.ling.upenn.edu/~wlabov/L165/ANAE2.pdf)。北美 `/r/` 前元音合流及方言差异；不提供 Mero 品牌的使用频率证据。
3. [Cambridge：hey 发音](https://dictionary.cambridge.org/pronunciation/english/hey)。`/heɪ/`。
4. [Cambridge：hay 发音](https://dictionary.cambridge.org/pronunciation/english/hay)。`/heɪ/`。

来源核实日期：2026-11-18。词典与音系文献支持碰撞分析，不替代本品牌的真人使用调查。
