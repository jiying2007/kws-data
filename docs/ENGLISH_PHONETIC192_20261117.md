# 第二个独立 192 条英文合成批次：直接音素与新来源

当前为 **disabled 源码**。只有绑定实际已合并政策 commit 和新的 data main predecessor 后，公开 Actions 才执行这个新的有限批次。本次源码检查没有合成、ASR、训练或 KWS 推理。

## 已发生的证据与这次改动

第一个批次实际 run `44098312998`、head `45a69bd8600cfb59c9e1b5c7342e4405d72145db` 已生成 192 条不同 PCM、24 个来源。保存的 Piper G2P 表明自然 `Hey Mero.` 请求成为 `hˈeɪ mˈiəɹoʊ.`，`Hey Meero.` 成为 `hˈeɪ mˈɪɹoʊ.`，自然 `Hey Vosen.` 成为 `/z/` 形式。请求文本和预设假说并未保证送入模型的音素。

第一批的独立 XLSR 192 次观察全部保存。按未改变的全词规则，Mero 两个 cell 各只有 6/24 具备 `/i/` 家族支持，`/ɛ/` 为零。仅音素证据能支持两词和普通负例的来源上界为 train 4、development 2、calibration 2、locked-test 1，低于既定 6/2/2/2。其他 ASR 只能进一步限制这些数量，无法补上缺少的全词音素证据。旧 192 条音频、全部观察和 UNKNOWN 均保留，不改原请求或标签，不重新生成旧 cell。

首批最终三 ASR 和音素联合准入为 silver 72、weak 12、UNKNOWN 108；平衡来源只有 train 1，其余角色为 0，`training_snapshot_allowed=false`。Mero 48 条中有 22 条文本至少两票、但音素不满足全词规则；Vosen 48 条中有 30 条音素符合、但旧文本别名不足两票。后者包括三个 ASR 同写 `Hey Vosin`、独立音素却为 `h eɪ v oː z ə n` 的记录。这是两个不同的瓶颈，不能把全部失败解释为缺少文本别名。

这次改动是明确控制四个正例请求的音素，并换用未使用过的 24 个来源。四个直接请求都带 `Hey` 的原生主重音及句末句点，与已保存的自然 `Hey Vosen.` G2P 控制格式一致。旧自然 Vosen 音素匹配 22/24，原无 Hey 重音且无句点的 raw-z 为 16/24；这个观察支持尝试相同控制格式，但不是因果效果证明。`length_scale` 统一预登记为 1.2；它是现有 Piper `SynthesisConfig` 参数，直接进入原 ONNX `scales` 的第二项。尚未运行新批次，不能声称实际时长必然增加恰好 20% 或识别质量必然提高。

## 固定设计

`campaign_id` / `source_design_id` 为 `piper-libritts-direct-34-57-v1`。共 24 个来源 × 8 个 cell = 192 条。新 ID 前缀 `enp192-`，ID 包含来源、cell、模型、campaign 和完整 synthesis_config 的哈希。

| Cell | requested_text | 请求方式 | 原始音素 | 假说 |
|---|---|---|---|---|
| mero-direct-eh | Hey Mero. | direct phoneme IDs | `hˈeɪ mˈɛɹoʊ.` | mero-us-eh-v1 |
| mero-direct-i | Hey Mero. | direct phoneme IDs | `hˈeɪ mˈiːɹoʊ.` | mero-us-i-v1 |
| vosen-direct-s | Hey Vosen. | direct phoneme IDs | `hˈeɪ vˈoʊsən.` | vosen-us-s-v1 |
| vosen-direct-z | Hey Vosen. | direct phoneme IDs | `hˈeɪ vˈoʊzən.` | vosen-us-z-v1 |
| ordinary-time | What time is it? | 原 en-us G2P | 实际输出保存 | ordinary-negative-v1 |
| ordinary-light | Please turn on the kitchen light. | 原 en-us G2P | 实际输出保存 | ordinary-negative-v1 |
| ordinary-package | The package is by the front door. | 原 en-us G2P | 实际输出保存 | ordinary-negative-v1 |
| ordinary-window | Could you open the window? | 原 en-us G2P | 实际输出保存 | ordinary-negative-v1 |

四个 wake cell 从 `list(raw_phonemes)` 调用真实 `PiperVoice.phonemes_to_ids`，再调用 `phoneme_ids_to_audio`；不经过文本 G2P 或 `[[...]]` parser。保存的实际 config 已包含每一个字符，包括 `ɛ`、`i`、`ː`、`ɹ` 和重音符号。每个直接请求的完整 ID 列表也已从保存 config 计算并固定；实际返回列表不一致时在模型调用之前拒绝。

模型仍是固定 revision `de3dcfdf1912bb49726dc4aa11c26017ce2ac62a` 的 `en_US-libritts-high`，ONNX SHA256 `9127a559e11603f10b366d1a20ac7426826081dbc521de4c2420c57728d73f0f`。实际保存 config 为 20,163 bytes，SHA256 `2efdc6d7f954588b8180132cbd9b8001933fdd00932c92bc92fd0d2028a9eb3d`；130 symbols，最大 ID 129，904 speaker slots。

`recipe.SYNTHESIS` 是唯一参数来源：length_scale 1.2、noise_scale 0.333、noise_w_scale 0.333、normalize_audio false、volume 1.0。recipe、计划、实际调用、每条 STARTED/COMPLETED 和读回检查使用相同值。

## 新来源与角色

| 角色 | Piper indices | Speaker IDs |
|---|---|---|
| train | 34–45 | p3370, p7704, p8506, p8410, p6904, p5655, p2204, p501, p7314, p1027, p5054, p534 |
| development | 46–49 | p2853, p5935, p2404, p7874 |
| calibration | 50–53 | p816, p2053, p8066, p16 |
| locked-test | 54–57 | p4586, p1923, p2592, p1265 |

从真实保存的 `speaker_id_map` 核对全部 24 个映射。排除旧 40 个 LibriSpeech speaker IDs 和 Piper indices 0–33，共 74 个不同 ID；新来源无交集。所有来源每类都生成正例和普通负例，失败或 UNKNOWN 不通过移动来源角色来补齐。音色共享同一个 Piper 预训练模型，不声称是新增独立真人验证。

## 固定观察器与前瞻文本兼容规则

仍使用原固定 Qwen / Whisper / Parakeet 三个盲 ASR 和同一 XLSR 音素模型，每条成功生成的原始 WAV 最多各观察一次。模型 revision、weights SHA、decode_config SHA、全词音素模式、音素归一化、silver/weak 权重及 6/2/2/2 最小来源数均保持不变。

仅对尚未生成的这个独立批次，预登记 `english-synthetic-machine-admission-v2`：增加整句 `Vosin` / `Vozin` 拼写及 `Hey` / `Hay` 同音拼写兼容，同时把所有矛盾词核也扩展到 Hey/Hay，保留 Rosen 等否决；不接受缺失 Hey 的 `A Vosen`、不同起始辅音 `Bozen` 或任意近似拼写。RULES SHA256 为 `0a02fbef84f3de275b67908038a9fcd5c91a579138ce65d47037dffa48c60383`。严格的 schwa、Mero ɛ/i、Vosen s/z 全词音素要求没有放宽，规则仍不声称验证重音或元音长度。

新独立准入模块为 `kws-pipeline/training/english_phonetic_silver.py`，限定本批来源设计；旧模块和旧批次准入结果保持原字节，不用 v2 重新准入旧数据。包内仍保存为 `policy/english_synthetic_silver.py`，其真实上游路径由 `policy/IDENTITY.json` 记录，以保持通用格式的下游兼容性。profile 继续使用 v3，Mero 最终母语发音仍未冻结。直接音素输入只证明请求是什么，独立声学观察仍须判断实际发出了什么。

三路都支持且全词音素支持才是 silver；两路支持且满足冲突规则的为明确 weak；UNKNOWN 不当背景。近似发音不作为硬负例，四个 ordinary cell 仍需三份完整观察、声明句子支持及目标缺席证据。

## 有界执行和格式

公开 main push、attempt 1、精确 predecessor、固定 source hashes 才可运行。生成上限 192 次/900 秒；每个 observer 上限 192 次/3600 秒。每次调用在开始前 create-once 写 STARTED；中断、空输出和失败保留原收据，不自动 retry。所有 192 个请求始终保留分母；绿色 combine 仅说明证据包保存，训练还要看实际 ADMISSION。

工作流为 `english-phonetic192-once.yml`：source-contract → generate → 四个独立 observe job → combine。artifact 为 `phonetic192-source`、`phonetic192-observer-<provider>`、相应 runtime 记录，以及最终 `english-phonetic192-corpus-<runid>`。

JSON schema 继续使用既有 `english-synthetic192-*` 格式版本。`RECIPE.json` 顶层新增 campaign_id；`EXECUTION_PLAN.json` 顶层 campaign_id 和 `silver_admission.source_design_id`；`CORPUS_MANIFEST.json` 顶层 campaign_id 与 source_design_id，全部为上述同一个 ID。原 `FILES.json` 闭合清单、PCM/WAV、captured ASR/phone、ADMISSION 路径保持格式兼容。

## 源码检查

```sh
python3 -I -S -B automation/test_english_phonetic192.py
python3 -I -S -B automation/english_phonetic192_once.py --plan automation/english_phonetic192_plan.json --source-check
```

另以 `-O`、`-OO` 运行同组契约检查。这些命令没有模型调用。旧 192 的代码、音频、观察和准入规则不改；新模型训练、native/ARM fullPCM、SDK 输出仍由实际后续结果决定。
