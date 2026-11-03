# BadCM-3P / Bad-MFL-3P

独立方法：`attacks/badcm_3p`、`attacks/bad_mfl_3p`。三个数据集入口
`experiment/{crisis-mmd,hateful_memes,upmc_food101}/train.py` 均可使用：

| 参数 `--attack_type` | 同义名称 |
| --- | --- |
| `badcm_3p` | `badcm-3path`、`badcm_3path`、`badcm-3p` |
| `bad_mfl_3p` | `bad-mfl-3path`、`bad_mfl_3path`、`bad-mfl-3p` |

原有 `badcm`、`bad_mfl` 的实现保留。新客户端继承原类；正常客户端调用父类训练分支。
当前支持 `missing_strategy=zero`、`pepsy` 或 `pmcmfl`、多模态单标签分类，以及原有 FedAvg/FedOpt/FedProx 优化器。
融合结构、联邦采样和聚合算法沿用原实现。启用攻击时使用原方法的融合配置（6 heads、dropout=0.2）。

## 三路径训练

仅通过样本 key 和 `poisoned_keys` 确认毒化身份；干净的目标类别样本不会加入三路径监督。

| 原始可用模态 | 毒化样本训练路径 |
| --- | --- |
| 图像和文本 | J：双模态；I：毒化图像/零文本；T：零图像/毒化文本 |
| 仅图像 | I |
| 仅文本 | T |
| 两者均缺失 | 跳过 |

删除模态时同时清零特征与有效长度。不为缺失模态补造触发器。
双模态样本的损失为 `(CE_J + CE_I + CE_T)/3`，单模态样本只有对应路径的 CE；
先按单个样本的路径数平均，再对毒化样本平均。
混合 batch 为 `poison_weight * L_poison + (1-poison_weight) * L_clean`；
全毒化/全干净 batch 只使用对应组损失。`poison_weight` 继承原方法：
默认等于 `bd_poison_ratio`，支持原有 `badcm_attack_weight` / `bad_mfl_attack_weight` 覆盖规则。
共用同一个模型，损失合并后反向传播一次、梯度裁剪一次、优化一步。
聚合样本数按原始有效样本计数，三路径不会将样本数乘三。

PEPSY 的每条合法路径都经过原有提示池、特征重建与融合。三路径只扩展毒化分类监督；
原始 batch 的 PEPSY 对比损失与提示正则各加入一次，默认权重分别为0.2和0.1。
恶意客户端上传原始可用模态标记，供 PEPSY 原有的按模态聚合使用；
人工删除模态产生的附加路径不改变客户端模态身份。全缺失样本仍跳过。

PMCMFL 的每条合法路径沿用原有编码器、原型填充和融合逻辑。三路径只扩展毒化分类损失，
原始 batch 的图像/文本/融合原型对比正则只计算一次。全局原型由原训练入口传入；
本地原型与模态统计在原始 dataloader 上重新计算后上传，附加路径不会增加聚合样本数或原型计数。

## 先生成，再提取

在服务器执行：

```bash
conda activate fed
cd /hy-tmp/fed
export PYTHONHASHSEED=2024

# BadCM-3P：raw 阶段只保存可用模态的 PNG/TXT 和清单。
python -m attacks.badcm_3p.prepare_crisis_mmd --stage raw \
  --missing_modailty_rate 0.1 --bd_target_label 0
python -m attacks.badcm_3p.prepare_crisis_mmd --stage extract \
  --missing_modailty_rate 0.1 --bd_target_label 0

# Bad-MFL-3P：原 CLIP/ITP/TTP/GAN 生成流程。
python -m attacks.bad_mfl_3p.prepare_crisismmd --stage raw \
  --missing_modailty_rate 0.1 --bd_target_label 0
python -m attacks.bad_mfl_3p.prepare_crisismmd --stage extract \
  --missing_modailty_rate 0.1 --bd_target_label 0
```

两种方法均提供 `prepare_hateful_memes`、`prepare_upmc_food101` 模块和通用
`prepare_features --dataset ...` 模块。`--stage all` 默认按 raw → extract 顺序执行。
raw 和 extract 的配置参数必须一致。`--force` 重新生成当前配置；原方法的 force 参数也兼容。
默认参数：alpha=5、恶意比例=0.1、毒化比例=0.1、目标标签=0、seed=2024、q=0.1。
q>0 必须使用对应 simulation 文件，不重新抽取缺失掩码，并严格校验样本 key/顺序。

raw 不实例化受害模型的特征提取模块，也不生成毒化特征。
Bad-MFL 的 raw 阶段允许使用 CLIP 执行原有生成计算。
`metadata.raw.json` 表示原始样本阶段结束；extract 校验清单和 PNG/TXT 的 SHA-256，
重新读取保存文件，用 MobileNetV2/MobileBERT 提取特征。
全部特征成功写出后，原子发布 `metadata.json`，训练入口拒绝未完成的制品。
特征文件只保存被毒化的 key/特征/标签；加载器在内存中覆盖对应干净记录。
不复制干净图像、预览图、干净特征或划分文件。原有干净数据文件不会被修改。

输出路径示例：

```text
output/attacks/badcm_3p/crisis-mmd/alpha50/mm01/mr01_pr01_tl0_seed2024/
output/attacks/bad_mfl_3p/crisis-mmd/alpha50/mm01/mr01_pr01_tl0_seed2024/
```

BadCM 保留当前项目的有界图像 pattern 与直接文本触发词适配版本，默认 `4/255`、`cf`。
沿用 q 下可用客户端的分层选择；预算允许时覆盖完整/仅图像/仅文本客户端。
Bad-MFL 保留从全部训练客户端固定随机选择恶意客户端的规则；全缺失客户端跳过投毒且不补抽。
它的离线生成阶段沿用访问完整原图文对的假设，之后只保存和训练实际可用的模态。

Bad-MFL 默认 CLIP 为 `openai/clip-vit-base-patch32`，概念为 retro/cinematic/warm/cool，
网格3×3、box_scale=0.32、Top-k各3、ITP/TTP=0.5/0.5、支持集64、生成器2 epochs、
batch8、lr=0.001、eps=0.18，损失为 `3 L_rec + 0.2 L_adv + L_align + 4 L_sem`，T5默认关闭。
模型默认只从本地缓存加载；需要下载时显式使用 `--bad_mfl_allow_remote_model_download`。
生成器、关键区域、文本候选缓存在配置和划分哈希一致时跨 q 复用。
必须在启动 Python 前设置 `PYTHONHASHSEED`，否则原生成器使用的 `hash()` 无法固定。

**支持集隔离修正：** 新 Bad-MFL-3P 明确排除 dev/test/val/validation 和与其重复的 key，
记录 `support_policy=train_only_v1`、支持集 key、原代码哈希和缓存路径。
这修正了旧版展开 lookup 后无法排除 holdout 的问题；不应声称与含该问题的历史实验结果完全一致。

## 训练与验收

```bash
python experiment/crisis-mmd/train.py \
  --attack_type badcm-3path --en_backdoor_attack --en_missing_modality \
  --missing_modailty_rate 0.1 --missing_strategy zero \
  --bd_malicious_ratio 0.1 --bd_poison_ratio 0.1 --bd_target_label 0 --bd_seed 2024 \
  --alpha 5.0 --fed_alg fed_avg --sample_rate 0.1 --learning_rate 0.05 \
  --hid_size 128 --batch_size 16 --local_epochs 1 --num_epochs 200 --num_folds 5
```

Bad-MFL-3P 改用 `--attack_type bad-mfl-3path`，先生成其独立制品。
PEPSY 改用 `--missing_strategy pepsy`；raw/extract 阶段沿用同一攻击生成方法。
q=0.1/0.3/0.5/0.7/0.9 分别准备并运行；该命令不会自动开始批量实验。
三个数据集入口默认5 folds，随机种子沿用8、16、24、32、40。
正式结果仍按干净 dev Macro-F1 最佳轮次选择，ACC/ASR/IA/TA 来自同一轮；
最终 `result.json` 含 `average` 和样本标准差 `std`（ddof=1；仅1 fold时为null）。
每轮 `three_path_fold{fold}.jsonl` 记录设备、CUDA名称、fold、轮次、有效/跳过客户端、
原始样本数和 J/I/T 训练路径计数；没有有效更新的轮次也会记录。
攻击测试集只选非目标类样本，数量取完整测试集大小×毒化比例的四舍五入值并限于候选数。
ASR 为双毒化输入，IA/TA 为删除另一模态（特征和长度均零）的输入。

```bash
python -m unittest attacks.tests.test_three_path -v
```

测试覆盖混合可用模态的逐样本平均、全干净/全毒化边界、key 判定、单模态与父类更新一致、
正常客户端分支、全缺失跳过、一次更新、原始样本数、raw/extract 隔离、文件校验、
稀疏合并、客户端选择、支持集隔离、跨 q 缓存和 ddof=1。
PEPSY 额外验证辅助损失的数值与梯度、单模态更新与父类一致、模态聚合标记、一次更新和全缺失跳过。
