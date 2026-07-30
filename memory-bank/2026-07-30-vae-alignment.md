# 2026-07-30 VAE 对齐与优化记录

## 结论

- 窗口特征 VAE 的 200k 低分并不能否定 VAE：主要问题是窗口化输入、顺序截断采样和时间切分共同改变了正常数据分布。
- 对齐 CARE-AE 的原始行级输入与随机 normal 切分后，VAE 的 AUC 和 CARE 均显著恢复；当前应以 raw CARE-aligned VAE 作为主 VAE 路线。

## CARE-aligned raw VAE 数据协议

- Farm B 使用 252 个数值 SCADA 传感器列；CARE 数据说明中的 257 包含元字段。
- 只使用 normal 数据集、`train` 分区、运行状态 0/2 的原始行。
- 从所有可用 normal 运行行随机抽取 `cap-train` 条；75% 用于训练，25% 用于正常 validation。
- 标准化为逐列 Z-score，均值/标准差只由训练 75% 计算；缺失值在 normal 数据内以中位数填充，prediction 缺失值以训练均值填充。
- 不使用窗口特征工程；每条 10 分钟 SCADA 行直接作为 VAE 输入。

## VAE 架构与评分

- 当前架构：`252 → 512 → 512 → (mu/logvar: 128) → 512 → 512 → 252`，训练损失为 sum-MSE + `beta=0.1` KL。
- 推理使用 posterior mean；异常分数已由 mean-MSE 改为全维重构 L2 范数，和 CARE-AE 的重构评分形式对齐。
- adaptive threshold NN 以正常输入预测预期 L2 重构误差；最终分数为 `actual_l2 - expected_normal_l2`，gamma 用正常 validation 残差 P99 标定。

## 结果（Farm B）

| 输入与配置 | AUC | CARE |
|---|---:|---:|
| windowed `avg_only`、60k、历史最佳 | 0.7163 | 0.6131 |
| raw CARE-aligned、60k、mean-MSE | 0.6931 | 0.5579 |
| raw CARE-aligned、60k、L2 | 0.6974 | 0.6150 |
| raw CARE-aligned、200k、L2 | **0.7731** | **0.7517** |

200k L2 主结果的 CARE 分项：Coverage F0.5=0.5881，Accuracy=0.9678，Reliability F0.5=0.9091，Earliness=0.3258。其优势是低误报和高事件级可靠性；下一步若优化 CARE，应扫描阈值以换取 Coverage/Earliness，而不重训模型。

## 当前实验

- 正在运行 raw CARE-aligned、200k、L2 配置下的 `latent=64` 对照；除 latent 从 128 降至 64 外其余参数不变。
- 目的：验证较小 latent 是否减少异常过度重构并进一步提高 AUC。此前窗口特征下的 latent 消融不能外推到本数据协议。

## Transformer 运维修复

- RAM-only FP16 window cache 曾在训练结束后与 adaptive threshold / prediction 评分叠加，32GB 主机可能被 OOM killer 终止。
- 已在 `f03ae38` 中于训练结束后关闭 persistent DataLoader workers、释放窗口缓存；在 `74aa778` 中仅缓存训练窗口，validation 保持惰性切片以降低内存峰值。
