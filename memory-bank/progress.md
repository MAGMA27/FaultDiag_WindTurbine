# 实验进展

最后更新：2026-07-29

## 数据与评估

- CARE 已转换为 95 个分序列 Parquet；每行是 10 分钟 SCADA 统计值。
- 训练使用 normal event 的 `train` 分区；最终报告仅使用 `prediction` 分区。
- CARE evaluator 输出 Coverage、Accuracy、Reliability、Earliness、CARE、逐事件时间线、事件可评分性和月度误报。
- 运行状态会显著改变可评分事件范围：Farm B 的结果必须结合 `event_eligibility` 报表解释。

## Farm B 当前代表性结果

| 模型 | 配置摘要 | AUC-ROC | CARE |
|---|---|---:|---:|
| PCA | window 48、99% variance | 0.5738 | 0.3737 |
| VAE | `avg_only`、window 576、hidden 512、latent 128、beta 0.1 | 0.7163 | **0.6131** |
| Transformer-AE | `stat_aware`、window 576、seq 144、cross-attention | 0.7191 | 0.5868 |
| VAE + Transformer | validation percentile-rank 等权集成 | **0.7375** | 0.5926 |

结果文件：

- `results/20260717_2357_vae_optimized_result.json`
- `results/20260727_203619_689562_transformer_optimized_result.json`
- `results/20260729_150407_958358_vae_transformer_ensemble_result.json`

## 结论与决策

- `stat_aware` 将 Transformer 输入从机械展开的 1764 维缩减为 1071 维，并提升 AUC 与 CARE。
- VAE 单模型的 CARE 更高；VAE+Transformer 在 AUC、覆盖和提前量上更高。两者应并列报告，不用单一数值掩盖取舍。
- Dense AE 的窗口特征版本并非 CARE 官方 baseline，已降级为诊断实验，不作为正式对照。
- CARE 原始 benchmark 的 AE=0.66 是所有 95 个子序列的整体结果；不能与单一 Farm B 的严格运行状态结果直接等同。

## 进行中

- 新增 `scripts/run_care_official_ae.py`：对齐 CARE 论文 Farm A/B 的原始特征、AE 层宽、L2 重构误差和 adaptive threshold NN。
- 论文未公开风速/功率辅助过滤的具体阈值；首轮运行会把该缺口写入结果，后续作为敏感性分析处理。

## 后续

1. 运行并审计 CARE Farm B official adaptive-AE。
2. 补齐 Farm C 的固定阈值路径，完成全风场 benchmark 对照。
3. 汇总同协议模型表，补充传感器贡献与事件案例解释。
