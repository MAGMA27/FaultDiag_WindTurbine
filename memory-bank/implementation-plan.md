# 实施计划

## 已完成

- [x] CARE CSV 转 Parquet、事件信息与字段字典读取。
- [x] `train_test == prediction` 的评估边界与 status 0/2 事件资格审计。
- [x] 时间连续分段，避免窗口跨长时间缺口。
- [x] CARE score、逐事件告警、24/48 小时覆盖和月度误报报表。
- [x] PCA、LightGBM 条件残差、VAE、LSTM-AE、Transformer-AE 与 VAE+Transformer 集成。
- [x] `stat_aware` 特征工程：区分原始 10 分钟平均值与 min/max/std 统计量。

## 当前优先级

1. **CARE benchmark AE 对齐**：原始特征、论文表 4.2.2 的每风场 AE 参数、L2 误差与 adaptive threshold NN。
2. **基线复现审计**：记录原文未公开的风速/功率过滤阈值与其影响；不要将近似实现称为完全复现。
3. **模型对比定稿**：以同一数据协议输出 PCA、条件残差、VAE、Transformer 和集成的 CARE 报表。
4. **解释层**：输出传感器重构误差贡献、关键变量趋势和事件案例卡片。

## 不作为当前优先级

- 继续扩大 LSTM-AE 搜索空间；当前表现未超过 VAE/Transformer。
- 使用 prediction 结果挑选阈值或集成权重。
- 将异常检测结果直接表述为故障根因诊断。
