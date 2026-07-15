# 系统架构 (Architecture)

## 高层数据流

```
CARE csv (per WT)
   │  DataLoader: 解析 `;` 分隔、时间戳、按 event_info 划分事件窗口、train/test
   ▼
特征工程 FeatureEngineering
   │  时序指标 + 统计指标 + 频域指标（Welch/FFT）
   ▼
模型集成 Models（无监督，仅正常数据训练）
   │  多 AE 变体（Dense/LSTM/CNN）+ Transformer 变体
   ▼
异常打分 Scoring
   │  各模型重构误差 → 集成异常分数
   ▼
自适应阈值 Thresholding
   │  百分位 τ ∈ [95, 99]（论文 Algorithm 1）
   ▼
评估 Evaluation
      AUC-ROC（分风场）、故障前早期检测小时数
```

## 模块划分（对应 `src/faultdiagnose/`）

- `data/`: CARE 加载器（读取 `datasets/*.csv`、`event_info.csv`、`feature_description.csv`，处理 `;` 分隔与匿名时间戳，按风场映射特征）
- `features/`: 特征工程（时序 / 统计 / 频域指标）
- `models/`: 自编码器与 Transformer 变体及其集成
- `training/`: 训练循环、早停、归一化统计
- `evaluation/`: 异常打分、自适应阈值、AUC-ROC 与早期检测指标

## 关键设计点

- **无监督**: 仅用正常数据训练，异常事件用于评估（与论文一致）
- **集成**: 多个结构不同的深度模型捕捉不同时间/上下文模式，集成打分提升鲁棒性
- **自适应阈值**: 不需要故障标签即可设定告警阈值（百分位法），契合无监督设定
- **可复现**: 种子固定、归一化统计随模型保存、实验输出写入 `results/`
