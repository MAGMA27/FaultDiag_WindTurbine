# 实施计划 (Implementation Plan)

基于论文原文 (Nair & Babu 2025, arXiv:2510.15010v1) 的方法拆解，作为复现路线。

## Phase 0 — 项目搭建 ✅
- [x] `uv` 管理依赖；`AGENTS.md` / `README.md` / `memory-bank/`

## Phase 1 — CARE 数据加载器 ✅
- [x] `src/faultdiagnose/data/load_care.py`：`convert_care_to_parquet` + 读取端
- [x] 已转 Parquet：95 数据集 / 5,242,948 行（A:22 / B:15 / C:58）
- [x] `tests/test_data_loader.py`：4 passed

## Phase 2 — 特征工程 (`src/faultdiagnose/features/`) ✅
论文 Section II.A：对**原始 SCADA 信号**在滑动窗口上提取三类指标
- 时序：移动平均、导数（局部动态）
- 统计：均值、方差、偏度(skewness)、峰度(kurtosis)
- 频域：FFT 特征（机械振动）
- 归一化：z-score，μ/σ 在训练集上全局计算
- **窗口长度 T 论文未给 → 需自定**（建议 24h=144 步 或 12h=72 步，做成可配置）
- 输入：各风场原始传感器列（A=81 / B=252 / C=952 维）

## Phase 3 — 模型 ✅ (VAE baseline done, LSTM/Transformer/ensemble pending) (`src/faultdiagnose/models/`) 三个 AE 变体
- VAE：encoder q(z|x) + decoder p(x|z)；loss = α·L_rec + β·D_KL
- LSTM-AE：LSTM 编码时序 → 隐状态 → FC → z → 解码重构
- Transformer-AE：自注意力捕长期依赖
- 异常分数 = 重构误差
- **超参（隐维度 / 层数 / epoch / αβ / 集成权重）论文未给 → 合理设定并写明**

## Phase 4 — 训练 + 打分 + 集成 + 阈值
- 无监督：仅用 **51 个 normal dataset** 训练
- 重构误差 → 各模型异常分数
- 集成：加权融合三模型分数
- 自适应阈值：τ = percentile(score, p)，p∈[95,99]（论文 Algorithm 1）

## Phase 5 — 评估 (`src/faultdiagnose/evaluation/` + `results/`)
- AUC-ROC（分风场，主指标）
- F1 / Precision / Recall
- 早期检测：故障前 24/48/72/96h 检出率（用 `events.parquet` 的 `event_start` 定位）
- 对照论文基线：Ensemble AUC A=0.912 / B=0.947 / C=0.963（均值 0.947）；早期检测 24h=92.2%、48h=88.6%

## Phase 6 — 作品集打磨
- `notebooks/` 探索；结果图（ROC、早期检测表）；可一键复现说明

## 已知坑 / 决策点
- 窗口 T、超参论文缺失 → 自定并文档化（ponytail: 先用合理默认值跑通）
- 数据集版本：我们 CARE（zenodo 15846963）实际 5.24M 行，论文表 I 写 8.99M（其引用链接 `CARE2024` 是占位符）→ **复现方法 + 趋势，不追求数字逐字一致**
- "正常训练集" = 51 个 normal dataset；44 个 anomaly dataset 用于测试打分


