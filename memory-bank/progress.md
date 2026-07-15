# 进度 (Progress)

## 当前阶段：Phase 3 VAE 基线跑通，下一步补 LSTM-AE / Transformer-AE 并做集成

## 已完成
- **Phase 0** 项目搭建
- **Phase 1** CARE 数据加载器 → Parquet（95 数据集 / 5.2M 行）
- **Phase 2 特征工程**（滑动窗口均值/标准差/偏度/峰度/导数；FFT 可选禁用）
- **Phase 3 VAE 端到端**
  - `src/faultdiagnose/models/vae.py`：MLP-VAE，per-time-step 重构，loss = α·L_rec + β·D_KL
  - `training/vae_trainer.py`：Adam + grad clip + 早停预留
  - `evaluation/anomaly.py`：AUC（ROC，event-window 标注）
  - `scripts/run_vae.py`：一键训练+评估，可配参数
  - **Farm A VAE 基线 AUC-ROC = 0.5503**（60k 训练向量，20 epochs，无 FFT 特征；单模型，未集成）
  - 端到端耗时 ~68s（CPU），无 FFT 瓶颈
  - 全量测试 `uv run pytest` → 12 passed；`uv run ruff check` → clean

## VAE 基线 (Farm A)
| 配置 | 值 |
|------|-----|
| 特征 | 81 原始列 × 5(无 FFT) = 405 维 |
| 训练 | 60k 向量，20 epochs，batch 256 |
| 测试 | 22 datasets，1.2M 向量，event-window 标注阳性 18k |
| AUC-ROC | **0.5503** (论文 ensemble=0.912) |

## 下一步
- Phase 3 补充 LSTM-AE + Transformer-AE
- Phase 4 集成打分 + 自适应阈值
- 性能优化：FFT 特征向量化（当前 rolling.apply 是瓶颈）
- 多风场评估 + 早期检测指标

## 目标指标（对照论文）
- Ensemble AUC-ROC ≈ 0.947（A=0.912/B=0.947/C=0.963）；早期检测 24h=92.2%、48h=88.6%

## 已知坑
- 单 VAE 基线 AUC 0.55 vs 集成 0.912；差 0.36 需要 LSTM/Transformer + 集成 + 调参来追
- FFT 特征需向量化重写（当前太慢）
