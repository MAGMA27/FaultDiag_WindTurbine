# FaultDiagnose — 风电机组早期故障检测（论文复现）

> 求职作品集项目：复现 Nair & Babu (2025) *Hybrid Autoencoder-Based Framework for Early Fault Detection in Wind Turbines*，在公开 CARE 数据集上做无监督早期故障检测。

## 论文与数据集

- **论文**: Hybrid Autoencoder-Based Framework for Early Fault Detection in Wind Turbines, Nair & Babu, 2025 (arXiv:2510.15010v1)
- **数据集**: CARE（"CARE to Compare"）— Zenodo [10.5281/zenodo.15846963](https://zenodo.org/records/15846963)
  - 89 年 SCADA、3 个风场、36 台机、95 个序列（44 异常 + 51 正常）
  - 论文报告：AUC-ROC **0.947**，故障前最早 **48h** 预警
- **方法要点**: 多种自编码器 / Transformer 变体集成 + 特征工程（时序 / 统计 / 频域）+ 自适应百分位阈值（τ∈[95,99]），无监督训练于正常数据

## 环境搭建（uv）

```bash
uv sync                 # 创建 .venv 并安装依赖（torch 为 CPU 版）
uv run python -c "import torch; print(torch.__version__)"
```

数据集需另行下载并解压到 `CARE_To_Compare/`（已被 `.gitignore` 忽略，不入库）。

## 目录结构

```
src/faultdiagnose/   主包（data / features / models / training / evaluation）
tests/               测试
memory-bank/         项目记忆：设计 / 架构 / 计划 / 进度
notebooks/           探索性分析
results/             生成的指标与图表（可复现）
```

## 复现路线

详见 `memory-bank/implementation-plan.md`：数据加载 → 特征工程 → 多模型集成 → 异常打分与自适应阈值 → 评估（AUC-ROC / 早期检测窗口）。

## 当前进度

见 `memory-bank/progress.md`。
