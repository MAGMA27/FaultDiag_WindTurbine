# FaultDiagnose

基于公开 CARE 风机 SCADA 数据集的无监督早期异常检测复现与实验审计。

项目比较统计重构、工况条件残差与深度时序自编码器，并以 CARE 事件级指标评估告警的覆盖、可靠性、误报与提前量。

## 数据与参考

- 数据集：CARE — *CARE to Compare: A real-world dataset for anomaly detection in wind turbine data*，Zenodo [10.5281/zenodo.15846963](https://zenodo.org/records/15846963)
- 深度模型参考：Nair & Babu (2025), *Hybrid Autoencoder-Based Framework for Early Fault Detection in Wind Turbines*
- CARE 原始 benchmark：Gück et al. (2024)，提供 AE、Isolation Forest 与 CARE score 对照

CARE 包含 3 个风场、36 台机组、95 个序列（44 个异常事件、51 个正常序列）。每行是一条 10 分钟 SCADA 统计记录；`event_info.csv` 给出异常事件的时间窗。

## 方法与协议

训练只使用正常序列的 `train` 分区；评估仅使用 `prediction` 分区。运行状态、时间连续性和阈值校准均显式记录。

```text
正常 train SCADA
  ├─ PCA reconstruction baseline
  ├─ 工况条件残差（LightGBM）
  ├─ VAE / LSTM-AE / Transformer-AE
  └─ CARE 原始 adaptive-AE baseline
          ↓
验证期正常分数校准阈值
          ↓
prediction 分区 → AUC-ROC（辅助）+ CARE 事件级报表（主指标）
```

CARE 报表包括 Coverage、Accuracy、Reliability、Earliness、逐事件首次告警时间、24/48 小时覆盖和月度误报 episode。

## 当前实验快照

Farm B 的冻结 prediction-only 协议下：

| 模型 | 输入 | AUC-ROC | CARE | 说明 |
|---|---:|---:|---:|---|
| VAE | `avg_only`, 长窗口 | 0.7163 | **0.6131** | 单模型 CARE 最优 |
| Transformer-AE | `stat_aware`, seq=144 | 0.7191 | 0.5868 | 单模型时序对照 |
| VAE + Transformer | validation percentile rank 集成 | **0.7375** | 0.5926 | AUC、覆盖与提前量最优 |

这些结果是当前项目协议下的实验结果，并不等同于论文在不同切分和实现细节下报告的数值。CARE 原始 AE baseline 正在按论文的原始特征、L2 重构误差与自适应阈值机制复现。

## 快速开始

### 1. 环境

```powershell
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```


```bash
git clone <repository-url> FaultDiagnose
cd FaultDiagnose
uv python install 3.13
uv sync
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

服务器的系统 CUDA toolkit 不必与项目 wheel 的 CUDA runtime 完全一致；`uv sync` 会按锁定依赖创建隔离环境。运行前仍应通过 `nvidia-smi` 确认 NVIDIA 驱动和 GPU 可见。

### 2. 准备数据

将 CARE 解压到 `CARE_To_Compare/`，随后转换为分序列 Parquet：

```powershell
uv run python src/faultdiagnose/data/load_care.py
```

### 3. 运行代表性实验

PCA 基线：

```powershell
uv run python scripts/run_pca_baseline.py --farms B --window 48 --variance 0.99
```

CARE Farm B 原始 adaptive-AE baseline：

```powershell
uv run python scripts/run_care_official_ae.py --farms B --cap-train 0
```

VAE 与 Transformer 集成复评：

```powershell
uv run python scripts/run_vae_transformer_ensemble.py --vae-weight 0.5 --batch 512
```

每个 runner 在 `results/` 写入配置、标量结果及 CARE JSON/CSV 报表。

## 仓库结构

```text
src/faultdiagnose/
  data/          CARE 读取与状态约定
  features/      窗口特征与特征审计
  models/        AE、VAE、LSTM-AE、Transformer-AE
  evaluation/    AUC、CARE 与集成校准
scripts/         可复现实验入口
tests/           单元与回归测试
memory-bank/     设计、计划与实验进展
```

## 开发检查

```powershell
uv run ruff check .
uv run pytest
```
