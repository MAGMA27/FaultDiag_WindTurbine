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

## Phase 4 — 冻结实验协议（当前最高优先级）
- [x] 训练/验证只使用 normal event 的 `train_test == "train"` 数据；评估只使用
  `train_test == "prediction"`。`run_gpu_tune.py` 已实现并有 helper 回归测试。
- [x] 筛选 `status_type_id in {0, 2}` 后再按时间缺口分段，禁止滚动特征或时序窗口跨段。
- [ ] 将协议提取为公共数据切分与评估接口，所有训练脚本统一复用；缺少 `train_test` 列时应显式报错，
  不可静默回退为全量数据。
- [ ] 阈值及其超参仅可由训练/验证期正常数据选择；prediction 数据不得用于调参。
- [ ] 每个正式结果保存数据版本、事件 ID、配置、seed、阈值校准数据与代码版本。
- **历史结果说明**：此前多次依据 Farm A prediction AUC 选择 beta、窗口和 latent，故这些结果仅是
  开发期探索证据，不得表述为独立最终测试。锁定选择规则后，以此前未参与调参的 Farm B/C 做一次性
  外部评估；Farm A 结果须标注为 development/audited result。

## Phase 5 — 事件级 CARE 评估（主交付）
- [x] 实现 `src/faultdiagnose/evaluation/care.py`，严格复现 Gück et al. (2024) CARE score；输入为每条
  prediction 时间序列的二值告警、事件标签、时间戳和运行状态。
- 只在正常状态点（CARE 定义的 status 0/2）打分：
  - Coverage：异常事件数据集的平均 `F_0.5`；
  - Accuracy：正常数据集的 `TN / (TN + FP)`；
  - Reliability：按 criticality 算法聚合事件，阈值 `t_c=72`（12 小时），再计算事件级 `F_0.5`；
  - Earliness：事件前半段权重为 1、后半段线性降至 0 的加权检出率；
  - CARE：上述平均子分数按论文权重 `(1, 1, 1, 2)` 合成，并实现“无告警为 0、Accuracy < 0.5
    时返回 Accuracy”两个特例。
- AUC-ROC 保留为无阈值的**辅助排序诊断**，不再作为主要模型选择目标；报告阈值、CARE 及四个子分数。
- [x] 将 CARE evaluator 接入正式训练/推理 runner；每次正式评估写出逐事件告警时间线/首次报警时间、
  24h 与 48h 预警覆盖率、每月误报次数。后三者是运维补充指标，不冒充 CARE 原始定义。
- [x] 每次 CARE 结果附带 `event_eligibility` 审计：事件窗内 prediction 行数、status 0/2 行数、
  可评分比例和点级资格。点级 AUC/Coverage/Earliness 仅解释为有资格事件的结果；全部事件仍用于
  Reliability。

## Phase 6 — 有意义的模型对照
- [ ] 统计基线：PCA（保留 99% 方差）+ Isolation Forest；与 CARE benchmark 的简单基线对齐。
- [ ] 工况条件残差基线：仅用正常训练数据，以风速、功率及 `feature_description.csv` 可识别的环境/工况变量
  预测关键传感器；以残差构造异常分数。变量与目标按风场配置，禁止假定匿名字段跨风场同名。
- [ ] 保留一个强时序模型（优先 LSTM-AE 或 Transformer-AE），在同一冻结协议和阈值机制下比较。
- [ ] 最后才做 VAE + LSTM + Transformer 集成，并将它作为消融结论：只有在单模型与简单基线都完成后，
  才评估其额外复杂度是否合理。

## Phase 7 — 可解释预警与作品集叙事
- [ ] 每个告警输出逐传感器重构误差贡献 Top-k、关键变量趋势及事件案例卡片；匿名特征仅按字段字典允许的
  语义解释。
- [ ] 项目定位为“无监督早期异常预警 / fault detection”，而非已能定位故障机理的“fault diagnosis”。
- [ ] README 与最终报告围绕以下主张组织：运行状态、时间断裂与特征冗余会扭曲点级指标；在严格事件级
  CARE 评估下，统计、工况残差与深度时序模型的取舍才可被可信比较。

## 已知坑 / 决策点
- 窗口 T、超参论文缺失 → 自定并文档化（ponytail: 先用合理默认值跑通）
- 数据集版本：我们 CARE（zenodo 15846963）实际 5.24M 行，论文表 I 写 8.99M（其引用链接 `CARE2024` 是占位符）→ **复现方法 + 趋势，不追求数字逐字一致**
- "正常训练集" = 51 个 normal dataset；44 个 anomaly dataset 用于测试打分
- CARE benchmark 论文是指标定义的权威来源：Gück et al. (2024), *CARE to Compare: A real-world
  dataset for anomaly detection in wind turbine data*；本地 PDF 已保存于 Zotero。


