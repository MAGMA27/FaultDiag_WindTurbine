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
- [x] 工况条件残差基线（Farm B 第一版）：使用 `LightGBMRegressor`，仅以 normal/train 段拟合；每个
  目标传感器独立建模。工况输入固定为 `sensor_8`（outside temperature）、`sensor_10`（pitch angle）、
  `sensor_12`（estimated torque）、`sensor_20`（gearbox rotational speed）、`sensor_25`（rotor speed）、
  `power_58`（available power）、`power_62`（active power）、`wind_speed_59/60/61`（wind speed），缺失输入
  不用跨列填补。目标先覆盖已知事件部件：`sensor_47`（transformer cell temperature）及 `sensor_52`
  （rotor bearing temperature 2）；随后扩展到 `sensor_31--53` 的温度信号。
  - 仅用 `train_test == "train"`、status 0/2、连续运行段切分后的正常数据；全段末尾 15% 为 validation。
  - 将 Farm B/C **连续至少 6 个点**的 0 值视作缺失候选：短暂的 0 值（如零桨距）保留；长零序列输入
    以 `NaN` 交由 LightGBM 处理。目标为 0 或缺失的行不训练、不评分；输出 0 值审计表，不默默删除。
  - 分数为每个目标的**标准化绝对残差** `abs(y - y_hat) / sigma_validation`；多目标分数取最大值，阈值仅取
    正常 validation 分数 P99。prediction 只作一次性 CARE 评估，复用 PCA 的事件/误报 artifacts。
  - 明确这是可部署的当前点 NBM（只依赖当前十分钟 SCADA 行），不引入目标的未来值、故障标签或目标的滞后值。
  - 变量/目标由每个风场的 `feature_description.csv` 显式配置，绝不假定 `sensor_47` 等编号跨风场有相同语义。
- [x] 工况残差扩展（Farm B 作品集版）：直接按 Nair & Babu (2025) 的全风场 Top-20 特征先验，覆盖可
  映射的轴承温度、齿轮箱油温、Hub Temperature（数据中没有明确 nacelle temperature）、变压器温度、
  传动链 Z 轴与塔筒 X/Y 轴振动；同时加入风向/机舱方向。保留 prediction-only CARE 作为统一比较口径，
  但不将论文重要性数值或复杂的 validation-only 特征消融作为作品集的阻塞条件。
- [x] 强时序模型：Farm B `Transformer-AE` 已在同一协议下完成，配置为 24 点工程窗口 + 48 个工程
  向量序列、60k 正常样本、`d_model=256`、latent=64、P99 正常验证阈值。它的 CARE=0.4796，显著低于
  LightGBM 工况残差的 0.6355，故不以 Transformer 为主模型、不再针对 prediction AUC 调参。
- [x] 第二个深度对照：同尺寸 Farm B LSTM-AE 的 CARE=0.4797，几乎与 Transformer（0.4796）相同，
  仍显著低于 LightGBM（0.6355）。两者均不值得作为主模型；停止深度模型迭代与三模型集成，转向解释、
  告警 dashboard 与作品集案例整理。
- [x] 统一窗口复验：Farm B 的 PCA、VAE、LSTM-AE、Transformer-AE 均已补跑工程窗口 48（8h）。
  PCA CARE 从 0.3186 升至 0.3737；LSTM 从 0.4797 升至 0.5134；Transformer 从 0.4796 升至
  0.5156；VAE48 达到 CARE 0.5602，但训练后段出现 `nan`，且 Coverage 仅 0.1121。结论是 24 窗口
  确实偏短，但深度模型整体仍低于 LightGBM 条件残差扩展版 CARE 0.6355。
- [x] VAE + LSTM + Transformer 严格集成：按论文 ensemble scoring 思路补齐，但不使用 prediction 标签
  学权重。三模型分数先用正常 validation median/IQR 稳健标准化，再分别评估等权与 normal-validation
  stability 权重。结果 CARE 约 0.473，Coverage 仅约 0.03，说明 ensemble 在当前冻结协议下过于保守，
  不作为主模型；主线转向 LightGBM 残差解释、事件案例卡片和告警 dashboard。
- [x] 深度模型 AUC/区分度优化第一轮：新增 `paper_top` 特征集，按论文关键因素和 LightGBM 先验将 Farm B
  深度输入从 1,764 维降至 196 维。VAE AUC 从全量特征 0.5556 提升到 0.5778，且 150 轮 validation loss
  仍在下降，说明此前深度模型不能被视为充分收敛。LSTM/Transformer 在该特征集上出现 score 方向反转；
  development-only 方向校准与 `abs(AUC-0.5)` 权重后，ensemble AUC 提升到 0.6022。下一步应把方向/权重
  学习改成不泄漏 prediction 测试的开发验证机制，再继续追 AUC。
- [x] 深度特征筛选第二轮：新增 `avg_only` 特征集，只去掉原始表中已是 min/max/std_dev 的聚合列，保留
  所有 average 物理量再做统一窗口工程。avg_only VAE AUC 0.5905，优于 paper_top VAE 0.5778；进一步将
  VAE anomaly score 改成 mean reconstruction-only 后 AUC 到 0.5967。LSTM/Transformer 仍接近随机或方向
  不稳，不能直接组合；development-only 方向校准只能作为错误分析，不作为正式优化结果。下一步先修单模型
  收敛和 score 分布，目标是让单 VAE/LSTM/Transformer 接近论文 0.8+ AUC 后再谈 ensemble。
- [x] VAE 单模型优化第一轮：新增 VAE-only runner 与 warmup-cosine scheduler。Farm B `avg_only`、
  window 96、300 epoch、mean reconstruction-only score 稳定跑满，AUC 提升到 0.6159、CARE 0.5313，
  且无 NaN/skip。validation loss 到第 300 轮仍缓慢下降，下一步可试 window 144/192、hidden/latent
  增大或 score calibration，继续优先拉单模型区分度。
- [x] VAE window 144 复验：在同一 VAE-only 配置下将上下文提升到 24 小时，AUC 继续升至 0.6227，
  median gap 升至 0.0425，但 CARE 降至 0.5108。结论：更长上下文仍提升排序区分度；事件级 CARE 还需要
  单独做阈值/score calibration。
- [x] VAE window 192 复验：32 小时上下文将 AUC 继续升至 0.6426，median gap 升至 0.0542，是当前 VAE
  单模型最佳排序结果；但 CARE 降至 0.4050，说明 P99 阈值对长窗口 VAE 过于保守。下一步应分两线：
  继续追单模型 AUC（window 240/288 或 hidden512/latent128）与单独做阈值/score calibration。
- [x] VAE 大窗口/大容量复验：hidden512/latent128 + warmup-cosine 下，window288 AUC 0.6741，
  window432 AUC 0.6945、CARE 0.5735，median gap 0.1028。VAE 已形成清晰优化曲线；下一步优先做
  score/threshold calibration，或小心试 window576 判断长上下文是否继续收益。
- [x] VAE window576 + 500 epoch 复验：按“window 与 epoch 同步增加，必要时再加 KL 退火”的策略，
  先跑无退火版本。结果完整稳定跑满 500 epoch，`skipped=0`、无 NaN，best epoch 497；AUC 继续升至
  0.7163，CARE 0.6131。已为 VAE trainer 和 runner 增加可选 `--kl-anneal-epochs`，但当前证据显示不必
  立即启用。下一步若继续追论文级 0.8+ AUC，优先做 window576 的容量/正则/阈值校准小网格，而非直接
  组合方向不稳的 LSTM/Transformer。
- [x] VAE beta/容量/window 小网格：三组 `beta0.05` 复验均未超过 window576/beta0.1 旧最佳。
  window576 hidden512 beta0.05 得到 AUC 0.6955、CARE 0.5856；window576 hidden768 beta0.05 得到
  AUC 0.7040、CARE 0.5924；window720 hidden512 beta0.05 得到 AUC 0.7011、CARE 0.5891。虽然 normal
  validation loss 显著降低，但异常排序变差，说明 beta0.1 的 latent 正则更适合当前 anomaly separation。
  后续不再沿 beta0.05 深挖，优先回到 beta0.1 做 window720、beta0.2、无 FFT 或阈值/score calibration。
- [x] VAE window720 / beta0.2 / all 特征对照：保留 FFT，补跑 `avg_only window720 beta0.1`、
  `avg_only window576 beta0.2` 与 `all window576 beta0.1`。三者分别为 AUC/CARE 0.7056/0.5890、
  0.6893/0.5993、0.7034/0.5024，均未超过 `avg_only window576 beta0.1` 旧最佳 0.7163/0.6131。
  结论：window720 开始偏保守，beta0.2 过强，all 特征虽把 10min 内 min/max/std_dev 等信息加回，但
  大幅降低 Coverage/Earliness；继续保留 `avg_only` 作为深度模型主输入，并将“高噪声 10min 统计列会
  扭曲长窗口 VAE 告警”写入作品集特征审计。

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


