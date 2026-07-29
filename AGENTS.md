# 项目说明 (Project Overview)

复现 Nair & Babu (2025) *Hybrid Autoencoder-Based Framework for Early Fault Detection in Wind Turbines*，在公开的 CARE 风电场 SCADA 数据集上实现无监督的早期故障检测。本项目定位为**求职作品集项目**，强调可复现、代码整洁、结果可解释。

- 论文: arXiv:2510.15010v1（Hybrid Autoencoder 框架：多种 AE / Transformer 变体集成 + 自适应阈值）
- 数据集: CARE（"CARE to Compare"，Zenodo 10.5281/zenodo.15846963），89 年 SCADA、3 风场、36 台机、95 个序列（44 异常 + 51 正常）
- 论文报告指标: AUC-ROC 0.947，故障前最早 48h 预警

# 环境管理 (uv)

本仓库用 `uv` 管理依赖与虚拟环境，**不要**用 `pip install` 直接装包。

- 同步环境: `uv sync`
- 加依赖: `uv add <pkg>`（torch 走 CUDA 源 `pytorch-cu124`，见 `pyproject.toml` 的 `[tool.uv.sources]`，无需手动指定 index）
- 运行脚本: `uv run python <script>.py`
- 跑测试: `uv run pytest`

Python 固定 `>=3.13.9,<3.14`；torch 默认安装 CUDA 版（`+cu124`，当前 `.venv` 内为 `2.6.0+cu124`），`uv run` 开箱即用 GPU。如需退回纯 CPU，改 `pyproject.toml` 的 `[tool.uv.sources]` 把 torch 指向 CPU index 后重新 `uv sync`。

# 代码风格 (Code Style)

- 校验: `uv run ruff check .`（规则 E/F/I/W，行宽 100，target py313）
- 公共函数与类**必须**加类型注解；模块级文档用简洁 docstring
- 命名: `snake_case` 函数/变量，`PascalCase` 类，模块全小写
- 优先复用本仓库已有工具，避免过度抽象（无单一实现的接口 / 工厂等）
- 不要把探索性代码留在一堆临时脚本里；可复用的逻辑进 `src/faultdiagnose/`

# 目录结构 (Directory Structure)

- `src/faultdiagnose/` 主包（data / features / models / training / evaluation 子模块将在实现阶段按 `memory-bank/architecture.md` 落地）
- `tests/` 单元测试与最小可运行验证
- `memory-bank/` 项目记忆（设计、架构、计划、进度）
- `data/`、`CARE_To_Compare/` 数据集（已被 `.gitignore` 忽略，不入库）
- `notebooks/` 探索性分析
- `results/` 生成的指标与图表（不入库，可复现生成）

# 数据处理约定 (Data Conventions)

- CARE 数据解压到仓库根目录的 `CARE_To_Compare/`；原始压缩包为 `CARE_To_Compare.zip`
- 文件为 **`;` 分隔**，时间戳格式 `YYYY-MM-DD HH:MM:SS`，已匿名
- 三风场特征数不同：Farm A=86、Farm B=257、Farm C=957；字段名各风场不一致，需按 `feature_description.csv` 映射
- 每风场含 `datasets/<id>.csv`、`event_info.csv`（事件窗口与标签）、`feature_description.csv`（字段字典）
- `status_type_id`: 0 正常发电 / 1 限功率 / 2 空转 / 3 维护 / 4 停机 / 5 其他（0、2 视为正常运行）
- 不要硬编码本机绝对路径；路径通过配置或相对路径（`DATA_ROOT` 约定）传入

# 复现与实验规范 (Reproducibility)

- 固定随机种子；每个实验的输出（指标 csv、图）写入 `results/` 并在 `memory-bank/progress.md` 记录
- 实现阶段对照论文 Algorithm 1：重构误差 → 异常分数 → 百分位阈值 τ∈[95,99] 自适应
- 评估指标：AUC-ROC（分风场）、故障前早期检测小时数；与论文基线（0.947 / 48h）对比

# Git 工作流

- 分支: `feature/<描述>`；提交遵循 Conventional Commits
- 不提交大文件、不提交密钥；数据集靠 `.gitignore` 排除

# 安全

- 仅使用公开 CARE 数据集；无密钥、无外部凭证
- 不在代码中硬编码他人机器路径


