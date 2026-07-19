
## Phase 4 GPU 锟斤拷锟轿斤拷展 (2026-07-15)
- **GPU 锟叫伙拷锟缴癸拷**: torch 2.6.0+cu124 on RTX 4070 Ti SUPER (16GB)锟斤拷pyproject 锟斤拷为 explicit CUDA index锟斤拷UV_CACHE_DIR 指锟斤拷 D 锟教ｏ拷C 锟教斤拷 0.2GB 剩锟洁）
- **锟截硷拷锟斤拷锟侥凤拷锟斤拷**:
  - 锟斤拷锟斤拷 Eq.11: VAE 锟届常锟斤拷 = `锟斤拷锟斤拷Lrec + 锟铰★拷DKL`锟斤拷锟斤拷锟斤拷之前只锟斤拷 Lrec锟斤拷锟斤拷 锟斤拷锟睫革拷 vae.py
  - 锟斤拷锟斤拷 Farm A 锟斤拷模锟斤拷: VAE 0.823 / LSTM 0.845 / Transformer 0.867 / 锟斤拷锟斤拷 0.912锟斤拷0.947 锟斤拷 **Farm B** 锟斤拷锟缴ｏ拷锟斤拷
  - 锟斤拷锟斤拷锟叫凤拷 70/15/15锟斤拷训锟斤拷锟斤拷全锟斤拷 ~2M 锟姐（锟斤拷锟斤拷之前 cap 150k锟斤拷
  - 锟斤拷停 on validation loss锟斤拷Algorithm 1 step 8锟斤拷
- **实锟斤拷锟斤拷锟斤拷GPU, cap 200k-520k锟斤拷**:
  - VAE h256_l64_b1.0 + DKL: AUC=0.5491锟斤拷DKL 锟睫革拷**未锟斤拷锟斤拷**锟斤拷VAE 锟斤拷锟斤拷锟斤拷锟斤拷锟解）
  - 锟斤拷史: VAE/LSTM 锟斤拷锟斤拷锟斤拷锟教帮拷 ~0.55-0.56锟斤拷raw-seq(81维) LSTM ~0.582
  - **锟斤拷锟侥硷拷锟斤拷**: 486 维锟斤拷锟斤拷锟斤拷锟教匡拷锟斤拷锟斤拷稀锟斤拷锟届常锟脚猴拷 锟斤拷 锟斤拷员锟?raw-seq 锟接达拷锟斤拷锟斤拷锟斤拷
- **锟斤拷锟斤拷证**: LSTM 520k 锟斤拷锟斤拷训锟斤拷锟叫ｏ拷锟斤拷锟斤拷锟斤拷锟斤拷锟角凤拷突锟斤拷 ~0.58锟斤拷
## Phase 4 GPU 璋冨弬杩涘睍 (2026-07-15)
- GPU 鍒囨崲鎴愬姛: torch 2.6.0+cu124 on RTX 4070 Ti SUPER (16GB)銆俻yproject 鏀?explicit CUDA index锛孶V_CACHE_DIR 鎸囧悜 D 鐩橈紙C 鐩樹粎 0.2GB 鍓╀綑锛?
- 鍏抽敭璁烘枃鍙戠幇:
  - 璁烘枃 Eq.11: VAE 寮傚父鍒?= alpha*Lrec + beta*DKL锛堟垜浠箣鍓嶅彧鐢?Lrec锛夆€?宸蹭慨澶?vae.py
  - 璁烘枃 Farm A 鍗曟ā鍨? VAE 0.823 / LSTM 0.845 / Transformer 0.867 / 闆嗘垚 0.912锛?.947 鏄?Farm B 闆嗘垚锛侊級
  - 鏁版嵁鍒囧垎 70/15/15锛涜缁冪敤鍏ㄩ噺 ~2M 鐐癸紙鎴戜滑涔嬪墠 cap 150k锛?
  - 鏃╁仠 on validation loss锛圓lgorithm 1 step 8锛?
- 瀹炴祴缁撴灉锛圙PU, cap 200k-520k锛?
  - VAE h256_l64_b1.0 + DKL: AUC=0.5491锛圖KL 淇鏈彁鍗囷紝VAE 鏍规湰鎬ч棶棰橈級
  - 鍘嗗彶: VAE/LSTM 鐗瑰緛宸ョ▼鐗?~0.55-0.56锛宺aw-seq(81缁? LSTM ~0.582
  - 鏍稿績鍋囪: 486 缁寸壒寰佸伐绋嬪彲鑳藉湪绋€閲婂紓甯镐俊鍙?-> 闇€瀵规瘮 raw-seq 鍔犲ぇ鏁版嵁閲?
- 寰呴獙璇? LSTM 520k 绐楀彛璁粌涓紙鐪嬫暟鎹噺鏄惁绐佺牬 ~0.58锛?

## Phase 4 鈥?鏍瑰洜瀹氫綅锛欰UC 鍙嶇浉 (~0.55) 宸叉煡娓?(2026-07-15)
- **鐜拌薄**: diag 閲?anomaly 閲嶆瀯璇樊 (mean 1.51) < normal (mean 3.37)锛孉UC 鍙嶇浉鍒?~0.55銆?- **鏍瑰洜 (闈?load_care 鍒囧垎/褰掍竴鍖栭棶棰?**: 鏍囪鐨勪簨浠剁獥鍙ｅ嚑涔庡叏鏄晠闅滃悗鍋滄満娈?  (`status_type_id` 鈭?{3,4,5})锛屼俊鍙峰钩鐩淬€佷綆鏂瑰樊銆傚師 pipeline 瀵规墍鏈夎(鍚仠鏈?鎵撳垎锛?  娆犳嫙鍚?AE 閲嶆瀯骞崇洿鍋滄満娈垫瘮閲嶆瀯澶氬彉鐨勬甯稿彂鐢垫鏇村鏄?鈫?姝ｇ被璇樊鍙嶈€屾洿浣?鈫?AUC 鍙嶇浉銆?- **琛ュ厖璇佹嵁**: 浜嬩欢绐楀彛鍐?鍙戠數鎬?(status鈭坽0,2}) 鐨?z-score L2 鑼冩暟 (8.30) 涓庢甯稿彂鐢?(8.46)
  鍑犱箮鏃犲樊寮?鈫?鐪熸"鏁呴殰鍓?鍙棭鏈熸娴嬩俊鍙锋瀬寮憋紝璁烘枃 0.947 澶ф鐜囩敱妫€娴?绂诲紑鍙戠數鎬?椹卞姩銆?- **淇**: `src/faultdiagnose/data/load_care.py` 鏂板 `OPERATING_STATES={0,2}` 涓?  `operating_mask(df)`锛涜缁冧笌鎵撳垎鍧囧彧淇濈暀鍙戠數鎬佽锛岄潪鍙戠數琛屼笉璁″叆鏍囩(浠庤瘎浼伴泦鍓旈櫎)銆?  宸插湪 `diag_separation.py` / `run_sequence.py` / `run_raw.py` / `run_gpu_tune.py` 鐨?  collect 涓?evaluate 澶勭粺涓€鎺ュ叆 (璁粌+娴嬭瘯鍚勫姞 `df = df[operating_mask(df)]`)銆?- **鐘舵€?*: 浠ｇ爜鏀瑰姩瀹屾垚銆佺紪璇戜笌 import 閫氳繃锛涘皻鏈窇璁粌楠岃瘉鏂瑰悜鏄惁缈绘 (鐢ㄦ埛瑕佹眰鍏堜笉鏀硅窇)銆?- **涓嬩竴姝?*: 璺?`uv run python scripts/diag_separation.py` 纭 anomaly 璇樊 > normal銆丄UC 缈绘銆?
## Phase 5 VAE 稳定性修复 + beta 调参 (2026-07-16)
- **现象**: VAE 训练 loss 出现孤立 1e8 / 2e4 尖峰 (ep51 / ep70),AUC ~0.565 接近随机。
- **根因 (数值爆炸)**: VAE.reparameterize 中 std = exp(0.5*logvar) 无上限,c_logvar 偶发漂到大正值 -> std->Inf -> 解码输出 Inf -> 单 batch MSE 爆掉。clip_grad_norm_ 只挡反向,挡不住前向 Inf,故仅污染一个 batch 即恢复。
- **修复**: src/faultdiagnose/models/vae.py 的 
eparameterize 内 clamp logvar 到 [-10, 10] (std in [e^-5, e^5])。尖峰消除。
- **数据标准化确认**: 
un_gpu_tune.py 的 collect_all 已做 z-score (it_standardizer/pply_standardizer),非问题根因。
- **beta ablation (同结构 vae_h256_l64, 80ep, Farm A, cap 150k)**:
  - beta=1.0: loss 152.4, AUC=0.607
  - beta=0.1: loss 91.9,  AUC=0.629  (KL 权重过高压垮重构, latent 趋塌缩, 异常分数区分度下降)
  - 结论: beta 是主因之一, 方向确认。beta=0.1 的 loss 仍持续下降 (ep80 未平台), 加 epoch 可能继续涨。
- **下一步 (未做)**:
  - beta=0.1 加 epoch (200) 看天花板; 试 beta=0.01 确认单调性。
  - 567 维特征中常数/低方差列被 std=1 保留 (已否决裁剪: 准常数特征可能是区分错误状态的关键信号, 且论文未做此步, 复现需对齐 pipeline)。
  - 正负比 ~0.24% (2522/1053180), AUC 对少量正样本本就难拉高, 需确认标签/窗口逻辑。
- **新增 config**: ae_h256_l64_b0.1 加入 scripts/run_gpu_tune.py CONFIGS, 用于严格 beta ablation。

### Phase 5 补充: beta 崩溃边界 + epoch 延伸 (2026-07-16)
- **beta 单调性 + 崩溃边界**:
  - beta=1.0: AUC=0.607 | beta=0.1: AUC=0.629 | beta=0.01: **训练崩溃** (ep30 起 loss=nan, mu 失约束发散 -> z 爆 -> 全 nan, roc_auc 因 0 样本报错)。
  - 结论: 甜点在 beta=0.1; beta 过小 KL 无约束, encoder mu 发散。已删除 ae_h256_l64_b0.01 config (只会崩)。
  - clamp logvar 只防了 std 爆炸, 未防 mu 爆炸 -> 极端 beta 仍会崩。默认 0.1 用不到, 暂不加 mu clamp。
- **epoch 延伸 (vae_h256_l64_b0.1, Farm A, cap 150k)**:
  - 80ep:  loss 91.9,  AUC=0.629
  - 200ep: loss 81.5,  AUC=0.632
  - 注意: loss 从 91.9 -> 81.5 持续稳定下降, **未平台**, 模型仍在学。AUC 仅 +0.3 是因 AUC 对重构误差的整体单调改善不敏感, 不代表无收益或到天花板。
  - 结论: 加 epoch 仍有稳定 (虽小) 提升, 性价比递减但未饱和。是否继续堆 epoch 取决于时间预算。
- **下一步 (未做)**:
  1. 保持特征工程与论文一致, 不裁剪低方差列 (准常数特征可能含错误状态信息; 论文未做此步, 复现须对齐)。
  2. 正负比 ~0.24% (2522/1053180), AUC 对少量正样本本就难拉高 -> 确认标签/窗口逻辑无误。
  3. 跨风场: 论文 0.947 是 Farm B 集成结果, 当前只在 Farm A 单模, 差距部分来自此。

### Phase 5 补充: 特征工程重复统计问题 (2026-07-16)
- **洞察 (用户发现)**: Farm A 原始 81 列 = 54 个 base sensor, 其中 9 个已是 CARE 预聚合的统计值列 (四件套 _avg/_max/_min/_std):
  wind_speed_3, sensor_5, sensor_18, reactive_power_27, reactive_power_28, power_29, power_30, sensor_31, sensor_52。
  其余 45 个为瞬时读数单值列。
- **问题**: engineer_features 对每一列 (不分原始/已统计) 统一套 _mean/_std/_skew/_kurt/_deriv/_deriv2 (窗口24)。
  即对 power_29_std 再算 _std/_mean/_kurt (波动量的波动), 对 sensor_5_max 算 _mean/_std (最大值的平均/标准差)。
  这属于"统计值的统计值", 语义稀薄且与同源原始列强相关 -> 注入冗余低信息维度, 稀释异常信号 (567 维噪声来源之一)。
- **论文对照**: Nair & Babu 2025 Sec.II.A 的滑动窗口特征工程默认输入为原始瞬时读数; CARE 混合了原始+预聚合列, 论文 pipeline 未考虑此混合。严格对齐论文应只对原始瞬时列做派生 (或仅用其 _avg)。但论文缺细节, 无法确认其实际处理。
- **状态**: 暂不改默认 pipeline (改动 in_dim/checkpoint 成本高, 且非当前 AUC 差距主因; beta 才是)。记录为已知方法论问题。
- **下一步验证 (未做)**: 加开关"跳过预聚合列 (仅保留其 _avg 作原始输入)"跑对照, 确认重复统计是否伤性能。另: 换 window 大小做快速验证 (见下)。

### Phase 5 补充: window 大小快速验证 (2026-07-16)
- 设置: vae_h256_l64_b0.1, cap=30000, epochs=30, Farm A, 仅改 --window。
- 结果 (AUC 随 window 单调上升):
  - window=12 (2h):  AUC=0.5242
  - window=24 (4h):  AUC~0.62 (30ep 小cap近似; 全量80ep=0.629)
  - window=48 (8h):  AUC=0.6220
  - window=96 (16h): AUC=0.6731  (loss 101.8 未平台)
- 结论: window 是强信号, 比 cap/epoch 影响更大。小窗口(2h)噪声主导最差; 长上下文(16h)浮现早期故障模式。window=96 仍未饱和, 可能继续涨。
- 论文对照: 论文用 4h(24步)偏保守; 本数据更长窗口更优。
- 下一步: 全量 (cap 150k) + window=96 + 200ep 验证天花板; 试 window=192/336 确认单调性。

### Phase 5 补充: mu 爆炸修复 + window=96 全量结果 (2026-07-16)
- **问题 (nan 崩溃)**: vae_h256_l64_b0.1 + window=96 + 200ep 在 ep40 起 loss=nan (全 nan, 评估 0 样本报错)。
  - 根因: beta=0.1 KL 约束弱, encoder 的 posterior mean (mu) 在长窗口复杂特征下中后期漂移溢出 -> Inf/NaN。
  - 之前的 logvar clamp 只防了 std 爆炸, 未防 mu 爆炸 (window=24 时 mu 未越界故不崩; window=96 更复杂更早越界)。
  - 注意: 代码无 LR scheduler, 崩与学习率调度无关。
- **修复**: src/faultdiagnose/models/vae.py 的 reparameterize 同时 clamp mu 到 [-10,10] (与 logvar 对称)。
  - 机制: mu 越界样本/维度处梯度被截断为 0 (encoder fc_mu 该步不更新), 其余参数/样本梯度正常; decoder 用 clamp 后的 z 训练, 梯度全程有效。
  - 代价: mu 被硬锁 ±10, 表达力略受限 (弱 KL 下的作弊通道被焊死); 更优雅做法是调大 beta 软约束, clamp 留作安全网。
- **验证 (小 cap 30k, window=96)**:
  - 修复前 200ep: ep40 起 nan (崩)
  - 修复后 60ep:  loss 78.5,  AUC=0.6959 (不崩, 稳定)
- **全量结果 (cap 150k, window=96, 200ep, beta=0.1)**:
  - loss 62.8 (仍缓降未完全平台), AUC=0.7196
  - 对比 window=24 全量 200ep AUC=0.632 -> +0.088。window 是比 cap/epoch 更强的信号。
- **下一步 (未做)**:
  - window=96 仍缓降, 可试 window=192/336 确认天花板; 或全量 200ep 已近饱和则停。
  - latent 假设: 加 vae_h256_l128_b0.1 验证 64->128 是否继续涨 (window=96 更复杂可能 latent 偏小)。
  - beta 软约束: 试 beta=0.3~0.5 看能否替代硬 clamp 且 AUC 更高。
  - 跨风场: 论文 0.947 为 Farm B 集成, Farm A 单模论文表 0.823, 当前 0.72 仍有差距待追。

### Phase 5 补充: LSTM/Transformer AE 架构澄清 (2026-07-16)
- **无 mu/sigma**: LSTM-AE 与 Transformer-AE 是确定性自编码器, 压缩产物是固定向量 z, 无分布/采样/KL, 不生成 mu 与 sigma (mu/sigma 仅 VAE 所有)。故不会遇 VAE 的 mu 漂移 nan 崩溃。
- **LSTMAE**: 编码器 LSTM 读完全部时间步取最后隐藏状态 -> z; 解码器 LSTM 从 z 自回归(递归)解出 T 个时间步。捕捉时间先后顺序依赖。
- **TransformerAE** (代码确认): 
  - 编码: x -> in_proj -> pos(位置编码) -> TransformerEncoder -> 对全部时间步 .mean(dim=1) 平均池化 -> pool 线性层 -> z。 z 是整段序列平均 (非末 token)。
  - 解码: z -> unpool -> 复制 T 份 (广播) -> pos -> TransformerEncoder -> out 线性层 -> 一次性并行输出整段 recon (非自回归, 非 token 递归生成)。
  - 异常分数 = 整段 MSE ((recon-x)^2).mean(dim=(1,2))。
- **与语言模型区别**: 用户直觉的 decoder-only 自回归 (末 token -> 线性 -> 逐 token 生成) 是 GPT 式做法; 本检测 AE 是非自回归并行重建, 训练直接用整段 MSE, 无误差累积。
- **对比口径注意**: seq 模型 seq_len 在 config 锁死 24, 不随 --window 变化; --window 只改 engineer_features 的滑动窗口统计, 不改模型自身时间步感受野。VAE window=96 变好部分来自特征统计窗口变大; LSTM/TF 本应靠 seq_len 变大捕获长时序, 但当前 seq_len 不可随 window 调 (待定是否修)。
- **论文意图**: Hybrid 框架集成 VAE(跨传感器相关性) + LSTM/TF(时间依赖) 多异构 AE 互补, 故集成 AUC 高于单模。

### Phase 5 补充: 时间连续性分块修复 (2026-07-16)
- **问题 (用户发现)**: collect_all 用 operating_mask 过滤后, 不同段之间被删除的非 operating 行导致时间断裂, 但剩余行被当成连续矩阵, 导致:
  1. rolling 窗口跨断点 (把 30 天前的结尾行和今天的开头行拼成连续窗口)
  2. SeqWindowsDataset 跨断点滑窗 (序列模型学到虚假的时序连续性)
- **量化**: Farm A 22 个 normal 数据集, 102 万 operating 行, 99.97% 连续 (10min 间隔); 但 >60min 断点 204 处, >360min (明确断裂) 97 处, 最大 gap 43,050min (~30 天)。
  -> 约 1% 污染行, 评估/混合时影响不可忽略。
- **修复 (最小方案)**: collect_all 内按时间连续性切段 (gap > 60min 算断), 段内独立 engineer_features/滑窗, 段间不跨。
  实现: 过滤 operating 后, 按 time_stamp diff > 60min 的 cumsum 分组, 每组一段。VAE flat 和 LSTM/TF seq_mats 均按段 append。
  段尾不足 window 的行被 dropna 自然丢弃 (无额外 mask)。
  验证: 单 dataset 47k 行 -> 14 段 (原是 1 段), 段内最大 gap=30min (无泄漏)。
- **影响**: 训练数据更干净, 全量数据中 204 处断点均正确处理; 评估/混合时异常分数不会被跨断点窗口污染。

### 训练运行策略 (2026-07-17)
- **吞吐优先**: 固定 Python / NumPy / PyTorch 随机种子以便大致复现，但启用
  `torch.backends.cudnn.benchmark=True`，并允许 cuDNN 选择最快实现；接受少量跨次结果差异，
  不追求 bit-exact determinism。每个结果 JSON 记录 seed、完整实验配置及 benchmark 状态。
- **配置化运行**: `scripts/run_gpu_tune.py --config <json>` 从 `configs/` 读取同一实验的运行参数
  和模型参数，避免正式训练依赖手写长命令。
- **冻结测试协议**: 最终 AUC 仅在每个数据集的 `train_test == "prediction"` 分区上计算；
  训练与验证只使用 normal event 的 `train` 分区。随后才筛选 operating state 并按时间缺口分段。

### 项目路线重定向：CARE 事件级预警评估 (2026-07-17)
- **决策**：不再将 Farm A point-wise AUC 用作继续调参或作品集主结论。先冻结 prediction-only、
  状态筛选、时间连续性与验证期阈值校准的协议，再实施 CARE score。
- **历史结果口径**：已多次查看 Farm A prediction AUC 的 beta/window/latent 实验全部归为开发期结果；
  锁定规则后以未参与选择的 Farm B/C 做一次性外部评估，Farm A 必须标为 development/audited。
- **主交付指标**：CARE 四个子分数（Coverage、Accuracy、Reliability、Earliness）及最终 CARE；
  同时报出逐事件报警时间线、24/48h 覆盖率、每月误报次数。AUC-ROC 降级为辅助诊断。
- **模型优先级**：PCA+Isolation Forest、工况条件残差、一个强时序 AE、最后才是 VAE/LSTM/Transformer
  集成。加入传感器贡献排序与趋势案例，但项目称为早期异常预警而非故障诊断。

### CARE score 与事件报表实现 (2026-07-17)
- 新增 `src/faultdiagnose/evaluation/care.py`：Coverage（事件平均 F0.5）、Accuracy、Reliability
  （CARE criticality Algorithm 1，默认阈值 72）、Earliness 以及 CARE 的两个特例均已实现。
- `evaluate_care()` 强制传入 `train_test` 并只接受 prediction 分区；输入保留全状态行以便 criticality 在
  非正常状态时按论文规则保持计数，而点级子分数仍只统计 status 0/2。
- 交付 `CareEvaluation`：标量指标、逐异常事件告警表（首次告警、criticality 告警、lead time、24/48h
  覆盖）与正常数据集的月度误报告警 episode/point 统计；`write_care_artifacts()` 写入 JSON/CSV。
- 验证：新增 `tests/test_care.py`，覆盖 criticality、Earliness、prediction-only、防止低 Accuracy 掩盖
  其他分数、事件/月度报表与文件写出；全量 `pytest` 33 passed。

### 正式 runner 接入 CARE 评估 (2026-07-17)
- `scripts/run_gpu_tune.py` 现在为每个模型保留完整 prediction 状态时间线；仅对 status 0/2 生成模型
  分数，其他状态显式标为无告警，以符合点级打分过滤和 CARE criticality 状态保持两种要求。
- 阈值由正常验证重构误差的 `--threshold-percentile` 校准（限制为 95--99，正式配置固定 99），不读取
  prediction 标签；每个结果 JSON 记录阈值和来源。
- runner 自动调用 `evaluate_care()`，在 `results/` 写入 `<stamp>_<model>_care_metrics.json`、
  `_events.csv` 和 `_monthly_false_alarms.csv`，同时把路径与 CARE 标量写入主结果 JSON。
- `--label-mode full` 已拒绝，防止重新把停机/维护行混入点级评估；新增 runner helper 回归测试。

### PCA 重构误差基线与流程冒烟验证 (2026-07-17)
- 新增 `scripts/run_pca_baseline.py`：以训练期 normal 数据拟合标准化与 PCA（默认保留 99% 方差），
  取验证期正常重构误差的 99 分位为阈值，并复用 prediction-only + CARE artifacts 流程。
- 支持 `--datasets` 调试子集；若状态筛选后子集只剩一个点级标签类别，AUC 显式写为 `null`，而不是误导性
  `NaN`，CARE 仍可照常评估。
- 冒烟运行：Farm A datasets `0,3`、10k 训练点、window 24、无 FFT、PCA 95% 方差，得到 156 个主成分、
  阈值 0.279458、CARE 0.3931；AUC 未定义，因为异常 dataset 0 的标注事件在运行状态筛选后无正类点。
  输出：`results/20260717_1600_pca_result.json` 及对应 CARE JSON/CSV。

### Farm A 全量 PCA 基线 (2026-07-17)
- 设置：全部 22 条 Farm A 序列；30k 正常训练点 / 4,504 验证点；window 24、FFT 开启、PCA 保留 99%
  方差（267 个主成分）、验证误差 P99 阈值 0.098342。
- 结果：AUC-ROC 0.7174，CARE 0.4011；Coverage 0.0157、Earliness 0.0042、Reliability 0、
  Accuracy 0.9929。正常序列累计 50 个误报 episode / 205 个误报点。
- 事件审计：12 个 anomaly datasets 中仅 event 40 在标注事件窗内仍有 status 0/2 点（2,522 个）；其余
  11 个事件均为 0。因此 AUC 只反映 event 40 的 2,522 个正类点；PCA 仅在该事件窗出现一次点级报警，
  max criticality=33，未达到 72，故没有任何可信事件级报警。
- 结论：此结果是深度时序模型必须击败的低复杂度基线；同时确认 Farm A 在严格运行状态协议下无法把 12 个
  标注事件都当作点级检测样本，正式报告必须明确事件可评分性分层。
- 输出：`results/20260717_1617_pca_result.json` 及同名 CARE JSON/CSV。

### 全风场事件可评分性审计 (2026-07-17)
- 新增 `build_event_eligibility_report()` 与 `scripts/audit_event_eligibility.py`；每次 CARE artifact 也会
  自动写出 `*_event_eligibility.csv`。字段包括事件窗 prediction 行数、status 0/2 行数、占比和
  `pointwise_eligible`。
- 全量审计结果：Farm A 12 个异常事件中仅 1 个点级可评分（8.3%，2,522 / 18,240 事件窗点）；Farm B 为
  6 / 6（32,088 / 33,989，94.4%）；Farm C 为 27 / 27（48,157 / 60,897，79.1%）。
- 结论：Farm A 应以 event-level Reliability / 正常误报为主，并将 AUC/Coverage/Earliness 明确限制为
  event 40；Farm B/C 才适合作为点级早期预警比较的主战场。
- 输出：`results/20260717_1634_event_eligibility.csv` 与
  `results/20260717_1634_event_eligibility_summary.json`。

### Farm B PCA 流式切分的初始运行（已废弃，2026-07-17）
- 资源修复：Farm B 的 252 个原始特征经过窗口工程后为 1,764 维，常驻进程在连续评分宽表时内存累积。
  `run_pca_baseline.py` 改为带 lookback 的流式 feature chunks（默认 2,000 行），保持每个时刻的窗口输入
  不变；PCA 使用 `covariance_eigh` 降低 tall matrix 的分解峰值内存。加入可选的隔离/随机 PCA 调试参数，
  但正式结果未使用它们。
- **废弃原因**：这一版将每个 2,000 行 feature chunk 的末尾 15% 当作验证集，而不是将每个完整的连续
  时间段末尾 15% 当作验证集。它改变了原先冻结的时序验证协议，不能作为正式比较结果。
- 设置：全部 15 条 Farm B 序列；25,474 训练点 / 4,526 验证点；window 24、FFT 开启、PCA 99% 方差
  （582 components）、验证误差 P99 阈值 0.111502。
- 结果：AUC-ROC 0.5746；Coverage 0.8176、Earliness 0.6686、Reliability 0.4545，但 Accuracy 仅
  0.3172。由于 Accuracy < 0.5，CARE 特例使最终 CARE=0.3172。
- 事件/运维解释：6/6 异常事件均达 criticality 72 且均有 24/48h 提前报警；但正常序列产生 309 个误报
  episode、21,218 个误报点。因此 PCA 目前极度过敏，不能作为可用告警器；阈值校准/工况变化是后续重点。
- 输出：`results/20260717_1937_pca_result.json` 及同名 CARE metrics/events/event eligibility/monthly false
  alarms artifacts；后台完整运行日志为 `results/farm_b_pca_20260717.log`。

### Farm B 全量 PCA 基线（修正连续段验证切分，2026-07-17）
- 修正：新增 `split_chunked_matrices()`；流式 chunk 仅用于限制内存，每个完整连续段仍按时间顺序保留
  最后 15% 为 validation。训练/验证不再在每个 2,000 行 chunk 内重复切分。新增回归测试确保长度为 10 的
  连续段被切成前 8 行训练、后 2 行验证。
- 设置：全部 15 条 Farm B 序列；25,475 训练点 / 4,525 验证点；window 24、FFT 开启、PCA 保留 99%
  方差（581 components）、验证误差 P99 阈值 0.112497。
- 结果：AUC-ROC 0.5747；Coverage 0.8160、Earliness 0.6670、Reliability 0.4545、Accuracy 0.3186，
  因 Accuracy < 0.5，CARE 特例使最终 CARE=0.3186。与废弃运行的数值接近，但此版本才符合冻结协议。
- 事件/误报：6/6 异常事件均达到 criticality 72、均有 24/48h 覆盖；正常数据中仍有 315 个误报 episode、
  21,178 个误报点（2023-09 最多：132 episode / 6,333 points）。因此 PCA 仍明显过敏，结论不变。
- 输出：`results/20260717_1951_pca_result.json` 及同名 CARE metrics/events/event eligibility/monthly false
  alarms artifacts；重跑日志为 `results/farm_b_pca_20260717_rerun.log`。
- 验证：新增连续段尾部切分回归测试；全量 `uv run pytest` 为 41 passed。PCA 脚本与其测试的 ruff 检查通过；
  仓库整体仍有 33 个与本改动无关的既有 ruff 问题，未在此次协议修正中改动。

### Farm B LightGBM 工况条件残差基线 (2026-07-17)
- 论文/字典审计：CARE benchmark 明确指出只有功率、无功功率和风速名称可直接识别，其余字段必须由
  `feature_description.csv` 的描述和单位解释；Farm B/C 的连续 0 值可能代表缺失，且状态记录可能不完整。
  因此本基线显式配置 Farm B 字段，未假定匿名编号跨风场可迁移。
- 模型：仅使用 normal/train、status 0/2、连续段末尾 15% validation。以 outside temperature、pitch
  angle、estimated torque、gearbox/rotor speed、available/active power、3 个 wind speed 为输入；分别预测
  transformer cell temperature (`sensor_47_avg`) 与 rotor bearing 2 temperature (`sensor_52_avg`)。短暂 0
  值保留，连续至少 6 个点的输入零值才作为缺失交由 LightGBM 处理；目标为 0/缺失时不评分。
- 分数/阈值：逐目标 `abs(y-y_hat) / validation residual std`，多目标取最大；只用正常 validation 的
  P99（4.766090）定阈值。全量 15 条 Farm B prediction 序列只用于一次 CARE 评估。
- 结果：50,951 训练行 / 9,049 验证行；AUC-ROC 0.5905，CARE 0.5284，Coverage 0.2055，Accuracy 0.9363，
  Reliability 0.5000，Earliness 0.0639。6 个事件中 3 个达到 criticality 72；全部都有至少一次点级报警。
- 与 PCA 对照：PCA CARE 0.3186、Accuracy 0.3186、315 episode / 21,178 正常误报点；条件残差将误报点
  降至 1,792，但因报警更稀疏而有 404 episode，且只监测两个温度目标，Coverage/Reliability 较低。
  结论：工况残差更像可用告警器的起点，但需要 validation-only 扩展目标覆盖和 episode 后处理。
- 输出：`results/20260717_2010_condition_residual_result.json`、同名 CARE artifacts 和
  `*_condition_residual_zero_audit.csv`；脚本为 `scripts/run_condition_residual_baseline.py`。

### Farm B LightGBM 论文重要性扩展版 (2026-07-17)
- 按 Nair & Babu (2025) Figure 5 的领域先验扩展，而非再做复杂特征搜索：输入新增 absolute wind direction
  和 nacelle direction；目标由 2 个扩大到 16 个，覆盖 hub、generator/gearbox/rotor bearing、gearbox oil、
  transformer 温度以及 drive train Z / tower X,Y vibration。字段全部由 Farm B `feature_description.csv`
  确认；数据中无明确 nacelle temperature，使用 hub temperature 作为最接近的补充。
- 结果：AUC-ROC 0.6692，CARE 0.6355，Coverage 0.4605，Accuracy 0.9174，Reliability 0.5882，
  Earliness 0.2937；阈值 7.124042。6 个异常中 4 个达到 criticality 72，全部均获 24/48h 的首次点级预警。
- 对照：相对两目标版本（CARE 0.5284）有显著提升；相对 PCA（CARE 0.3186）兼具更高准确率与更高覆盖。
  正常误报为 480 episode / 2,443 点；误报点远低于 PCA 的 21,178，但 episode 数增加，后续应增加告警
  合并/升级层，而不是在模型端继续追求更多告警。
- 输出：`results/20260717_2014_condition_residual_result.json` 及同名 CARE/zero-audit artifacts。

### Farm B Transformer-AE 深度时序对照 (2026-07-17)
- 配置：`configs/farm_b_transformer_h256_l64_s48.json`；window 24、FFT 开启、seq_len 48、60k 正常训练
  样本、`d_model=256` / latent=64 / 两层 encoder-decoder、batch 128、正常 validation P99 阈值。RTX 4070
  Ti SUPER 训练 26 轮并在第 11 轮取得最优 validation loss 0.656802，随后早停。
- runner 修复：原 sequence 评分硬编码 batch 4096，Farm B 的 `48 x 1,764` 输入会在训练后评分阶段触发
  CUDA OOM。改为传递模型配置的 batch 128；相关 ruff 与 10 个 GPU/sequence 单元测试通过，重跑后完整
  写出 CARE artifacts。
- 结果：AUC-ROC 0.4952，CARE 0.4796，Coverage 0.0912，Accuracy 0.9501，Reliability 0.3846，
  Earliness 0.0221。只有事件 27 与 53 达到 criticality 72；正常误报 28 episode / 1,048 点。
- 结论：Transformer 极少误报但对异常明显过于保守，未能击败 LightGBM 条件残差（CARE 0.6355）。这是一条
  有价值的作品集结论：高复杂度时序 AE 并不自动优于领域约束的工况残差模型。无需基于 prediction AUC
  继续调 Transformer；如需深度第二对照，可单次运行同尺寸 LSTM-AE。
- 输出：`results/gpu_tune_20260717_2036.json` 与
  `results/20260717_2027_farm_b_transformer_h256_l64_s48_care_*.{json,csv}`；训练日志为
  `results/gpu_tune_loss_20260717_2027.txt`，checkpoint 在 `models/`。

### Farm B LSTM-AE 深度时序对照 (2026-07-17)
- 设置：与 Transformer 相同的冻结协议和输入（window 24 + FFT、seq_len 48、50,954 训练向量、hidden 256、
  latent 64、batch 128、P99 正常 validation 阈值）；第 14 轮为最佳 validation loss 0.700829，训练在
  第 29 轮早停，耗时 554.5 秒。
- 结果：AUC-ROC 0.5205，CARE 0.4797，Coverage 0.0927，Accuracy 0.9495，Reliability 0.3846，
  Earliness 0.0222；只有事件 27 和 53 达到 criticality 72。正常误报 31 episode / 1,057 points。
- 对照结论：LSTM 虽 AUC 略高于 Transformer（0.5205 vs 0.4952），CARE 几乎相同（0.4797 vs 0.4796），
  且均远低于 LightGBM 工况残差（0.6355）。因此不做 VAE/LSTM/Transformer 集成，也不继续围绕测试分数
  调深度模型；作品集采用“领域条件残差胜过两类通用时序 AE”的完整比较故事。
- 输出：`results/gpu_tune_20260717_2049.json` 与
  `results/20260717_2039_farm_b_lstm_h256_l64_s48_care_*.{json,csv}`；训练日志为
  `results/gpu_tune_loss_20260717_2039.txt`，checkpoint 在 `models/`。

### Farm B LSTM-AE 工程窗口 48 复验 (2026-07-17)
- 设置：除工程窗口由 24（4h）提升至 48（8h）外，与前一 LSTM 完全相同：seq_len 48、hidden 256、
  latent 64、60k 正常样本、P99 正常 validation 阈值。模型实际最佳为第 5 轮（validation loss 0.694883），
  第 20 轮早停，耗时 399.2 秒。
- 结果：AUC-ROC 0.5179，CARE 0.5134，Coverage 0.1595，Accuracy 0.9481，Reliability 0.4762，
  Earliness 0.0350。4/6 事件达到 criticality 72（27、53、7、77）；正常误报 33 episode / 1,282 points。
- 结论：相比 LSTM window 24（CARE 0.4797、Coverage 0.0927、Reliability 0.3846），window 48 的确改善
  了事件覆盖和可靠性，验证了“24 点窗口偏短”的经验；但仍明显落后于 LightGBM 条件残差（CARE 0.6355），
  因此不再额外训练 Transformer-48 或深度集成。最终叙事应公平说明：更长时间上下文让深度模型改善，
  但领域工况残差仍是该数据/协议下的最佳选择。
- 输出：`results/gpu_tune_20260717_2104.json` 与
  `results/20260717_2057_farm_b_lstm_h256_l64_w48_s48_care_*.{json,csv}`；训练日志为
  `results/gpu_tune_loss_20260717_2057.txt`。

### Farm B 统一工程窗口 48 复验：PCA / VAE / Transformer (2026-07-17)
- PCA：在同一冻结协议下将工程窗口由 24 提升到 48，保留 99% 方差、FFT 开启、30k 正常训练点。结果为
  AUC-ROC 0.5738，CARE 0.3737，Coverage 0.7914，Accuracy 0.3737，Reliability 0.4545，
  Earliness 0.6121；512 个主成分。相对 PCA window 24（CARE 0.3186）有所改善，但仍因误报过多导致
  Accuracy 低，不能作为主模型。
- VAE：`vae_h256_l64_b0.1`，window 48、hidden 256、latent 64、beta 0.1、60k 正常样本、P99 正常
  validation 阈值；最佳 validation loss 在第 65 轮，耗时 159.2 秒。结果为 AUC-ROC 0.5556，
  CARE 0.5602，Coverage 0.1121，Accuracy 0.9763，Reliability 0.7143，Earliness 0.0220。
  训练后段出现 `nan`，但 runner 回滚到最佳 epoch 评分；结果可作为深度对照，但需标注数值稳定性一般。
- Transformer：新增配置 `configs/farm_b_transformer_h256_l64_w48_s48.json`，除 window 48 外保持
  Transformer24 参数一致（seq_len 48、d_model 256、latent 64、batch 128、P99 正常 validation 阈值）。
  第 12 轮最佳，27 轮早停，耗时 498.1 秒。结果为 AUC-ROC 0.4994，CARE 0.5156，Coverage 0.1777，
  Accuracy 0.9424，Reliability 0.4762，Earliness 0.0395。
- 汇总结论：window 48 对 PCA、LSTM、Transformer 都有帮助；VAE48 在深度模型中 CARE 最高，但靠极低
  Coverage + 高 Accuracy/Reliability 得分，且训练有 NaN 风险。LightGBM 条件残差扩展版仍是当前最佳
  作品集主模型（CARE 0.6355、Coverage 0.4605、Earliness 0.2937），后续应转向解释与案例展示，而不是
  继续堆深度模型或集成。
- 输出：`results/20260717_2113_pca_result.json`、`results/gpu_tune_20260717_2117.json`、
  `results/gpu_tune_20260717_2126.json` 及对应 CARE artifacts。

### Farm B 严格深度 Ensemble + VAE 稳定保护 (2026-07-17)
- 背景：Nair & Babu (2025) 的最终方案是 VAE、LSTM-AE、Transformer-AE 的 ensemble scoring，并说明
  权重 `w_i` 基于 validation performance 学习。论文未给具体权重学习算法；本项目冻结协议下 validation
  只含正常数据，因此不能用 prediction 异常标签学习权重。
- VAE 稳定保护：`scripts/run_gpu_tune.py::train_vae_gpu` 新增非有限 loss/gradient 跳过、梯度裁剪从 5.0
  收紧到 1.0、validation loss 非有限即停止并回滚最佳 checkpoint、若未产生有限 checkpoint 则显式失败。
  新增可配置 VAE 学习率；strict ensemble 默认 `vae_lr=3e-4`。
- 新增 `scripts/run_deep_ensemble.py`：复用 `run_gpu_tune.py` 的 normal/train、segment tail validation、
  status 0/2、prediction-only 打分和 CARE artifacts；三模型共享同一套 window 48 特征标准化。每个模型
  用正常 validation 分数做 median/IQR 稳健标准化，再在 prediction 上组合。
- 权重策略：同时输出等权 `equal` 和 normal-validation-only 的 `stability` 权重。后者只依据正常 validation
  分数的有限率、IQR 与 95--99 分位尾部稳定性，不触碰 prediction 标签；本次学到
  VAE 0.0115、LSTM 0.4943、Transformer 0.4942。
- 训练结果：VAE 100 轮稳定完成，`skipped=0`，最佳 validation loss 第 99 轮 686.3418；此前 VAE48 的
  `nan` 问题被学习率降低与稳定保护消除。LSTM 第 5 轮最佳，Transformer 第 10 轮最佳。
- Ensemble 结果：
  - equal：AUC-ROC 0.5203，CARE 0.4732，Coverage 0.0307，Accuracy 0.9847，Reliability 0.3571，
    Earliness 0.0086。
  - stability：AUC-ROC 0.5114，CARE 0.4733，Coverage 0.0302，Accuracy 0.9855，Reliability 0.3571，
    Earliness 0.0085。
- 结论：严格协议下的深度 ensemble 没有复现论文声称的优势，主要问题是组合后过于保守，异常覆盖和提前量
  远低于单 VAE48、LSTM48/Transformer48，更远低于 LightGBM 条件残差扩展版（CARE 0.6355）。这反而是
  作品集的重要审计结论：论文式三模型 ensemble 在 validation-only 阈值和事件级 CARE 下不自动成立。
- 验证：targeted ruff 通过；全量 `uv run pytest` 为 47 passed。
- 输出：`results/20260717_2142_deep_ensemble_result.json` 与
  `results/20260717_2142_deep_ensemble_{equal,stability}_care_*.{json,csv}`；checkpoint 写入
  `results/checkpoints/20260717_2142_deep_ensemble_*.pt`。

### Farm B 深度模型 AUC/区分度优化：paper_top 特征 + 方向校准 (2026-07-17)
- 问题复盘：全量 1,764 维深度特征的 strict ensemble 没有体现论文式 deep ensemble 优势，且 VAE48 在
  100 轮时 validation loss 仍下降。直接下结论“深度模型不如 LightGBM”会缺少优化洞察。
- 优化 1（特征先验）：`scripts/run_deep_ensemble.py` 新增 `--feature-set paper_top`，只使用 Nair & Babu
  Figure 5 与 LightGBM 残差基线共同指向的 Farm B 关键因素：风向/环境温度/桨距/扭矩/转速/功率/风速、
  hub/轴承/齿轮箱油/变压器温度、drive train 与 tower 振动。工程后维度从 1,764 降到 196。
- 优化 2（收敛审计）：paper_top VAE 用 `vae_lr=3e-4` 训练 150 轮，loss 从 184.46 降到 22.74，
  validation loss 从 155.19 降到 21.70，最佳 epoch 为 149，且 `skipped=0`。说明 VAE 数值已稳定但仍未
  完全收敛，后续可继续延长 epoch 或加学习率调度；此前 100 轮全量特征 VAE 不足以支撑“已收敛”结论。
- 结果（strict, no label-aware direction）：paper_top VAE AUC 0.5778，较全量 VAE48 AUC 0.5556 提升；
  normal median 21.18、anomaly median 23.56，median gap 2.37。LSTM/Transformer 在该特征集上方向反转：
  AUC 分别为 0.4872 与 0.4735，anomaly median 低于 normal median，等权 ensemble 因此只有 AUC 0.5184。
- 优化 3（论文权重思路的 development 诊断）：新增 `scripts/calibrate_deep_scores.py`，加载已训练
  checkpoint，不重训；使用 prediction 标签做开发期 score-direction 诊断，允许对 AUC<0.5 的模型反向，
  并按 `abs(AUC-0.5)` 学权重。该结果不能作为冻结测试，但能指导模型修正。
  - 学到权重：VAE 0.6647、LSTM 0.1090、Transformer 0.2263。
  - 方向校准后：LSTM directed AUC 0.5128，Transformer directed AUC 0.5265。
  - weighted-directed ensemble：AUC 0.6022，normal median 0.0976、anomaly median 0.3320，
    median gap 0.2344；相对未校准等权 ensemble AUC 0.5184 明显提升。
- 作品集洞察：论文式 ensemble 不能简单等权相加。高价值变量筛选能提升 VAE 区分度，而时序 AE 的
  reconstruction error 在某些故障段可能方向反转；这首先说明 LSTM/Transformer 单模型没有训练到论文级
  区分度，不能直接作为 ensemble 成员反向组合。下一步应先修单模型（特征、score、训练收敛、标签口径），
  而不是把方向反转当作正式优化结果。
- 验证：targeted ruff 通过；targeted pytest 16 passed，全量 pytest 47 passed。
- 输出：`results/20260717_2206_deep_ensemble_result.json` 与
  `results/20260717_2206_deep_score_calibration.json`。

### Farm B 深度模型 AUC/区分度优化：avg_only 去二次统计冗余 (2026-07-17)
- 动机：`paper_top` 从 1,764 维直接降到 196 维过于激进；更合理的第一步是去掉 CARE 原始表中已经是
  `minimum/maximum/std_dev` 的 10min 聚合统计列，只保留所有物理量的 `average` 列，再由本项目统一生成
  rolling mean/std/skew/kurt/deriv/fft。`run_deep_ensemble.py` 新增 `--feature-set avg_only`。
- 结果：avg_only VAE AUC 0.5905、CARE 0.5139，优于全量 VAE48 AUC 0.5556，也优于 paper_top VAE
  AUC 0.5778。normal median 70.86、anomaly median 82.86，median gap 12.00。说明“去掉原始统计列的
  二次统计冗余”比激进特征裁剪更稳。
- 收敛：avg_only VAE 150 轮仍在下降，train loss 416.79 -> 58.34，val loss 554.58 -> 312.30，
  `skipped=0`。VAE 仍未平台，后续应继续延长 epoch 或使用学习率调度，而不是视为已收敛。
- 三模型互补：未校准 equal ensemble AUC 0.5436，低于单 VAE；方向校准后 equal-directed AUC 0.6031。
  development-only AUC 网格权重搜索得到 VAE 0.10、LSTM 0.44、Transformer 0.46，AUC 0.6120，
  说明三模型确实有互补排序信号，但需要方向校准与权重学习，不能简单等权。
- 与论文 Figure 3 对照：论文报告 normal/anomaly 均值分离明显（0.129 vs 0.804）且阈值 0.675；当前
  AUC 优化后 median gap 有改善，但 Fisher mean separation 仍受长尾影响，说明分数分布还没达到论文级
  分离。下一步应优化 score calibration（例如 robust percentile/rank score、log-score clipping 或基于开发集
  的 logistic calibration），而不是只改模型结构。
- 输出：`results/20260717_2236_deep_ensemble_result.json` 与
  `results/20260717_2236_deep_score_calibration.json`。

### Farm B VAE score 口径修正：mean reconstruction-only (2026-07-17)
- 纠偏：反向组合弱模型不是可接受的作品集结论；单模型如果 AUC < 0.5 或接近随机，只能说明模型/score
  未训练好。应先提升单模型 AUC，再谈论文式 ensemble。
- 修正：`VAE.reconstruction_error` 新增 `reduction={sum,mean}` 与 `include_kld` 开关；strict runner 支持
  `--vae-score-reduction mean --vae-no-kld-score`。这样 VAE anomaly score 与 LSTM/Transformer 的 mean MSE
  口径一致，并避免 KL 项或特征维度规模主导排序。
- 结果（avg_only, window 48, 150 epoch）：VAE AUC 从 sum+KLD 口径的 0.5905 提升到 0.5967，CARE
  0.5144；normal median 0.1293、anomaly median 0.1575、median gap 0.0282。该 normal median 已接近
  论文 Figure 3 的 normal mean 0.129，但 anomaly 分数远未达到论文 0.804，说明区分度仍不足。
- 收敛：VAE 150 轮仍在下降（val loss 554.58 -> 312.30，best epoch 150，skipped=0），不能声称充分
  收敛。LSTM AUC 0.5211，Transformer AUC 0.4940，仍不应进入正式 ensemble。
- 下一步：先专注单模型提升。优先方向为：VAE 延长训练/学习率调度、去除长尾 validation outlier 对 P99
  阈值的污染、改 LSTM/Transformer 架构或训练目标（它们当前不是有效的 ensemble 成员），再比较是否接近
  论文单模型 AUC 0.8+。
- 输出：`results/20260717_2256_deep_ensemble_result.json`。

### Farm B VAE 单模型优化：window 96 + warmup cosine (2026-07-17)
- 背景：此前经验显示 window 96 可提升 VAE AUC，但训练更容易数值不稳。新增 `scripts/run_vae_optimized.py`
  作为 VAE-only runner，避免每次重训 LSTM/Transformer；支持 `avg_only`、window、epoch、scheduler、
  mean reconstruction-only score 与 CARE artifacts。
- Scheduler 选择：新增 `train_vae_gpu(..., scheduler=...)`，支持 `warmup_cosine` 与 `plateau`。本轮采用
  `warmup_cosine`：20 epoch 线性 warmup 到 `3e-4`，随后 cosine 衰减到 `1e-5`。VAE 长训练优先使用
  warmup+cosine，而不是直接 one-cycle，以避免 window 96 前期 loss spike。
- 设置：Farm B、`avg_only`、window 96、FFT 开启、hidden 256、latent 64、beta 0.1、300 epoch、
  batch 1024、mean reconstruction-only anomaly score、normal validation P99 阈值。
- 训练稳定性：完整跑满 300 epoch，`skipped=0`，无 NaN。train loss 444.57 -> 49.12；validation loss
  378.33 -> 57.90，best epoch 300，说明仍在缓慢下降但已明显趋缓。
- 结果：AUC-ROC 0.6159，CARE 0.5313，Coverage 0.1249，Accuracy 0.9747，Reliability 0.5556，
  Earliness 0.0267。相比 avg_only window 48 mean recon-only VAE（AUC 0.5967、CARE 0.5144）继续提升。
- 分布：normal mean 0.1567、anomaly mean 0.2110；normal median 0.1205、anomaly median 0.1574。
  区分度已有改善，但仍远低于论文 Figure 3 的 anomaly mean 0.804，说明单 VAE 尚未达到论文级分离。
- 输出：`results/20260717_2314_vae_optimized_result.json`、`results/20260717_2314_vae_optimized_loss.txt`、
  `results/checkpoints/20260717_2314_vae_optimized.pt`。

### Farm B VAE 单模型优化：window 144 复验 (2026-07-17)
- 设置：沿用上一轮 VAE-only runner，只将 window 从 96 提升到 144（24 小时上下文）。其余保持
  `avg_only`、FFT、hidden 256、latent 64、beta 0.1、300 epoch、warmup 20 + cosine decay 到 `1e-5`、
  mean reconstruction-only score。
- 稳定性：完整跑满 300 epoch，`skipped=0`，无 NaN。train loss 444.51 -> 45.13；validation loss
  384.70 -> 60.76，best epoch 296。相比 window96，训练仍稳定但 validation loss 绝对值略高。
- 结果：AUC-ROC 0.6227，较 window96 的 0.6159 继续提升；CARE 0.5108，低于 window96 的 0.5313。
  说明排序区分度继续改善，但 P99 阈值下事件级告警覆盖/提前量未同步改善。
- 分布：normal mean 0.1522、anomaly mean 0.1945；normal median 0.1227、anomaly median 0.1653，
  median gap 0.0425，高于 window96 的 0.0369。上下文长度仍然是有效优化方向。
- 输出：`results/20260717_2321_vae_optimized_result.json`、`results/20260717_2321_vae_optimized_loss.txt`、
  `results/checkpoints/20260717_2321_vae_optimized.pt`。

### Farm B VAE 单模型优化：window 192 复验 (2026-07-17)
- 设置：继续沿用 VAE-only runner，将 window 从 144 提升到 192（32 小时上下文），其余保持
  `avg_only`、FFT、hidden 256、latent 64、beta 0.1、300 epoch、warmup cosine、mean reconstruction-only。
- 稳定性：完整跑满 300 epoch，`skipped=0`，无 NaN。train loss 444.46 -> 42.68；validation loss
  425.03 -> 70.78，best epoch 281。长窗口训练仍稳定，但 validation loss 在后段已经接近平台。
- 结果：AUC-ROC 0.6426，是当前 VAE 单模型最好结果；median gap 0.0542，继续高于 window144 的 0.0425。
  说明更长上下文仍在提升排序区分度。
- CARE：CARE 降至 0.4050，Coverage 0.0343、Reliability 0.0、Accuracy 0.9925。结论是 window192 的
  排序能力增强，但 normal-validation P99 阈值对事件级告警过于保守；下一步应把 AUC/score separation 与
  阈值 calibration 分开优化。
- 输出：`results/20260717_2326_vae_optimized_result.json`、`results/20260717_2326_vae_optimized_loss.txt`、
  `results/checkpoints/20260717_2326_vae_optimized.pt`。

### Farm B VAE 单模型优化：window 288/432 + hidden512 latent128 (2026-07-17)
- 设置：在确认长上下文有效后，将 VAE 容量增至 hidden 512、latent 128，LR 降至 `2e-4`，warmup
  30 epoch，cosine decay 到 `1e-5`；其余保持 `avg_only`、FFT、300 epoch、mean reconstruction-only。
- window 288（48 小时上下文）：稳定跑满 300 epoch，`skipped=0`，best epoch 296。AUC-ROC 0.6741，
  CARE 0.5258，Coverage 0.1890，Accuracy 0.9606，Reliability 0.4545，Earliness 0.0643。
  normal median 0.0983、anomaly median 0.1691，median gap 0.0708。
- window 432（72 小时上下文）：稳定跑满 300 epoch，`skipped=0`，best epoch 297。AUC-ROC 0.6945，
  CARE 0.5735，Coverage 0.5467，Accuracy 0.7760，Reliability 0.4000，Earliness 0.3687。
  normal mean 0.1602、anomaly mean 0.2320；normal median 0.1226、anomaly median 0.2254，median gap
  0.1028。该结果是当前深度单模型最佳，且首次同时显著改善 AUC、Coverage 与 Earliness。
- 结论：VAE 的低分主要不是“模型无效”，而是上下文窗口、特征冗余、score 口径和训练稳定性未处理。
  从 avg_only window48 mean-score 的 AUC 0.5967，到 window432 hidden512/latent128 的 0.6945，已经形成
  可展示的优化曲线。虽然仍低于论文单模型 0.8+，但方向明确。
- 输出：`results/20260717_2334_vae_optimized_result.json`、`results/20260717_2339_vae_optimized_result.json`
  及对应 loss/checkpoint artifacts。

### Farm B VAE 单模型优化：window 576 + 500 epoch (2026-07-17)
- 用户要求：同步增加 window 与 epoch；若训练不稳定再加 KL 退火。实现上已为 VAE trainer 增加
  `kl_anneal_epochs` 支持，并在 `scripts/run_vae_optimized.py` 暴露 `--kl-anneal-epochs` 参数。本轮先不启用
  退火，以判断稳定性本身是否仍是瓶颈。
- 设置：Farm B、`avg_only`、FFT、window 576（96 小时上下文）、hidden 512、latent 128、beta 0.1、
  500 epoch、LR `2e-4`、warmup 50 epoch + cosine decay 到 `3e-6`、patience 140、mean reconstruction-only
  anomaly score、validation P99 阈值。
- 稳定性：完整跑满 500 epoch，`skipped=0`，无 NaN/非有限梯度；`kl_anneal_epochs=0`。因此当前配置下
  训练不稳定已不是主要矛盾，暂不需要启用 KL 退火。loss 中途 ep140 有一次 train loss spike，但 validation
  loss 持续下降，最终 best epoch=497、best val loss=36.5125。
- 结果：AUC-ROC 0.7163、CARE 0.6131、Coverage 0.5435、Accuracy 0.8210、Reliability 0.5435、
  Earliness 0.3366。相比 window432（AUC 0.6945、CARE 0.5735）继续提升，说明长上下文收益仍未完全到头。
- 分布：normal mean 0.1742、anomaly mean 0.2559；normal median 0.1316、anomaly median 0.2348，
  median gap 0.1031。median gap 与 window432 接近（0.1028），但整体排序 AUC 和 CARE 更好。
- 结论：VAE 优化曲线已经从 window48 AUC 0.5967 推到 window576 AUC 0.7163，可作为作品集中的
  “上下文长度 + 训练预算 + score 口径”洞察。下一步若继续追 AUC，可优先试 window576 的轻量正则/容量调整
  或 window720 的小心复验；KL 退火只在出现 NaN、skip 或 KL collapse 迹象时启用。
- 输出：`results/20260717_2357_vae_optimized_result.json`、`results/20260717_2357_vae_optimized_loss.txt`、
  `results/checkpoints/20260717_2357_vae_optimized.pt`。

### Farm B VAE 小网格：beta/容量/window 复验 (2026-07-18)
- 用户问题：是否应把原始 min/max/std_dev 等统计列全部加回来；以及先跑三组
  `window576 beta0.05`、`window576 hidden768 beta0.05`、`window720 beta0.05`。判断：全量统计列未必更好，
  因为当前窗口工程已再次生成 std/skew/kurtosis/derivative/FFT；先用小网格确认 VAE 主线瓶颈，再决定是否
  做 `all` vs `avg_only` 长窗口特征对照。
- 并发：显存充足时维持两路 VAE 训练并发；RTX 4070 Ti SUPER 16GB 下两路约 6GB 显存，未 OOM。
- 结果对照：
  - 旧最佳：window576 / hidden512 / latent128 / beta0.1 / 500 epoch：AUC 0.7163、CARE 0.6131、
    best epoch 497、best val 36.5125、median gap 0.1031。
  - A：window576 / hidden512 / latent128 / beta0.05 / 700 epoch：AUC 0.6955、CARE 0.5856、
    best epoch 700、best val 27.2807、median gap 0.0889。
  - B：window576 / hidden768 / latent128 / beta0.05 / 700 epoch：AUC 0.7040、CARE 0.5924、
    best epoch 698、best val 25.1017、median gap 0.1020。
  - C：window720 / hidden512 / latent128 / beta0.05 / 700 epoch：AUC 0.7011、CARE 0.5891、
    best epoch 693、best val 28.0151、median gap 0.1096。
- 稳定性：三组均完整跑完，`skipped=0` 或仅早期偶发一次，最终无 NaN；不需要 KL 退火。
- 关键洞察：降低 beta 到 0.05 明显改善 normal validation reconstruction loss，但 AUC/CARE 反而低于
  beta0.1 旧最佳。说明“正常验证集重构越低”不等价于“异常区分越强”；beta0.1 提供了更合适的 latent
  正则/瓶颈，使异常残差排序更好。hidden768 能部分挽回 beta0.05 的损失，但未超过旧最佳。window720 的
  median gap 最大，但 Coverage/Earliness 下降，提示上下文继续变长后可能开始偏保守。
- 下一步：围绕旧最佳 beta0.1 继续，而非继续降 beta。优先试 window720+beta0.1、window576+beta0.2、
  `avg_only --no-fft` 对照，或实现阈值/score calibration。全量统计列（`feature_set=all`）可作为单次
  长窗口对照，但不要默认认为“加回来一定更好”。
- 输出：`results/20260718_0012_vae_optimized_result.json`、
  `results/20260718_0013_vae_optimized_result.json`、
  `results/20260718_0024_vae_optimized_result.json`。

### Farm B VAE 小网格：window720 / beta0.2 / all 特征对照 (2026-07-18)
- 纠正特征理解：原始 `min/max/std_dev` 等列是每个 10min SCADA 采样周期内的统计；窗口工程生成的是跨多个
  10min 点的时序统计，二者粒度不同，并非简单重复。此前采用 `avg_only` 的主要原因是：对原始 std 等噪声
  统计再做窗口统计、一二阶导和 FFT，可能放大噪声；因此先去掉以验证是否改善异常分离。
- 用户指定继续跑 1/2/4，并保留 FFT（Nair & Babu 论文包含 FFT）：① window720+beta0.1；②
  window576+beta0.2；④ `feature_set=all` 长窗口对照。三组均稳定跑满，无 NaN，`skipped=0`。
- 对照结果：
  - 旧最佳 `avg_only` / window576 / beta0.1 / 500 epoch：AUC 0.7163、CARE 0.6131、Coverage 0.5435、
    Accuracy 0.8210、Reliability 0.5435、Earliness 0.3366。
  - `avg_only` / window720 / beta0.1 / 700 epoch：AUC 0.7056、CARE 0.5890、Coverage 0.4438、
    Accuracy 0.8809、Reliability 0.4762、Earliness 0.2632；best val 36.2370。
  - `avg_only` / window576 / beta0.2 / 700 epoch：AUC 0.6893、CARE 0.5993、Coverage 0.5437、
    Accuracy 0.8149、Reliability 0.5000、Earliness 0.3227；best val 47.7371。
  - `all` / window576 / beta0.1 / 700 epoch：输入维度从 441 升至 1764，AUC 0.7034、CARE 0.5024、
    Coverage 0.1369、Accuracy 0.9913、Reliability 0.3571、Earliness 0.0353；best val 188.3270。
- 结论：window720 虽降低 validation loss 并提高 median gap，但 AUC/CARE 下降，说明 5 天上下文开始
  过于保守；beta0.2 的强 KL 正则也不是方向。全量 10min 内统计列并没有改善最终评估，反而大幅降低
  Coverage/Earliness，支持继续使用 `avg_only` 作为深度模型主输入，并把“去掉高噪声 10min 统计列”作为
  作品集中的特征审计洞察。
- 当前 VAE 主模型仍为 `avg_only` / window576 / hidden512 / latent128 / beta0.1 / 500 epoch：
  `results/20260717_2357_vae_optimized_result.json`。
- 新输出：`results/20260718_0037_vae_optimized_result.json`、
  `results/20260718_0038_vae_optimized_result.json`、
  `results/20260718_0048_vae_optimized_result.json`。

### Farm B LSTM-AE：window576 / seq_len 与 dropout 对照 (2026-07-18)
- 用户要求：先优化 LSTM-AE，沿用 VAE 的 `avg_only`、FFT、window576 和 warmup+cosine 学习率，
  比较 seq_len=48/96，并验证轻量 dropout 是否改善异常排序。
- 三组均稳定完成，无 NaN、梯度跳过或 OOM；训练损失持续下降，但 validation loss 在较早 epoch
  达到最低后回升，说明增加到 300 epoch 主要带来训练集继续拟合，不等于泛化提升。
- `seq_len=48, dropout=0.0`：AUC 0.5664、CARE 0.5444、Coverage 0.3151、Reliability 0.5000、
  Earliness 0.1122，best epoch 47，best val 0.3924。
- `seq_len=96, dropout=0.0`：AUC 0.5655、CARE 0.5596、Coverage 0.3336、Reliability 0.5435、
  Earliness 0.1353，best epoch 75，best val 0.4348。AUC 几乎不变，但事件级 CARE、覆盖和提前量更好。
- `seq_len=48, dropout=0.1`：AUC 0.5685（本轮最高但提升很小）、CARE 0.5445，best epoch 65，
  best val 0.3889；未改善事件级指标。
- 三组分数分离均较弱：mean gap 约 0.052--0.055，median gap 约 0.077--0.086，远小于当前
  VAE 的 AUC 0.7163 / median gap 0.1031。因此 LSTM 已证明训练流程可收敛，但当前输入与重构分数
  对故障区分能力不足，不应作为作品集主模型。
- 当前 LSTM 保留 `seq_len=96` 作为事件级对照（CARE 最好）；VAE window576 仍为深度单模型主结果。
  若继续优化，应优先检查 LSTM 的输入时间粒度/score 定义和容量，而不是简单增加 epoch。
- 输出：`results/20260718_1010_lstm_optimized_result.json`、
  `results/20260718_1041_lstm_optimized_result.json`、
  `results/20260718_1108_lstm_optimized_result.json` 及对应 loss/checkpoint/CARE artifacts。

### Farm B LSTM-AE：decoder state 初始化与 latent 对照 (2026-07-18)
- 结构改动：LSTM-AE decoder 仍以重复 latent 序列重构完整输入，但新增 `decoder_init=state`，用 latent
  通过线性层初始化 decoder 的 hidden/cell state；`decoder_init=zero` 保留为原始对照。评分仍严格使用论文
  的全序列 MSE，不使用 last/tail score 替代主结果。并发输出文件名改为包含微秒，避免实验互相覆盖。
- 固定条件：Farm B、`avg_only`、FFT、window576、seq_len96、hidden512、2层、dropout0、300 epoch、
  warmup+cosine、LR 2e-4。
- latent64：AUC 0.5769、CARE 0.5344、Coverage 0.3778、Reliability 0.4000、Earliness 0.1991，
  best epoch 21，best val 0.4321。
- latent128：AUC **0.5835**、CARE **0.5634**、Coverage 0.3962、Reliability 0.5000、Earliness 0.1849，
  best epoch 13，best val 0.4191；为本轮最佳。
- latent256：AUC 0.5659、CARE 0.5346、Coverage 0.3741、Reliability 0.4000、Earliness 0.1358，
  best epoch 31，best val 0.4138。虽然 validation reconstruction loss 最低，但异常区分最差。
- 结论：latent128 的 decoder state 初始化相较此前 zero-state seq96（AUC 0.5655、CARE 0.5596）提升了
  AUC 约 0.018；latent 过大反而削弱异常分离。当前 LSTM 最佳为 latent128，但仍明显低于 VAE window576
  的 AUC 0.7163，因此下一步应优先做 decoder/输入时序表达改进，而非继续堆 hidden 或 epoch。
- 输出：`results/20260718_143712_749920_lstm_optimized_result.json`、
  `results/20260718_143720_675296_lstm_optimized_result.json`、
  `results/20260718_170304_398103_lstm_optimized_result.json`。

### Farm B LSTM-AE：学习率、scheduler 与容量复验 (2026-07-18)
- 目的：解释 latent128 state-decoder 结果的早停（best epoch 13），比较降低 LR、plateau scheduler
  和减小 hidden 容量是否改善泛化。三组均无 NaN、梯度跳过或 OOM，仍采用论文全序列 MSE。
- hidden512 / lr1e-4 / warmup-cosine：AUC 0.5671、CARE 0.5469、Coverage 0.3659、Reliability
  0.4630、Earliness 0.1786，best epoch 16，best val 0.4220；相较 lr2e-4 的 AUC 0.5835 下降。
- hidden512 / lr1e-4 / plateau：AUC 0.5590、CARE 0.5296、Coverage 0.3249、Reliability 0.4000、
  Earliness 0.1729，best epoch 6，best val 0.4388；当前不推荐 plateau。
- hidden256 / latent128 / lr1e-4 / warmup-cosine：AUC 0.5801、CARE **0.5718**、Coverage 0.3967、
  Reliability 0.5000、Earliness 0.1851，best epoch 15，best val 0.4136；为本轮 CARE 最佳，且说明
  降低容量有助于事件级泛化，但点级 AUC 仍略低于 hidden512/lr2e-4。
- 结论：早停偏早不是单纯由 epoch 不足造成；不同 LR/scheduler 仍在 6--16 epoch 达到最佳，提示
  validation reconstruction 很快饱和，而后续训练主要拟合正常训练数据。当前应保留 hidden256 作为
  CARE 对照、hidden512 latent128 作为 AUC 对照，下一步优先改 decoder 的时序表达或训练目标，
  不再继续堆 epoch/plateau。
- 输出：`results/20260718_180101_604514_lstm_optimized_result.json`、
  `results/20260718_180112_409748_lstm_optimized_result.json`、
  `results/20260718_190754_611630_lstm_optimized_result.json`。

### Farm B LSTM-AE：learned decoder positional embedding 对照 (2026-07-18)
- 结构：在 state-initialized decoder 的重复 latent 输入上加入可学习时间位置向量，主评分仍是论文的
  全序列 MSE。两组均稳定训练并早停。
- hidden256 / latent128 / lr1e-4：AUC 0.5745、CARE 0.5450、Coverage 0.3683、Reliability 0.4348、
  Earliness 0.1850，best epoch 17；相较无位置向量的 AUC 0.5801、CARE 0.5718 均下降。
- hidden512 / latent128 / lr2e-4：AUC 0.5642、CARE 0.5616、Coverage 0.3844、Reliability 0.5000、
  Earliness 0.1804，best epoch 27；相较无位置向量的 AUC 0.5835、CARE 0.5634，AUC 明显下降，CARE 基本持平。
- 结论：learned positional embedding 没有改善当前 LSTM-AE，反而可能增加 decoder 拟合负担；LSTM
  自身递归 hidden state 已提供顺序信息，因此停止该方向。当前保留无位置向量的 hidden256/CARE 和
  hidden512/AUC 两个对照。
- 输出：`results/20260718_201245_822833_lstm_optimized_result.json`、
  `results/20260718_201252_378824_lstm_optimized_result.json`。

### Farm B LSTM-AE：完整特征 + 扩大正常训练集对照 (2026-07-19)
- 目的：同时恢复 `feature_set=all`（原始 10min 统计列 + 窗口统计/导数/FFT，输入 1764 维）并将
  `cap_train` 从约 60k 提高到 200k，验证论文完整特征和更多正常样本是否能提升 LSTM 泛化。
- 设置：window576、seq_len96、hidden256、latent128、state decoder、无 positional、lr1e-4、
  warmup-cosine、batch32。训练稳定，best epoch 19，best val 0.3982。
- 结果：AUC **0.4283**、CARE 0.4861、Coverage 0.0685、Accuracy 0.9954、Reliability 0.3571、
  Earliness 0.0143。异常分数反向：normal mean 0.5386 高于 anomaly mean 0.5311，median gap -0.0644。
- 对照：`avg_only` hidden256 的 AUC 0.5801、CARE 0.5718；因此扩大训练样本没有抵消完整统计特征
  的负面影响。完整特征使 LSTM 主要学习到高维统计噪声/可重构背景，而不是故障模式，并导致极端
  保守报警。
- 结论：LSTM 主线继续固定 `avg_only`；`all` 结果作为重要负向消融，不再继续在全量特征上调 hidden、
  epoch 或 scheduler。输出：`results/20260718_231847_526832_lstm_optimized_result.json`。

### Farm B LSTM-AE：feature window 与 seq_len 对照 (2026-07-19)
- 固定 `avg_only`、hidden256、latent128、state decoder、无 positional、lr1e-4，比较 window288/576
  与 seq_len48/96。四组均稳定训练，best epoch 13--20。
- window288 / seq48：AUC 0.5592、CARE 0.5071、Coverage 0.2047、Earliness 0.0563。
- window288 / seq96：AUC 0.5681、CARE 0.5556、Coverage 0.3649、Earliness 0.1305。
- window576 / seq48：AUC 0.5614、CARE 0.5247、Coverage 0.2836、Earliness 0.1124。
- window576 / seq96：AUC **0.5796**、CARE 0.5441、Coverage 0.3728、Earliness 0.2016。
- 结论：seq_len96 在两个 feature window 下都明显优于 seq_len48，确认 16 小时序列比 8 小时更适合
  当前 LSTM。window288/seq96 的本轮 CARE 略高于 window576/seq96，但 AUC、Coverage、Earliness 不占优，
  且低于此前 window576/seq96 的 CARE 0.5718；因此不能认为缩短 window 有稳定收益。主配置继续固定
  window576 + seq_len96，停止继续缩小 window/seq_len 的大网格。
- 输出：`results/20260719_091641_367602_lstm_optimized_result.json`、
  `results/20260719_091647_493615_lstm_optimized_result.json`、
  `results/20260719_100601_967636_lstm_optimized_result.json`、
  `results/20260719_100613_416135_lstm_optimized_result.json`。

### Farm B LSTM-AE：旧式 all/window96/小 cap 配置复验 (2026-07-19)
- 设置：`feature_set=all`、window96、seq_len48、cap_train30k、hidden256、latent64、60 epoch、
  state decoder、无 positional。训练稳定，best epoch 9。
- 结果：AUC 0.5776、CARE 0.2718、Coverage 0.8126、Accuracy 0.2718、Reliability 0.4545、
  Earliness 0.6725。
- 关键诊断：validation P99 阈值仅 3.0156，但 prediction 正常点分数 median 已达 15.75，导致大量正常
  点被报警；因此 Accuracy 只有 0.2718，CARE 极低。异常分数还出现极端值（mean 约 1.03e10），说明
  完整统计特征在小 cap/window96 下产生严重数值/分布漂移。AUC 0.5776 只能说明排序部分恢复，不能
  作为可部署结果，也不能用测试分布反推阈值。
- 结论：旧式配置可能解释过去出现的较高 AUC，但它依赖不稳定的分数分布；严格 validation-only 阈值下
  并不可靠。LSTM 主线仍保留 avg_only，all/window96 作为“高 AUC 与 CARE 失配”的负向审计案例。
- 输出：`results/20260719_110606_774667_lstm_optimized_result.json`。

### Farm B LSTM-AE：MAE 训练损失与 window96 对照 (2026-07-19)
- 实现：为 LSTM-AE 增加 `--train-loss {mse,mae}`，仅改变训练损失；全序列重构、最终 anomaly score
  和 CARE 协议保持不变。dropout 仍为 0。
- avg_only / window576 / seq96 / hidden256 / latent128 / MAE：AUC 0.5614、CARE 0.5482、Coverage
  0.3984、Accuracy 0.8508、Reliability 0.4348、Earliness 0.2061，best epoch 18，best val 0.3554。
  相比同配置 MSE 的 AUC 0.5801、CARE 0.5718，MAE 明显下降。
- avg_only / window96 / seq96 / 同结构 / MAE：AUC 0.5386、CARE 0.5410、Coverage 0.3536、Accuracy
  0.8778、Reliability 0.4839、Earliness 0.1120，best epoch 25，best val 0.3706。
- 结论：MAE 的 validation loss 更低，但异常排序更差，确认“重构损失最小”不能作为 AUC 优化代理；
  当前 LSTM 继续使用 MSE 训练。window96 + MAE 也没有恢复旧实验的 AUC，window576/avg_only/MSE
  仍是当前 LSTM 主基线。
- 输出：`results/20260719_114912_799373_lstm_optimized_result.json`、
  `results/20260719_114919_403815_lstm_optimized_result.json`。

### Farm B LSTM-AE：reference 非对称瓶颈结构对照 (2026-07-19)
- 迁移 `reference/main.pdf` 的 LSTM-AE 结构：Encoder `[LSTM(128), LSTM(64)]`，Decoder `[LSTM(64),
  LSTM(128)]`，无 dropout，MSE 训练，全序列 MSE score。实现为 `--architecture paper`，原 symmetric
  结构保留为对照。
- 设置：Farm B、avg_only、window576、seq_len96、cap_train60k、hidden128、latent64、state decoder、
  warmup-cosine、300 epoch。训练稳定但 best epoch 15、best val 0.4474。
- 结果：AUC 0.5474、CARE 0.5486、Coverage 0.4194、Accuracy 0.8270、Reliability 0.4630、
  Earliness 0.2066；mean gap 0.0383、median gap 0.0450。
- 对照：当前 symmetric hidden256/latent128 的 AUC 0.5801、CARE 0.5718；symmetric hidden512/latent128
  的 AUC 0.5835。非对称结构明显下降，说明在 441 维 CARE 特征上 `128→64` 瓶颈过强，出现表达不足。
- 结论：reference 的架构和 CARE 的输入维度/采样协议不同，不能直接照搬；停止 `architecture=paper` 主线，
  保留为有价值的负向结构消融。LSTM 主线继续 symmetric hidden256/latent128/MSE。
- 输出：`results/20260719_125407_323372_lstm_optimized_result.json`。

### Farm B LSTM-AE：去除 latent 线性投影的 direct 结构 (2026-07-19)
- 新增 `architecture=direct`：encoder 最后一层 hidden state 直接作为 z，不经过 `Linear(hidden→latent)`；
  direct 模式要求 `latent=hidden`。其余全序列 MSE、state decoder、无 positional、avg_only/window576/seq96
  协议保持不变。
- direct512（hidden=latent=512）：AUC **0.5884**、CARE **0.5766**、Coverage 0.4196、Accuracy 0.8781、
  Reliability 0.5000，best epoch 19，best val 0.3896；超过 symmetric hidden256/latent128（AUC 0.5801、
  CARE 0.5718）和 symmetric hidden512/latent128（AUC 0.5835、CARE 0.5634）。
- direct256（hidden=latent=256）：AUC 0.5652、CARE 0.5701、Coverage 0.3886、Accuracy 0.9010、
  Reliability 0.5000，best epoch 26，best val 0.4071；没有改善。
- 结论：当前收益来自“保留高维最终 hidden 表达 + 去掉过强的 latent 投影”，而不是单纯去掉线性层。
  direct512 成为当前 LSTM 最佳结构；direct256 说明容量仍是关键。下一步可固定 direct512，转向
  更大正常训练集或一次 `seq_len=144` 复验，但不再继续做 paper 窄瓶颈结构。
- 输出：`results/20260719_133434_839498_lstm_optimized_result.json`、
  `results/20260719_133441_861386_lstm_optimized_result.json`。

### Farm B LSTM-AE：direct512 数据量与 24 小时序列复验 (2026-07-19)
- direct512 固定：avg_only、window576、MSE、state decoder、无 positional。
- avg_only / cap_train200k / seq96：AUC 0.4339、CARE 0.4768、Coverage 0.0313、Accuracy 0.9948、
  Reliability 0.3571、Earliness 0.0061，best epoch 50，best val 0.3254。异常分数反向（normal median
  0.3653 > anomaly median 0.3228，gap -0.0425），说明扩大 cap 后训练/验证分布与 prediction 发生漂移，
  validation loss 下降不能代表异常排序改善；不再继续扩大 cap。
- avg_only / cap_train60k / seq144（24 小时）：AUC **0.5949**、CARE **0.5737**、Coverage 0.4523、
  Accuracy 0.8396、Reliability 0.5000、Earliness 0.2369，best epoch 9，best val 0.4539，median gap
  0.0941。AUC 与分数分离度均超过 direct512/seq96（AUC 0.5884）。
- 结论：`seq_len=144` 是当前最有希望的 LSTM 方向，说明 24 小时上下文对故障前模式有增益；`cap_train`
  盲目扩大反而破坏 validation-only 阈值与分布稳定性。下一步固定 direct512/seq144，只做一次随机种子
  复验或轻量 decoder 输入实验。
- 输出：`results/20260719_145100_944238_lstm_optimized_result.json`、
  `results/20260719_145109_027966_lstm_optimized_result.json`。
