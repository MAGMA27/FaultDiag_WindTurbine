
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
