
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
- **修复**: src/faultdiagnose/models/vae.py 的 eparameterize 内 clamp logvar 到 [-10, 10] (std in [e^-5, e^5])。尖峰消除。
- **数据标准化确认**: un_gpu_tune.py 的 collect_all 已做 z-score (it_standardizer/pply_standardizer),非问题根因。
- **beta ablation (同结构 vae_h256_l64, 80ep, Farm A, cap 150k)**:
  - beta=1.0: loss 152.4, AUC=0.607
  - beta=0.1: loss 91.9,  AUC=0.629  (KL 权重过高压垮重构, latent 趋塌缩, 异常分数区分度下降)
  - 结论: beta 是主因之一, 方向确认。beta=0.1 的 loss 仍持续下降 (ep80 未平台), 加 epoch 可能继续涨。
- **下一步 (未做)**:
  - beta=0.1 加 epoch (200) 看天花板; 试 beta=0.01 确认单调性。
  - 567 维特征中常数/低方差列被 std=1 保留为噪声, 稀释异常信号 -> 需丢弃低方差列再标准化 (改动面较大, 待定)。
  - 正负比 ~0.24% (2522/1053180), AUC 对少量正样本本就难拉高, 需确认标签/窗口逻辑。
- **新增 config**: ae_h256_l64_b0.1 加入 scripts/run_gpu_tune.py CONFIGS, 用于严格 beta ablation。
