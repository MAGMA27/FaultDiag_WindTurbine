# 云端训练记录

最后更新：2026-07-29

## 已确认结果

- CARE 官方风格 adaptive-AE（Farm B）在云端得到 `CARE=0.6585`：Coverage F0.5=0.7355、Accuracy=0.6933、Reliability F0.5=0.7143、Earliness=0.4560。AUC=0.5493；该 runner 的 AUC 标签范围与 CARE 运行状态筛选不同，不能用 AUC 单独否定该事件级结果。
- VAE 的 200k 样本、1500 epoch 试验已完成：`AUC=0.5584`、`CARE=0.4904`、best epoch=1157、best validation loss=29.2002。末段验证损失已平台化，不再通过继续加 epoch 优化这一配置。
- 历史 VAE 参考配置为 `avg_only`、window 576、hidden 512、latent 128、beta 0.1、60k 样本、batch 1024、LR 2e-4、warmup 50；结果为 AUC=0.7163、CARE=0.6131。下一轮保持 200k 样本，仅把 batch/LR/warmup 回退到该有效区间，以拆分数据规模与训练动力学的影响。

## 云端数据管线

- RAM-only sequence window cache（提交 `dfcaf13`）先物化滑动序列窗口，再从连续内存取样；避免每个 batch 反复在 Python/CPU 中动态切片。
- 4090 建议使用 `--ram-window-cache --window-cache-dtype float16 --max-window-cache-gb 24 --num-workers 2`。该路径不占硬盘缓存；60k、seq_len 144 的窗口约需 17GB RAM（FP16），适合 32GB RAM、磁盘仅约 5GB 剩余的服务器。
- DataLoader shuffle 仅打乱窗口样本顺序，不破坏单个窗口内部的连续时间顺序。`num_workers=2` 与 prefetch 由 PyTorch DataLoader 原生管理，worker 只提前准备未来 batch，不改变 sampler 给出的训练顺序。

## 特征配置

- `raw_stat_compact` profile（提交 `2ef838b`）保留所有原始 10 分钟统计量为 `raw`，但只对 `*_avg` 信号计算滚动统计、导数和 FFT。Farm B 输入预计从 `stat_aware` 的约 1086 维降至约 698 维。
- 该 profile 对应“保留 10 分钟内 min/max/std 信息，避免再对其做跨 10 分钟二次统计”；待与 Transformer-AE 当前最优配置对照。
