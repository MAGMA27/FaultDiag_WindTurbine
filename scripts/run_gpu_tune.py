"""GPU-optimized tuning script for all 3 AE variants on CARE Farm A.

Key departures from the CPU scripts:
- Larger epochs (50-100), batch sizes (512-2048), hidden dims (128-512)
- More training data (cap=150k; CARE Farm A has millions of rows)
- Per-epoch AUC recording for learning-curve analysis
- Saves model checkpoints for ensemble reuse
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from faultdiagnose.data import list_datasets, load_care, load_dataset, load_events, operating_mask
from faultdiagnose.evaluation.anomaly import compute_auc
from faultdiagnose.features import (
    apply_standardizer,
    engineer_features,
    fit_standardizer,
    select_feature_columns,
)
from faultdiagnose.models import LSTMAE, VAE, SeqWindowsDataset, TransformerAE

OUT = load_care.OUT_DEFAULT
RESULTS = Path(__file__).resolve().parents[1] / "results"
MODELS_DIR = RESULTS / "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 锟斤拷锟斤拷 data pipeline 锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷

def collect_all(farms, window, seq_len, cap, use_fft=False):
    """Collect both VAE (flat) mats and LSTM (sequence) mats from normal data."""
    events = load_events(OUT)
    normal_ids: dict[str, list[str]] = {}
    for r in events.itertuples():
        if r.event_label == "normal":
            normal_ids.setdefault(r.farm, []).append(str(r.event_id))

    flat, seq_mats = [], []
    total = 0
    for farm in farms:
        for did in normal_ids.get(farm, []):
            df = load_dataset(OUT, farm, did)
            if "train_test" in df.columns:
                df = df[df["train_test"] == "train"]
            df = df[operating_mask(df)]
            cols = select_feature_columns(df)
            feats = engineer_features(df, window=window, columns=cols, use_fft=use_fft, fillna="ffill", clip_sigma=5.0).dropna()
            X = feats.values.astype(np.float32)
            flat.append(X)
            total += len(X)
            if total >= cap:
                break
        if total >= cap:
            break

    X_all = np.concatenate(flat, axis=0)
    mean, std = fit_standardizer(X_all)
    zmats = [apply_standardizer(m, mean, std).astype(np.float32) for m in flat]
    vae_X = np.concatenate(zmats, axis=0)
    return vae_X, zmats, mean, std


def evaluate_vae(model, farms, window, mean, std, ev_map, use_fft=False, label_mode="operating"):
    scores_list, labels_list = [], []
    for farm in farms:
        for did in list_datasets(OUT, farm):
            df = load_dataset(OUT, farm, did)
            if label_mode == "operating":
                df = df[operating_mask(df)]
            cols = select_feature_columns(df)
            feats = engineer_features(df, window=window, columns=cols, use_fft=use_fft, fillna="ffill", clip_sigma=5.0).dropna()
            times = df.loc[feats.index, "time_stamp"]
            Z = apply_standardizer(feats, mean, std).astype(np.float32)
            with torch.no_grad():
                scores = model.reconstruction_error(torch.from_numpy(Z).to(DEVICE)).cpu().numpy()
            ev = ev_map.get((farm, did))
            labels = _make_labels(ev, times, len(feats))
            scores_list.append(scores)
            labels_list.append(labels)
    return np.concatenate(scores_list), np.concatenate(labels_list)


def evaluate_seq(model, farms, window, seq_len, mean, std, ev_map, use_fft=False, batch=4096, label_mode="operating"):
    scores_list, labels_list = [], []
    for farm in farms:
        for did in list_datasets(OUT, farm):
            df = load_dataset(OUT, farm, did)
            if label_mode == "operating":
                df = df[operating_mask(df)]
            cols = select_feature_columns(df)
            feats = engineer_features(df, window=window, columns=cols, use_fft=use_fft, fillna="ffill", clip_sigma=5.0)
            times = df.loc[feats.index, "time_stamp"]
            feats = feats.dropna()
            times = times.loc[feats.index]
            if len(feats) < seq_len:
                continue
            Z = apply_standardizer(feats, mean, std).astype(np.float32)
            ds = SeqWindowsDataset([Z], seq_len)
            dl = DataLoader(ds, batch_size=batch, shuffle=False)
            win_err = np.empty(len(ds), dtype=np.float32)
            pos = 0
            with torch.no_grad():
                for xb in dl:
                    e = model.reconstruction_error(xb.to(DEVICE))
                    win_err[pos : pos + len(e)] = e.cpu().numpy()
                    pos += len(e)
            N = len(feats)
            scores = np.full(N, np.nan, dtype=np.float32)
            scores[seq_len - 1 :] = win_err
            scores = pd.Series(scores).bfill().ffill().to_numpy()
            ev = ev_map.get((farm, did))
            labels = _make_labels(ev, times, N)
            scores_list.append(scores)
            labels_list.append(labels)
    return np.concatenate(scores_list), np.concatenate(labels_list)


def _make_labels(ev, times, n):
    if ev is not None and ev.event_label == "anomaly":
        start, end = pd.to_datetime(ev.event_start), pd.to_datetime(ev.event_end)
        return ((times >= start) & (times <= end)).astype(int).values
    return np.zeros(n, dtype=int)


# 锟斤拷锟斤拷 training with per-epoch AUC 锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷

def train_vae_gpu(model, X, epochs, batch_size, lr=1e-3, name="", loss_f=None):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dl = DataLoader(torch.from_numpy(X).float(), batch_size=batch_size, shuffle=True)
    losses = []
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for xb in dl:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            recon, mu, logvar = model(xb)
            loss, _, _ = model.loss(xb, recon, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            n += 1
        losses.append(tot / n)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  ep {ep + 1}/{epochs} loss={losses[-1]:.4f}")
        if loss_f is not None:
            loss_f.write(f"name={name} ep={ep+1} loss={losses[-1]:.6f}\n")
            loss_f.flush()
    return losses


def train_seq_gpu(model, zmats, seq_len, epochs, batch_size, lr=1e-3, name="", loss_f=None):
    model.to(DEVICE)
    ds = SeqWindowsDataset(zmats, seq_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for xb in dl:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            recon, _ = model(xb)
            loss = model.loss(xb, recon)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            n += 1
        losses.append(tot / n)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  ep {ep + 1}/{epochs} loss={losses[-1]:.4f}")
        if loss_f is not None:
            loss_f.write(f"name={name} ep={ep+1} loss={losses[-1]:.6f}\n")
            loss_f.flush()
    return losses


# 锟斤拷锟斤拷 hyperparameter configs 锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷

CONFIGS = [
{"name": "vae_h256_l32_b0.1",   "model": "vae",  "hidden": 256, "latent": 32,  "beta": 0.1,  "epochs": 80, "batch": 1024},
{"name": "vae_h512_l64_b0.1",   "model": "vae",  "hidden": 512, "latent": 64,  "beta": 0.1,  "epochs": 80, "batch": 1024},
{"name": "vae_h256_l64_b1.0",   "model": "vae",  "hidden": 256, "latent": 64,  "beta": 1.0,  "epochs": 80, "batch": 1024},
{"name": "vae_h256_l64_b0.1",   "model": "vae",  "hidden": 256, "latent": 64,  "beta": 0.1,  "epochs": 80, "batch": 1024},
{"name": "vae_h512_l64_b1.0",   "model": "vae",  "hidden": 512, "latent": 64,  "beta": 1.0,  "epochs": 80, "batch": 1024},
{"name": "vae_h512_l128_b1.0",  "model": "vae",  "hidden": 512, "latent": 128, "beta": 1.0,  "epochs": 80, "batch": 1024},
{"name": "lstm_h128_l64",       "model": "lstm", "hidden": 128, "latent": 64,  "epochs": 80, "batch": 512,  "seq_len": 24},
{"name": "lstm_h256_l64",       "model": "lstm", "hidden": 256, "latent": 64,  "epochs": 80, "batch": 512,  "seq_len": 24},
{"name": "lstm_h128_l128",      "model": "lstm", "hidden": 128, "latent": 128, "epochs": 80, "batch": 512,  "seq_len": 24},
{"name": "lstm_h256_l128",      "model": "lstm", "hidden": 256, "latent": 128, "epochs": 80, "batch": 512,  "seq_len": 24},
{"name": "tf_h128_l64",         "model": "transformer", "hidden": 128, "latent": 64,  "epochs": 80, "batch": 256,  "seq_len": 24},
{"name": "tf_h256_l64",         "model": "transformer", "hidden": 256, "latent": 64,  "epochs": 80, "batch": 256,  "seq_len": 24},
{"name": "tf_h128_l128",        "model": "transformer", "hidden": 128, "latent": 128, "epochs": 80, "batch": 256,  "seq_len": 24}

]


def run_config(cfg, farms, window, cap, use_fft, label_mode="operating", loss_f=None, args=None):
    name = cfg["name"]
    print(f"\n{'='*60}\n  {name}\n{'='*60}")

    seq_len = cfg.get("seq_len", 24)
    epochs = args.epochs if args is not None and getattr(args, "epochs", None) else cfg["epochs"]
    vae_X, zmats, mean, std = collect_all(farms, window, seq_len, cap, use_fft)
    in_dim = vae_X.shape[1]
    n_train = len(vae_X)
    n_win = sum(len(m) - seq_len + 1 for m in zmats)
    print(f"  in_dim={in_dim}, train vecs={n_train}, train windows={n_win}")

    events = load_events(OUT)
    ev_map = {(r.farm, str(r.event_id)): r for r in events.itertuples()}

    t0 = time.time()

    if cfg["model"] == "vae":
        model = VAE(in_dim, latent=cfg["latent"], hidden=cfg["hidden"], beta=cfg["beta"])
        train_vae_gpu(model, vae_X, epochs=epochs, batch_size=cfg["batch"], name=name, loss_f=loss_f)
        scores, labels = evaluate_vae(model, farms, window, mean, std, ev_map, use_fft, label_mode)
    elif cfg["model"] == "lstm":
        model = LSTMAE(in_dim, seq_len, latent=cfg["latent"], hidden=cfg["hidden"], num_layers=2)
        train_seq_gpu(model, zmats, seq_len, epochs=epochs, batch_size=cfg["batch"], name=name, loss_f=loss_f)
        scores, labels = evaluate_seq(model, farms, window, seq_len, mean, std, ev_map, use_fft, label_mode)
    else:  # transformer
        nhead = 4 if cfg["hidden"] <= 128 else 8
        model = TransformerAE(in_dim, seq_len, latent=cfg["latent"],
                              d_model=cfg["hidden"], nhead=nhead, num_layers=2)
        train_seq_gpu(model, zmats, seq_len, epochs=epochs, batch_size=cfg["batch"], name=name, loss_f=loss_f)
        scores, labels = evaluate_seq(model, farms, window, seq_len, mean, std, ev_map, use_fft, label_mode)

    elapsed = time.time() - t0
    finite = np.isfinite(scores)
    scores, labels = scores[finite], labels[finite]
    auc = compute_auc(scores, labels)

    result = {
        **cfg,
        "in_dim": in_dim, "n_train": n_train, "n_test": int(len(scores)),
        "n_positives": int(labels.sum()), "auc_roc": float(auc),
        "elapsed_s": round(elapsed, 1), "device": DEVICE,
    }
    print(f"  >> AUC={auc:.4f}  ({elapsed:.0f}s)")

    # Save checkpoint + its frozen normalization stats (self-contained artifact)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / f"{name}.pt")
    np.savez(MODELS_DIR / f"{name}_norm.npz", mean=mean, std=std)

    return result


def main():
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--farms", default="A")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--cap-train", type=int, default=150000)
    ap.add_argument("--no-fft", action="store_true", help="disable FFT features (enabled by default)")
    ap.add_argument("--label-mode", choices=["operating", "full"], default="operating",
                   help="operating: score/label only operating rows; full: label entire event window incl. downtime")
    ap.add_argument("--cfg", help="run single config by name, e.g. vae_h256_l32_b0.1")
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    ap.add_argument("--device", default=DEVICE, help="override device")
    args = ap.parse_args()
    DEVICE = args.device

    if not OUT.exists():
        raise SystemExit("processed data missing; run convert_care_to_parquet first")
    print(f"Device: {DEVICE}  ({torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'})")

    farms = [f.strip() for f in args.farms.split(",")]
    use_fft = not args.no_fft
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    loss_path = RESULTS / f"gpu_tune_loss_{stamp}.txt"
    loss_f = open(loss_path, "w", encoding="utf-8")
    print(f"  loss log -> {loss_path}")
    configs = CONFIGS if not args.cfg else [c for c in CONFIGS if c["name"] == args.cfg]
    if not configs:
        raise SystemExit(f"config '{args.cfg}' not found")

    results = []
    for cfg in configs:
        try:
            r = run_config(cfg, farms, args.window, args.cap_train, use_fft, args.label_mode, loss_f, args)
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({**cfg, "error": str(e)})

    loss_f.close()

    # Save all results
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    path = RESULTS / f"gpu_tune_{stamp}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    for r in results:
        if "auc_roc" in r:
            print(f"  {r['name']:30s}  AUC={r['auc_roc']:.4f}  ({r['elapsed_s']:.0f}s)")
        else:
            print(f"  {r['name']:30s}  FAILED: {r.get('error', '?')}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()


