"""Diagnostic: can the AE separate normal vs anomaly scores at all?

Trains a small LSTM-AE on normal data, then compares reconstruction-error
distributions for normal vs anomaly test windows. Also compares raw (81-dim)
vs engineered (486-dim) inputs to test the feature-engineering hypothesis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from faultdiagnose.data import list_datasets, load_care, load_dataset, load_events, operating_mask
from faultdiagnose.features import (
    apply_standardizer,
    engineer_features,
    fit_standardizer,
    select_feature_columns,
)
from faultdiagnose.models import LSTMAE, SeqWindowsDataset

OUT = load_care.OUT_DEFAULT
RESULTS = Path(__file__).resolve().parents[1] / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 24
WINDOW = 24


def collect_normal(farm, use_fft, cap=200000):
    events = load_events(OUT)
    normal_ids = [str(r.event_id) for r in events.itertuples()
                  if r.event_label == "normal" and r.farm == farm]
    mats, total = [], 0
    for did in normal_ids:
        df = load_dataset(OUT, farm, did)
        if "train_test" in df.columns:
            df = df[df["train_test"] == "train"]
        df = df[operating_mask(df)]
        cols = select_feature_columns(df)
        if use_fft:
            feats = engineer_features(df, window=WINDOW, columns=cols, use_fft=True, fillna="ffill", clip_sigma=5.0).dropna()
        else:
            feats = engineer_features(df, window=WINDOW, columns=cols, fillna="ffill", clip_sigma=5.0).dropna()
        X = feats.values.astype(np.float32)
        mats.append(X)
        total += len(X)
        if total >= cap:
            break
    all_X = np.concatenate(mats, axis=0)
    mean, std = fit_standardizer(all_X)
    return [apply_standardizer(m, mean, std).astype(np.float32) for m in mats], mean, std


def collect_test(farm, use_fft, mean, std, label_mode="operating"):
    events = load_events(OUT)
    ev_map = {(r.farm, str(r.event_id)): r for r in events.itertuples()}
    out = []
    for did in list_datasets(OUT, farm):
        df = load_dataset(OUT, farm, did)
        if label_mode == "operating":
            df = df[operating_mask(df)]
        cols = select_feature_columns(df)
        if use_fft:
            feats = engineer_features(df, window=WINDOW, columns=cols, use_fft=True, fillna="ffill", clip_sigma=5.0)
        else:
            feats = engineer_features(df, window=WINDOW, columns=cols, fillna="ffill", clip_sigma=5.0)
        times = df.loc[feats.index, "time_stamp"]
        feats = feats.dropna()
        times = times.loc[feats.index]
        if len(feats) < SEQ_LEN:
            continue
        Z = apply_standardizer(feats, mean, std).astype(np.float32)
        ev = ev_map.get((farm, did))
        if ev is not None and ev.event_label == "anomaly":
            start, end = pd.to_timestamp(ev.event_start) if hasattr(pd, "to_timestamp") else pd.Timestamp(ev.event_start), pd.Timestamp(ev.event_end)
            labels = ((times >= start) & (times <= end)).astype(int).values
        else:
            labels = np.zeros(len(feats), dtype=int)
        out.append((Z, labels, did, ev.event_label if ev else "normal"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-mode", choices=["operating", "full"], default="operating",
                   help="operating: label only operating rows in event window; "
                        "full: label entire event window incl. downtime")
    args = ap.parse_args()
    label_mode = args.label_mode
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    loss_path = RESULTS / f"diag_loss_{stamp}.txt"
    loss_f = open(loss_path, "w", encoding="utf-8")
    print(f"  loss log -> {loss_path}")
    farm = "A"
    for use_fft in (False, True):
        for mode in ("engineered", "raw"):
            print(f"\n{'='*50}\n  mode={mode}, use_fft={use_fft}\n{'='*50}")
            if mode == "raw":
                # raw: use original columns, z-scored
                events = load_events(OUT)
                normal_ids = [str(r.event_id) for r in events.itertuples()
                              if r.event_label == "normal" and r.farm == farm]
                mats, total = [], 0
                for did in normal_ids:
                    df = load_dataset(OUT, farm, did)
                    if "train_test" in df.columns:
                        df = df[df["train_test"] == "train"]
                    df = df[operating_mask(df)]
                    cols = select_feature_columns(df)
                    X = df[cols].dropna().values.astype(np.float32)
                    mats.append(X)
                    total += len(X)
                    if total >= 200000:
                        break
                all_X = np.concatenate(mats, axis=0)
                mean, std = fit_standardizer(all_X)
                zmats = [apply_standardizer(m, mean, std).astype(np.float32) for m in mats]
                in_dim = all_X.shape[1]
            else:
                zmats, mean, std = collect_normal(farm, use_fft)
                in_dim = zmats[0].shape[1]

            print(f"  in_dim={in_dim}")
            model = LSTMAE(in_dim, SEQ_LEN, latent=64, hidden=128, num_layers=1).to(DEVICE)
            ds = SeqWindowsDataset(zmats, SEQ_LEN)
            dl = DataLoader(ds, batch_size=512, shuffle=True)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            for ep in range(30):
                model.train()
                tot, n = 0.0, 0
                for xb in dl:
                    xb = xb.to(DEVICE)
                    opt.zero_grad()
                    recon, _ = model(xb)
                    loss = model.loss(xb, recon)
                    loss.backward()
                    opt.step()
                    tot += loss.item(); n += 1
                avg = tot / n
                print(f"  ep {ep+1}/30 loss={avg:.4f}")
                loss_f.write(f"mode={mode} use_fft={use_fft} ep={ep+1} loss={avg:.6f}\n")
                loss_f.flush()
            print(f"  trained 30 epochs")

            # Evaluate on test
            test_data = collect_test(farm, use_fft, mean, std, label_mode) if mode == "engineered" else None
            if mode == "raw":
                # rebuild test with raw
                events = load_events(OUT)
                ev_map = {(r.farm, str(r.event_id)): r for r in events.itertuples()}
                test_data = []
                for did in list_datasets(OUT, farm):
                    df = load_dataset(OUT, farm, did)
                    if label_mode == "operating":
                        df = df[operating_mask(df)]
                    cols = select_feature_columns(df)
                    feats = df[cols].dropna()
                    times = df.loc[feats.index, "time_stamp"]
                    if len(feats) < SEQ_LEN:
                        continue
                    Z = apply_standardizer(feats, mean, std).astype(np.float32)
                    ev = ev_map.get((farm, did))
                    if ev is not None and ev.event_label == "anomaly":
                        start, end = pd.Timestamp(ev.event_start), pd.Timestamp(ev.event_end)
                        labels = ((times >= start) & (times <= end)).astype(int).values
                    else:
                        labels = np.zeros(len(feats), dtype=int)
                    test_data.append((Z, labels, did, ev.event_label if ev else "normal"))

            all_err, all_lab = [], []
            model.eval()
            for Z, labels, did, label in test_data:
                sds = SeqWindowsDataset([Z], SEQ_LEN)
                sdl = DataLoader(sds, batch_size=2048, shuffle=False)
                win = np.empty(len(sds), dtype=np.float32)
                pos = 0
                with torch.no_grad():
                    for xb in sdl:
                        win[pos:pos+len(xb)] = model.reconstruction_error(xb.to(DEVICE)).cpu().numpy()
                        pos += len(xb)
                N = len(Z)
                scores = np.full(N, np.nan, dtype=np.float32)
                scores[SEQ_LEN-1:] = win
                scores = pd.Series(scores).bfill().ffill().to_numpy()
                all_err.append(scores)
                all_lab.append(labels)
            err = np.concatenate(all_err)
            lab = np.concatenate(all_lab)
            f = np.isfinite(err)
            err, lab = err[f], lab[f]
            from faultdiagnose.evaluation.anomaly import compute_auc
            auc = compute_auc(err, lab)
            # distributions
            norm_err = err[lab == 0]
            anom_err = err[lab == 1]
            print(f"  AUC={auc:.4f}")
            print(f"  normal:  mean={norm_err.mean():.4f}  p50={np.percentile(norm_err,50):.4f}  p95={np.percentile(norm_err,95):.4f}  n={len(norm_err)}")
            print(f"  anomaly: mean={anom_err.mean():.4f}  p50={np.percentile(anom_err,50):.4f}  p95={np.percentile(anom_err,95):.4f}  n={len(anom_err)}")


    loss_f.close()


if __name__ == "__main__":
    main()
