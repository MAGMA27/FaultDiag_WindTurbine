"""Raw-sensor (no feature engineering) LSTM-AE training for AUC tuning.

Hypothesis from diag_separation.py: engineered 486-dim INVERTS the anomaly
signal (anomaly error < normal error), while raw 81-dim has correct direction
(anomaly > normal) but too-small separation at 200k training vectors. This
script scales raw-mode training data to test whether volume recovers AUC.
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
from faultdiagnose.features import apply_standardizer, fit_standardizer, select_feature_columns
from faultdiagnose.models import LSTMAE, SeqWindowsDataset

OUT = load_care.OUT_DEFAULT
RESULTS = Path(__file__).resolve().parents[1] / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 24
WINDOW = 24


def collect_raw(farms, cap):
    events = load_events(OUT)
    normal_ids = [str(r.event_id) for r in events.itertuples()
                  if r.event_label == "normal" and r.farm in farms]
    mats, total = [], 0
    for did in normal_ids:
        farm = next(f for f in farms if load_events(OUT) is not None)  # placeholder
    # iterate per farm
    per_farm = {f: [] for f in farms}
    for r in events.itertuples():
        if r.event_label == "normal" and r.farm in farms:
            per_farm[r.farm].append(str(r.event_id))
    all_mats, total = [], 0
    for farm in farms:
        for did in per_farm[farm]:
            df = load_dataset(OUT, farm, did)
            if "train_test" in df.columns:
                df = df[df["train_test"] == "train"]
            df = df[operating_mask(df)]
            cols = select_feature_columns(df)
            X = df[cols].dropna().values.astype(np.float32)
            all_mats.append(X)
            total += len(X)
            if total >= cap:
                break
        if total >= cap:
            break
    all_X = np.concatenate(all_mats, axis=0)
    mean, std = fit_standardizer(all_X)
    return [apply_standardizer(m, mean, std).astype(np.float32) for m in all_mats], mean, std


def evaluate(model, farms, mean, std, ev_map, batch=4096, label_mode="operating"):
    scores_list, labels_list = [], []
    for farm in farms:
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
            N = len(feats)
            ds = SeqWindowsDataset([Z], SEQ_LEN)
            dl = DataLoader(ds, batch_size=batch, shuffle=False)
            win = np.empty(len(ds), dtype=np.float32)
            pos = 0
            with torch.no_grad():
                for xb in dl:
                    win[pos:pos+len(xb)] = model.reconstruction_error(xb.to(DEVICE)).cpu().numpy()
                    pos += len(xb)
            s = np.full(N, np.nan, dtype=np.float32)
            s[SEQ_LEN-1:] = win
            s = pd.Series(s).bfill().ffill().to_numpy()
            ev = ev_map.get((farm, did))
            if ev is not None and ev.event_label == "anomaly":
                start, end = pd.Timestamp(ev.event_start), pd.Timestamp(ev.event_end)
                labels = ((times >= start) & (times <= end)).astype(int).values
            else:
                labels = np.zeros(N, dtype=int)
            scores_list.append(s)
            labels_list.append(labels)
    return np.concatenate(scores_list), np.concatenate(labels_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--farms", default="A")
    ap.add_argument("--cap-train", type=int, default=600000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--label-mode", choices=["operating", "full"], default="operating",
                   help="operating: score/label only operating rows; full: label entire event window incl. downtime")
    args = ap.parse_args()
    print(f"Device: {DEVICE}")

    farms = [f.strip() for f in args.farms.split(",")]
    zmats, mean, std = collect_raw(farms, args.cap_train)
    in_dim = zmats[0].shape[1]
    n_win = sum(len(m) - SEQ_LEN + 1 for m in zmats)
    print(f"Raw in_dim={in_dim}, train windows={n_win}")

    model = LSTMAE(in_dim, SEQ_LEN, latent=args.latent, hidden=args.hidden,
                   num_layers=args.num_layers).to(DEVICE)
    ds = SeqWindowsDataset(zmats, SEQ_LEN)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, pin_memory=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    for ep in range(args.epochs):
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
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"  ep {ep+1}/{args.epochs} loss={tot/n:.4f}")

    events = load_events(OUT)
    ev_map = {(r.farm, str(r.event_id)): r for r in events.itertuples()}
    scores, labels = evaluate(model, farms, mean, std, ev_map, label_mode=args.label_mode)
    f = np.isfinite(scores)
    scores, labels = scores[f], labels[f]
    auc = compute_auc(scores, labels)
    el = time.time() - t0
    print(f"  >> AUC={auc:.4f} ({el:.0f}s)")

    # distributions
    norm = scores[labels == 0]
    anom = scores[labels == 1]
    print(f"  normal:  mean={norm.mean():.4f} p50={np.percentile(norm,50):.4f} p95={np.percentile(norm,95):.4f}")
    print(f"  anomaly: mean={anom.mean():.4f} p50={np.percentile(anom,50):.4f} p95={np.percentile(anom,95):.4f}")

    out = {"mode": "raw", "in_dim": in_dim, "cap": args.cap_train, "epochs": args.epochs,
           "hidden": args.hidden, "latent": args.latent, "num_layers": args.num_layers,
           "auc_roc": float(auc), "elapsed_s": round(el, 1), "device": DEVICE}
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    (RESULTS / f"raw_lstm_{stamp}.json").write_text(json.dumps(out, indent=2))
    print(f"  saved -> {RESULTS / f'raw_lstm_{stamp}.json'}")


if __name__ == "__main__":
    main()
