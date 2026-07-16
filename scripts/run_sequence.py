from __future__ import annotations

import argparse
import json
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
from faultdiagnose.models import LSTMAE, SeqWindowsDataset, TransformerAE
from faultdiagnose.training.seq_trainer import train_seq_ae

OUT = load_care.OUT_DEFAULT
RESULTS = Path(__file__).resolve().parents[1] / "results"


def collect_training_seq(farms, window, seq_len, cap, use_fft=False):
    events = load_events(OUT)
    normal_ids: dict[str, list[str]] = {}
    for r in events.itertuples():
        if r.event_label == "normal":
            normal_ids.setdefault(r.farm, []).append(str(r.event_id))
    mats, total = [], 0
    for farm in farms:
        for did in normal_ids.get(farm, []):
            df = load_dataset(OUT, farm, did)
            df = df[operating_mask(df)]
            cols = select_feature_columns(df)
            feats = engineer_features(df, window=window, columns=cols, use_fft=use_fft, fillna="ffill", clip_sigma=5.0).dropna()
            X = feats.values.astype(np.float32)
            if total + len(X) > cap:
                X = X[: cap - total]  # honor cap within a single large dataset
            mats.append(X)
            total += len(X)
            if total >= cap:
                break
        if total >= cap:
            break
    X_all = np.concatenate(mats, axis=0)
    mean, std = fit_standardizer(X_all)
    zmats = [apply_standardizer(m, mean, std).astype(np.float32) for m in mats]
    return zmats, mean, std


def evaluate_seq(farms, window, seq_len, model, mean, std, ev_map, device,
                  use_fft=False, batch=2048, label_mode="operating"):
    all_scores, all_labels = [], []
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
                    e = model.reconstruction_error(xb.to(device))
                    win_err[pos : pos + len(e)] = e.cpu().numpy()
                    pos += len(e)
            N = len(feats)
            scores = np.full(N, np.nan, dtype=np.float32)
            scores[seq_len - 1 :] = win_err  # window i -> last time-step i+T-1
            scores = pd.Series(scores).bfill().ffill().to_numpy()
            ev = ev_map.get((farm, did))
            if ev is not None and ev.event_label == "anomaly":
                start, end = pd.to_datetime(ev.event_start), pd.to_datetime(ev.event_end)
                labels = ((times >= start) & (times <= end)).astype(int).values
            else:
                labels = np.zeros(N, dtype=int)
            all_scores.append(scores)
            all_labels.append(labels)
    return np.concatenate(all_scores), np.concatenate(all_labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    ap.add_argument("--farms", default="A")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--cap-train", type=int, default=60000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-fft", action="store_true", help="disable FFT features (enabled by default)")
    ap.add_argument("--label-mode", choices=["operating", "full"], default="operating",
                   help="operating: score/label only operating rows; full: label entire event window incl. downtime")
    args = ap.parse_args()
    use_fft = not args.no_fft
    label_mode = args.label_mode

    if not OUT.exists():
        raise SystemExit("processed data missing; run convert first")

    farms = [f.strip() for f in args.farms.split(",")]
    zmats, mean, std = collect_training_seq(
        farms, args.window, args.seq_len, args.cap_train, use_fft
    )
    in_dim = zmats[0].shape[1]
    n_train = sum(len(m) - args.seq_len + 1 for m in zmats)

    if args.model == "lstm":
        model = LSTMAE(in_dim, args.seq_len, latent=args.latent, hidden=args.hidden)
    else:
        model = TransformerAE(in_dim, args.seq_len, latent=args.latent, d_model=args.hidden)
    print(f"Training {args.model}: in_dim={in_dim}, seq_len={args.seq_len}, "
          f"train_windows={n_train}, farms={farms}")
    losses = train_seq_ae(model, zmats, args.seq_len, epochs=args.epochs,
                          batch_size=args.batch, device=args.device)

    events = load_events(OUT)
    ev_map = {(r.farm, str(r.event_id)): r for r in events.itertuples()}
    scores, labels = evaluate_seq(farms, args.window, args.seq_len, model, mean, std,
                                  ev_map, args.device, use_fft, label_mode=args.label_mode)
    finite = np.isfinite(scores)
    if not finite.all():
        n_nan = int((~finite).sum())
        print(f"WARNING: {n_nan} NaN scores dropped ({n_nan / len(scores) * 100:.1f}%)")
        scores = scores[finite]
        labels = labels[finite]
    auc = compute_auc(scores, labels)
    print(f"AUC-ROC = {auc:.4f}  (test vectors={len(scores)}, positives={labels.sum()})")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {
        "model": args.model, "farms": farms, "window": args.window,
        "seq_len": args.seq_len, "in_dim": in_dim, "latent": args.latent,
        "hidden": args.hidden, "epochs": args.epochs, "n_train_windows": int(n_train),
        "n_test": int(len(scores)), "n_positives": int(labels.sum()),
        "auc_roc": float(auc), "final_loss": float(losses[-1]),
    }
    path = RESULTS / f"{args.model}_baseline.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
