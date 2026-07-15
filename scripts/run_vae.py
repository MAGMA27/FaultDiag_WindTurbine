from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from faultdiagnose.data import list_datasets, load_care, load_dataset, load_events
from faultdiagnose.evaluation.anomaly import compute_auc
from faultdiagnose.features import (
    apply_standardizer,
    engineer_features,
    fit_standardizer,
    select_feature_columns,
)
from faultdiagnose.models import VAE
from faultdiagnose.training.vae_trainer import train_vae

OUT = load_care.OUT_DEFAULT
RESULTS = Path(__file__).resolve().parents[1] / "results"


def collect_training(farms, window, cap, use_fft=False):
    events = load_events(OUT)
    normal_ids: dict[str, list[str]] = {}
    for r in events.itertuples():
        if r.event_label == "normal":
            normal_ids.setdefault(r.farm, []).append(str(r.event_id))
    vecs, total = [], 0
    for farm in farms:
        for did in normal_ids.get(farm, []):
            df = load_dataset(OUT, farm, did)
            cols = select_feature_columns(df)
            feats = engineer_features(df, window=window, columns=cols, use_fft=use_fft).dropna()
            vecs.append(feats.values.astype(np.float32))
            total += len(feats)
            if total >= cap:
                break
        if total >= cap:
            break
    X = np.concatenate(vecs, axis=0)
    if len(X) > cap:
        idx = np.random.default_rng(0).choice(len(X), cap, replace=False)
        X = X[idx]
    return X


def evaluate(farms, window, model, mean, std, ev_map, use_fft=False):
    all_scores, all_labels, n = [], [], 0
    for farm in farms:
        for did in list_datasets(OUT, farm):
            df = load_dataset(OUT, farm, did)
            cols = select_feature_columns(df)
            feats = engineer_features(df, window=window, columns=cols, use_fft=use_fft)
            times = df.loc[feats.index, "time_stamp"]
            feats = feats.dropna()
            times = times.loc[feats.index]
            Z = apply_standardizer(feats, mean, std).astype(np.float32)
            with torch.no_grad():
                scores = model.reconstruction_error(torch.from_numpy(Z)).numpy()
            ev = ev_map.get((farm, did))
            if ev is not None and ev.event_label == "anomaly":
                start, end = pd.to_datetime(ev.event_start), pd.to_datetime(ev.event_end)
                labels = ((times >= start) & (times <= end)).astype(int).values
            else:
                labels = np.zeros(len(feats), dtype=int)
            all_scores.append(scores)
            all_labels.append(labels)
            n += len(scores)
    return np.concatenate(all_scores), np.concatenate(all_labels), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--farms", default="A")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--cap-train", type=int, default=60000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--use-fft", action="store_true")
    args = ap.parse_args()

    if not OUT.exists():
        raise SystemExit("processed data missing; run convert first")

    farms = [f.strip() for f in args.farms.split(",")]
    X = collect_training(farms, args.window, args.cap_train, args.use_fft)
    mean, std = fit_standardizer(pd.DataFrame(X))
    Xz = apply_standardizer(pd.DataFrame(X), mean, std).astype(np.float32)
    in_dim = Xz.shape[1]

    model = VAE(in_dim, latent=args.latent, hidden=args.hidden)
    n_feats = 6 if args.use_fft else 5
    print(f"Training VAE: {len(Xz)} vectors, in_dim={in_dim} ({n_feats}/col), farms={farms}")
    losses = train_vae(model, Xz, epochs=args.epochs, batch_size=args.batch, device=args.device)

    events = load_events(OUT)
    ev_map = {(r.farm, str(r.event_id)): r for r in events.itertuples()}
    scores, labels, n_test = evaluate(farms, args.window, model, mean, std, ev_map, args.use_fft)
    finite = np.isfinite(scores)
    if not finite.all():
        n_nan = (~finite).sum()
        print(f"WARNING: {n_nan} NaN scores dropped ({n_nan / len(scores) * 100:.1f}%)")
        scores = scores[finite]
        labels = labels[finite]
    auc = compute_auc(scores, labels)
    print(f"AUC-ROC = {auc:.4f}  (test vectors={n_test}, positives={labels.sum()})")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {
        "model": "VAE", "farms": farms, "window": args.window,
        "in_dim": in_dim, "latent": args.latent, "hidden": args.hidden,
        "epochs": args.epochs, "n_train": len(Xz), "n_test": int(n_test),
        "n_positives": int(labels.sum()), "auc_roc": auc,
        "final_loss": float(losses[-1]),
    }
    (RESULTS / "vae_baseline.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved -> {RESULTS / 'vae_baseline.json'}")


if __name__ == "__main__":
    main()


