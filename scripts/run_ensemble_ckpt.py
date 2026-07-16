"""Ensemble evaluation over saved GPU-tuned checkpoints (paper Algorithm 1).

Loads model checkpoints produced by run_gpu_tune.py, evaluates each on the test
set (per-dataset scores retained for early-detection), combines via validation-AUC
weighting, applies adaptive percentile threshold, and reports ensemble AUC + lead time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from faultdiagnose.data import list_datasets, load_care, load_dataset, load_events
from faultdiagnose.evaluation.anomaly import compute_auc
from faultdiagnose.evaluation.ensemble import (
    adaptive_threshold,
    combine,
    early_detection_report,
    lead_time_hours,
    onset_to_detection,
    validation_weights,
)
from faultdiagnose.features import (
    apply_standardizer,
    engineer_features,
    fit_standardizer,
    select_feature_columns,
)
from faultdiagnose.models import LSTMAE, SeqWindowsDataset, TransformerAE, VAE

OUT = load_care.OUT_DEFAULT
RESULTS = Path(__file__).resolve().parents[1] / "results"
CKPT_DIR = RESULTS / "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint(name: str, in_dim: int, seq_len: int):
    p = CKPT_DIR / f"{name}.pt"
    if not p.exists():
        return None, None
    if name.startswith("vae_"):
        parts = name.split("_")
        hidden, latent = int(parts[1][1:]), int(parts[2][1:])
        return VAE(in_dim, latent=latent, hidden=hidden), "vae"
    if name.startswith("lstm_"):
        parts = name.split("_")
        hidden, latent = int(parts[1][1:]), int(parts[2][1:])
        return LSTMAE(in_dim, seq_len, latent=latent, hidden=hidden, num_layers=2), "lstm"
    if name.startswith("tf_"):
        parts = name.split("_")
        hidden, latent = int(parts[1][1:]), int(parts[2][1:])
        nhead = 4 if hidden <= 128 else 8
        return TransformerAE(in_dim, seq_len, latent=latent, d_model=hidden, nhead=nhead, num_layers=2), "transformer"
    return None, None


def evaluate(model, kind, farm, did, window, seq_len, mean, std, ev_map, use_fft=False):
    df = load_dataset(OUT, farm, did)
    cols = select_feature_columns(df)
    feats_all = engineer_features(df, window=window, columns=cols, use_fft=use_fft, fillna="ffill", clip_sigma=5.0)
    times_all = df.loc[feats_all.index, "time_stamp"]
    feats_all = feats_all.dropna()
    times_all = times_all.loc[feats_all.index]
    if len(feats_all) < seq_len:
        return None
    Z = apply_standardizer(feats_all, mean, std).astype(np.float32)
    N = len(feats_all)
    with torch.no_grad():
        if kind == "vae":
            s = model.reconstruction_error(torch.from_numpy(Z).to(DEVICE)).cpu().numpy()
        else:
            ds = SeqWindowsDataset([Z], seq_len)
            dl = DataLoader(ds, batch_size=4096, shuffle=False)
            win = np.empty(len(ds), dtype=np.float32)
            pos = 0
            for xb in dl:
                win[pos : pos + len(xb)] = model.reconstruction_error(xb.to(DEVICE)).cpu().numpy()
                pos += len(xb)
            s = np.full(N, np.nan, dtype=np.float32)
            s[seq_len - 1 :] = win
            s = pd.Series(s).bfill().ffill().to_numpy()
    ev = ev_map.get((farm, did))
    if ev is not None and ev.event_label == "anomaly":
        start, end = pd.to_datetime(ev.event_start), pd.to_datetime(ev.event_end)
        labels = ((times_all >= start) & (times_all <= end)).astype(int).values
    else:
        labels = np.zeros(N, dtype=int)
    return s, labels, times_all.reset_index(drop=True), ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--farms", default="A")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--tau", type=float, default=99.0)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--use-fft", action="store_true")
    args = ap.parse_args()

    if not OUT.exists():
        raise SystemExit("processed data missing")
    farms = [f.strip() for f in args.farms.split(",")]
    events = load_events(OUT)
    ev_map = {(r.farm, str(r.event_id)): r for r in events.itertuples()}

    normal_ids = [str(r.event_id) for r in events.itertuples() if r.event_label == "normal"]
    feats_all = []
    for did in normal_ids:
        df = load_dataset(OUT, "A", did)
        if "train_test" in df.columns:
            df = df[df["train_test"] == "train"]
        cols = select_feature_columns(df)
        feats_all.append(engineer_features(df, window=args.window, columns=cols, use_fft=args.use_fft, fillna="ffill", clip_sigma=5.0).dropna().values)
    X_all = np.concatenate(feats_all, axis=0)
    mean, std = fit_standardizer(X_all)
    in_dim = X_all.shape[1]

    ckpts = sorted(CKPT_DIR.glob("*.pt"))
    per_model = {}   # name -> {scores, labels, records}
    for p in ckpts:
        name = p.stem
        model, kind = load_checkpoint(name, in_dim, args.seq_len)
        if model is None:
            continue
        model.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
        scores_list, labels_list, records = [], [], []
        for farm in farms:
            for did in list_datasets(OUT, farm):
                r = evaluate(model, kind, farm, did, args.window, args.seq_len,
                             mean, std, ev_map, args.use_fft)
                if r is None:
                    continue
                s, labels, times, ev = r
                scores_list.append(s)
                labels_list.append(labels)
                if ev is not None and ev.event_label == "anomaly":
                    records.append((s, times, ev.event_start))
        per_model[name] = {
            "scores": np.concatenate(scores_list),
            "labels": np.concatenate(labels_list),
            "records": records,
        }
        f = np.isfinite(per_model[name]["scores"])
        auc = compute_auc(per_model[name]["scores"][f], per_model[name]["labels"][f])
        print(f"  {name}: AUC={auc:.4f}")

    if not per_model:
        raise SystemExit("no checkpoints found")

    # rank by AUC
    ranked = sorted(per_model.keys(),
                    key=lambda n: compute_auc(per_model[n]["scores"][np.isfinite(per_model[n]["scores"])],
                                              per_model[n]["labels"][np.isfinite(per_model[n]["labels"])]),
                    reverse=True)
    top = ranked[: args.topk]
    print(f"\nTop-{args.topk}: {top}")

    sub = {n: per_model[n]["scores"] for n in top}
    norm = {n: (float(np.min(per_model[n]["scores"])), float(np.max(per_model[n]["scores"]))) for n in top}
    val_aucs = {n: compute_auc(per_model[n]["scores"][np.isfinite(per_model[n]["scores"])],
                               per_model[n]["labels"][np.isfinite(per_model[n]["labels"])]) for n in top}
    weights = validation_weights(val_aucs)

    ens = combine(sub, weights, norm)
    tl = np.concatenate([per_model[n]["labels"] for n in top])
    f = np.isfinite(ens)
    auc_ens = compute_auc(ens[f], tl[f])
    tau = adaptive_threshold(ens, args.tau)

    print(f"\nEnsemble AUC-ROC = {auc_ens:.4f}")
    print(f"tau (p={args.tau}) = {tau:.4f}")
    print(f"weights = {weights}")

    lead_times, onset_lats = [], []
    for n in top:
        for s, times, es in per_model[n]["records"]:
            lead_times.append(lead_time_hours(s, times, es, tau))
            onset_lats.append(onset_to_detection(s, times, es, tau))
    ed = early_detection_report(lead_times)
    od = early_detection_report(onset_lats)
    print(f"Early detection (pre-fault lead): {ed}")
    print(f"Detection latency (after onset): {od}")

    out = {
        "ensemble_auc_roc": float(auc_ens), "tau": tau, "tau_percentile": args.tau,
        "top_models": top, "weights": weights, "per_model_auc": val_aucs,
        "early_detection_pre_fault_lead": ed, "detection_latency_after_onset": od,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    path = RESULTS / f"ensemble_eval_{stamp}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
