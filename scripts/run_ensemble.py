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
from faultdiagnose.models import LSTMAE, VAE, SeqWindowsDataset, TransformerAE
from faultdiagnose.training.seq_trainer import train_seq_ae
from faultdiagnose.training.vae_trainer import train_vae

OUT = load_care.OUT_DEFAULT
RESULTS = Path(__file__).resolve().parents[1] / "results"
MODEL_NAMES = ("vae", "lstm", "transformer")


def collect_normal(farms, window, cap, use_fft=False):
    events = load_events(OUT)
    normal_ids: dict[str, list[str]] = {}
    for r in events.itertuples():
        if r.event_label == "normal":
            normal_ids.setdefault(r.farm, []).append(str(r.event_id))
    mats, total = [], 0
    for farm in farms:
        for did in normal_ids.get(farm, []):
            df = load_dataset(OUT, farm, did)
            if "train_test" in df.columns:
                df = df[df["train_test"] == "train"]
            cols = select_feature_columns(df)
            feats = engineer_features(df, window=window, columns=cols, use_fft=use_fft, fillna="ffill", clip_sigma=5.0).dropna()
            X = feats.values.astype(np.float32)
            if total + len(X) > cap:
                X = X[: cap - total]
            mats.append(X)
            total += len(X)
            if total >= cap:
                break
        if total >= cap:
            break
    X_all = np.concatenate(mats, axis=0)
    mean, std = fit_standardizer(X_all)
    zmats = [apply_standardizer(m, mean, std).astype(np.float32) for m in mats]
    vae_X = np.concatenate(zmats, axis=0)
    return vae_X, zmats, mean, std


def window_to_timestep(win_err, N, seq_len):
    scores = np.full(N, np.nan, dtype=np.float32)
    if len(win_err):
        scores[seq_len - 1 :] = win_err
    return pd.Series(scores).bfill().ffill().to_numpy()


def evaluate(farms, window, seq_len, models, mean, std, ev_map, device, use_fft=False, batch=2048):
    pm_val = {n: [] for n in MODEL_NAMES}
    pm_test = {n: [] for n in MODEL_NAMES}
    val_labels, test_labels, test_records = [], [], []
    for farm in farms:
        for did in list_datasets(OUT, farm):
            df = load_dataset(OUT, farm, did)
            cols = select_feature_columns(df)
            feats_all = engineer_features(df, window=window, columns=cols, use_fft=use_fft, fillna="ffill", clip_sigma=5.0)
            times_all = df.loc[feats_all.index, "time_stamp"]
            feats_all = feats_all.dropna()
            times_all = times_all.loc[feats_all.index]
            if "train_test" in df.columns:
                tt = df.loc[feats_all.index, "train_test"].values
                keep = tt == "prediction"
            else:
                keep = np.ones(len(feats_all), dtype=bool)
            feats = feats_all[keep]
            times = times_all[keep]
            if len(feats) < seq_len:
                continue
            Z = apply_standardizer(feats, mean, std).astype(np.float32)
            N = len(feats)
            with torch.no_grad():
                s_vae = models["vae"].reconstruction_error(
                    torch.from_numpy(Z).to(device)
                ).cpu().numpy()
                seq_ds = SeqWindowsDataset([Z], seq_len)
                seq_dl = DataLoader(seq_ds, batch_size=batch, shuffle=False)
                win_lstm = np.empty(len(seq_ds), dtype=np.float32)
                win_tf = np.empty(len(seq_ds), dtype=np.float32)
                pos = 0
                for xb in seq_dl:
                    xb = xb.to(device)
                    win_lstm[pos : pos + len(xb)] = (
                        models["lstm"].reconstruction_error(xb).cpu().numpy()
                    )
                    win_tf[pos : pos + len(xb)] = (
                        models["transformer"].reconstruction_error(xb).cpu().numpy()
                    )
                    pos += len(xb)
            s_lstm = window_to_timestep(win_lstm, N, seq_len)
            s_tf = window_to_timestep(win_tf, N, seq_len)
            ev = ev_map.get((farm, did))
            if ev is not None and ev.event_label == "anomaly":
                start, end = pd.to_datetime(ev.event_start), pd.to_datetime(ev.event_end)
                labels = ((times >= start) & (times <= end)).astype(int).values
            else:
                labels = np.zeros(N, dtype=int)
            k = N // 2
            pm_val["vae"].append(s_vae[:k])
            pm_test["vae"].append(s_vae)
            pm_val["lstm"].append(s_lstm[:k])
            pm_test["lstm"].append(s_lstm)
            pm_val["transformer"].append(s_tf[:k])
            pm_test["transformer"].append(s_tf)
            val_labels.append(labels[:k])
            test_labels.append(labels)
            if ev is not None and ev.event_label == "anomaly":
                ds_test = {n: pm_test[n][-1] for n in MODEL_NAMES}
                test_records.append(
                    (ds_test, times.reset_index(drop=True), ev.event_start)
                )
    return pm_val, pm_test, val_labels, test_labels, test_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--farms", default="A")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--cap-train", type=int, default=20000)
    ap.add_argument(
        "--tau", type=float, default=99.0,
        help="percentile p in [95,99] for adaptive threshold",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--use-fft", action="store_true")
    args = ap.parse_args()

    if not OUT.exists():
        raise SystemExit("processed data missing; run convert first")

    farms = [f.strip() for f in args.farms.split(",")]
    vae_X, zmats, mean, std = collect_normal(farms, args.window, args.cap_train, args.use_fft)
    in_dim = vae_X.shape[1]

    models = {}
    models["vae"] = VAE(in_dim, latent=args.latent, hidden=args.hidden)
    print(f"Training VAE: {len(vae_X)} vectors, in_dim={in_dim}")
    train_vae(models["vae"], vae_X, epochs=args.epochs, batch_size=args.batch, device=args.device)

    models["lstm"] = LSTMAE(in_dim, args.seq_len, latent=args.latent, hidden=args.hidden)
    n_win = sum(len(m) - args.seq_len + 1 for m in zmats)
    print(
        f"Training LSTM-AE: in_dim={in_dim}, seq_len={args.seq_len}, "
        f"train_windows={n_win}"
    )
    train_seq_ae(
        models["lstm"], zmats, args.seq_len,
        epochs=args.epochs, batch_size=args.batch, device=args.device,
    )

    models["transformer"] = TransformerAE(
        in_dim, args.seq_len, latent=args.latent, d_model=args.hidden
    )
    print(f"Training Transformer-AE: in_dim={in_dim}, seq_len={args.seq_len}")
    train_seq_ae(
        models["transformer"], zmats, args.seq_len,
        epochs=args.epochs, batch_size=args.batch, device=args.device,
    )

    events = load_events(OUT)
    ev_map = {(r.farm, str(r.event_id)): r for r in events.itertuples()}
    pm_val, pm_test, val_labels, test_labels, test_records = evaluate(
        farms, args.window, args.seq_len, models, mean, std, ev_map, args.device, args.use_fft
    )

    val_aucs, norm = {}, {}
    for name in MODEL_NAMES:
        v = np.concatenate(pm_val[name])
        vl = np.concatenate(val_labels)
        f = np.isfinite(v)
        val_aucs[name] = float(compute_auc(v[f], vl[f])) if f.any() else 0.5
        norm[name] = (float(np.min(v)), float(np.max(v)))
    weights = validation_weights(val_aucs)
    print(f"val AUC: {val_aucs}")
    print(f"weights: {weights}")

    ens_val = combine({n: np.concatenate(pm_val[n]) for n in MODEL_NAMES}, weights, norm)
    tau = adaptive_threshold(ens_val, args.tau)
    print(f"tau (p={args.tau}) = {tau:.4f}")

    ens_test = combine({n: np.concatenate(pm_test[n]) for n in MODEL_NAMES}, weights, norm)
    tl_all = np.concatenate(test_labels)
    f = np.isfinite(ens_test)
    auc = compute_auc(ens_test[f], tl_all[f])
    print(
        f"Ensemble AUC-ROC = {auc:.4f} (test vectors={len(ens_test)}, "
        f"positives={int(tl_all.sum())})"
    )

    lead_times, onset_lats = [], []
    for ds_test, t, es in test_records:
        ds_score = combine(ds_test, weights, norm)
        lead_times.append(lead_time_hours(ds_score, t, es, tau))
        onset_lats.append(onset_to_detection(ds_score, t, es, tau))
    ed = early_detection_report(lead_times)
    od = early_detection_report(onset_lats)
    print(f"Early detection (pre-fault lead): {ed}")
    print(f"Detection latency (after onset): {od}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {
        "model": "ensemble", "farms": farms, "window": args.window, "seq_len": args.seq_len,
        "in_dim": in_dim, "latent": args.latent, "hidden": args.hidden, "epochs": args.epochs,
        "tau_percentile": args.tau, "tau": tau, "weights": weights, "val_auc": val_aucs,
        "ensemble_auc_roc": float(auc),
        "n_test": int(len(ens_test)), "n_test_positives": int(tl_all.sum()),
        "early_detection_pre_fault_lead": ed,
        "detection_latency_after_onset": od,
    }
    path = RESULTS / "ensemble_baseline.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
