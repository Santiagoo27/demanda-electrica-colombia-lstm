"""
final_model.py
==============
Full experiment of the article: LSTM for hourly electricity demand forecasting in Colombia,
with input ablation and baselines, in a single reproducible run.

  - The MinMaxScalers are fitted ONLY on the rows used for training (no data leakage).
  - Fixed seeds, several per configuration (mean and standard deviation are reported).
  - Ablation: full model, without solar/wind, without demand lags, without cyclic encodings.
  - Baselines (persistence, seasonal naive, gradient boosting) with the same information as the LSTM.
  - Metrics in MW and on the normalised scale are computed from the same run.
  - Test targets with zero demand (the last 24 rows of the file, dated 2015-01-01) are missing
    data and are excluded from every metric. The full predictions are still saved in
    runs/*.npz, together with the "valid" mask.
  - Each run is saved as soon as it finishes: if interrupted, re-running resumes where it stopped.

Architecture: LSTM 128 + Dropout 0.2 + LSTM 64 + BatchNormalization + Dense 32 + Dense 1,
Adam 0.001, 80 epochs, batch 32, validation_split 0.2, 72-hour window.

Usage:
  python src/final_model.py          (defaults: --data-dir data/raw, --output results/new_run)
  python src/final_model.py --seeds 1 --epochs 3 --output <absolute path>   (quick test)
  python src/final_model.py --baselines-only                                (baselines only, ~1 minute)
  python src/final_model.py --n-rows 8760                                   (truncate to 8,760 rows)

Results: folder <data-dir>/<output> (an absolute --output is used as is). Summary in summary.txt.
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
# Default paths, relative to the repository
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(_REPO, "data", "raw")
# New runs go to their own folder so that earlier results are never overwritten
DEFAULT_OUTPUT = os.path.join(_REPO, "results", "new_run")
FILE_SOLAR = "4. Solar_generation_ENSO-NINO.txt"
FILE_WIND = "6. Wind_generation_ENSO-NINO.txt"
FILE_DEMAND = "Electricity demand Colombia 2014 [MW].txt"

SEQ_LEN = 72
BATCH = 32
VAL_SPLIT = 0.2
LR = 0.001

CYCLIC = ["hour_sin", "hour_cos", "day_sin", "day_cos", "year_sin", "year_cos"]
LAGS = ["demand_lag_1", "demand_lag_24", "demand_lag_168"]
ALL = ["solar", "wind"] + CYCLIC + ["is_weekend", "is_working_hour"] + LAGS

# key -> (label, input variables)
CONFIGS = {
    "full": ("Full model (13 variables)", ALL),
    "no_solar_wind": ("Without solar and wind (11)", [f for f in ALL if f not in ("solar", "wind")]),
    "no_lags": ("Without demand lags (10)", [f for f in ALL if f not in LAGS]),
    "no_cyclic": ("Without cyclic encodings (7)", [f for f in ALL if f not in CYCLIC]),
}
# Configuration keys used by earlier (Spanish) versions of this script
LEGACY_KEYS = {"completo": "full", "sin_solar_wind": "no_solar_wind",
               "sin_lags": "no_lags", "sin_ciclicas": "no_cyclic"}


# ----------------------------------------------------------------------------
# Data and features
# ----------------------------------------------------------------------------
def load_data(data_dir, f_solar, f_wind, f_demand, n_rows=None):
    solar = pd.read_csv(os.path.join(data_dir, f_solar), header=None, names=["solar"])
    wind = pd.read_csv(os.path.join(data_dir, f_wind), header=None, names=["wind"])
    demand = pd.read_csv(os.path.join(data_dir, f_demand), header=None, names=["demand"])
    print(f"Files: solar={f_solar} | wind={f_wind} | demand={f_demand}")
    print(f"Rows read: solar={len(solar)}, wind={len(wind)}, demand={len(demand)}")
    n = min(len(solar), len(wind), len(demand))
    if n_rows:
        n = min(n, n_rows)
    if n != 8760:
        print(f"NOTE: using {n} rows, while 2014 has 8,760 hours. The extra rows are dated 2015.")
    data = pd.concat([solar[:n], wind[:n], demand[:n]], axis=1)
    data.index = pd.date_range(start="2014-01-01", periods=len(data), freq="60min")
    d = data["demand"].values
    print(f"Demand: min={d.min():.0f} MW, max={d.max():.0f} MW, mean={d.mean():.0f} MW, zeros={int((d == 0).sum())}")

    data["hour"] = data.index.hour
    data["day_of_week"] = data.index.dayofweek
    data["day_of_year"] = data.index.dayofyear
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)
    data["year_sin"] = np.sin(2 * np.pi * data["day_of_year"] / 365)
    data["year_cos"] = np.cos(2 * np.pi * data["day_of_year"] / 365)
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["is_working_hour"] = ((data["hour"] >= 8) & (data["hour"] <= 18)).astype(int)
    data["demand_lag_1"] = data["demand"].shift(1)
    data["demand_lag_24"] = data["demand"].shift(24)
    data["demand_lag_168"] = data["demand"].shift(168)
    data.dropna(inplace=True)
    return data


def split_info(n_rows):
    """Sequences of SEQ_LEN rows and chronological 80/20 split. t = row of the target."""
    n_seq = n_rows - SEQ_LEN
    split = int(0.8 * n_seq)
    train_t = np.arange(SEQ_LEN, split + SEQ_LEN)
    test_t = np.arange(split + SEQ_LEN, n_rows)
    return n_seq, split, train_t, test_t


def valid_test(data):
    """Mask of the valid test targets (demand > 0; zeros are missing data)."""
    _, _, _, test_t = split_info(len(data))
    return data["demand"].values[test_t] > 0


def metrics_mw(y, p):
    return {
        "MAE_MW": float(mean_absolute_error(y, p)),
        "RMSE_MW": float(np.sqrt(np.mean((y - p) ** 2))),
        "R2": float(r2_score(y, p)),
    }


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------
def run_baselines(data, out):
    n_rows = len(data)
    _, _, train_t, test_t = split_info(n_rows)
    test_t = test_t[valid_test(data)]
    d = data["demand"].values
    y_test = d[test_t]
    rows = {
        "Persistence (last known demand, t-2)": metrics_mw(y_test, d[test_t - 2]),
        "Seasonal naive 24 h (t-24)": metrics_mw(y_test, d[test_t - 24]),
        "Seasonal naive 168 h (t-168)": metrics_mw(y_test, d[test_t - 168]),
    }
    for name, cols in [
        ("Gradient boosting, 13 variables", ALL),
        ("Gradient boosting, without solar and wind (11)", CONFIGS["no_solar_wind"][1]),
    ]:
        X = data[cols].values
        gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0)
        gb.fit(X[train_t - 1], d[train_t])
        rows[name] = metrics_mw(y_test, gb.predict(X[test_t - 1]))
    table = pd.DataFrame(rows).T
    table.index.name = "model"
    table.round(3).to_csv(os.path.join(out, "baselines.csv"), encoding="utf-8-sig")
    return table


# ----------------------------------------------------------------------------
# LSTM
# ----------------------------------------------------------------------------
def prepare(data, cols):
    n_rows = len(data)
    n_seq, split, _, _ = split_info(n_rows)
    fit_end = split + SEQ_LEN  # rows seen during training (inputs and targets)
    sx, sy = MinMaxScaler(), MinMaxScaler()
    sx.fit(data[cols].iloc[:fit_end])                  # training rows ONLY
    sy.fit(data[["demand"]].iloc[:fit_end])
    xs = sx.transform(data[cols])
    ys = sy.transform(data[["demand"]]).ravel()
    X = np.stack([xs[i:i + SEQ_LEN] for i in range(n_seq)])
    y = ys[SEQ_LEN:SEQ_LEN + n_seq]
    return X, y, sy, split


def train_one(data, key, seed, epochs, out):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization

    label, cols = CONFIGS[key]
    X, y, sy, split = prepare(data, cols)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]

    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    model = Sequential([
        Input(shape=(SEQ_LEN, len(cols))),
        LSTM(128, return_sequences=True, activation="tanh"),
        Dropout(0.2),
        LSTM(64, activation="tanh"),
        BatchNormalization(),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss="mse")
    t0 = time.time()
    hist = model.fit(Xtr, ytr, epochs=epochs, batch_size=BATCH, validation_split=VAL_SPLIT, verbose=0)
    pred_n = model.predict(Xte, verbose=0).ravel()

    real = sy.inverse_transform(yte.reshape(-1, 1)).ravel()
    pred = sy.inverse_transform(pred_n.reshape(-1, 1)).ravel()
    ok = valid_test(data)
    m = metrics_mw(real[ok], pred[ok])
    e = yte[ok] - pred_n[ok]
    m.update({
        "MSE_norm": float(np.mean(e ** 2)),
        "RMSE_norm": float(np.sqrt(np.mean(e ** 2))),
        "MAE_norm": float(np.mean(np.abs(e))),
        "MAE_pct_mean": float(100 * m["MAE_MW"] / np.mean(real[ok])),
        "seconds": round(time.time() - t0, 1),
    })
    np.savez(os.path.join(out, "runs", f"{key}_seed{seed}.npz"),
             real=real, pred=pred, valid=ok,
             loss=np.array(hist.history["loss"]), val_loss=np.array(hist.history["val_loss"]))
    return label, m


# ----------------------------------------------------------------------------
# Summary, tables and figures
# ----------------------------------------------------------------------------
def postprocess(data, out, baselines, args):
    csv = os.path.join(out, "metrics_per_seed.csv")
    if not os.path.exists(csv):
        return None
    df = pd.read_csv(csv)
    order = [CONFIGS[k][0] for k in CONFIGS if CONFIGS[k][0] in set(df["config"])]
    cols = ["MAE_MW", "RMSE_MW", "R2", "MSE_norm", "RMSE_norm", "MAE_norm", "MAE_pct_mean"]
    g = df.groupby("config")[cols].agg(["mean", "std"]).reindex(order)
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g["n_seeds"] = df.groupby("config").size().reindex(order)
    g.round(4).to_csv(os.path.join(out, "lstm_summary.csv"), encoding="utf-8-sig")

    comp = []
    for lab in order:
        r = g.loc[lab]
        comp.append({"model": "LSTM: " + lab,
                     "MAE_MW": r["MAE_MW_mean"], "MAE_std": r["MAE_MW_std"],
                     "RMSE_MW": r["RMSE_MW_mean"], "RMSE_std": r["RMSE_MW_std"],
                     "R2": r["R2_mean"], "R2_std": r["R2_std"]})
    for name, r in baselines.iterrows():
        comp.append({"model": name, "MAE_MW": r["MAE_MW"], "MAE_std": np.nan,
                     "RMSE_MW": r["RMSE_MW"], "RMSE_std": np.nan, "R2": r["R2"], "R2_std": np.nan})
    comp = pd.DataFrame(comp)
    comp.round(3).to_csv(os.path.join(out, "comparison_table.csv"), index=False, encoding="utf-8-sig")

    # Figure: loss curves and prediction (full model, seed 0)
    f0 = os.path.join(out, "runs", "full_seed0.npz")
    if os.path.exists(f0):
        z = np.load(f0)
        plt.figure(figsize=(10, 4))
        plt.plot(z["loss"], label="Training loss", color="steelblue", linewidth=1.5)
        plt.plot(z["val_loss"], label="Validation loss", color="tomato", linewidth=1.5, linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("MSE (normalized scale)")
        plt.title("Convergence curves of the LSTM model")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out, "loss_curves.png"), dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(12, 5))
        plt.plot(z["real"][:500], label="Actual demand")
        plt.plot(z["pred"][:500], label="LSTM prediction")
        plt.legend()
        plt.title("Electric demand prediction (first 500 test hours)")
        plt.xlabel("Hours")
        plt.ylabel("Demand (MW)")
        plt.tight_layout()
        plt.savefig(os.path.join(out, "demand_prediction.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # Figure: ablation
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, ttl in zip(axes, ["R2", "RMSE_MW"], ["R² (mean ± SD)", "RMSE in MW (mean ± SD)"]):
        ax.bar(range(len(order)), g[f"{col}_mean"].values, yerr=g[f"{col}_std"].fillna(0).values, capsize=4,
               color="steelblue")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([o.split(" (")[0] for o in order], rotation=20, ha="right")
        ax.set_title(ttl)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "ablation.png"), dpi=200, bbox_inches="tight")
    plt.close()
    return g, comp, df


def write_summary(out, data, args, baselines, post, versions):
    n_rows = len(data)
    _, _, train_t, test_t = split_info(n_rows)
    lines = []
    lines.append("SUMMARY (final_model.py)")
    lines.append("=" * 60)
    lines.append(f"Files: solar={args.solar} | wind={args.wind} | demand={args.demand}")
    lines.append(f"Rows after dropna: {n_rows} | training: {len(train_t)} | test: {len(test_t)}")
    lines.append(f"Test period: {data.index[test_t[0]]} -> {data.index[test_t[-1]]}")
    ok = valid_test(data)
    lines.append(f"Test targets evaluated (demand > 0): {ok.sum()} of {len(test_t)} "
                 f"(up to {data.index[test_t[ok][-1]]})")
    lines.append(f"Seeds: {args.seeds} | epochs: {args.epochs} | batch: {BATCH} | window: {SEQ_LEN} h")
    lines.append(f"Versions: {json.dumps(versions, ensure_ascii=False)}")
    lines.append("")
    lines.append("BASELINES (MW)")
    lines.append(baselines.round(3).to_string())
    if post is not None:
        g, comp, df = post
        lines.append("")
        lines.append("LSTM, mean and standard deviation per configuration")
        lines.append(g.round(4).to_string())
        lines.append("")
        lines.append("COMPARISON TABLE")
        lines.append(comp.round(3).to_string(index=False))
        lines.append("")
        lines.append("METRICS PER SEED")
        lines.append(df.round(4).to_string(index=False))
    txt = "\n".join(lines)
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    return txt


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="LSTM ablation experiment and baselines.")
    # Spanish option names from earlier versions are kept as aliases
    ap.add_argument("--data-dir", "--ruta", dest="data_dir", default=DEFAULT_DATA_DIR,
                    help="folder with the .txt data files")
    ap.add_argument("--solar", default=FILE_SOLAR)
    ap.add_argument("--wind", default=FILE_WIND)
    ap.add_argument("--demand", default=FILE_DEMAND)
    ap.add_argument("--seeds", type=int, default=5, help="seeds per configuration")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--n-rows", "--n-filas", dest="n_rows", type=int, default=None,
                    help="truncate the files to N rows (e.g. 8760)")
    ap.add_argument("--configs", default=",".join(CONFIGS), help="comma-separated configuration keys")
    ap.add_argument("--baselines-only", "--solo-baselines", dest="baselines_only", action="store_true")
    ap.add_argument("--output", "--salida", dest="output", default=DEFAULT_OUTPUT,
                    help="output folder (a relative path is taken relative to --data-dir)")
    args = ap.parse_args()

    out = os.path.join(args.data_dir, args.output)
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)

    data = load_data(args.data_dir, args.solar, args.wind, args.demand, args.n_rows)
    _, _, train_t, test_t = split_info(len(data))
    print(f"Training: {len(train_t)} targets | Test: {len(test_t)} targets")
    print(f"Test period: {data.index[test_t[0]]} -> {data.index[test_t[-1]]}")

    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    baselines = run_baselines(data, out)
    print("\n=== Baselines (MW) ===")
    print(baselines.round(3).to_string())

    csv = os.path.join(out, "metrics_per_seed.csv")
    if not args.baselines_only:
        try:
            import tensorflow as tf
            versions["tensorflow"] = tf.__version__
        except ImportError:
            sys.exit("TensorFlow not found. Use --baselines-only or install TensorFlow.")
        requested = [LEGACY_KEYS.get(k.strip(), k.strip()) for k in args.configs.split(",")]
        keys = [k for k in requested if k in CONFIGS]
        done = set()
        if os.path.exists(csv):
            prev = pd.read_csv(csv)
            done = set(zip(prev["key"], prev["seed"]))
        total = len(keys) * args.seeds
        k = 0
        for seed in range(args.seeds):
            for key in keys:
                k += 1
                if (key, seed) in done:
                    print(f"[{k}/{total}] {key} seed {seed}: already done, skipped")
                    continue
                label, m = train_one(data, key, seed, args.epochs, out)
                row = {"key": key, "config": label, "seed": seed, **m}
                pd.DataFrame([row]).to_csv(csv, mode="a", header=not os.path.exists(csv),
                                           index=False, encoding="utf-8-sig")
                print(f"[{k}/{total}] {label} | seed {seed}: MAE={m['MAE_MW']:.1f} MW  "
                      f"RMSE={m['RMSE_MW']:.1f} MW  R2={m['R2']:.3f}  ({m['seconds']} s)")

    post = postprocess(data, out, baselines, args)
    json.dump({"args": vars(args), "versions": versions, "rows": len(data)},
              open(os.path.join(out, "config.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    txt = write_summary(out, data, args, baselines, post, versions)
    print("\n" + txt)
    print(f"\nDone. Files in: {out}")


if __name__ == "__main__":
    main()
