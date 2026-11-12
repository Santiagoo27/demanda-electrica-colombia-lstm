"""
evaluate_without_zeros.py
=========================
Re-evaluates a run of final_model.py excluding the test targets with zero demand.

final_model.py already excludes those targets itself. This script re-evaluates, without
retraining, runs made before that change (for example, the run behind the article).

Why
  The demand file has 8,784 rows; the last 24 (dated 2015-01-01) are 0 MW. They are not
  hours of 2014 and a national demand of 0 MW is not physical, so they are missing data.
  They lie at the end of the series and only appear as TEST TARGETS: they never enter
  training, the scaler fit, or the input window of any other target. Excluding them from
  the evaluation is therefore exactly equivalent to evaluating on clean data, with no retraining.

What it does (no LSTM retraining; about 1 minute)
  - LSTM: reads the saved predictions in <results>/runs/*.npz (or the legacy layout
    <results>/corridas/<legacy key>_semilla<n>.npz).
  - Baselines: recomputed with the same code and seed as final_model.py.
  - Computes the metrics on the test targets with demand > 0.

Usage:
  python src/evaluate_without_zeros.py --results <run folder> --output <folder>
  (defaults: --results results/final, --output results/without_zeros)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_model as fm  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LEGACY_FROM_KEY = {v: k for k, v in fm.LEGACY_KEYS.items()}


def run_file(results, key, seed):
    """Saved predictions of one run, in the current or the legacy layout (None if missing)."""
    for path in (os.path.join(results, "runs", f"{key}_seed{seed}.npz"),
                 os.path.join(results, "corridas", f"{LEGACY_FROM_KEY[key]}_semilla{seed}.npz")):
        if os.path.exists(path):
            return path
    return None


def main():
    ap = argparse.ArgumentParser(description="Re-evaluate a run without the zero-demand test targets.")
    ap.add_argument("--data-dir", "--ruta", dest="data_dir", default=fm.DEFAULT_DATA_DIR)
    ap.add_argument("--results", "--resultados", dest="results",
                    default=os.path.join(fm._REPO, "results", "final"))
    ap.add_argument("--output", "--salida", dest="output",
                    default=os.path.join(fm._REPO, "results", "without_zeros"))
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    data = fm.load_data(args.data_dir, fm.FILE_SOLAR, fm.FILE_WIND, fm.FILE_DEMAND)
    n_seq, split, train_t, test_t = fm.split_info(len(data))
    d = data["demand"].values
    valid = fm.valid_test(data)
    print(f"Test targets: {len(test_t)} | zero demand: {(~valid).sum()} | valid: {valid.sum()}")

    # Demand scaler, exactly as in final_model.prepare (training rows only)
    sy = MinMaxScaler().fit(data[["demand"]].iloc[:split + fm.SEQ_LEN].values)
    norm = lambda v: sy.transform(np.asarray(v).reshape(-1, 1)).ravel()

    # --- LSTM, from the saved predictions
    rows = []
    for key, (label, _) in fm.CONFIGS.items():
        seed = 0
        while (f := run_file(args.results, key, seed)) is not None:
            z = np.load(f)
            real, pred = z["real"][valid], z["pred"][valid]
            assert np.allclose(real, d[test_t][valid], atol=1e-6), "predictions do not match the test set"
            m = fm.metrics_mw(real, pred)
            e = norm(real) - norm(pred)
            m.update({"MSE_norm": float(np.mean(e ** 2)), "RMSE_norm": float(np.sqrt(np.mean(e ** 2))),
                      "MAE_norm": float(np.mean(np.abs(e))),
                      "MAE_pct_mean": float(100 * m["MAE_MW"] / np.mean(real))})
            rows.append({"key": key, "config": label, "seed": seed, **m})
            seed += 1
    if not rows:
        sys.exit(f"No saved runs found in {args.results}")
    per_seed = pd.DataFrame(rows)
    per_seed.round(4).to_csv(os.path.join(args.output, "metrics_per_seed.csv"), index=False, encoding="utf-8-sig")

    cols = ["MAE_MW", "RMSE_MW", "R2", "MSE_norm", "RMSE_norm", "MAE_norm", "MAE_pct_mean"]
    order = [fm.CONFIGS[k][0] for k in fm.CONFIGS if fm.CONFIGS[k][0] in set(per_seed["config"])]
    g = per_seed.groupby("config")[cols].agg(["mean", "std"]).reindex(order)
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g["n_seeds"] = per_seed.groupby("config").size().reindex(order)
    g.round(4).to_csv(os.path.join(args.output, "lstm_summary.csv"), encoding="utf-8-sig")

    # --- Baselines (same code and seed as final_model.run_baselines)
    base = fm.run_baselines(data, args.output)

    comp = [{"model": "LSTM: " + lab, "MAE_MW": g.loc[lab, "MAE_MW_mean"], "MAE_std": g.loc[lab, "MAE_MW_std"],
             "RMSE_MW": g.loc[lab, "RMSE_MW_mean"], "RMSE_std": g.loc[lab, "RMSE_MW_std"],
             "R2": g.loc[lab, "R2_mean"], "R2_std": g.loc[lab, "R2_std"]} for lab in order]
    comp += [{"model": n, "MAE_MW": r["MAE_MW"], "MAE_std": np.nan, "RMSE_MW": r["RMSE_MW"],
              "RMSE_std": np.nan, "R2": r["R2"], "R2_std": np.nan} for n, r in base.iterrows()]
    comp = pd.DataFrame(comp)
    comp.round(3).to_csv(os.path.join(args.output, "comparison_table.csv"), index=False, encoding="utf-8-sig")

    tt = test_t[valid]
    full = per_seed[per_seed["key"] == "full"]
    lines = [
        "EVALUATION WITHOUT THE 24 ZERO-DEMAND TEST TARGETS (evaluate_without_zeros.py)",
        "=" * 70,
        f"Source run: {args.results}",
        f"Valid test targets: {valid.sum()} of {len(test_t)} ({data.index[tt[0]]} -> {data.index[tt[-1]]})",
        "", "COMPARISON TABLE", comp.round(3).to_string(index=False),
        "", "LSTM, mean and standard deviation", g.round(4).to_string(),
    ]
    if len(full):
        lines += ["", f"Full model per seed: R2 {full['R2'].min():.3f}-{full['R2'].max():.3f}, "
                      f"MAE {full['MAE_MW'].min():.1f}-{full['MAE_MW'].max():.1f} MW"]
    lines += ["", "METRICS PER SEED", per_seed.round(4).to_string(index=False), ""]
    txt = "\n".join(lines)
    open(os.path.join(args.output, "summary.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
