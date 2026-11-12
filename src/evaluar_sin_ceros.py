"""
evaluar_sin_ceros.py
====================
Reevalúa una corrida de modelo_final.py excluyendo los objetivos de test con demanda = 0.

modelo_final.py ya excluye esos objetivos por sí mismo. Este script sirve para reevaluar, sin
reentrenar, corridas hechas antes de ese cambio (por ejemplo, la que respalda el artículo).

Por qué
  El archivo de demanda tiene 8,784 filas; las últimas 24 (fechadas 2015-01-01) valen 0 MW.
  No son horas de 2014 y una demanda nacional de 0 MW no es física, así que son datos faltantes.
  Caen al final de la serie, solo como OBJETIVOS de test: nunca entran al entrenamiento, al
  ajuste de los escaladores ni a la ventana de entrada de ningún otro objetivo. Por eso excluirlos
  de la evaluación equivale exactamente a evaluar con datos limpios, sin reentrenar.

Qué hace (no reentrena el LSTM; tarda alrededor de 1 minuto)
  - LSTM: lee las predicciones guardadas en <resultados>/corridas/*.npz.
  - Baselines: los recalcula con el mismo código y la misma semilla que modelo_final.py.
  - Calcula las métricas sobre los objetivos de test con demanda > 0.

Uso:
  python src/evaluar_sin_ceros.py                               (lee results/final)
  python src/evaluar_sin_ceros.py --resultados <carpeta> --salida <carpeta>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modelo_final as mf  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruta", default=mf.RUTA_DEFECTO)
    ap.add_argument("--resultados", default=os.path.join(mf._REPO, "results", "final"))
    ap.add_argument("--salida", default=os.path.join(mf._REPO, "results", "sin_ceros"))
    args = ap.parse_args()
    os.makedirs(args.salida, exist_ok=True)

    data = mf.load_data(args.ruta, mf.FILE_SOLAR, mf.FILE_WIND, mf.FILE_DEMAND)
    n_seq, split, train_t, test_t = mf.split_info(len(data))
    d = data["demand"].values
    valid = d[test_t] > 0
    print(f"Objetivos de test: {len(test_t)} | con demanda 0: {(~valid).sum()} | válidos: {valid.sum()}")

    # Escalador de la demanda, igual que en modelo_final.prepare (solo entrenamiento)
    sy = MinMaxScaler().fit(data[["demand"]].iloc[:split + mf.SEQ_LEN].values)
    norm = lambda v: sy.transform(np.asarray(v).reshape(-1, 1)).ravel()

    # --- LSTM desde las predicciones guardadas
    rows = []
    for key, (label, _) in mf.CONFIGS.items():
        seed = 0
        while os.path.exists(f := os.path.join(args.resultados, "corridas", f"{key}_semilla{seed}.npz")):
            z = np.load(f)
            real, pred = z["real"][valid], z["pred"][valid]
            assert np.allclose(real, d[test_t][valid], atol=1e-6), "las predicciones no corresponden al test"
            m = mf.metrics_mw(real, pred)
            e = norm(real) - norm(pred)
            m.update({"MSE_norm": float(np.mean(e ** 2)), "RMSE_norm": float(np.sqrt(np.mean(e ** 2))),
                      "MAE_norm": float(np.mean(np.abs(e))),
                      "MAE_pct_media": float(100 * m["MAE_MW"] / np.mean(real))})
            rows.append({"clave": key, "config": label, "seed": seed, **m})
            seed += 1
    per_seed = pd.DataFrame(rows)
    per_seed.round(4).to_csv(os.path.join(args.salida, "metricas_por_semilla.csv"), index=False, encoding="utf-8-sig")

    cols = ["MAE_MW", "RMSE_MW", "R2", "MSE_norm", "RMSE_norm", "MAE_norm", "MAE_pct_media"]
    order = [mf.CONFIGS[k][0] for k in mf.CONFIGS if mf.CONFIGS[k][0] in set(per_seed["config"])]
    g = per_seed.groupby("config")[cols].agg(["mean", "std"]).reindex(order)
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g["n_semillas"] = per_seed.groupby("config").size().reindex(order)
    g.round(4).to_csv(os.path.join(args.salida, "resumen_lstm.csv"), encoding="utf-8-sig")

    # --- Baselines (mismo código que modelo_final.run_baselines, evaluados en los objetivos válidos)
    tt = test_t[valid]
    y = d[tt]
    base = {
        "Persistencia (última demanda conocida, t-2)": mf.metrics_mw(y, d[tt - 2]),
        "Seasonal naive 24 h (t-24)": mf.metrics_mw(y, d[tt - 24]),
        "Seasonal naive 168 h (t-168)": mf.metrics_mw(y, d[tt - 168]),
    }
    for name, fcols in [("Gradient boosting, 13 variables", mf.ALL),
                        ("Gradient boosting, sin solar ni wind (11)", mf.CONFIGS["sin_solar_wind"][1])]:
        X = data[fcols].values
        gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0)
        gb.fit(X[train_t - 1], d[train_t])
        base[name] = mf.metrics_mw(y, gb.predict(X[tt - 1]))
    base = pd.DataFrame(base).T
    base.index.name = "modelo"
    base.round(3).to_csv(os.path.join(args.salida, "baselines.csv"), encoding="utf-8-sig")

    comp = [{"modelo": "LSTM: " + lab, "MAE_MW": g.loc[lab, "MAE_MW_mean"], "MAE_std": g.loc[lab, "MAE_MW_std"],
             "RMSE_MW": g.loc[lab, "RMSE_MW_mean"], "RMSE_std": g.loc[lab, "RMSE_MW_std"],
             "R2": g.loc[lab, "R2_mean"], "R2_std": g.loc[lab, "R2_std"]} for lab in order]
    comp += [{"modelo": n, "MAE_MW": r["MAE_MW"], "MAE_std": np.nan, "RMSE_MW": r["RMSE_MW"],
              "RMSE_std": np.nan, "R2": r["R2"], "R2_std": np.nan} for n, r in base.iterrows()]
    comp = pd.DataFrame(comp)
    comp.round(3).to_csv(os.path.join(args.salida, "tabla_comparacion.csv"), index=False, encoding="utf-8-sig")

    full = per_seed[per_seed["clave"] == "completo"]
    txt = "\n".join([
        "EVALUACIÓN SIN LOS 24 OBJETIVOS CON DEMANDA 0 (evaluar_sin_ceros.py)",
        "=" * 70,
        f"Resultados de origen: {args.resultados}",
        f"Objetivos de test válidos: {valid.sum()} de {len(test_t)} "
        f"({data.index[tt[0]]} -> {data.index[tt[-1]]})",
        "", "TABLA DE COMPARACIÓN", comp.round(3).to_string(index=False),
        "", "LSTM, media y desviación", g.round(4).to_string(),
        "", f"Modelo completo por semilla: R2 {full['R2'].min():.3f}–{full['R2'].max():.3f}, "
            f"MAE {full['MAE_MW'].min():.1f}–{full['MAE_MW'].max():.1f} MW",
        "", "MÉTRICAS POR SEMILLA", per_seed.round(4).to_string(index=False), "",
    ])
    open(os.path.join(args.salida, "resumen.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
