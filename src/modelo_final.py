"""
modelo_final.py
===============
Experimento completo del artículo: LSTM para el pronóstico horario de la demanda eléctrica de
Colombia, con ablación de variables y baselines, en una sola ejecución reproducible.

  - Los MinMaxScaler se ajustan SOLO con las filas usadas en el entrenamiento (sin fuga de datos).
  - Semillas fijas, varias por configuración (se reporta media y desviación estándar).
  - Ablación: modelo completo, sin solar/wind, sin lags, sin codificación cíclica.
  - Baselines (persistencia, seasonal naive, gradient boosting) con la misma información que el LSTM.
  - Métricas en MW y en escala normalizada calculadas de la misma corrida.
  - Los objetivos de test con demanda 0 (las últimas 24 filas del archivo, fechadas 2015-01-01)
    son datos faltantes y se excluyen de todas las métricas. Las predicciones completas se
    guardan igualmente en corridas/*.npz, junto con la máscara "valido".
  - Guarda cada corrida al terminar: si se interrumpe, al volver a ejecutar continúa donde quedó.

Arquitectura: LSTM 128 + Dropout 0.2 + LSTM 64 + BatchNormalization + Dense 32 + Dense 1,
Adam 0.001, 80 épocas, lote 32, validation_split 0.2, ventana de 72 h.

Uso:
  python src/modelo_final.py          (por defecto: --ruta data/raw, --salida results/nueva_corrida)
  python src/modelo_final.py --seeds 1 --epochs 3 --salida <ruta absoluta>   (prueba rápida)
  python src/modelo_final.py --solo-baselines                                (solo baselines, ~1 minuto)
  python src/modelo_final.py --n-filas 8760                                  (recorta a 8,760 filas)

Resultados: carpeta <ruta>/<salida> (una --salida absoluta se usa tal cual). El resumen queda en resumen.txt.
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
# CONFIGURACIÓN
# ----------------------------------------------------------------------------
# Rutas por defecto relativas al repositorio
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_DEFECTO = os.path.join(_REPO, "data", "raw")
# results/final guarda la corrida verificada del manuscrito; las corridas nuevas van a otra carpeta
SALIDA_DEFECTO = os.path.join(_REPO, "results", "nueva_corrida")
FILE_SOLAR = "4. Solar_generation_ENSO-NINO.txt"
FILE_WIND = "6. Wind_generation_ENSO-NINO.txt"
FILE_DEMAND = "Electricity demand Colombia 2014 [MW].txt"

SEQ_LEN = 72
BATCH = 32
VAL_SPLIT = 0.2
LR = 0.001

CICLICAS = ["hour_sin", "hour_cos", "day_sin", "day_cos", "year_sin", "year_cos"]
LAGS = ["demand_lag_1", "demand_lag_24", "demand_lag_168"]
ALL = ["solar", "wind"] + CICLICAS + ["is_weekend", "is_working_hour"] + LAGS

# clave -> (etiqueta, variables)
CONFIGS = {
    "completo": ("Completo (13 variables)", ALL),
    "sin_solar_wind": ("Sin solar ni wind (11)", [f for f in ALL if f not in ("solar", "wind")]),
    "sin_lags": ("Sin lags de demanda (10)", [f for f in ALL if f not in LAGS]),
    "sin_ciclicas": ("Sin codificación cíclica (7)", [f for f in ALL if f not in CICLICAS]),
}


# ----------------------------------------------------------------------------
# Datos y features
# ----------------------------------------------------------------------------
def load_data(ruta, f_solar, f_wind, f_demand, n_filas=None):
    solar = pd.read_csv(os.path.join(ruta, f_solar), header=None, names=["solar"])
    wind = pd.read_csv(os.path.join(ruta, f_wind), header=None, names=["wind"])
    demand = pd.read_csv(os.path.join(ruta, f_demand), header=None, names=["demand"])
    print(f"Archivos: solar={f_solar} | wind={f_wind} | demanda={f_demand}")
    print(f"Filas leídas: solar={len(solar)}, wind={len(wind)}, demanda={len(demand)}")
    n = min(len(solar), len(wind), len(demand))
    if n_filas:
        n = min(n, n_filas)
    if n != 8760:
        print(f"AVISO: se usan {n} filas, y 2014 tiene 8,760 horas. Las filas sobrantes quedan fechadas en 2015.")
    data = pd.concat([solar[:n], wind[:n], demand[:n]], axis=1)
    data.index = pd.date_range(start="2014-01-01", periods=len(data), freq="60min")
    d = data["demand"].values
    print(f"Demanda: mín={d.min():.0f} MW, máx={d.max():.0f} MW, media={d.mean():.0f} MW, ceros={int((d == 0).sum())}")
    if len(d) >= 48:
        print(f"Últimas 24 filas vs. las 24 anteriores: correlación={np.corrcoef(d[-24:], d[-48:-24])[0, 1]:.3f}, "
              f"idénticas={bool(np.allclose(d[-24:], d[-48:-24]))}")

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
    """Secuencias de SEQ_LEN filas y partición cronológica 80/20. t = fila del objetivo."""
    n_seq = n_rows - SEQ_LEN
    split = int(0.8 * n_seq)
    train_t = np.arange(SEQ_LEN, split + SEQ_LEN)
    test_t = np.arange(split + SEQ_LEN, n_rows)
    return n_seq, split, train_t, test_t


def valid_test(data):
    """Máscara de los objetivos de test válidos (demanda > 0; los ceros son datos faltantes)."""
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
        "Persistencia (última demanda conocida, t-2)": metrics_mw(y_test, d[test_t - 2]),
        "Seasonal naive 24 h (t-24)": metrics_mw(y_test, d[test_t - 24]),
        "Seasonal naive 168 h (t-168)": metrics_mw(y_test, d[test_t - 168]),
    }
    for name, cols in [
        ("Gradient boosting, 13 variables", ALL),
        ("Gradient boosting, sin solar ni wind (11)", CONFIGS["sin_solar_wind"][1]),
    ]:
        X = data[cols].values
        gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0)
        gb.fit(X[train_t - 1], d[train_t])
        rows[name] = metrics_mw(y_test, gb.predict(X[test_t - 1]))
    table = pd.DataFrame(rows).T
    table.index.name = "modelo"
    table.round(3).to_csv(os.path.join(out, "baselines.csv"), encoding="utf-8-sig")
    return table


# ----------------------------------------------------------------------------
# LSTM
# ----------------------------------------------------------------------------
def prepare(data, cols):
    n_rows = len(data)
    n_seq, split, _, _ = split_info(n_rows)
    fit_end = split + SEQ_LEN  # filas vistas en el entrenamiento (entradas y objetivos)
    sx, sy = MinMaxScaler(), MinMaxScaler()
    sx.fit(data[cols].iloc[:fit_end])                  # SOLO entrenamiento
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
        "MAE_pct_media": float(100 * m["MAE_MW"] / np.mean(real[ok])),
        "segundos": round(time.time() - t0, 1),
    })
    np.savez(os.path.join(out, "corridas", f"{key}_semilla{seed}.npz"),
             real=real, pred=pred, valido=ok,
             loss=np.array(hist.history["loss"]), val_loss=np.array(hist.history["val_loss"]))
    return label, m


# ----------------------------------------------------------------------------
# Resumen, tablas y figuras
# ----------------------------------------------------------------------------
def postprocess(data, out, baselines, args):
    csv = os.path.join(out, "metricas_por_semilla.csv")
    if not os.path.exists(csv):
        return None
    df = pd.read_csv(csv)
    order = [CONFIGS[k][0] for k in CONFIGS if CONFIGS[k][0] in set(df["config"])]
    cols = ["MAE_MW", "RMSE_MW", "R2", "MSE_norm", "RMSE_norm", "MAE_norm", "MAE_pct_media"]
    g = df.groupby("config")[cols].agg(["mean", "std"]).reindex(order)
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g["n_semillas"] = df.groupby("config").size().reindex(order)
    g.round(4).to_csv(os.path.join(out, "resumen_lstm.csv"), encoding="utf-8-sig")

    comp = []
    for lab in order:
        r = g.loc[lab]
        comp.append({"modelo": "LSTM: " + lab,
                     "MAE_MW": r["MAE_MW_mean"], "MAE_std": r["MAE_MW_std"],
                     "RMSE_MW": r["RMSE_MW_mean"], "RMSE_std": r["RMSE_MW_std"],
                     "R2": r["R2_mean"], "R2_std": r["R2_std"]})
    for name, r in baselines.iterrows():
        comp.append({"modelo": name, "MAE_MW": r["MAE_MW"], "MAE_std": np.nan,
                     "RMSE_MW": r["RMSE_MW"], "RMSE_std": np.nan, "R2": r["R2"], "R2_std": np.nan})
    comp = pd.DataFrame(comp)
    comp.round(3).to_csv(os.path.join(out, "tabla_comparacion.csv"), index=False, encoding="utf-8-sig")

    # Figura: curvas de pérdida (modelo completo, semilla 0)
    f0 = os.path.join(out, "corridas", "completo_semilla0.npz")
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
        plt.savefig(os.path.join(out, "curvas_perdida.png"), dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(12, 5))
        plt.plot(z["real"][:500], label="Actual demand")
        plt.plot(z["pred"][:500], label="LSTM prediction")
        plt.legend()
        plt.title("Electric demand prediction (first 500 test hours)")
        plt.xlabel("Hours")
        plt.ylabel("Demand (MW)")
        plt.tight_layout()
        plt.savefig(os.path.join(out, "demanda_prediccion.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # Figura: ablación
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, ttl in zip(axes, ["R2", "RMSE_MW"], ["R² (media ± desv.)", "RMSE en MW (media ± desv.)"]):
        ax.bar(range(len(order)), g[f"{col}_mean"].values, yerr=g[f"{col}_std"].fillna(0).values, capsize=4,
               color="steelblue")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([o.split(" (")[0] for o in order], rotation=20, ha="right")
        ax.set_title(ttl)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "ablacion.png"), dpi=200, bbox_inches="tight")
    plt.close()
    return g, comp, df


def write_summary(out, data, args, baselines, post, versions):
    n_rows = len(data)
    _, _, train_t, test_t = split_info(n_rows)
    lines = []
    lines.append("RESUMEN (modelo_final.py)")
    lines.append("=" * 60)
    lines.append(f"Archivos: solar={args.solar} | wind={args.wind} | demanda={args.demand}")
    lines.append(f"Filas usadas tras dropna: {n_rows} | entrenamiento: {len(train_t)} | test: {len(test_t)}")
    lines.append(f"Periodo de test: {data.index[test_t[0]]} -> {data.index[test_t[-1]]}")
    ok = valid_test(data)
    lines.append(f"Objetivos de test evaluados (demanda > 0): {ok.sum()} de {len(test_t)} "
                 f"(hasta {data.index[test_t[ok][-1]]})")
    lines.append(f"Semillas: {args.seeds} | épocas: {args.epochs} | lote: {BATCH} | ventana: {SEQ_LEN} h")
    lines.append(f"Versiones: {json.dumps(versions, ensure_ascii=False)}")
    lines.append("")
    lines.append("BASELINES (MW)")
    lines.append(baselines.round(3).to_string())
    if post is not None:
        g, comp, df = post
        lines.append("")
        lines.append("LSTM, media y desviación por configuración")
        lines.append(g.round(4).to_string())
        lines.append("")
        lines.append("TABLA DE COMPARACIÓN")
        lines.append(comp.round(3).to_string(index=False))
        lines.append("")
        lines.append("MÉTRICAS POR SEMILLA")
        lines.append(df.round(4).to_string(index=False))
    txt = "\n".join(lines)
    with open(os.path.join(out, "resumen.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    return txt


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruta", default=RUTA_DEFECTO)
    ap.add_argument("--solar", default=FILE_SOLAR)
    ap.add_argument("--wind", default=FILE_WIND)
    ap.add_argument("--demand", default=FILE_DEMAND)
    ap.add_argument("--seeds", type=int, default=5, help="semillas por configuración")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--n-filas", type=int, default=None, help="recorta los archivos a N filas (p. ej. 8760)")
    ap.add_argument("--configs", default=",".join(CONFIGS), help="claves separadas por coma")
    ap.add_argument("--solo-baselines", action="store_true")
    ap.add_argument("--salida", default=SALIDA_DEFECTO)
    args = ap.parse_args()

    out = os.path.join(args.ruta, args.salida)
    os.makedirs(os.path.join(out, "corridas"), exist_ok=True)

    data = load_data(args.ruta, args.solar, args.wind, args.demand, args.n_filas)
    _, _, train_t, test_t = split_info(len(data))
    print(f"Entrenamiento: {len(train_t)} objetivos | Test: {len(test_t)} objetivos")
    print(f"Periodo de test: {data.index[test_t[0]]} -> {data.index[test_t[-1]]}")

    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    baselines = run_baselines(data, out)
    print("\n=== Baselines (MW) ===")
    print(baselines.round(3).to_string())

    csv = os.path.join(out, "metricas_por_semilla.csv")
    if not args.solo_baselines:
        try:
            import tensorflow as tf
            versions["tensorflow"] = tf.__version__
        except ImportError:
            sys.exit("No se encontró TensorFlow. Usa --solo-baselines o instala TensorFlow.")
        keys = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
        done = set()
        if os.path.exists(csv):
            prev = pd.read_csv(csv)
            done = set(zip(prev["clave"], prev["seed"]))
        total = len(keys) * args.seeds
        k = 0
        for seed in range(args.seeds):
            for key in keys:
                k += 1
                if (key, seed) in done:
                    print(f"[{k}/{total}] {key} semilla {seed}: ya existe, se omite")
                    continue
                label, m = train_one(data, key, seed, args.epochs, out)
                row = {"clave": key, "config": label, "seed": seed, **m}
                pd.DataFrame([row]).to_csv(csv, mode="a", header=not os.path.exists(csv),
                                           index=False, encoding="utf-8-sig")
                print(f"[{k}/{total}] {label} | semilla {seed}: MAE={m['MAE_MW']:.1f} MW  "
                      f"RMSE={m['RMSE_MW']:.1f} MW  R2={m['R2']:.3f}  ({m['segundos']} s)")

    post = postprocess(data, out, baselines, args)
    json.dump({"args": vars(args), "versions": versions, "filas": len(data)},
              open(os.path.join(out, "config.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    txt = write_summary(out, data, args, baselines, post, versions)
    print("\n" + txt)
    print(f"\nListo. Archivos en: {out}")


if __name__ == "__main__":
    main()
