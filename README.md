# Pronóstico horario de la demanda eléctrica de Colombia con LSTM

Código del artículo:

> S. F. Obando Rosero y O. J. Suárez, *Hourly Electrical Demand Forecasting of the Colombian Energy System Using LSTM Neural Networks: Ablation of Demand Lags, Cyclic Encodings, and Renewable Generation Series*. Enviado a *Ingeniería y Desarrollo* (Universidad del Norte). **Estado: en revisión.**

**English summary.** Code to train and evaluate a two-layer stacked LSTM (128/64 units, 72-hour window) that forecasts Colombian hourly national electricity demand. Inputs are demand lags, cyclic calendar encodings and modelled solar and wind output series (2014 data, Pupo-Roncallo et al., 2020). The code includes an ablation study over five seeds and persistence, seasonal-naive and gradient-boosting baselines. Results are reported in the article.

## Estructura

```
├── src/
│   ├── modelo_final.py          # experimento: ablación (5 semillas) + baselines
│   └── evaluar_sin_ceros.py     # reevalúa corridas antiguas sin las 24 horas con demanda 0
├── data/raw/                    # series horarias del dataset (txt sin encabezado, 8,784 filas)
├── docs/NOTAS_METODO.md         # decisiones del método
├── requirements.txt
└── LICENSE
```

Los scripts escriben sus salidas (tablas, métricas por semilla, figuras) en `results/`. Esa carpeta se genera localmente y no forma parte del repositorio.

## Datos

Pupo-Roncallo, O., Campillo, J., Ingham, D., Hughes, K. y Pourkashanian, M. (2020). *Renewable energy production and demand dataset for the energy system of Colombia*. Data in Brief, 28, 105084. https://doi.org/10.1016/j.dib.2019.105084. Mendeley Data: https://doi.org/10.17632/w6cwkn4jm7.2. Licencia: **CC BY 4.0**. Los archivos se redistribuyen sin cambios; ver [data/raw/README.md](data/raw/README.md).

Archivos que se usan:

- `Electricity demand Colombia 2014 [MW].txt`: demanda nacional horaria.
- `4. Solar_generation_ENSO-NINO.txt`: generación solar **modelada**, no medida.
- `6. Wind_generation_ENSO-NINO.txt`: generación eólica **modelada**, no medida.

**Ojo:** el archivo de demanda tiene 8,784 filas y las últimas 24 (2015-01-01) valen 0 MW. Son datos faltantes que caen en el conjunto de test; `modelo_final.py` los excluye de todas las métricas. Ver [docs/NOTAS_METODO.md](docs/NOTAS_METODO.md).

## Cómo reproducir

```bash
pip install -r requirements.txt

# Experimento completo (4 configuraciones x 5 semillas x 80 épocas: tarda varias horas)
python src/modelo_final.py
```

Otras opciones:

```bash
# Prueba rápida (ruta de salida absoluta: una relativa se interpretaría respecto a --ruta)
python src/modelo_final.py --seeds 1 --epochs 3 --salida "$PWD/results/prueba"

# Solo baselines (alrededor de 1 minuto, no necesita TensorFlow)
python src/modelo_final.py --solo-baselines
```

`modelo_final.py` es **reanudable**: anexa cada corrida a `metricas_por_semilla.csv` y omite las combinaciones (configuración, semilla) que ya existen.

| Argumento | Por defecto | Descripción |
|---|---|---|
| `--ruta` | `data/raw` | carpeta con los `.txt` |
| `--solar`, `--wind`, `--demand` | archivos ENSO-NIÑO y demanda 2014 | nombres de los archivos de datos |
| `--seeds` | 5 | semillas por configuración |
| `--epochs` | 80 | épocas de entrenamiento |
| `--n-filas` | todas | recorta los archivos a N filas (p. ej. 8760) |
| `--configs` | `completo,sin_solar_wind,sin_lags,sin_ciclicas` | configuraciones de la ablación, separadas por coma |
| `--solo-baselines` | — | calcula solo los baselines |
| `--salida` | `results/nueva_corrida` | carpeta de salida; una ruta relativa se interpreta respecto a `--ruta`, así que conviene pasarla absoluta |

`evaluar_sin_ceros.py` reevalúa, sin reentrenar, una corrida hecha antes de que `modelo_final.py` excluyera los ceros. Acepta `--ruta` (datos), `--resultados` (carpeta de la corrida, con su subcarpeta `corridas/`) y `--salida`.

## Método (resumen)

- 13 variables de entrada: solar, viento, codificación seno/coseno de hora, día de la semana y día del año, fin de semana, hora laboral (8–18 h) y lags de demanda de 1, 24 y 168 h.
- Ventana de 72 h. Con la construcción de las secuencias, el último dato de demanda disponible es el de t-2, así que el horizonte efectivo es de 2 horas.
- LSTM(128, return_sequences) → Dropout(0.2) → LSTM(64) → BatchNormalization → Dense(32, ReLU) → Dense(1). Adam (lr 0.001), MSE, 80 épocas, lote 32, `validation_split=0.2`.
- Partición cronológica 80/20. Los escaladores MinMax se ajustan **solo con el tramo de entrenamiento**.

Más detalle en [docs/NOTAS_METODO.md](docs/NOTAS_METODO.md).

## Limitaciones

- Un solo año de datos y horizonte de 2 h: es un estudio de factibilidad.
- Solar y viento son series modeladas del dataset, no mediciones.

## Licencia

- **Código:** MIT. Ver [LICENSE](LICENSE).
- **Datos** (`data/raw/`): CC BY 4.0, de Pupo-Roncallo et al. (2020). Se redistribuyen sin modificaciones; ver [data/raw/README.md](data/raw/README.md).
