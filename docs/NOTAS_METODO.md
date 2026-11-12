# Notas del método

Decisiones de diseño de `src/modelo_final.py` y su porqué. Todo lo que aparece aquí se puede comprobar en el código.

## Datos

- Tres series horarias sin encabezado del dataset de Pupo-Roncallo et al. (2020): demanda nacional 2014, solar ENSO-NIÑO y eólica ENSO-NIÑO. Las dos últimas son **distribuciones de salida modeladas**, no mediciones.
- Se usan las 8,784 filas. El índice temporal arranca el 2014-01-01 00:00 con frecuencia horaria, así que las últimas 24 filas quedan fechadas en 2015-01-01 (en ellas la demanda vale 0; ver «Objetivos con demanda 0» más abajo).
- Tras crear los lags (hasta 168 h) y eliminar los NaN quedan 8,616 filas.

## Variables (13)

| Grupo | Variables |
|---|---|
| Renovables | `solar`, `wind` |
| Codificación cíclica | seno/coseno de la hora (24), del día de la semana (7) y del día del año (365) |
| Calendario binario | `is_weekend` (sáb–dom), `is_working_hour` (8–18 h) |
| Lags de demanda | `demand_lag_1`, `demand_lag_24`, `demand_lag_168` |

## Secuencias y horizonte

- Ventana de 72 h (`SEQ_LEN = 72`). La secuencia `i` usa las filas `i … i+71` y su objetivo es la demanda de la fila `i+72`.
- En la última fila de la ventana (t-1), `demand_lag_1` contiene la demanda de t-2. La demanda de t-1 nunca entra como variable, así que el **horizonte efectivo es de 2 h**. Por coherencia, la persistencia usa t-2.

## Partición y escalado

- Partición cronológica 80/20 sobre las secuencias: 6,835 objetivos de entrenamiento y 1,709 de test (2014-10-22 19:00 → 2015-01-01 23:00).
- `MinMaxScaler` para X e y ajustado **solo con las filas del tramo de entrenamiento**. Ajustarlos con todos los datos provocaría fuga de información del test hacia el entrenamiento.
- Dentro del entrenamiento, Keras reserva el último 20 % como validación (`validation_split=0.2`) para las curvas de pérdida. El test no participa.

## Modelo

`LSTM(128, return_sequences=True, tanh) → Dropout(0.2) → LSTM(64, tanh) → BatchNormalization → Dense(32, ReLU) → Dense(1)`. Adam (lr 0.001), pérdida MSE, 80 épocas, lote 32. La arquitectura es la misma que la de la versión del manuscrito enviada a la revista.

## Semillas y reproducibilidad

- 5 semillas (0–4) por configuración con `tf.keras.utils.set_random_seed`. Se reporta media ± desviación estándar, porque una sola corrida puede variar bastante entre semillas.
- Cada corrida guarda `real`, `pred`, `loss` y `val_loss` en `<salida>/corridas/<config>_semilla<n>.npz`, y una fila en `metricas_por_semilla.csv`. Si el script se interrumpe, reanuda sin repetir las corridas hechas.

## Ablación

| Clave | Variables retiradas | Nº variables |
|---|---|---|
| `completo` | — | 13 |
| `sin_solar_wind` | solar, wind | 11 |
| `sin_lags` | 3 lags de demanda | 10 |
| `sin_ciclicas` | 6 codificaciones seno/coseno | 7 |

## Baselines (mismo test, misma información disponible)

- Persistencia: demanda en t-2.
- Seasonal naive: demanda en t-24 y en t-168.
- `HistGradientBoostingRegressor` (400 iteraciones, lr 0.05, `random_state=0`) con las variables de la última fila de la ventana, con 13 y con 11 variables.

## Métricas

MAE, RMSE y R² en MW (después de invertir el escalado), y MSE, RMSE y MAE en escala normalizada [0, 1]. Todas salen de la misma corrida. También se calcula el MAE como % de la demanda media del test.

## Objetivos con demanda 0 (decisión)

- Las últimas 24 filas del archivo de demanda (fechadas 2015-01-01) valen 0 MW. No son horas de 2014 y una demanda nacional de 0 MW no es física, así que se tratan como **datos faltantes**.
- Por estar al final de la serie, solo aparecen como **objetivos de test** (24 de 1,709). Nunca entran al entrenamiento, al ajuste de los escaladores ni a la ventana de entrada de otro objetivo. Incluirlas no cambia el entrenamiento, pero distorsiona las métricas: un error de miles de MW en esas horas domina el RMSE y el R².
- Decisión: las métricas se calculan sobre los **1,685 objetivos válidos**. `modelo_final.py` aplica la exclusión a todos los modelos (función `valid_test`), y equivale exactamente a evaluar con datos limpios manteniendo la misma partición. Para corridas hechas antes de este cambio, `src/evaluar_sin_ceros.py` reutiliza las predicciones guardadas y recalcula los baselines, sin reentrenar; los resultados son idénticos.
- No se usa `--n-filas 8760`: desplazaría el punto de corte 80/20 y obligaría a reentrenar sin aportar nada.
