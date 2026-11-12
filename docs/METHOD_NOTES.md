# Method notes

Design decisions of `src/final_model.py` and the reasons behind them. Everything stated here can be checked in the code.

## Data

- Three headerless hourly series from the Pupo-Roncallo et al. (2020) dataset: 2014 national demand, ENSO-NIÑO solar and ENSO-NIÑO wind. The last two are **modelled output distributions**, not measurements.
- All 8,784 rows are used. The time index starts on 2014-01-01 00:00 with hourly frequency, so the last 24 rows are dated 2015-01-01 (their demand is 0; see "Zero-demand targets" below).
- After building the lags (up to 168 h) and dropping NaN values, 8,616 rows remain.

## Variables (13)

| Group | Variables |
|---|---|
| Renewables | `solar`, `wind` |
| Cyclic encoding | sine/cosine of the hour (24), day of week (7) and day of year (365) |
| Binary calendar | `is_weekend` (Sat–Sun), `is_working_hour` (8–18 h) |
| Demand lags | `demand_lag_1`, `demand_lag_24`, `demand_lag_168` |

## Sequences and horizon

- 72-hour window (`SEQ_LEN = 72`). Sequence `i` uses rows `i … i+71` and its target is the demand at row `i+72`.
- In the last row of the window (t-1), `demand_lag_1` holds the demand at t-2. The demand at t-1 is never an input, so the **effective horizon is 2 hours**. For consistency, the persistence baseline uses t-2.

## Split and scaling

- Chronological 80/20 split of the sequences: 6,835 training targets and 1,709 test targets (2014-10-22 19:00 → 2015-01-01 23:00), of which 1,685 are valid (see below).
- `MinMaxScaler` for X and y fitted **only on the rows of the training portion**. Fitting them on all the data would leak information from the test set into training.
- Within the training portion, Keras holds out the last 20% for validation (`validation_split=0.2`) to monitor the loss curves. The test set is not involved.

## Model

`LSTM(128, return_sequences=True, tanh) → Dropout(0.2) → LSTM(64, tanh) → BatchNormalization → Dense(32, ReLU) → Dense(1)`. Adam (lr 0.001), MSE loss, 80 epochs, batch 32. The architecture is the same as in the version of the manuscript submitted to the journal.

## Seeds and reproducibility

- 5 seeds (0–4) per configuration via `tf.keras.utils.set_random_seed`. Mean ± standard deviation is reported, because a single run can vary considerably between seeds.
- Each run saves `real`, `pred`, `valid`, `loss` and `val_loss` to `<output>/runs/<config>_seed<n>.npz`, and one row to `metrics_per_seed.csv`. If the script is interrupted, it resumes without repeating finished runs.

## Ablation

| Key | Variables removed | No. of variables |
|---|---|---|
| `full` | — | 13 |
| `no_solar_wind` | solar, wind | 11 |
| `no_lags` | the 3 demand lags | 10 |
| `no_cyclic` | the 6 sine/cosine encodings | 7 |

## Baselines (same test set, same available information)

- Persistence: demand at t-2.
- Seasonal naive: demand at t-24 and at t-168.
- `HistGradientBoostingRegressor` (400 iterations, lr 0.05, `random_state=0`) on the variables of the last row of the window, with 13 and with 11 variables.

## Metrics

MAE, RMSE and R² in MW (after inverting the scaling), and MSE, RMSE and MAE on the normalised [0, 1] scale. All of them come from the same run. The MAE as a percentage of the mean test demand is also computed.

## Zero-demand targets (decision)

- The last 24 rows of the demand file (dated 2015-01-01) are 0 MW. They are not hours of 2014 and a national demand of 0 MW is not physical, so they are treated as **missing data**.
- Being at the end of the series, they only appear as **test targets** (24 of 1,709). They never enter training, the scaler fit or the input window of another target. Including them does not change training but distorts the metrics: an error of thousands of MW in those hours dominates the RMSE and R².
- Decision: the metrics are computed on the **1,685 valid targets**. `final_model.py` applies the exclusion to every model (function `valid_test`), which is exactly equivalent to evaluating on clean data while keeping the same split. For runs made before this change, `src/evaluate_without_zeros.py` reuses the saved predictions and recomputes the baselines without retraining; the results are identical.
- `--n-rows 8760` is not used: it would shift the 80/20 split point and require retraining without any benefit.
