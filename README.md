# Hourly electricity demand forecasting for Colombia with LSTM

Code for the article:

> S. F. Obando Rosero and O. J. Suárez, *Hourly Electrical Demand Forecasting of the Colombian Energy System Using LSTM Neural Networks: Ablation of Demand Lags, Cyclic Encodings, and Renewable Generation Series*. Submitted to *Ingeniería y Desarrollo* (Universidad del Norte). **Status: under review.**

The code trains and evaluates a two-layer stacked LSTM (128/64 units, 72-hour window) that forecasts the hourly national electricity demand of Colombia. The inputs are demand lags, cyclic calendar encodings and modelled solar and wind output series (2014 data, Pupo-Roncallo et al., 2020). It includes an ablation study over five seeds and persistence, seasonal-naive and gradient-boosting baselines. Results are reported in the article.

## Repository structure

```
├── src/
│   ├── final_model.py              # full experiment: ablation (5 seeds) + baselines
│   └── evaluate_without_zeros.py   # re-evaluates older runs without the 24 zero-demand hours
├── data/raw/                       # hourly series from the dataset (headerless txt, 8,784 rows)
├── docs/METHOD_NOTES.md            # method design decisions
├── requirements.txt
└── LICENSE
```

The scripts write their outputs (tables, per-seed metrics, figures) to `results/`. That folder is created locally and is not part of the repository.

## Data

Pupo-Roncallo, O., Campillo, J., Ingham, D., Hughes, K. and Pourkashanian, M. (2020). *Renewable energy production and demand dataset for the energy system of Colombia*. Data in Brief, 28, 105084. https://doi.org/10.1016/j.dib.2019.105084. Mendeley Data: https://doi.org/10.17632/w6cwkn4jm7.2. License: **CC BY 4.0**. The files are redistributed unmodified; see [data/raw/README.md](data/raw/README.md).

Files used:

- `Electricity demand Colombia 2014 [MW].txt`: hourly national demand.
- `4. Solar_generation_ENSO-NINO.txt`: **modelled** (not measured) solar output.
- `6. Wind_generation_ENSO-NINO.txt`: **modelled** (not measured) wind output.

**Note:** the demand file has 8,784 rows and the last 24 (dated 2015-01-01) are 0 MW. They are missing data that fall in the test set; `final_model.py` excludes them from every metric. See [docs/METHOD_NOTES.md](docs/METHOD_NOTES.md).

## How to reproduce

```bash
pip install -r requirements.txt

# Full experiment (4 configurations x 5 seeds x 80 epochs: takes several hours)
python src/final_model.py
```

Other options:

```bash
# Quick test (use an absolute output path: a relative one is taken relative to --data-dir)
python src/final_model.py --seeds 1 --epochs 3 --output "$PWD/results/test"

# Baselines only (about 1 minute, TensorFlow not needed)
python src/final_model.py --baselines-only
```

`final_model.py` can be **resumed**: it appends each finished run to `metrics_per_seed.csv` and skips the (configuration, seed) pairs that already exist.

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `data/raw` | folder with the `.txt` files |
| `--solar`, `--wind`, `--demand` | ENSO-NIÑO files and 2014 demand | data file names |
| `--seeds` | 5 | seeds per configuration |
| `--epochs` | 80 | training epochs |
| `--n-rows` | all | truncate the files to N rows (e.g. 8760) |
| `--configs` | `full,no_solar_wind,no_lags,no_cyclic` | comma-separated ablation configurations |
| `--baselines-only` | — | compute only the baselines |
| `--output` | `results/new_run` | output folder; a relative path is taken relative to `--data-dir`, so an absolute path is recommended |

`evaluate_without_zeros.py` re-evaluates, without retraining, a run made before `final_model.py` excluded the zero-demand hours. Arguments: `--data-dir`, `--results` (the run folder, with its `runs/` subfolder) and `--output`.

## Method (summary)

- 13 input variables: solar, wind, sine/cosine encodings of hour, day of week and day of year, weekend flag, working-hour flag (8–18 h) and demand lags of 1, 24 and 168 h.
- 72-hour window. Given how the sequences are built, the most recent demand value available is that of t-2, so the effective horizon is 2 hours.
- LSTM(128, return_sequences) → Dropout(0.2) → LSTM(64) → BatchNormalization → Dense(32, ReLU) → Dense(1). Adam (lr 0.001), MSE loss, 80 epochs, batch 32, `validation_split=0.2`.
- Chronological 80/20 split. The MinMax scalers are fitted **on the training portion only**.

More details in [docs/METHOD_NOTES.md](docs/METHOD_NOTES.md).

## Limitations

- A single year of data and a 2-hour horizon: this is a feasibility study.
- Solar and wind are modelled series from the dataset, not measurements.

## License

- **Code:** MIT. See [LICENSE](LICENSE).
- **Data** (`data/raw/`): CC BY 4.0, by Pupo-Roncallo et al. (2020). Redistributed unmodified; see [data/raw/README.md](data/raw/README.md).
