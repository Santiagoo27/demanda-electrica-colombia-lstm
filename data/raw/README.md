# Source data

These files are redistributed **unmodified** under the original license of the dataset.

- **Dataset:** O. Pupo-Roncallo, J. Campillo, D. Ingham, K. Hughes and M. Pourkashanian, *Data on electricity demand and renewable energy production for the Colombian energy system*, Mendeley Data, version 2 (9 Dec 2019). https://doi.org/10.17632/w6cwkn4jm7.2
- **Data article:** O. Pupo-Roncallo et al., "Renewable energy production and demand dataset for the energy system of Colombia", *Data in Brief*, 28, 105084, 2020. https://doi.org/10.1016/j.dib.2019.105084
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), as stated by Mendeley Data for versions 1 and 2 (checked on 2026-11-11). It allows copying and redistribution with attribution to the authors and an indication of the license.

| File | Use in this repository |
|---|---|
| `Electricity demand Colombia 2014 [MW].txt` | **used** (demand) |
| `4. Solar_generation_ENSO-NINO.txt` | **used** (solar, modelled series) |
| `6. Wind_generation_ENSO-NINO.txt` | **used** (wind, modelled series) |
| `1. Estimated Solar PV generation - Installed cap. 520 MW.txt` | not used |
| `2. Estimated Wind generation - Installed cap. 658 MW.txt` | not used |
| `3. Wind_generation_Colombia2014 - Installed cap. 19.5 MW.txt` | not used |
| `5. Solar_generation_ENSO- NINA.txt` | not used (La Niña scenario) |
| `7. Wind_generation_ENSO-NINA.txt` | not used (La Niña scenario) |

Format: plain text without header, one value per line, 8,784 hourly rows. The last 24 rows of the demand file are 0 and are excluded from the evaluation (see `docs/METHOD_NOTES.md`).
