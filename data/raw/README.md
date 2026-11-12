# Datos de origen

Estos archivos se redistribuyen **sin modificaciones** bajo la licencia original del dataset.

- **Dataset:** O. Pupo-Roncallo, J. Campillo, D. Ingham, K. Hughes y M. Pourkashanian, *Data on electricity demand and renewable energy production for the Colombian energy system*, Mendeley Data, versión 2 (9-dic-2019). https://doi.org/10.17632/w6cwkn4jm7.2
- **Artículo de datos:** O. Pupo-Roncallo et al., "Renewable energy production and demand dataset for the energy system of Colombia", *Data in Brief*, 28, 105084, 2020. https://doi.org/10.1016/j.dib.2019.105084
- **Licencia:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Es la licencia que declara Mendeley Data para las versiones 1 y 2; se consultó el 2026-11-11. Permite copiar y redistribuir citando a los autores e indicando la licencia.

| Archivo | Uso en este repositorio |
|---|---|
| `Electricity demand Colombia 2014 [MW].txt` | **usado** (demanda) |
| `4. Solar_generation_ENSO-NINO.txt` | **usado** (solar, serie modelada) |
| `6. Wind_generation_ENSO-NINO.txt` | **usado** (eólica, serie modelada) |
| `1. Estimated Solar PV generation - Installed cap. 520 MW.txt` | no usado |
| `2. Estimated Wind generation - Installed cap. 658 MW.txt` | no usado |
| `3. Wind_generation_Colombia2014 - Installed cap. 19.5 MW.txt` | no usado |
| `5. Solar_generation_ENSO- NINA.txt` | no usado (escenario NIÑA) |
| `7. Wind_generation_ENSO-NINA.txt` | no usado (escenario NIÑA) |

Formato: texto plano sin encabezado, un valor por línea, 8,784 filas horarias. Las últimas 24 filas del archivo de demanda valen 0 y se excluyen de la evaluación (ver `docs/NOTAS_METODO.md`).
