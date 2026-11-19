# EcoWatt AI: 24-Hour Household Electricity Forecasting

**Course:** Professional Training (Intermediate) — YN3012170034
**Project type:** Individual machine-learning application
**Assigned GitHub folder:** `mam032`
**Framework:** Python, Streamlit, Plotly and scikit-learn

## 1. Project overview

EcoWatt AI is an interactive application for forecasting household electricity use for the next 24 hours. It uses recorded hourly energy data in kWh. The application compares three forecasting methods and can also check a user's hourly CSV file.

The project uses two public datasets as separate experiments:

- **XKY 2025:** one French household with smart-meter measurements from 2025.
- **UCI Household Electric Power Consumption:** one French household with measurements from 2006 to 2010.

EcoWatt AI is not a live smart-meter service. It does not receive live household data.

## 2. Main features

- A Streamlit application with five sections: Overview, Forecast Studio, Data Hub, Model Lab and Help.
- A 24-hour forecast from the selected model.
- Three forecasting methods and a comparison of their results.
- Hourly CSV upload with checks for column names, timestamps, missing hours and invalid energy values.
- Charts, tables and a forecast CSV download.
- Household energy profiles and simple planning scenarios.
- Separate results for the XKY and UCI datasets.
- Data preparation, model training and verification scripts.

## 3. Models used

The application compares these methods:

**Seasonal Naive**

This baseline uses the recorded value from the same hour on the previous day. It does not need model training.

**Ridge Regression**

This is a linear regression model with L2 regularisation. The features are standardised before training. The regularisation value is `alpha=10`.

**HistGradientBoosting**

This model uses a series of small decision trees to learn non-linear patterns. It uses 220 boosting iterations, a learning rate of 0.06, up to 24 leaves per tree, at least 30 samples per leaf and L2 regularisation of 0.2. The random seed is 42. Random early stopping is disabled so that model selection follows the chronological validation plan.

The trained models use recent energy values, 24-hour, 48-hour and 168-hour lags, rolling statistics, calendar features and the forecast horizon. All input features use information available at or before the forecast origin.

The application selects the recommended trained model by the lowest validation MAE. It does not use the held-out test set to select a model.

## 4. Model capabilities and limitations

The models can:

- forecast 24 hourly energy values from a valid hourly history;
- compare the three methods on the selected benchmark;
- create a forecast from an uploaded hourly CSV after the local models have been prepared;
- show benchmark metrics and export forecast results.

The current evidence has important limits:

- Each benchmark represents one household only.
- The XKY test period is small.
- The reported scores apply to the completed 23:00 daily forecast schedule.
- The scores do not prove accuracy for another household or a future period.
- Forecasts for an uploaded household use a model trained on the selected benchmark. Accuracy for that upload is not verified.
- Rare high-demand events may be difficult to predict.
- Cost estimates depend on the rate entered by the user.

## 5. Data and evaluation method

The XKY experiment uses Subject 14 from the public XKY dataset. The selected `BASE` field is a cumulative energy counter in Wh. The preparation script calculates hourly grid-drawn energy from valid past-only counter readings near each hour boundary. A retained XKY hour is an estimate, not a live reading.

The UCI source records minute-average active power in kW. The preparation script converts valid minute readings into hourly energy in kWh. Hours with too few valid readings are not treated as zero.

The two datasets are processed, trained and evaluated separately. The project uses chronological training, validation and test periods. Model selection uses validation MAE. Final MAE, RMSE and R² are calculated on the held-out test period. The evaluation uses completed 23:00 hourly origins. Results from other origin hours are exploratory.

## 6. Reported test results

The following results are recorded in the project evaluation:

| Dataset | Model | MAE (kWh) | RMSE (kWh) | R² |
|---|---|---:|---:|---:|
| XKY 2025 | HistGradientBoosting | 0.1885 | 0.3451 | 0.8150 |
| XKY 2025 | Ridge | 0.2193 | 0.3483 | 0.8116 |
| XKY 2025 | Seasonal Naive | 0.2207 | 0.4085 | 0.7408 |
| UCI 2006–2010 | HistGradientBoosting | 0.4166 | 0.5775 | 0.3341 |
| UCI 2006–2010 | Ridge | 0.4595 | 0.6106 | 0.2555 |
| UCI 2006–2010 | Seasonal Naive | 0.5109 | 0.7600 | -0.1532 |

These scores describe the two source households only. They should not be read as expected accuracy for every household.

## 7. System requirements

- Python 3.12 is recommended.
- Use a 64-bit version of Python.
- A normal laptop or desktop CPU is sufficient. A GPU is not required.
- An internet connection is required to download the full public datasets.
- Allow enough free disk space for the downloaded data and locally trained model files.

The project uses the dependency versions listed in `requirements.txt`.

## 8. Installation

Run these commands from the `mam032` project folder.

**macOS or Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 9. Dataset sources and attribution

The full datasets are not included in this repository. The preparation scripts download the source data from the official providers.

| Dataset | Official source | Licence |
|---|---|---|
| UCI Household Electric Power Consumption | [Dataset DOI: 10.24432/C58K54](https://doi.org/10.24432/C58K54) | CC BY 4.0 |
| XKY study | [Dataset DOI: 10.57745/MA43FS](https://doi.org/10.57745/MA43FS) | Licence Ouverte / Open Licence 2.0 |
| XKY Subject 14, 2025 source file | [File DOI: 10.57745/OILZFV](https://doi.org/10.57745/OILZFV) | Licence Ouverte / Open Licence 2.0 |

Small, attributed example CSV files are included in `examples/`. They are provided for the application demo and input-format checks. They are not replacements for the full datasets.

## 10. Download and prepare the datasets

Run the preparation command for each dataset you want to use.

**XKY 2025**

```bash
python scripts/prepare_xky_data.py
```

**UCI Household Electric Power Consumption**

```bash
python scripts/prepare_data.py
```

The scripts download the official source data, prepare hourly energy data and create local data-quality information. The prepared files are saved under `data/`. The generated files are not part of the GitHub submission.

## 11. Train the models

Train each dataset separately.

**XKY 2025**

```bash
python scripts/train_models.py --dataset xky_2025
```

**UCI Household Electric Power Consumption**

```bash
python scripts/train_models.py --dataset uci_household
```

Training creates the local model bundle, evaluation results and training metadata under `artifacts/`. The application needs these local files for full forecasting and benchmark results.

## 12. Verify the experiments

Run the baseline comparison and release checks after data preparation and model training.

```bash
python scripts/run_baseline_check.py --dataset xky_2025
python scripts/run_baseline_check.py --dataset uci_household

python scripts/verify_release.py --dataset xky_2025 --write-manifests
python scripts/verify_release.py --dataset uci_household --write-manifests
```

Optional local checks:

```bash
python scripts/run_diagnostics.py --dataset all
python scripts/verify_app_workflows.py
python scripts/verify_decision_workflows.py
python scripts/check_ui_contrast.py
```

The Streamlit workflow checks use the locally trained experiment files. They do not create new benchmark results.

## 13. Run the application

Start Streamlit from the project folder:

```bash
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

Open this address in a browser:

```text
http://127.0.0.1:8501/
```

To check the application health while it is running, open a second terminal and run:

```bash
python scripts/check_health.py
```

## 14. CSV input format

The uploaded file must contain hourly energy data. It must have a timestamp column and an energy column.

Example:

```csv
timestamp,energy_kwh
2024-01-01 00:00:00,0.52
2024-01-01 01:00:00,0.47
2024-01-01 02:00:00,0.49
```

The application accepts `timestamp` or `datetime` as the time column. It accepts `energy_kwh` or `consumption_kwh` as the energy column. The data must be in ascending order and contain consecutive hourly records. The application checks the file before forecasting.

## 15. Example usage

1. Install the requirements.
2. Download and prepare at least one full dataset.
3. Train the models for that dataset.
4. Verify the experiment.
5. Start the Streamlit application.
6. Select a benchmark dataset in the sidebar.
7. Open Forecast Studio and choose a model.
8. Use the built-in example or upload an hourly CSV.
9. Generate the 24-hour forecast.
10. Review the chart and table, then download the forecast CSV.

The example files in `examples/` can be used to check the expected CSV format. Full model forecasts require the local training steps above.

## 16. Troubleshooting

**The model file is missing**

Run the preparation command for the selected dataset, then run its training command. The model file is created locally under `artifacts/`.

**Benchmark metrics are not available**

Train the selected dataset and check that `test_metrics.csv` was created. Then run the matching release verification command.

**The prepared data is missing**

Run the dataset preparation script again. Check that the download completed and that the source file is available in the local `data/` folder.

**The CSV upload is rejected**

Check the column names, timestamp format, ascending order, hourly gaps, missing values and energy values. The file must contain enough consecutive hourly history for a 24-hour forecast.

## 17. Project structure

```text
mam032/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .streamlit/
├── src/
├── scripts/
├── config/
├── examples/
├── assets/
└── tests/
```

Main source modules:

- `src/data.py` — source download, data preparation and CSV validation
- `src/recent_data.py` — XKY source download and hourly preparation
- `src/datasets.py` — dataset definitions and file locations
- `src/features.py` — forecasting features
- `src/training.py` — model training and evaluation
- `src/forecasting.py` — 24-hour forecasting
- `src/evaluation.py` — evaluation metrics and chronological split
- `src/decision_support.py` — household profiles and planning calculations
- `src/views.py` and `src/ui.py` — Streamlit screens and interface
- `scripts/` — preparation, training, evaluation and verification commands
- `tests/` — automated tests
- `examples/` — small attributed example CSV files

## 18. References

1. Hebrail, G., and Berard, A. (2006). *Individual Household Electric Power Consumption*. UCI Machine Learning Repository. [https://doi.org/10.24432/C58K54](https://doi.org/10.24432/C58K54). CC BY 4.0.
2. Hong, T., and Fan, S. (2016). Probabilistic electric load forecasting: A tutorial review. *International Journal of Forecasting*, 32(3), 914–938. [https://doi.org/10.1016/j.ijforecast.2015.11.011](https://doi.org/10.1016/j.ijforecast.2015.11.011).
3. Kong, W., Dong, Z. Y., Jia, Y., Hill, D. J., Xu, Y., and Zhang, Y. (2019). Short-Term Residential Load Forecasting Based on LSTM Recurrent Neural Network. *IEEE Transactions on Smart Grid*, 10(1), 841–851. [https://doi.org/10.1109/TSG.2017.2753802](https://doi.org/10.1109/TSG.2017.2753802).
4. Kiprijanovska, I., Stankoski, S., Ilievski, I., Jovanovski, S., Gams, M., and Gjoreski, H. (2020). HousEEC: Day-Ahead Household Electrical Energy Consumption Forecasting Using Deep Learning. *Energies*, 13(10), 2672. [https://doi.org/10.3390/en13102672](https://doi.org/10.3390/en13102672).
5. Hyeon, J., Lee, H., Ko, B., and Choi, H.-J. (2020). Deep learning-based household electric energy consumption forecasting. *The Journal of Engineering*, 2020(13), 639–642. [https://doi.org/10.1049/joe.2019.1219](https://doi.org/10.1049/joe.2019.1219).
6. Xia, Z., Zhang, R., Ma, H., and Saha, T. K. (2024). Day-Ahead Electricity Consumption Prediction of Individual Household-Capturing Peak Consumption Pattern. *IEEE Transactions on Smart Grid*, 15(3), 2971–2984. [https://doi.org/10.1109/TSG.2023.3332281](https://doi.org/10.1109/TSG.2023.3332281).
7. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. [https://jmlr.org/papers/v12/pedregosa11a.html](https://jmlr.org/papers/v12/pedregosa11a.html).
8. Scikit-learn developers. *Ridge* and *HistGradientBoostingRegressor* API documentation, version 1.6.1. [Ridge](https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.Ridge.html); [HistGradientBoostingRegressor](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html); [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html).
9. Streamlit. *API reference*. [https://docs.streamlit.io/develop/api-reference](https://docs.streamlit.io/develop/api-reference).
10. Plotly. *Plotly Python documentation*. [https://plotly.com/python/](https://plotly.com/python/).
11. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 56–61. [https://doi.org/10.25080/Majora-92bf1922-00a](https://doi.org/10.25080/Majora-92bf1922-00a).
12. Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585, 357–362. [https://doi.org/10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
13. Ferrari, J., Osonuga, S., Malbert, T., Imard, V., Wurtz, F., Delinchant, B., and Llerena, D. (2026). *Long-term energy data from Linky meters with sub-minute sampling (XKY study / 108 French households / 2024 and 2025)*. Recherche Data Gouv. [Dataset DOI: 10.57745/MA43FS](https://doi.org/10.57745/MA43FS). Licence Ouverte / Open Licence 2.0.
14. Ferrari, J., et al. (2026). *Subject14_DataLinkyRAW_2025.csv*. File in the XKY dataset. [File DOI: 10.57745/OILZFV](https://doi.org/10.57745/OILZFV).
15. XKY dataset authors (2026). *DataDictionnaryRAW.tab*. [Official data dictionary](https://entrepot.recherche.data.gouv.fr/api/access/datafile/769168?format=original).
16. Etalab. *Licence Ouverte / Open Licence 2.0*. [Licence text](https://spdx.org/licenses/etalab-2.0.html).

## 19. Submission scope

This folder is the source-code submission for the Professional Training GitHub Pull Request. It contains the application source, required scripts, configuration, runtime assets, example inputs, dependencies, tests and this README.

Full datasets, trained model files and generated experiment outputs are created locally after setup. They are not included in this GitHub source submission. The technical report and project demonstration are separate deliverables.
