# SAMNN A/H1N1 Influenza Forecasting Code

This repository contains the analysis scripts accompanying the manuscript:

> **Stage-aware deep learning with adversarial data synthesis enables prospective season-scale A/H1N1 influenza forecasting in China**

The released scripts cover the season-specific SAMNN implementations, five baseline models, adversarial synthetic-covariate generation, the 2024/25 scenario analysis, and the SHAP Sankey-bubble visualization.

## Repository contents

| File | Purpose |
|---|---|
| `samnn_2018_2019.py` | SAMNN evaluation for the 2018/19 season |
| `samnn_2022_2023.py` | SAMNN evaluation for the 2022/23 season |
| `samnn_2023_2024.py` | Stage-aware weighted-fusion SAMNN evaluation for the 2023/24 season |
| `samnn_2024_2025_scenario.py` | Season-scale 2024/25 scenario forecasting |
| `baseline_lightgbm.py` | LightGBM baseline |
| `baseline_xgboost.py` | XGBoost baseline |
| `baseline_svr.py` | Support vector regression baseline |
| `baseline_lstm.py` | Long short-term memory baseline |
| `baseline_transformer.py` | Transformer baseline |
| `generate_synthetic_covariates.py` | WGAN-GP and CTGAN synthetic-covariate generation and diagnostics |
| `plot_shap_sankey_bubble.py` | Sankey-bubble visualization from precomputed SHAP summaries |

## Forecasting design

The historical evaluation seasons are:

- 2018/19
- 2022/23
- 2023/24

The forecast horizon is controlled by `FORECAST_LEAD`.

For SAMNN, LightGBM, SVR, LSTM, and Transformer:

- `FORECAST_LEAD = 1`: epidemiological lags 1-4
- `FORECAST_LEAD = 2`: epidemiological lags 2-5
- `FORECAST_LEAD = 4`: epidemiological lags 4-7

The final XGBoost implementation uses three horizon-specific epidemiological lags:

- `FORECAST_LEAD = 1`: lags 1-3
- `FORECAST_LEAD = 2`: lags 2-4
- `FORECAST_LEAD = 4`: lags 4-6

This model-specific configuration is retained to reproduce the analyses reported in the manuscript.

## Model-specific implementations

The repository preserves the original season-specific and architecture-specific implementations used in the study.

- The 2018/19 and 2022/23 SAMNN scripts use direct fusion of epidemiological, climatic, and social-context representations.
- The 2023/24 SAMNN script adds a time-period input, learnable softmax-weighted fusion, and increased weighting of post-pandemic training observations.
- The 2024/25 script uses the scenario-analysis implementation provided with the study.
- Baseline models retain their original sliding-window or sequence construction and season-specific hyperparameter settings.

All historical models are evaluated on the prespecified held-out weekly targets for the corresponding epidemic seasons.

## Required input data

The scripts expect a prepared weekly analysis table. The original analysis used the following legacy local filenames:

- `2011-2024AH1N1预测数据(返修).csv`
- `2011-2024AH1N1预测数据.csv`

These filenames are retained in the scripts for compatibility with the original workflow. Users may replace the paths with local English filenames without changing the analysis.

The prepared table should contain, as applicable:

- `time`
- `time_part`
- `time_part_code`
- `H1N1_ILI%×positive%_national`
- epidemiological lag variables for weeks 1-7
- 12 climatic variables
- `holiday`
- `trend_stage`
- `H1N1_flu_season`

The source surveillance and meteorological data are not redistributed in this repository. Access and reuse should follow the policies of the original data providers.

## Running the scripts

1. Place the prepared analysis table in the working directory or update the input path in the relevant script.
2. Set `FORECAST_LEAD` to `1`, `2`, or `4`.
3. Run the required season-specific SAMNN or baseline script.
4. Use `generate_synthetic_covariates.py` to generate synthetic climatic, epidemiological, and social-context samples.
5. Use `samnn_2024_2025_scenario.py` with the prepared scenario input table for the 2024/25 scenario analysis.
6. Use `plot_shap_sankey_bubble.py` after preparing the three season-specific SHAP summary files:
   - `shap_sankey_2018_2019.csv`
   - `shap_sankey_2022_2023.csv`
   - `shap_sankey_2023_2024.csv`

The SHAP plotting script visualizes precomputed `mean_abs_shap` summaries. It does not train a model or calculate SHAP values.

## Metric and interval conventions

The scripts report R2, MAE, RMSE, and Pearson correlation where applicable.

In the residual plots:

> residual = observed - predicted

Residual-based uncertainty limits are obtained from the dispersion of training residuals. These limits should be interpreted as empirical uncertainty intervals rather than formal parameter confidence intervals.

The 2024/25 scenario script saves unsmoothed weekly predictions and applies a three-week rolling mean to the displayed scenario trajectory and interval bounds.

## Synthetic-data workflow

`generate_synthetic_covariates.py` uses:

- WGAN-GP for climatic variables
- CTGAN for epidemiological and social-context variables

The script also calculates distribution-distance metrics, nearest-neighbor overlap, diversity, direct-leak diagnostics, and real-versus-synthetic classification performance.

Some internal variable names containing the legacy prefix `mia_` are retained for traceability. The corresponding output text describes the implemented procedure as real-versus-synthetic classification rather than a formal membership-inference attack.

## Software

The scripts require Python and the packages listed in `requirements.txt`.

Because deep-learning and synthetic-data results may depend on software, hardware, and random-number implementation details, users should record the full execution environment when performing an exact replication.

## Translation and code integrity

The public versions translate Chinese comments, docstrings, console messages, plot labels, and revision-process wording into neutral technical English. Model structures, mathematical operations, numeric constants, training settings, feature selections, date ranges, and control flow were not modified.

See `TRANSLATION_AND_INTEGRITY_REPORT.md` for details.

## Citation

Please cite the associated manuscript when using this code. Full bibliographic information should be added after publication.
