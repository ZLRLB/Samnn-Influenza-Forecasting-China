import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.stats import pearsonr
import seaborn as sns

myfont = fm.FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
plt.rcParams['axes.unicode_minus'] = False

BASE_CLIMATE = [
    'absolute_humidity','uvindex','solarenergy','solarradiation',
    'visibility','cloudcover','winddir','windspeed','humidity',
    'feelslike','temp','DTR'
]
BASE_SOCIAL = ['holiday','trend_stage','H1N1_flu_season']

TARGET = 'H1N1_ILI%×positive%_national'

# ============================================================
# Forecast-horizon configuration
# Set to 1, 2, or 4 for the corresponding forecast horizon
# ============================================================
FORECAST_LEAD = 4   # Allowed values: 1, 2, or 4
N_EPI_INPUTS = 4    # Number of epidemiological lag inputs used by this implementation

def find_lag_column(df, lag):
    """
    Identify the lag-column names available in the input table.
    Supported naming patterns:
    1. lag_1_week
    2. H1N1_ILI%×positive%_national_lag_1_week
    3. H1N1_ILI%×positive%_national_lag_1_week_smoothed
    """
    candidates = [
        f"lag_{lag}_week",
        f"{TARGET}_lag_{lag}_week",
        f"{TARGET}_lag_{lag}_week_smoothed",
        f"{TARGET}_lag_{lag}_week_for_horizon",
        f"{TARGET}_lag_{lag}_week_smoothed_generated"
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Lag {lag}-week column was not found. Check whether the table contains one of the following names: {candidates}"
    )


def get_epi_cols_for_lead(df, lead, n_epi_inputs=4):
    """
    Select epidemiological lag columns according to the forecast horizon.

    lead=1: lag1-lag4
    lead=2: lag2-lag5
    lead=4: lag4-lag7
    """
    if lead == 1:
        selected_lags = [1, 2, 3, 4]
    elif lead == 2:
        selected_lags = [2, 3, 4, 5]
    elif lead == 4:
        selected_lags = [4, 5, 6, 7]
    else:
        raise ValueError("FORECAST_LEAD must be set to 1, 2, or 4")

    selected_lags = selected_lags[:n_epi_inputs]
    epi_cols = [find_lag_column(df, lag) for lag in selected_lags]

    print(f"\nCurrent configuration: {lead}-week ahead forecasting")
    print("Epidemiological lag columns:")
    for col in epi_cols:
        print("  ", col)

    return epi_cols


WINDOW_SIZE = 3


def build_features(df):
    df = df.dropna(subset=[TARGET]).copy()
    df = df.sort_values("time").reset_index(drop=True)

    # Meteorological variables: retain forward and backward filling
    for feat in BASE_CLIMATE:
        if feat in df.columns:
            df[feat] = df[feat].ffill().bfill()
        else:
            raise ValueError(f"Missing meteorological variable: {feat}")

    # Epidemiological lag variables:
    # Backward filling is not applied to epidemiological lags because it could introduce future information.
    # Rows with missing lag values are removed; this affects only the initial weeks of the training series.
    for feat in BASE_EPIDEMIOLOGY:
        if feat in df.columns:
            df[feat] = pd.to_numeric(df[feat], errors="coerce")
        else:
            raise ValueError(f"Missing epidemiological lag variable: {feat}")

    # Social-context variables
    for col in BASE_SOCIAL:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
            df[col] = df[col].astype("category").cat.codes
        else:
            raise ValueError(f"Missing social-context variable: {col}")

    # Remove rows with missing lag values required for the selected horizon
    df = df.dropna(subset=BASE_EPIDEMIOLOGY).copy()
    df = df.sort_values("time").reset_index(drop=True)

    return df


def make_tabular_samples_full_window(df, window, target_times):
    features, targets, times = [], [], []
    df = df.reset_index(drop=True)
    time_idx_map = {row['time']: idx for idx, row in df.iterrows()}
    for t in target_times:
        idx = time_idx_map.get(t)
        if idx is not None and idx >= window-1:
            row_feats = []
            for w in range(window):
                row_feats += df.iloc[idx-window+1+w][BASE_CLIMATE].tolist()
                row_feats += df.iloc[idx-window+1+w][BASE_SOCIAL].tolist()
                row_feats += df.iloc[idx-window+1+w][BASE_EPIDEMIOLOGY].tolist()
            features.append(row_feats)
            targets.append(df.iloc[idx][TARGET])  # Use the original target
            times.append(df.iloc[idx]['time'])
    return np.array(features), np.array(targets), np.array(times)

period_params = [
    {"train_start": "2011-01-05", "train_end": "2018-09-26", "test_start": "2018-10-03", "test_end": "2019-04-03", "name": "2018-2019"},
    {"train_start": "2011-01-05", "train_end": "2022-10-26", "test_start": "2022-11-02", "test_end": "2023-04-05", "name": "2022-2023"},
    {"train_start": "2011-01-05", "train_end": "2023-09-27", "test_start": "2023-10-04", "test_end": "2024-04-03", "name": "2023-2024"},
    {"train_start": "2011-01-05", "train_end": "2024-05-01", "test_start": "2024-05-09", "test_end": "2025-04-02", "name": "2024-2025_scenario"}
]

df = pd.read_csv("2011-2024AH1N1预测数据(返修).csv", parse_dates=['time'])

# Select lag columns for the current FORECAST_LEAD
BASE_EPIDEMIOLOGY = get_epi_cols_for_lead(
    df,
    lead=FORECAST_LEAD,
    n_epi_inputs=N_EPI_INPUTS
)

# Exclude the COVID period
df = df[df['time_part'] != 'covid']

# Apply feature preprocessing
df = build_features(df)
raw = df.sort_values('time').reset_index(drop=True)
results = []

for params in period_params:
    train_mask = (raw['time'] >= pd.to_datetime(params["train_start"])) & (raw['time'] < pd.to_datetime(params["train_end"]))
    test_mask  = (raw['time'] >= pd.to_datetime(params["test_start"])) & (raw['time'] <= pd.to_datetime(params["test_end"]))

    df_train = raw[train_mask].reset_index(drop=True)
    df_test  = raw[test_mask].reset_index(drop=True)

    # Prepend historical observations to complete the input window
    df_all = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    target_times = df_test['time'].tolist()
    X_train, y_train, ts_train = make_tabular_samples_full_window(df_train, window=WINDOW_SIZE, target_times=df_train['time'].tolist())
    X_test, y_test, ts_test = make_tabular_samples_full_window(df_all, window=WINDOW_SIZE, target_times=target_times)

    if X_test.shape[0] == 0:
        print(f"Period {params['name']} has insufficient samples and will be skipped.")
        continue

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if params["test_start"] == "2023-10-04" and params["test_end"] == "2024-04-03":
        print("Using the season-specific SVR parameter configuration for 2023-10-04 to 2024-04-03.")
        svr = SVR(kernel='rbf', C=10, epsilon=0.001, gamma=0.005)
    else:
        svr = SVR(kernel='rbf', C=10, epsilon=0.001, gamma=0.005)
    svr.fit(X_train, y_train)
    pred = svr.predict(X_test)

    stderr = (y_train - svr.predict(X_train)).std(ddof=1)
    ci95 = 1.96 * stderr
    lower = pred - ci95
    upper = pred + ci95
    lower = np.nan_to_num(lower, nan=0.0)
    upper = np.nan_to_num(upper, nan=0.0)
    print(f"Period {params['name']} Mean bounds of the residual-based uncertainty interval: [{lower.mean():.4f}, {upper.mean():.4f}]")

    pred_df = pd.DataFrame({
        "date": pd.to_datetime(ts_test),
        "y_true": y_test if not all(np.isnan(y_test)) else [np.nan] * len(pred),
        "y_pred": pred,
        "SVR_pred_lower": lower,
        "SVR_pred_upper": upper
    })
    pred_df.to_csv(f"revised_SVR_preds_{params['name']}.csv", index=False, encoding="utf-8-sig")

    if not all(np.isnan(y_test)):
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        rmse = mean_squared_error(y_test, pred, squared=False)
        pearson_corr, _ = pearsonr(y_test, pred)
    else:
        r2 = mae = rmse = pearson_corr = np.nan

    print(f"\nSVR {params['name']}: {params['test_start']} ~ {params['test_end']}")
    print('  R2:', r2)
    print('  MAE:', mae)
    print('  RMSE:', rmse)
    print('  Pearson correlation:', pearson_corr)

    sns.set(style="whitegrid")
    x_dates = pd.to_datetime(ts_test)
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.22)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax1.plot(x_dates, y_test, label="Observed", marker='o', color="#0072B2", linewidth=2)
    ax1.plot(x_dates, pred, label="SVR prediction", marker='x', color="#D55E00", linewidth=2)
    ax1.fill_between(x_dates, lower, upper, color='gray', alpha=0.2, label="Residual-based uncertainty interval")
    ax1.set_ylabel(TARGET, fontproperties=myfont)
    ax1.set_title(f"Prediction interval {params['test_start']} - {params['test_end']}: observed and SVR predictions with uncertainty interval", fontproperties=myfont)
    ax1.legend(frameon=True, fontsize=12, prop=myfont)
    ax1.set_xticks(x_dates[::max(1, len(x_dates)//8)])
    ax1.tick_params(axis='x', rotation=45)

    residuals = y_test - pred
    bar = ax2.bar(x_dates, residuals, color="#F0E442", alpha=0.65, edgecolor="k", width=6)
    ax2.plot(x_dates, residuals, color='#009E73', marker='o', linewidth=2, label="Residual")
    ax2.axhline(residuals.mean(), color='r', linestyle='--', alpha=0.7, label='Mean residual')
    ax2.axhline(0, color='k', linestyle=':', alpha=0.7)
    ax2.set_ylabel("Residual (observed - predicted)", fontproperties=myfont)
    ax2.set_title('Weekly prediction residuals', fontproperties=myfont)
    ax2.set_xticks(x_dates[::max(1, len(x_dates)//8)])
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(frameon=True, fontsize=12, prop=myfont)
    plt.subplots_adjust(hspace=0.22)
    plt.show()

    results.append({
        "period": f"{params['test_start']} ~ {params['test_end']}",
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Pearson": pearson_corr
    })

print("\n===== SVR comparison summary =====")
for r in results:
    print(f"{r['period']}: R2={r['R2']:.4f}, MAE={r['MAE']:.4f}, RMSE={r['RMSE']:.4f}, Pearson correlation={r['Pearson']:.4f}")