import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.stats import pearsonr
import tensorflow as tf
import seaborn as sns

os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

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


DEFAULT_WINDOW_SIZE = 3


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


def make_sequence_samples_full_window(df, window, target_times):
    features, targets, times = [], [], []
    df = df.reset_index(drop=True)
    time_idx_map = {row['time']: idx for idx, row in df.iterrows()}
    for t in target_times:
        idx = time_idx_map.get(t)
        if idx is not None and idx >= window-1:
            f = []
            for w in range(window):
                f.append(df.iloc[idx-window+1+w][BASE_CLIMATE + BASE_SOCIAL + BASE_EPIDEMIOLOGY].values)
            features.append(f)
            targets.append(df.iloc[idx][TARGET])  # Use the original target
            times.append(df.iloc[idx]['time'])
    return np.array(features), np.array(targets), np.array(times)

period_params = [
    {"train_start": "2011-01-05", "train_end": "2018-09-26", "test_start": "2018-10-03", "test_end": "2019-04-03", "window": DEFAULT_WINDOW_SIZE, "name": "2018-2019", "lstm_units": 64, "epochs": 100},
    {"train_start": "2011-01-05", "train_end": "2022-10-26", "test_start": "2022-11-02", "test_end": "2023-04-05", "window": DEFAULT_WINDOW_SIZE, "name": "2022-2023", "lstm_units": 64, "epochs": 100},
    {"train_start": "2011-01-05", "train_end": "2023-09-27", "test_start": "2023-10-04", "test_end": "2024-04-03", "window": 10, "name": "2023-2024", "lstm_units": 220, "epochs": 100, "lstm_units2": 128, "dropout_rate": 0.5, "dense_units": 128},
    {"train_start": "2011-01-05", "train_end": "2024-05-01", "test_start": "2024-10-02", "test_end": "2025-04-02", "window": 10, "name": "2024-2025_scenario", "lstm_units": 220, "epochs": 100, "lstm_units2": 128, "dropout_rate": 0.5, "dense_units": 128}
]

df = pd.read_csv("2011-2024AH1N1预测数据(返修).csv", parse_dates=['time'])

BASE_EPIDEMIOLOGY = get_epi_cols_for_lead(
    df,
    lead=FORECAST_LEAD,
    n_epi_inputs=N_EPI_INPUTS
)

df = df[df['time_part'] != 'covid']
df = build_features(df)
raw = df.sort_values('time').reset_index(drop=True)
results = []

for params in period_params:
    window = params.get('window', DEFAULT_WINDOW_SIZE)
    lstm_units = params.get('lstm_units', 64)
    lstm_units2 = params.get('lstm_units2', 0)
    epochs = params.get('epochs', 100)
    batch_size = 8
    dropout_rate = params.get('dropout_rate', 0.5)
    dense_units = params.get('dense_units', 0)

    train_mask = (raw['time'] >= pd.to_datetime(params["train_start"])) & (raw['time'] < pd.to_datetime(params["train_end"]))
    test_mask  = (raw['time'] >= pd.to_datetime(params["test_start"])) & (raw['time'] <= pd.to_datetime(params["test_end"]))

    df_train = raw[train_mask].reset_index(drop=True)
    df_test  = raw[test_mask].reset_index(drop=True)

    # Prepend historical observations to complete the input window
    df_all = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    target_times_train = df_train['time'].tolist()
    target_times_test = df_test['time'].tolist()
    X_train_seq, y_train_seq, ts_train_seq = make_sequence_samples_full_window(df_train, window=window, target_times=target_times_train)
    X_test_seq, y_test_seq, ts_test_seq = make_sequence_samples_full_window(df_all, window=window, target_times=target_times_test)
    n_features = X_train_seq.shape[2]

    if X_test_seq.shape[0] == 0:
        print(f"Period {params['name']} has insufficient samples and will be skipped.")
        continue

    X_train_seq_2d = X_train_seq.reshape(-1, n_features)
    X_test_seq_2d  = X_test_seq.reshape(-1, n_features)
    scaler = StandardScaler()
    X_train_seq_2d = scaler.fit_transform(X_train_seq_2d)
    X_test_seq_2d  = scaler.transform(X_test_seq_2d)
    X_train_seq = X_train_seq_2d.reshape(X_train_seq.shape)
    X_test_seq  = X_test_seq_2d.reshape(X_test_seq.shape)

    tf.keras.backend.clear_session()
    if lstm_units2 > 0:
        lstm = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(window, n_features)),
            tf.keras.layers.LSTM(lstm_units, return_sequences=True),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.LSTM(lstm_units2),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(dense_units, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
    else:
        lstm = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(window, n_features)),
            tf.keras.layers.LSTM(lstm_units),
            tf.keras.layers.Dense(1)
        ])
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size, verbose=0)
    pred_lstm = lstm.predict(X_test_seq).reshape(-1)

    train_pred = lstm.predict(X_train_seq).reshape(-1)
    stderr = (y_train_seq - train_pred).std(ddof=1)
    ci95 = 1.96 * stderr
    lower = pred_lstm - ci95
    upper = pred_lstm + ci95
    lower = np.maximum(lower, 0)
    lower = np.nan_to_num(lower, nan=0.0)
    upper = np.nan_to_num(upper, nan=0.0)
    print(f"Period {params['name']} Mean bounds of the residual-based uncertainty interval: [{lower.mean():.4f}, {upper.mean():.4f}]")

    pred_df = pd.DataFrame({
        "date": pd.to_datetime(ts_test_seq),
        "y_true": y_test_seq if not all(np.isnan(y_test_seq)) else [np.nan] * len(pred_lstm),
        "y_pred": pred_lstm,
        "LSTM_pred_lower": lower,
        "LSTM_pred_upper": upper
    })
    pred_df.to_csv(f"revised_LSTM_preds_{params['name']}.csv", index=False, encoding="utf-8-sig")

    if not all(np.isnan(y_test_seq)):
        r2 = r2_score(y_test_seq, pred_lstm)
        mae = mean_absolute_error(y_test_seq, pred_lstm)
        rmse = mean_squared_error(y_test_seq, pred_lstm, squared=False)
        pearson_corr, _ = pearsonr(y_test_seq, pred_lstm)
    else:
        r2 = mae = rmse = pearson_corr = np.nan

    print(f"\nLSTM {params['name']}: {params['test_start']} ~ {params['test_end']}")
    print('  R2:', r2)
    print('  MAE:', mae)
    print('  RMSE:', rmse)
    print('  Pearson correlation:', pearson_corr)

    sns.set(style="whitegrid")
    x_dates = pd.to_datetime(ts_test_seq)
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.22)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax1.plot(x_dates, y_test_seq, label="Observed", marker='o', color="#0072B2", linewidth=2)
    ax1.plot(x_dates, pred_lstm, label="LSTM prediction", marker='x', color="#D55E00", linewidth=2)
    ax1.fill_between(x_dates, lower, upper, color='gray', alpha=0.2, label="95% residual-based uncertainty interval")
    ax1.set_ylabel(TARGET, fontproperties=myfont)
    ax1.set_title(f"Prediction interval {params['test_start']} - {params['test_end']} Observed and LSTM predictions with uncertainty interval", fontproperties=myfont)
    ax1.legend(frameon=True, fontsize=12, prop=myfont)
    ax1.set_xticks(x_dates[::max(1, len(x_dates)//8)])
    ax1.tick_params(axis='x', rotation=45)

    residuals = y_test_seq - pred_lstm
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

print("\n===== LSTM comparison summary =====")
for r in results:
    print(f"{r['period']}: R2={r['R2']:.4f}, MAE={r['MAE']:.4f}, RMSE={r['RMSE']:.4f}, Pearson correlation={r['Pearson']:.4f}")