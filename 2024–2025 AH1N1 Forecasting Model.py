# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, concatenate, Dropout
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from scipy.stats import t
import matplotlib.pyplot as plt
import sys
from matplotlib import font_manager
import matplotlib.dates as mdates
import seaborn as sns


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(42)

font_path = "C:/Windows/Fonts/msyh.ttc"
myfont = font_manager.FontProperties(fname=font_path)
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_CLIMATE = [
    'absolute_humidity', 'uvindex', 'solarenergy', 'solarradiation', 'visibility',
    'cloudcover', 'winddir', 'windspeed', 'humidity', 'feelslike', 'temp', 'DTR'
]
BASE_SOCIAL = ['holiday', 'trend_stage', 'H1N1_flu_season']
BASE_EPIDEMIOLOGY = [
    'H1N1_ILI%×positive%_national_lag_1_week',
    'H1N1_ILI%×positive%_national_lag_2_week',
    'H1N1_ILI%×positive%_national_lag_3_week',
    'H1N1_ILI%×positive%_national_lag_4_week'
]
STAGE_COL = 'time_part_code'

TARGET = 'H1N1_ILI%×positive%_national'
PREDICT_START = "2024-10-02"
PREDICT_END = "2025-04-02"
TRAIN_END = "2024-05-01"

EPOCHS = 100
BATCH_SIZE = 16
SEED = 42


def moving_average(series, window_size):
    return series.rolling(window_size, min_periods=1).mean()


def build_features(df, drop_target=True):
    window_size = 4
    if TARGET in df.columns:
        df['TARGET_smoothed'] = moving_average(df[TARGET], window_size)
    for col in BASE_CLIMATE:
        df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(df[col].mean())
    for col in BASE_SOCIAL:
        df[col] = df[col].astype('category').cat.codes.fillna(0)
    if "time_part" not in df.columns:
        df["time_part"] = np.where(df["time"].dt.year < 2020, "pre", "after")
    df[STAGE_COL] = df["time_part"].astype("category").cat.codes
    drop_cols = BASE_EPIDEMIOLOGY + BASE_CLIMATE + BASE_SOCIAL
    if drop_target and TARGET in df.columns:
        drop_cols = [TARGET] + drop_cols
    df = df.dropna(subset=drop_cols)
    return df


def assign_season_groups(df):
    df = df.sort_values('time').reset_index(drop=True)
    season_group = np.zeros(len(df), dtype=int)
    group = 0
    in_season = False
    for i in range(len(df)):
        if df.loc[i, 'H1N1_flu_season'] == 1:
            if not in_season:
                group += 1
                in_season = True
            season_group[i] = group
        else:
            in_season = False
    df['season_group'] = season_group
    return df


def is_anomaly_season(stat_mean, stat_var, mean_thres=0.01, var_thres=0.001):
    return (stat_mean < mean_thres) and (stat_var < var_thres)


def abnormal_season_groups(df):
    stats = df.groupby('season_group')[TARGET].agg(['mean', 'var'])
    abnormal = stats[(stats['mean'] < 0.01) & (stats['var'] < 0.001)].index.tolist()
    return abnormal


def load_train_data():
    df = pd.read_csv("2011-2024AH1N1预测数据.csv", parse_dates=["time"])
    df = build_features(df, drop_target=True)
    df = assign_season_groups(df)
    df = df[(df["time"] < pd.to_datetime(PREDICT_START)) & (df["time"] <= pd.to_datetime(TRAIN_END))]
    df_train = df[df['season_group'] > 0].reset_index(drop=True)
    abnormal = abnormal_season_groups(df_train)
    df_train = df_train[~df_train['season_group'].isin(abnormal)].reset_index(drop=True)
    print(f"Removed anomalous epidemic seasons: {abnormal}")
    return df_train


def load_full_data():
    df = pd.read_csv("2011-2024AH1N1预测数据.csv", parse_dates=["time"])
    df = build_features(df, drop_target=False)
    df = assign_season_groups(df)
    return df


def get_inputs(df):
    epi_cols = BASE_EPIDEMIOLOGY
    climate_cols = BASE_CLIMATE + [STAGE_COL]
    social_cols = BASE_SOCIAL
    X_epi = df[epi_cols].values.astype(np.float32)
    X_climate = df[climate_cols].values.astype(np.float32)
    X_social = df[social_cols].values.astype(np.int32)
    y = df[TARGET].values if TARGET in df.columns else np.zeros(len(df))
    return X_epi, X_climate, X_social, y


def build_multimodal_model():
    inp_epi = Input(shape=(len(BASE_EPIDEMIOLOGY),), name='epi_input')
    x_epi = Dense(16, activation='relu')(inp_epi)
    inp_climate = Input(shape=(len(BASE_CLIMATE) + 1,), name='climate_input')
    x_climate = Dense(16, activation='relu')(inp_climate)
    inp_social = Input(shape=(len(BASE_SOCIAL),), name='social_input')
    x_social = Embedding(input_dim=4, output_dim=4, input_length=len(BASE_SOCIAL))(inp_social)
    x_social = Flatten()(x_social)
    x_social = Dense(8, activation='relu')(x_social)
    x = concatenate([x_epi, x_climate, x_social])
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='relu')(x)
    model = Model(inputs=[inp_epi, inp_climate, inp_social], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss='mse',
        metrics=['mae']
    )
    return model


def calc_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse


def get_early_stopping(validation=False):
    if validation:
        return tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, monitor='val_loss'
        )
    else:
        return tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, monitor='loss'
        )


def get_sample_weight(df):
    return np.where(df['time_part'] == 'after', 2.0, 1.0)


def group_cv(df, plot_each_fold=False):
    X_epi, X_climate, X_social, y = get_inputs(df)
    groups = df['season_group'].values
    unique_groups = np.unique(groups)
    n_splits = len(unique_groups)
    gkf = GroupKFold(n_splits=n_splits)
    r2_list, mae_list, rmse_list = [], [], []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_epi, groups=groups)):
        val_group = np.unique(groups[val_idx])[0]
        val_df = df.iloc[val_idx]
        val_mean = val_df[TARGET].mean()
        val_var = val_df[TARGET].var()
        if is_anomaly_season(val_mean, val_var):
            print(f"Fold {fold + 1}: skipped anomalous epidemic season {val_group}")
            continue
        print(f"Fold {fold + 1}: train seasons:", np.unique(groups[train_idx]), "validation season:", val_group)
        scaler_epi = StandardScaler().fit(X_epi[train_idx])
        scaler_climate = StandardScaler().fit(X_climate[train_idx])
        Xtr_e, Xval_e = scaler_epi.transform(X_epi[train_idx]), scaler_epi.transform(X_epi[val_idx])
        Xtr_c, Xval_c = scaler_climate.transform(X_climate[train_idx]), scaler_climate.transform(X_climate[val_idx])
        Xtr_s, Xval_s = X_social[train_idx], X_social[val_idx]
        ytr, yval = y[train_idx], y[val_idx]
        sample_weight = get_sample_weight(df.iloc[train_idx])
        model = build_multimodal_model()
        early_stopping = get_early_stopping(validation=True)
        model.fit(
            [Xtr_e, Xtr_c, Xtr_s], ytr,
            validation_data=([Xval_e, Xval_c, Xval_s], yval),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping],
            verbose=0,
            sample_weight=sample_weight
        )
        y_pred = model.predict([Xval_e, Xval_c, Xval_s], batch_size=BATCH_SIZE).reshape(-1)
        r2, mae, rmse = calc_metrics(yval, y_pred)
        print(f"  R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
        r2_list.append(r2)
        mae_list.append(mae)
        rmse_list.append(rmse)
        if plot_each_fold:
            plt.figure(figsize=(10, 4))
            plt.plot(yval, label='Observed')
            plt.plot(y_pred, label='Predicted')
            plt.title(f'Fold {fold + 1}: Observed vs Predicted', fontproperties=myfont)
            plt.legend(prop=myfont)
            plt.show()

    r2_arr = np.array(r2_list)
    mean_r2 = r2_arr.mean() if len(r2_arr) else float('nan')
    std_r2 = r2_arr.std(ddof=1) if len(r2_arr) > 1 else float('nan')
    se_r2 = std_r2 / np.sqrt(len(r2_arr)) if len(r2_arr) > 1 else float('nan')
    tval = t.ppf(0.975, len(r2_arr) - 1) if len(r2_arr) > 1 else float('nan')
    ci_lower = mean_r2 - tval * se_r2 if len(r2_arr) > 1 else float('nan')
    ci_upper = mean_r2 + tval * se_r2 if len(r2_arr) > 1 else float('nan')

    print("\n====== Cross-validation summary by epidemic season ======")
    print(f"Mean R2: {mean_r2:.4f}")
    print(f"Standard deviation of R2: {std_r2:.4f}")
    print(f"95% confidence interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    for i, (r2, mae, rmse) in enumerate(zip(r2_list, mae_list, rmse_list), 1):
        print(f"  Fold {i}: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    return mean_r2, std_r2, (ci_lower, ci_upper)


def plot_forecast_with_ci(dates, y_pred, lower, upper, target_name, fontproperties, save_path=None):
    sns.set(style="whitegrid", font_scale=1.15)
    plt.figure(figsize=(13, 6))
    plt.plot(
        dates,
        y_pred,
        label="Predicted value",
        color="#D55E00",
        linewidth=2.5,
        marker='o',
        markersize=5,
        zorder=10
    )
    plt.fill_between(
        dates,
        lower,
        upper,
        color="#0072B2",
        alpha=0.18,
        label="95% prediction interval",
        zorder=2
    )
    plt.ylabel(target_name, fontproperties=fontproperties, fontsize=16)
    plt.xlabel("Date", fontproperties=fontproperties, fontsize=15)
    plt.title(
        "Prospective H1N1 forecast for the 2024–2025 epidemic season with 95% prediction intervals",
        fontproperties=fontproperties,
        fontsize=18,
        pad=15
    )
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    plt.xticks(fontsize=12, fontproperties=fontproperties, rotation=30)
    plt.yticks(fontsize=12, fontproperties=fontproperties)
    ax.grid(which='major', color='#e5e5e5', linewidth=1.2)
    ax.grid(which='minor', color='#f5f5f5', linewidth=0.8, linestyle='--')
    plt.legend(frameon=True, fontsize=13, prop=fontproperties, loc='upper right')
    sns.despine()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.show()


def train_final_and_predict(train_df, full_df, predict_start, predict_end, plot=True):
    X_epi, X_climate, X_social, y = get_inputs(train_df)
    scaler_epi = StandardScaler().fit(X_epi)
    scaler_climate = StandardScaler().fit(X_climate)
    X_epi_scaled = scaler_epi.transform(X_epi)
    X_climate_scaled = scaler_climate.transform(X_climate)
    sample_weight = get_sample_weight(train_df)
    model = build_multimodal_model()

    print("Model architecture:")
    model.summary()
    print(f"Key parameters: EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, random seed={SEED}")

    early_stopping = get_early_stopping(validation=False)
    model.fit(
        [X_epi_scaled, X_climate_scaled, X_social], y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1,
        sample_weight=sample_weight
    )

    df_predict = full_df.copy()
    X_epi_p, X_climate_p, X_social_p, _ = get_inputs(df_predict)
    X_epi_p = scaler_epi.transform(X_epi_p)
    X_climate_p = scaler_climate.transform(X_climate_p)
    y_pred = model.predict([X_epi_p, X_climate_p, X_social_p], batch_size=BATCH_SIZE).reshape(-1)
    y_all_pred = model.predict([X_epi_scaled, X_climate_scaled, X_social], batch_size=BATCH_SIZE).reshape(-1)
    residuals = y - y_all_pred
    stderr = residuals.std(ddof=1)
    ci95 = 1.96 * stderr
    lower = y_pred - ci95
    lower = np.maximum(lower, 0)
    upper = y_pred + ci95

    df_predict["date"] = pd.to_datetime(df_predict["time"]).dt.tz_localize(None)
    mask = (df_predict["date"] >= pd.to_datetime(predict_start)) & (df_predict["date"] <= pd.to_datetime(predict_end))
    df_predict = df_predict.loc[mask].reset_index(drop=True)
    y_pred_selected = y_pred[mask.values]
    lower_selected = lower[mask.values]
    upper_selected = upper[mask.values]

    if len(df_predict) == 0 or len(y_pred_selected) == 0:
        print("No data are available in the specified interval; plotting is not possible.")
        return None, None, None, None

    print("\nWeek-by-week forecasts for the 2024–2025 epidemic season with 95% prediction intervals:")
    for dt, pred, l, u in zip(df_predict["date"], y_pred_selected, lower_selected, upper_selected):
        print(f"{dt.strftime('%Y-%m-%d')}: predicted value={pred:.4f}, lower bound={l:.4f}, upper bound={u:.4f}")

    print("\nMean prediction interval: lower bound=%.4f, upper bound=%.4f" % (lower_selected.mean(), upper_selected.mean()))
    print("Mean predicted value: %.4f" % y_pred_selected.mean())

    # Save unsmoothed forecasts to CSV
    save_df = pd.DataFrame({
        "date": df_predict["date"].values,
        "y_pred": y_pred_selected,
        "ci_lower": lower_selected,
        "ci_upper": upper_selected
    })
    save_df.to_csv("forecast_2024-2025_raw.csv", index=False, encoding="utf-8-sig")
    print("\nUnsmoothed forecast results have been saved to forecast_2024-2025_raw.csv")

    # Apply 3-point moving-average smoothing to forecasts and prediction intervals
    window_size = 3
    y_pred_smoothed = pd.Series(y_pred_selected).rolling(window=window_size, min_periods=1).mean().values
    lower_smoothed = pd.Series(lower_selected).rolling(window=window_size, min_periods=1).mean().values
    upper_smoothed = pd.Series(upper_selected).rolling(window=window_size, min_periods=1).mean().values

    if plot:
        plot_forecast_with_ci(
            df_predict["date"],
            y_pred_smoothed,
            lower_smoothed,
            upper_smoothed,
            TARGET,
            myfont,
            save_path="forecast_2024-2025.png"
        )
    return y_pred_smoothed, lower_smoothed, upper_smoothed, df_predict["date"]


if __name__ == "__main__":
    print("Loading training data (excluding anomalous epidemic seasons, excluding the COVID period, up to 2024/05/01)...")
    train_df = load_train_data()
    if train_df.empty:
        print("No valid epidemic seasons remain in the training set. Cannot continue.")
        sys.exit()

    print("Running grouped cross-validation by epidemic season (excluding anomalous epidemic seasons)...")
    group_cv(train_df, plot_each_fold=False)

    print("Loading scenario-simulated feature data for 2024–2025...")
    full_df = load_full_data()
    if full_df.empty:
        print("The scenario-simulated dataset is empty. Prediction cannot proceed.")
        sys.exit()

    print("Generating the scenario-simulated forecast curve for the 2024–2025 epidemic season...")
    y_pred, lower, upper, pred_dates = train_final_and_predict(
        train_df,
        full_df,
        PREDICT_START,
        PREDICT_END,
        plot=True
    )