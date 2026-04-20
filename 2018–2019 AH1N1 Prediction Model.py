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
TARGET = 'H1N1_ILI%×positive%_national'
PREDICT_START = "2018-10-03"
PREDICT_END = "2019-04-03"
EPOCHS = 100
BATCH_SIZE = 16
SEED = 42

def moving_average(series, window_size):
    return series.rolling(window_size, min_periods=1).mean()

def build_features(df):
    window_size = 4
    df['TARGET_smoothed'] = moving_average(df[TARGET], window_size)
    for col in BASE_CLIMATE:
        df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(df[col].mean())
    for col in BASE_SOCIAL:
        df[col] = df[col].astype('category').cat.codes.fillna(0)
    if "time_part" not in df.columns:
        df["time_part"] = np.where(df["time"].dt.year < 2018, "pre",
                                   np.where(df["time"].dt.year < 2020, "covid", "after"))
    df = df.dropna(subset=BASE_EPIDEMIOLOGY + BASE_CLIMATE + BASE_SOCIAL)
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
    stats = df.groupby('season_group')[TARGET].agg(['mean','var'])
    abnormal = stats[(stats['mean'] < 0.01) & (stats['var'] < 0.001)].index.tolist()
    return abnormal

def load_train_data():
    df = pd.read_csv("2011-2024AH1N1预测数据.csv", parse_dates=["time"])
    df = build_features(df)
    df = assign_season_groups(df)
    mask = (df["time_part"] != "covid") & (df["time"] < pd.to_datetime(PREDICT_START))
    df_train = df[mask].reset_index(drop=True)
    df_train = df_train[df_train['season_group'] > 0].reset_index(drop=True)
    abnormal = abnormal_season_groups(df_train)
    df_train = df_train[~df_train['season_group'].isin(abnormal)].reset_index(drop=True)
    print(f"Removed anomalous epidemic seasons: {abnormal}")
    return df_train

def load_full_data():
    df = pd.read_csv("2011-2024AH1N1预测数据.csv", parse_dates=["time"])
    df = build_features(df)
    df = assign_season_groups(df)
    return df

def get_inputs(df):
    X_epi = df[BASE_EPIDEMIOLOGY].values.astype(np.float32)
    X_climate = df[BASE_CLIMATE].values.astype(np.float32)
    X_social = df[BASE_SOCIAL].values.astype(np.int32)
    y = df[TARGET].values
    return X_epi, X_climate, X_social, y

def build_multimodal_model():
    inp_epi = Input(shape=(len(BASE_EPIDEMIOLOGY),), name='epi_input')
    x_epi = Dense(16, activation='relu')(inp_epi)
    inp_climate = Input(shape=(len(BASE_CLIMATE),), name='climate_input')
    x_climate = Dense(16, activation='relu')(inp_climate)
    inp_social = Input(shape=(len(BASE_SOCIAL),), name='social_input')
    x_social = Embedding(input_dim=4, output_dim=2, input_length=len(BASE_SOCIAL))(inp_social)
    x_social = Flatten()(x_social)
    x_social = Dense(8, activation='relu')(x_social)
    x = concatenate([x_epi, x_climate, x_social])
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.1)(x)
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
            print(f"Fold {fold+1}: skipped anomalous epidemic season {val_group}")
            continue
        print(f"Fold {fold+1}: train seasons:", np.unique(groups[train_idx]), "val season:", val_group)
        scaler_epi = StandardScaler().fit(X_epi[train_idx])
        scaler_climate = StandardScaler().fit(X_climate[train_idx])
        Xtr_e, Xval_e = scaler_epi.transform(X_epi[train_idx]), scaler_epi.transform(X_epi[val_idx])
        Xtr_c, Xval_c = scaler_climate.transform(X_climate[train_idx]), scaler_climate.transform(X_climate[val_idx])
        Xtr_s, Xval_s = X_social[train_idx], X_social[val_idx]
        ytr, yval = y[train_idx], y[val_idx]
        model = build_multimodal_model()
        early_stopping = get_early_stopping(validation=True)
        model.fit([Xtr_e, Xtr_c, Xtr_s], ytr,
                  validation_data=([Xval_e, Xval_c, Xval_s], yval),
                  epochs=EPOCHS, batch_size=BATCH_SIZE,
                  callbacks=[early_stopping],
                  verbose=0)
        y_pred = model.predict([Xval_e, Xval_c, Xval_s], batch_size=BATCH_SIZE).reshape(-1)
        r2, mae, rmse = calc_metrics(yval, y_pred)
        print(f"  R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
        r2_list.append(r2)
        mae_list.append(mae)
        rmse_list.append(rmse)
        if plot_each_fold:
            plt.figure(figsize=(10,4))
            plt.plot(yval, label='Observed')
            plt.plot(y_pred, label='Predicted')
            plt.title(f'Fold {fold+1} Observed vs Predicted', fontproperties=myfont)
            plt.legend(prop=myfont)
            plt.show()
    r2_arr = np.array(r2_list)
    mean_r2 = r2_arr.mean() if len(r2_arr) else float('nan')
    std_r2 = r2_arr.std(ddof=1) if len(r2_arr) > 1 else float('nan')
    se_r2 = std_r2 / np.sqrt(len(r2_arr)) if len(r2_arr) > 1 else float('nan')
    tval = t.ppf(0.975, len(r2_arr)-1) if len(r2_arr) > 1 else float('nan')
    ci_lower = mean_r2 - tval * se_r2 if len(r2_arr) > 1 else float('nan')
    ci_upper = mean_r2 + tval * se_r2 if len(r2_arr) > 1 else float('nan')
    print("\n====== Cross-validation summary by epidemic season ======")
    print(f"Mean R2: {mean_r2:.4f}")
    print(f"Standard deviation of R2: {std_r2:.4f}")
    print(f"95%% confidence interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    for i, (r2, mae, rmse) in enumerate(zip(r2_list, mae_list, rmse_list), 1):
        print(f"  Fold {i}: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    return mean_r2, std_r2, (ci_lower, ci_upper)

def save_predict_result_csv(y_pred, lower, upper, y_true, dates, filename):
    df_save = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "y_true": y_true if y_true is not None else [np.nan]*len(y_pred),
        "y_pred": y_pred,
        "lower": lower,
        "upper": upper
    })
    df_save.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"Prediction results have been saved to {filename}")

def train_final_and_predict(train_df, full_df, predict_start, predict_end, plot=True):
    X_epi, X_climate, X_social, y = get_inputs(train_df)
    scaler_epi = StandardScaler().fit(X_epi)
    scaler_climate = StandardScaler().fit(X_climate)
    X_epi_scaled = scaler_epi.transform(X_epi)
    X_climate_scaled = scaler_climate.transform(X_climate)
    model = build_multimodal_model()
    print("Model architecture:")
    model.summary()
    print(f"Key parameters: EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, random seed={SEED}")
    early_stopping = get_early_stopping(validation=False)
    model.fit([X_epi_scaled, X_climate_scaled, X_social], y,
              epochs=EPOCHS, batch_size=BATCH_SIZE,
              callbacks=[early_stopping],
              verbose=1)
    full_df["date"] = pd.to_datetime(full_df["time"]).dt.tz_localize(None)
    mask = (full_df["date"] >= pd.to_datetime(predict_start)) & (full_df["date"] <= pd.to_datetime(predict_end)) \
           & (full_df["time_part"] != "covid") & (full_df["season_group"] > 0)
    df_predict = full_df.loc[mask].copy()
    abnormal = abnormal_season_groups(df_predict)
    df_predict = df_predict[~df_predict['season_group'].isin(abnormal)].reset_index(drop=True)
    if len(df_predict) == 0:
        print("No valid epidemic season is available in the prediction interval. No forecast will be produced.")
        return None, None, None, None, None
    X_epi_p, X_climate_p, X_social_p, y_p = get_inputs(df_predict)
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
    r2 = r2_score(y_p, y_pred)
    mae = mean_absolute_error(y_p, y_pred)
    rmse = np.sqrt(mean_squared_error(y_p, y_pred))
    print(f"\nPerformance metrics for the prediction interval {predict_start} - {predict_end}:")
    print(f"R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    print("Mean 95%% prediction interval: [%.4f, %.4f]" % (lower.mean(), upper.mean()))
    residual_pred = y_p - y_pred
    print("Residual mean: %.4f, standard deviation: %.4f" % (residual_pred.mean(), residual_pred.std()))
    corr = np.corrcoef(y_p, y_pred)[0, 1]
    print("Pearson correlation between observed and predicted values (trend correlation): %.4f" % corr)
    # Save CSV
    save_predict_result_csv(y_pred, lower, upper, y_p, df_predict["date"], f"Multimodal_preds_{predict_start}_{predict_end}.csv")
    if plot:
        import seaborn as sns
        sns.set(style="whitegrid")
        df_predict["date"] = pd.to_datetime(df_predict["date"]).dt.tz_localize(None)
        x_dates = df_predict["date"]
        fig = plt.figure(figsize=(13, 9))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.22)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        ax1.plot(x_dates, y_p, label="Observed", marker='o', color="#0072B2", linewidth=2)
        ax1.plot(x_dates, y_pred, label="Predicted", marker='x', color="#D55E00", linewidth=2)
        ax1.fill_between(x_dates, lower, upper, color='gray', alpha=0.2, label="95% prediction interval")
        ax1.set_ylabel(TARGET, fontproperties=myfont)
        ax1.set_title(f"Observed and predicted values with 95% prediction intervals for {predict_start} - {predict_end}",
                      fontproperties=myfont)
        ax1.legend(frameon=True, fontsize=12, prop=myfont)
        ax1.set_xticks(x_dates[::max(1,len(x_dates)//8)])
        ax1.tick_params(axis='x', rotation=45)

        bar = ax2.bar(x_dates, residual_pred, color="#F0E442", alpha=0.65, edgecolor="k", width=6)
        ax2.plot(x_dates, residual_pred, color='#009E73', marker='o', linewidth=2, label="Error")
        ax2.axhline(residual_pred.mean(), color='r', linestyle='--', alpha=0.7, label='Mean error')
        ax2.axhline(0, color='k', linestyle=':', alpha=0.7)
        ax2.set_ylabel("Error (observed-predicted)", fontproperties=myfont)
        ax2.set_title('Weekly prediction error', fontproperties=myfont)
        ax2.set_xticks(x_dates[::max(1,len(x_dates)//8)])
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(frameon=True, fontsize=12, prop=myfont)
        for rect in bar:
            height = rect.get_height()
            if np.abs(height) > 0.15:
                ax2.text(rect.get_x() + rect.get_width()/2.0, height, f'{height:.2f}',
                         ha='center', va='bottom' if height>=0 else 'top', fontsize=8, color='#333', fontproperties=myfont)
        plt.subplots_adjust(hspace=0.22)
        plt.show()
        # Save model and interpretation inputs
    model.save('multimodal_model_2018_2019.h5')
    np.savez('multimodal_inputs_2018_2019.npz',
             X_epi_p=X_epi_p,
             X_climate_p=X_climate_p,
             X_social_p=X_social_p,
             y_p=y_p)
    return y_pred, lower, upper, y_p, df_predict["date"]

if __name__ == "__main__":
    print("Loading training data (excluding anomalous epidemic seasons)...")
    train_df = load_train_data()
    if train_df.empty:
        print("No valid epidemic season remains in the training set. Cannot continue.")
        sys.exit()
    print("Running grouped cross-validation by epidemic season (excluding anomalous epidemic seasons)...")
    group_cv(train_df, plot_each_fold=False)
    print("Loading the full dataset...")
    full_df = load_full_data()
    if full_df.empty:
        print("The full dataset is empty. Prediction cannot proceed.")
        sys.exit()
    print("Forecasting the 2018-2019 epidemic season (excluding anomalous epidemic seasons)...")
    train_final_and_predict(train_df, full_df, PREDICT_START, PREDICT_END, plot=True)