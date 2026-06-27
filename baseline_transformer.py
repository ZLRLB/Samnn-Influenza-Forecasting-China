import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.stats import pearsonr
import seaborn as sns
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

myfont = fm.FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
plt.rcParams['axes.unicode_minus'] = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TARGET = 'H1N1_ILI%×positive%_national'
BASE_CLIMATE = [
    'absolute_humidity','uvindex','solarenergy','solarradiation',
    'visibility','cloudcover','winddir','windspeed','humidity',
    'feelslike','temp','DTR'
]
BASE_SOCIAL = ['holiday','trend_stage','H1N1_flu_season']
FORECAST_LEAD = 4  # Set to 1, 2, or 4 for the corresponding forecast horizon

df = pd.read_csv("2011-2024AH1N1预测数据(返修).csv", parse_dates=['time'])


def get_epi_cols_for_lead(lead):
    if lead == 1:
        return [
            'H1N1_ILI%×positive%_national_lag_1_week',
            'H1N1_ILI%×positive%_national_lag_2_week',
            'H1N1_ILI%×positive%_national_lag_3_week',
            'H1N1_ILI%×positive%_national_lag_4_week'
        ]
    elif lead == 2:
        return [
            'H1N1_ILI%×positive%_national_lag_2_week',
            'H1N1_ILI%×positive%_national_lag_3_week',
            'H1N1_ILI%×positive%_national_lag_4_week',
            'H1N1_ILI%×positive%_national_lag_5_week'
        ]
    elif lead == 4:
        return [
            'H1N1_ILI%×positive%_national_lag_4_week',
            'H1N1_ILI%×positive%_national_lag_5_week',
            'H1N1_ILI%×positive%_national_lag_6_week',
            'H1N1_ILI%×positive%_national_lag_7_week'
        ]
    else:
        raise ValueError("FORECAST_LEAD must be 1, 2, or 4")

BASE_EPIDEMIOLOGY = get_epi_cols_for_lead(FORECAST_LEAD)

print(f"\nCurrent configuration: {FORECAST_LEAD}-week ahead forecasting")
print("Epidemiological lag columns:")
for c in BASE_EPIDEMIOLOGY:
    print("  ", c)

ALL_VARS = BASE_CLIMATE + BASE_SOCIAL + BASE_EPIDEMIOLOGY

df = df[df['time_part'] != 'covid']
for feat in BASE_CLIMATE + BASE_EPIDEMIOLOGY:
    if feat in df.columns:
        df[feat] = df[feat].fillna(method='ffill').fillna(method='bfill')
for col in BASE_SOCIAL:
    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    df[col] = df[col].astype('category').cat.codes
raw = df.sort_values('time').reset_index(drop=True)

period_params = [
    {
        "train_start": "2011-01-05", "train_end": "2018-09-26",
        "test_start": "2018-10-03", "test_end": "2019-04-03", "name": "2018-2019",
        "model_params": {"d_model": 32, "nhead": 4, "num_layers": 2, "dropout": 0.1, "window_size": 12, "epochs": 30}
    },
    {
        "train_start": "2011-01-05", "train_end": "2022-10-26",
        "test_start": "2022-11-02", "test_end": "2023-04-05", "name": "2022-2023",
        "model_params": {"d_model": 32, "nhead": 4, "num_layers": 2, "dropout": 0.1, "window_size": 12, "epochs": 30}
    },
    {
        "train_start": "2011-01-05", "train_end": "2023-09-27",
        "test_start": "2023-10-04", "test_end": "2024-04-03", "name": "2023-2024",
        "model_params": {"d_model": 64, "nhead": 4, "num_layers": 2, "dropout": 0.1, "window_size": 12, "epochs": 30,
                         "lr": 0.007}
    },
    {
        "train_start": "2011-01-05", "train_end": "2024-05-01",
        "test_start": "2024-10-02", "test_end": "2025-04-02", "name": "2024-2025_scenario",
        "model_params": {"d_model": 32, "nhead": 4, "num_layers": 2, "dropout": 0.1, "window_size": 12, "epochs": 30}
    }
]

class SequenceDataset(Dataset):
    def __init__(self, df, input_cols, target_col, window_size):
        self.data = df
        self.input_cols = input_cols
        self.target_col = target_col
        self.window = window_size
        self.X, self.y = [], []
        for i in range(len(df) - window_size):
            self.X.append(df[input_cols].iloc[i:i+window_size].values)
            self.y.append(df[target_col].iloc[i+window_size])
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model=32, nhead=4, num_layers=2, dropout=0.1, window_size=12):
        super().__init__()
        self.input_linear = nn.Linear(input_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(window_size, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.input_linear(x) + self.pos_embedding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.fc(x).squeeze(-1)

def windowed_step_predict(model, test_with_history_df, input_cols, window_size, pred_steps):
    model.eval()
    preds = []
    for i in range(pred_steps):
        window = test_with_history_df[input_cols].iloc[i:i+window_size].values.astype(np.float32)
        window = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(window)
        preds.append(pred.item())
    return np.array(preds)

results = []

for idx, params in enumerate(period_params):
    mp = params["model_params"]
    window_size = mp["window_size"]

    train_mask = (raw['time'] >= pd.to_datetime(params["train_start"])) & (raw['time'] < pd.to_datetime(params["train_end"]))
    test_mask  = (raw['time'] >= pd.to_datetime(params["test_start"])) & (raw['time'] <= pd.to_datetime(params["test_end"]))
    train = raw[train_mask].reset_index(drop=True)
    test = raw[test_mask].reset_index(drop=True)

    if test.shape[0] < 1 or train.shape[0] < window_size:
        print(f"Period {params['name']} has insufficient samples and will be skipped.")
        continue

    # Prepend historical observations
    history = train.tail(window_size)
    test_with_history = pd.concat([history, test], ignore_index=True)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(train[ALL_VARS].values)
    train_inputs = scaler.transform(train[ALL_VARS].values)
    test_with_history_inputs = scaler.transform(test_with_history[ALL_VARS].values)
    train_df = train.copy()
    test_with_history_df = test_with_history.copy()
    train_df[ALL_VARS] = train_inputs
    test_with_history_df[ALL_VARS] = test_with_history_inputs

    train_dataset = SequenceDataset(train_df, ALL_VARS, TARGET, window_size)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = TimeSeriesTransformer(
        input_size=len(ALL_VARS),
        d_model=mp.get("d_model", 32),
        nhead=mp.get("nhead", 4),
        num_layers=mp.get("num_layers", 2),
        dropout=mp.get("dropout", 0.1),
        window_size=window_size
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=mp.get("lr", 0.002))
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(mp["epochs"]):
        epoch_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X)
        if epoch % 5 == 0:
            print(f"{params['name']} Epoch {epoch}, Loss {epoch_loss / len(train_loader.dataset):.5f}")

    pred_steps = test.shape[0]
    y_preds = windowed_step_predict(model, test_with_history_df, ALL_VARS, window_size, pred_steps)
    y_trues = test[TARGET].values
    ts_test = test['time'].values

    # ====== Residual-based interval calculation using sequential iteration over the training dataset======
    model.eval()
    train_preds_for_ci = []
    train_targets_for_ci = []
    with torch.no_grad():
        for i in range(len(train_dataset)):
            X, y = train_dataset[i]
            X = X.unsqueeze(0).to(DEVICE)  # batch_size=1
            pred = model(X).cpu().numpy()
            train_preds_for_ci.append(pred.item())
            train_targets_for_ci.append(y.item())
    train_preds_for_ci = np.array(train_preds_for_ci)
    train_targets_for_ci = np.array(train_targets_for_ci)
    stderr = (train_targets_for_ci - train_preds_for_ci).std(ddof=1)
    ci95 = 1.96 * stderr
    lower = y_preds - ci95
    upper = y_preds + ci95
    lower = np.maximum(lower, 0)
    lower = np.nan_to_num(lower, nan=0.0)
    upper = np.nan_to_num(upper, nan=0.0)

    r2 = r2_score(y_trues, y_preds)
    mae = mean_absolute_error(y_trues, y_preds)
    rmse = mean_squared_error(y_trues, y_preds, squared=False)
    pearson_corr, _ = pearsonr(y_trues, y_preds)

    print(f"\nTransformer sliding-window period {params['name']}: {params['test_start']} ~ {params['test_end']}")
    print('  R2:', r2)
    print('  MAE:', mae)
    print('  RMSE:', rmse)
    print('  Pearson correlation:', pearson_corr)
    print(f"Mean bounds of the residual-based uncertainty interval: [{lower.mean():.4f}, {upper.mean():.4f}]")

    pred_df = pd.DataFrame({
        "date": ts_test,
        "y_true": y_trues,
        "y_pred": y_preds,
        "Transformer_pred_lower": lower,
        "Transformer_pred_upper": upper
    })
    pred_df.to_csv(f"revised_Transformer_sliding_preds_{params['name']}.csv", index=False, encoding="utf-8-sig")

    sns.set(style="whitegrid")
    x_dates = pd.to_datetime(ts_test)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), height_ratios=[2, 1], sharex=True)

    ax1.plot(x_dates, y_trues, label="Observed", marker='o', color="#0072B2", linewidth=2)
    ax1.plot(x_dates, y_preds, label="Transformer sliding-window prediction", marker='x', color="#D55E00", linewidth=2)
    ax1.fill_between(x_dates, lower, upper, color='gray', alpha=0.2, label="95% residual-based uncertainty interval")
    ax1.set_ylabel(TARGET, fontproperties=myfont)
    ax1.set_title(f"Sliding-window prediction period {params['test_start']} - {params['test_end']}: observed and Transformer predictions", fontproperties=myfont)
    ax1.legend(frameon=True, fontsize=12, prop=myfont)
    ax1.set_xticks(x_dates[::max(1, len(x_dates)//8)])
    ax1.tick_params(axis='x', rotation=45)

    residuals = y_trues - y_preds
    bar = ax2.bar(x_dates, residuals, color="#F0E442", alpha=0.65, edgecolor="k", width=6)
    ax2.plot(x_dates, residuals, color='#009E73', marker='o', linewidth=2, label="Residual")
    ax2.axhline(residuals.mean(), color='r', linestyle='--', alpha=0.7, label='Mean residual')
    ax2.axhline(0, color='k', linestyle=':', alpha=0.7)
    ax2.set_ylabel("Residual (observed - predicted)", fontproperties=myfont)
    ax2.set_title('Weekly sliding-window prediction residuals', fontproperties=myfont)
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

print("\n===== Transformer sliding-window comparison summary =====")
for r in results:
    print(
        f"{r['period']}: R2={r['R2']:.4f}, MAE={r['MAE']:.4f}, RMSE={r['RMSE']:.4f}, Pearson correlation={r['Pearson']:.4f}")