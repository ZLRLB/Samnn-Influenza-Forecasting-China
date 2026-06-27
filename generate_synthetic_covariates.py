import os
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from scipy.stats import wasserstein_distance, entropy
from sklearn.manifold import TSNE
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# ==== Step 0. Set all random seeds ====
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==== Step 1. Load data and define features ====
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
ALL_FEATURES = BASE_CLIMATE + BASE_SOCIAL + BASE_EPIDEMIOLOGY

df = pd.read_csv('2011-2024AH1N1预测数据.csv')
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)
n_gen = 500

# ==== Step 2. Train WGAN-GP and sample climatic features ====
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )
    def forward(self, z):
        return self.model(z)

class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.model(x)

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    alpha = alpha.expand(real_samples.size())
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(real_samples.size(0), 1).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

climate_data = df[BASE_CLIMATE].astype(float).fillna(df[BASE_CLIMATE].mean())
scaler = MinMaxScaler()
climate_scaled = scaler.fit_transform(climate_data)
real_samples = torch.tensor(climate_scaled, dtype=torch.float32)

latent_dim = 32
output_dim = len(BASE_CLIMATE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
real_samples = real_samples.to(device)

G = Generator(latent_dim, output_dim).to(device)
D = Critic(output_dim).to(device)

batch_size = 128
epochs = 20000
lambda_gp = 10
lr = 1e-4
n_critic = 5

optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

# Store losses for diagnostic plots
loss_G_list = []
loss_D_list = []

for epoch in range(epochs):
    for _ in range(n_critic):
        idx = np.random.randint(0, climate_scaled.shape[0], batch_size)
        real_batch = torch.tensor(climate_scaled[idx], dtype=torch.float32).to(device)
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_batch = G(z)
        d_real = D(real_batch)
        d_fake = D(fake_batch)
        gradient_penalty = compute_gradient_penalty(D, real_batch, fake_batch, device)
        loss_D = -torch.mean(d_real) + torch.mean(d_fake) + lambda_gp * gradient_penalty
        optimizer_D.zero_grad()
        loss_D.backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=10)
        optimizer_D.step()
    # Train the generator
    z = torch.randn(batch_size, latent_dim).to(device)
    fake_batch = G(z)
    loss_G = -torch.mean(D(fake_batch))
    optimizer_G.zero_grad()
    loss_G.backward()
    torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=10)
    optimizer_G.step()
    # Record losses
    loss_G_list.append(loss_G.item())
    loss_D_list.append(loss_D.item())
    if torch.isnan(loss_D) or torch.isnan(loss_G):
        print(f"nan detected at epoch {epoch}, aborting training")
        break
    if epoch % 1000 == 0:
        print(f"WGAN-GP Epoch {epoch}/{epochs} | LossD: {loss_D.item():.4f} | LossG: {loss_G.item():.4f}")

with torch.no_grad():
    z = torch.randn(n_gen, latent_dim).to(device)
    gen_climate_scaled = G(z).cpu().numpy()
gen_climate = scaler.inverse_transform(gen_climate_scaled)
gen_climate_df = pd.DataFrame(gen_climate, columns=BASE_CLIMATE)

# ==== Step 3. Train CTGAN and sample social-context and epidemiological features ====
df_social = df[BASE_SOCIAL].copy()
metadata_social = SingleTableMetadata()
metadata_social.detect_from_dataframe(df_social)
social_ctgan = CTGANSynthesizer(metadata=metadata_social)
social_ctgan.fit(df_social)
gen_social_df = social_ctgan.sample(n_gen)

label_encoders = {}
social_data = df[BASE_SOCIAL].copy()
for col in BASE_SOCIAL:
    le = LabelEncoder()
    social_data[col] = le.fit_transform(social_data[col])
    label_encoders[col] = le

gen_social = pd.DataFrame()
for col in BASE_SOCIAL:
    mode = gen_social_df[col].mode()
    fill_value = mode.iloc[0] if not mode.empty else 0
    temp = gen_social_df[col].fillna(fill_value)
    temp = np.round(temp).clip(0, len(label_encoders[col].classes_)-1).astype(int)
    gen_social[col] = label_encoders[col].inverse_transform(temp)

df_epi = df[BASE_EPIDEMIOLOGY].copy()
metadata_epi = SingleTableMetadata()
metadata_epi.detect_from_dataframe(df_epi)
epi_ctgan = CTGANSynthesizer(metadata=metadata_epi)
epi_ctgan.fit(df_epi)
gen_epi_df = epi_ctgan.sample(n_gen)

# ==== Step 4. Combine generated feature blocks ====
gen_all = pd.concat([gen_climate_df.reset_index(drop=True),
                     gen_social.reset_index(drop=True),
                     gen_epi_df.reset_index(drop=True)], axis=1)

# ==== Step 5. Create the SVG output directory ====
img_dir = "synthetic_images_svg"
os.makedirs(img_dir, exist_ok=True)

# ==== Step 5.1 Compare univariate distributions and save SVG files ====
for col in ALL_FEATURES:
    if gen_all[col].var() == 0:
        print(f"{col} has zero variance; skipping the KDE plot")
        continue
    plt.figure(figsize=(6, 4))
    # Use distinct thick lines for real and generated distributions
    sns.kdeplot(df[col], label='Real', fill=True, color='blue', linewidth=3)
    sns.kdeplot(gen_all[col], label='Generated', fill=True, color='red', linestyle='--', linewidth=3)
    plt.title(f'Distribution comparison: {col}')
    plt.legend()
    plt.tight_layout()
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(axis='both', which='major', width=2.5, length=7, labelsize=18)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(18)
    file_path = os.path.join(img_dir, f'distribution_{col}.svg')
    plt.savefig(file_path, format='svg')
    plt.close()

# ==== Step 5.2 Visualize joint distributions with pair plots and save SVG files ====
real_pairplot = sns.pairplot(df[BASE_CLIMATE + BASE_SOCIAL], diag_kind='kde', plot_kws={'alpha':0.6, 'color':'blue'})
real_pairplot.fig.suptitle('Real Data Pairplot', y=1.02)
for ax in real_pairplot.axes.flatten():
    if ax is not None:
        for spine in ax.spines.values():
            spine.set_linewidth(3)
        ax.tick_params(axis='both', which='major', width=2.5, length=7, labelsize=18)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(18)
real_pairplot.savefig(os.path.join(img_dir, 'pairplot_real_climate_social.svg'), format='svg')
plt.close()

synthetic_pairplot = sns.pairplot(gen_all[BASE_CLIMATE + BASE_SOCIAL], diag_kind='kde', plot_kws={'alpha':0.6, 'color':'red'})
synthetic_pairplot.fig.suptitle('Synthetic Data Pairplot', y=1.02)
for ax in synthetic_pairplot.axes.flatten():
    if ax is not None:
        for spine in ax.spines.values():
            spine.set_linewidth(3)
        ax.tick_params(axis='both', which='major', width=2.5, length=7, labelsize=18)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(18)
synthetic_pairplot.savefig(os.path.join(img_dir, 'pairplot_synthetic_climate_social.svg'), format='svg')
plt.close()

# ==== Step 5.5 Visualize t-SNE embeddings and save SVG files ====
def plot_tsne(real, fake, features, filename):
    n_samples = min(len(real), len(fake), 200)
    tsne = TSNE(n_components=2, random_state=SEED)
    tsne_data = np.vstack([real[:n_samples], fake[:n_samples]])
    tsne_labels = np.array(['Real']*n_samples + ['Synthetic']*n_samples)
    tsne_result = tsne.fit_transform(tsne_data)
    plt.figure(figsize=(7, 6))
    # Use opaque markers without outlines
    palette = {'Real':'blue', 'Synthetic':'red'}
    sns.scatterplot(x=tsne_result[:,0], y=tsne_result[:,1], hue=tsne_labels,
                    palette=palette, alpha=1, edgecolor='none', s=40)
    plt.title(f't-SNE of Real vs Synthetic ({features})')
    plt.tight_layout()
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(axis='both', which='major', width=2.5, length=7, labelsize=18)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(18)
    plt.savefig(os.path.join(img_dir, filename), format='svg')
    plt.close()

plot_tsne(df[BASE_CLIMATE].values, gen_all[BASE_CLIMATE].values, "Climate", "tsne_climate_real_vs_synthetic.svg")
plot_tsne(df[BASE_SOCIAL].values, gen_all[BASE_SOCIAL].values, "Social", "tsne_social_real_vs_synthetic.svg")
plot_tsne(df[BASE_EPIDEMIOLOGY].values, gen_all[BASE_EPIDEMIOLOGY].values, "Epidemiology", "tsne_epidemiology_real_vs_synthetic.svg")

# ==== Step 6. Calculate univariate distribution-distance metrics and save CSV ====
def js_divergence(p, q, bins=50):
    p_hist, bin_edges = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bin_edges, density=True)
    p_hist = np.where(p_hist==0, 1e-8, p_hist)
    q_hist = np.where(q_hist==0, 1e-8, q_hist)
    m = 0.5 * (p_hist + q_hist)
    kl_pm = entropy(p_hist, m)
    kl_qm = entropy(q_hist, m)
    return 0.5 * (kl_pm + kl_qm)

metrics = []
for col in ALL_FEATURES:
    real = df[col].values
    fake = gen_all[col].values
    if np.all(np.isnan(real)) or np.all(np.isnan(fake)):
        continue
    real = real[~np.isnan(real)]
    fake = fake[~np.isnan(fake)]
    if len(real) == 0 or len(fake) == 0:
        continue
    wass = wasserstein_distance(real, fake)
    p_hist, bin_edges = np.histogram(real, bins=50, density=True)
    q_hist, _ = np.histogram(fake, bins=bin_edges, density=True)
    p_hist = np.where(p_hist==0, 1e-8, p_hist)
    q_hist = np.where(q_hist==0, 1e-8, q_hist)
    kl = entropy(p_hist, q_hist)
    js = js_divergence(real, fake, bins=50)
    metrics.append({'feature': col, 'wasserstein': wass, 'kl': kl, 'js': js})

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('synthetic_metrics.csv', index=False)
print("Feature-level distribution-distance metrics were saved to synthetic_metrics.csv")

# ==== Step 7. Assess overfitting, disclosure, and distinguishability ====
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

print("\n" + "=" * 60)
print("Synthetic-data diagnostic summary")
print("=" * 60)

# Prepare numerical features
real_data = df[ALL_FEATURES].fillna(df[ALL_FEATURES].mean()).values
gen_data = gen_all[ALL_FEATURES].fillna(gen_all[ALL_FEATURES].mean()).values

print(f"\nDataset dimensions:")
print(f"  Real samples: {len(real_data)}")
print(f"  Synthetic samples: {len(gen_data)}")
print(f"  Features: {real_data.shape[1]}")

# ---- 1. Diversity assessment ----
print("\n" + "=" * 60)
print("1. Diversity assessment")
print("=" * 60)
# Use pandas drop_duplicates to count unique synthetic rows
gen_df_temp = pd.DataFrame(gen_data, columns=ALL_FEATURES)
# Round to four decimal places to reduce floating-point sensitivity
gen_df_rounded = gen_df_temp.round(4)
unique_count = gen_df_rounded.drop_duplicates().shape[0]
diversity_ratio = unique_count / len(gen_data)
print(f"  Unique rows (four-decimal precision): {unique_count}/{len(gen_data)}")
print(f"  Diversity ratio: {diversity_ratio:.6f}")
if diversity_ratio > 0.95:
    diversity_eval = "Excellent"
elif diversity_ratio > 0.90:
    diversity_eval = "Acceptable"
else:
    diversity_eval = "Needs improvement"
print(f"  Assessment:  {diversity_eval}")

# ---- 2. Nearest-neighbor overlap ----
print("\n" + "=" * 60)
print("2. Nearest-neighbor overlap (bidirectional check)")
print("=" * 60)

# Standardize the data
scaler_privacy = StandardScaler()
real_scaled = scaler_privacy.fit_transform(real_data)
gen_scaled = scaler_privacy.transform(gen_data)

# Find the nearest real sample for each synthetic sample
nbrs_real = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(real_scaled)
distances_g2r, indices_g2r = nbrs_real.kneighbors(gen_scaled)

# Find the nearest synthetic sample for each real sample
nbrs_gen = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(gen_scaled)
distances_r2g, indices_r2g = nbrs_gen.kneighbors(real_scaled)

# Count mutual nearest-neighbor pairs between generated and real samples
overlap_count = 0
for i, nearest_real_idx in enumerate(indices_g2r.flatten()):
    if indices_r2g[nearest_real_idx, 0] == i:
        overlap_count += 1

overlap_rate = overlap_count / len(gen_data)
print(f"  Mutual nearest-neighbor pairs: {overlap_count}/{len(gen_data)}")
print(f"  Overlap rate: {overlap_rate:.6f}")
if overlap_rate < 0.05:
    overlap_eval = "Excellent"
elif overlap_rate < 0.15:
    overlap_eval = "Acceptable"
else:
    overlap_eval = "Needs improvement"
print(f"  Assessment: {overlap_eval}")

# ---- 3. Direct-leak diagnostic based on Euclidean distance ----
print("\n" + "=" * 60)
print("3. Direct-leak diagnostic (distance-threshold method)")
print("=" * 60)

# Calculate the minimum distance from each synthetic sample to the real dataset
from sklearn.metrics.pairwise import euclidean_distances

distances_matrix = euclidean_distances(gen_scaled, real_scaled)
min_distances = distances_matrix.min(axis=1)

# Summarize the minimum-distance distribution
print(f"  Minimum-distance summary:")
print(f"    Min:     {min_distances.min():.6f}")
print(f"    Median: {np.median(min_distances):.6f}")
print(f"    Mean:   {np.mean(min_distances):.6f}")
print(f"    Max:    {min_distances.max():.6f}")

# Set the threshold in standardized feature space
threshold = 0.01
leak_count = np.sum(min_distances < threshold)
leak_ratio = leak_count / len(gen_data)
print(f"  Distance threshold epsilon: {threshold}")
print(f"  Potential direct-leak samples (distance < epsilon): {leak_count}/{len(gen_data)}")
print(f"  Direct-leak rate: {leak_ratio:.6f}")
if leak_ratio == 0:
    leak_eval = "Excellent"
elif leak_ratio < 0.05:
    leak_eval = "Acceptable"
else:
    leak_eval = "Needs improvement"
print(f"  Assessment: {leak_eval}")

# ---- 4. Real-versus-synthetic distinguishability analysis ----
print("\n" + "=" * 60)
print("4. Real-versus-synthetic classification")
print("=" * 60)

# Create labels: real = 1 and synthetic = 0
real_labels = np.ones(len(real_scaled))
gen_labels = np.zeros(len(gen_scaled))

# Combine the datasets
X_all = np.vstack([real_scaled, gen_scaled])
y_all = np.hstack([real_labels, gen_labels])

# Repeat random train-test splits to assess stability
n_iterations = 10
mia_accuracies = []
mia_aucs = []

np.random.seed(SEED)
for i in range(n_iterations):
    # Use a 50/50 split with balanced classes
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.5, random_state=SEED + i, stratify=y_all
    )

    # Train a logistic-regression classifier
    clf = LogisticRegression(max_iter=1000, random_state=SEED + i, solver='lbfgs')
    clf.fit(X_train, y_train)

    # Generate predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    mia_accuracies.append(acc)
    mia_aucs.append(auc)

mia_acc_mean = np.mean(mia_accuracies)
mia_acc_std = np.std(mia_accuracies)
mia_auc_mean = np.mean(mia_aucs)
mia_auc_std = np.std(mia_aucs)

print(f"  Real-versus-synthetic classification accuracy: {mia_acc_mean:.4f} ± {mia_acc_std:.4f}")
print(f"  MIA AUC:        {mia_auc_mean:.4f} ± {mia_auc_std:.4f}")
print(f"  Absolute deviation from the random baseline (0.50): {abs(mia_acc_mean - 0.5):.4f}")
if abs(mia_acc_mean - 0.5) < 0.05:
    mia_eval = "Excellent"
elif abs(mia_acc_mean - 0.5) < 0.10:
    mia_eval = "Acceptable"
else:
    mia_eval = "Needs improvement"
print(f"  Assessment: {mia_eval} (Values closer to 0.50 indicate lower linear distinguishability)")

# ---- Overall diagnostic summary ----
print("\n" + "=" * 60)
print("Overall diagnostic summary")
print("=" * 60)
scores = [
    diversity_ratio > 0.95,
    overlap_rate < 0.05,
    leak_ratio == 0,
    abs(mia_acc_mean - 0.5) < 0.05
]
print(f"  Diagnostics meeting the predefined thresholds:  {sum(scores)}/4")
if sum(scores) == 4:
    overall_eval = "All predefined diagnostic thresholds met"
elif sum(scores) >= 3:
    overall_eval = "Most predefined diagnostic thresholds met"
elif sum(scores) >= 2:
    overall_eval = "Some predefined diagnostic thresholds were not met"
else:
    overall_eval = "Multiple predefined diagnostic thresholds were not met"
print(f"  Overall assessment: {overall_eval}")
print("=" * 60)

# ---- Save results ----
privacy_results = {
    'nearest_neighbor_overlap_rate': overlap_rate,
    'diversity_ratio': diversity_ratio,
    'direct_leak_ratio': leak_ratio,
    'mia_accuracy_mean': mia_acc_mean,
    'mia_accuracy_std': mia_acc_std,
    'mia_auc_mean': mia_auc_mean,
    'mia_auc_std': mia_auc_std,
    'min_distance_min': min_distances.min(),
    'min_distance_median': np.median(min_distances),
    'min_distance_mean': np.mean(min_distances)
}

privacy_df = pd.DataFrame([privacy_results])
privacy_df.to_csv('privacy_overfit_metrics.csv', index=False)
print("\nSynthetic-data diagnostic results were saved to privacy_overfit_metrics.csv")

# ==== Step 8. Save the generated data ====
gen_all.to_csv('synthetic_fused_data2.csv', index=False)
print("\nAll figures were saved as SVG files, all metrics were saved as CSV files, and the generated data were saved to synthetic_fused_data2.csv.")