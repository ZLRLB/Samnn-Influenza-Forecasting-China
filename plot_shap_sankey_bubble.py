# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(
    "ignore",
    message="Support for Kaleido versions less than 1.0.0 is deprecated*"
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# 0. Basic settings
# ============================================================

file_stage = [
    ("shap_sankey_2018_2019.csv", "2018/19 Season"),
    ("shap_sankey_2022_2023.csv", "2022/23 Season"),
    ("shap_sankey_2023_2024.csv", "2023/24 Season"),
]

OUTPUT_PREFIX = "Figure_Sankey_Bubble_final_snap_compact_bubble"

# Set to 6 or 8 if a more compact figure is required
# None retains all variables
TOP_N_PER_GROUP = None


# ============================================================
# 1. Feature groups and display labels
# ============================================================

feature_group_map = {
    "lag1": "Epi",
    "lag2": "Epi",
    "lag3": "Epi",
    "lag4": "Epi",

    "absolute_humidity": "Climate",
    "uvindex": "Climate",
    "solarenergy": "Climate",
    "solarradiation": "Climate",
    "visibility": "Climate",
    "cloudcover": "Climate",
    "winddir": "Climate",
    "windspeed": "Climate",
    "humidity": "Climate",
    "feelslike": "Climate",
    "temp": "Climate",
    "DTR": "Climate",

    "holiday": "Social",
    "trend_stage": "Social",
    "H1N1_flu_season": "Social",

    "time_part_code": "Special",
}

groups = ["Epi", "Climate", "Social", "Special"]

feature_label_map = {
    "lag1": "A/H1N1 CI lag 1 wk",
    "lag2": "A/H1N1 CI lag 2 wk",
    "lag3": "A/H1N1 CI lag 3 wk",
    "lag4": "A/H1N1 CI lag 4 wk",

    "absolute_humidity": "Absolute humidity",
    "uvindex": "UV index",
    "solarenergy": "Solar energy",
    "solarradiation": "Solar radiation",
    "visibility": "Visibility",
    "cloudcover": "Cloud cover",
    "winddir": "Wind direction",
    "windspeed": "Wind speed",
    "humidity": "Relative humidity",
    "feelslike": "Feels-like temperature",
    "temp": "Temperature",
    "DTR": "DTR",

    "holiday": "Holiday",
    "trend_stage": "Trend stage",
    "H1N1_flu_season": "A/H1N1 flu season",

    "time_part_code": "Time period code",
}

reverse_feature_label_map = {v: k for k, v in feature_label_map.items()}


def standardize_feature_name(x):
    """
    Support both original variable names and formatted display labels.
    """
    s = str(x).strip()

    if s in feature_group_map:
        return s

    if s in reverse_feature_label_map:
        return reverse_feature_label_map[s]

    s_lower = s.lower()

    lower_key_map = {k.lower(): k for k in feature_group_map.keys()}
    lower_label_map = {k.lower(): v for k, v in reverse_feature_label_map.items()}

    if s_lower in lower_key_map:
        return lower_key_map[s_lower]

    if s_lower in lower_label_map:
        return lower_label_map[s_lower]

    if "lag 1" in s_lower:
        return "lag1"
    if "lag 2" in s_lower:
        return "lag2"
    if "lag 3" in s_lower:
        return "lag3"
    if "lag 4" in s_lower:
        return "lag4"

    return s


# ============================================================
# 2. Read and organize input data
# ============================================================

stage_data = []
all_features = set()

for fname, stage in file_stage:
    df = pd.read_csv(fname)

    if "feature" not in df.columns:
        raise ValueError(f"{fname} is missing the feature column.")
    if "mean_abs_shap" not in df.columns:
        raise ValueError(f"{fname} is missing the mean_abs_shap column.")

    df = df.copy()
    df["feature"] = df["feature"].apply(standardize_feature_name)
    df["mean_abs_shap"] = pd.to_numeric(df["mean_abs_shap"], errors="coerce").fillna(0)
    df["mean_abs_shap"] = df["mean_abs_shap"].abs()
    df["stage"] = stage

    stage_data.append(df)
    all_features.update(df["feature"].unique())

all_features = sorted(list(all_features))

# Add missing features for each season
for i, df in enumerate(stage_data):
    missing = set(all_features) - set(df["feature"])

    if len(missing) > 0:
        add_df = pd.DataFrame({
            "feature": list(missing),
            "mean_abs_shap": [0.0] * len(missing),
            "stage": [df["stage"].iloc[0]] * len(missing)
        })
        df = pd.concat([df, add_df], ignore_index=True)

    df = df.sort_values("feature").reset_index(drop=True)
    stage_data[i] = df


# ============================================================
# 3. Calculate cross-season feature contributions and sort features
# ============================================================

feature_total_map = {}

for f in all_features:
    vals = []
    for df in stage_data:
        tmp = df.loc[df["feature"] == f, "mean_abs_shap"]
        vals.append(float(tmp.values[0]) if len(tmp) > 0 else 0.0)

    feature_total_map[f] = float(np.sum(np.abs(vals)))

# Remove features with zero total contribution
all_features = [f for f in all_features if feature_total_map.get(f, 0) > 0]

features_sorted = []

for g in groups:
    group_feats = [
        f for f in all_features
        if feature_group_map.get(f, "Special") == g
    ]

    group_feats = sorted(
        group_feats,
        key=lambda x: -feature_total_map.get(x, 0)
    )

    if TOP_N_PER_GROUP is not None:
        group_feats = group_feats[:TOP_N_PER_GROUP]

    features_sorted.extend(group_feats)

feature_total_map = {f: feature_total_map[f] for f in features_sorted}
feature_labels = [feature_label_map.get(f, f) for f in features_sorted]


# ============================================================
# 4. Node labels and indices
# ============================================================

group_labels = groups
stage_labels = [stage for _, stage in file_stage]

labels = group_labels + feature_labels + stage_labels

group_idx = {g: i for i, g in enumerate(groups)}
feature_idx = {f: len(groups) + i for i, f in enumerate(features_sorted)}
stage_idx = {s: len(groups) + len(features_sorted) + i for i, (_, s) in enumerate(file_stage)}


# ============================================================
# 5. Color settings
# ============================================================

group_color_map = {
    "Epi": "#5B8FF9",
    "Climate": "#61DDAA",
    "Social": "#F6BD16",
    "Special": "#7262FD",
}

feature_group_color_map = {
    "Epi": "#C6E5FF",
    "Climate": "#D3F261",
    "Social": "#FFD591",
    "Special": "#D3CEFD",
}

stage_color_map = {
    "2018/19 Season": "#FFB1B1",
    "2022/23 Season": "#A0E3F0",
    "2023/24 Season": "#FFE591",
}

node_colors = (
    [group_color_map[g] for g in groups] +
    [feature_group_color_map[feature_group_map.get(f, "Special")] for f in features_sorted] +
    [stage_color_map[s] for s in stage_labels]
)


def hex_to_rgba(hex_color, alpha=0.45):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ============================================================
# 6. Sankey links
# ============================================================
# Group → Feature: total contribution across the three seasons
# Feature → Season: season-specific contribution
# This preserves flow conservation at each feature node.

source = []
target = []
value = []
link_label = []
link_color = []
link_customdata = []

MIN_FLOW = 1e-8

# Group → Feature
for f in features_sorted:
    g = feature_group_map.get(f, "Special")
    total_v = float(feature_total_map.get(f, 0.0))

    if total_v <= 0:
        continue

    source.append(group_idx[g])
    target.append(feature_idx[f])
    value.append(max(total_v, MIN_FLOW))
    link_label.append("")
    link_color.append("rgba(145,145,145,0.24)")
    link_customdata.append(
        f"{g} → {feature_label_map.get(f, f)}<br>"
        f"Total contribution = {total_v:.4f}"
    )

# Feature → Season
max_v = max([df["mean_abs_shap"].max() for df in stage_data] + [1e-8])

for df_stage in stage_data:
    s = df_stage["stage"].iloc[0]

    for _, row in df_stage.iterrows():
        f = row["feature"]

        if f not in feature_idx:
            continue

        v = float(abs(row["mean_abs_shap"]))

        if v <= 0:
            continue

        g = feature_group_map.get(f, "Special")
        color_base = feature_group_color_map.get(g, "#CCCCCC")

        # Link opacity varies with contribution, while link width equals v
        alpha = 0.34 + 0.46 * (v / max_v)
        alpha = min(max(alpha, 0.34), 0.82)

        source.append(feature_idx[f])
        target.append(stage_idx[s])
        value.append(max(v, MIN_FLOW))
        link_label.append("")
        link_color.append(hex_to_rgba(color_base, alpha))
        link_customdata.append(
            f"{feature_label_map.get(f, f)} → {s}<br>"
            f"Contribution = {v:.4f}"
        )


# ============================================================
# 7. Bubble-plot data
# ============================================================

bubble_df = pd.DataFrame({
    "feature": features_sorted,
    "feature_label": feature_labels,
    "group": [feature_group_map.get(f, "Special") for f in features_sorted],
    "mean_abs_shap": [feature_total_map[f] for f in features_sorted],
})

# Manually position groups on the x-axis for a compact layout
group_x_map = {
    "Epi": 0.00,
    "Climate": 0.50,
    "Social": 1.00,
    "Special": 1.50
}

bubble_df["group_x"] = bubble_df["group"].map(group_x_map)

shap_min = bubble_df["mean_abs_shap"].min()
shap_max = bubble_df["mean_abs_shap"].max()

bubble_df["shap_norm"] = (
    (bubble_df["mean_abs_shap"] - shap_min) /
    (shap_max - shap_min + 1e-12)
)


def redblue(val):
    r1, g1, b1 = (91, 143, 249)
    r2, g2, b2 = (255, 75, 75)

    r = int(r1 + (r2 - r1) * val)
    g = int(g1 + (g2 - g1) * val)
    b = int(b1 + (b2 - b1) * val)

    return f"rgba({r},{g},{b},{0.50 + 0.45 * val})"


normed = bubble_df["shap_norm"].values
marker_colors = [redblue(v) for v in normed]

# Use larger bubbles
marker_sizes = 45 + 115 * normed

bubble_trace = go.Scatter(
    x=bubble_df["group_x"],
    y=bubble_df["feature_label"],
    mode="markers",
    marker=dict(
        size=marker_sizes,
        color=marker_colors,
        line=dict(width=1.9, color="rgba(80,80,80,0.45)"),
        symbol="circle",
        opacity=0.94
    ),
    showlegend=False,
    text=[
        f"{lbl}<br>Total contribution = {val:.4f}"
        for lbl, val in zip(
            bubble_df["feature_label"],
            bubble_df["mean_abs_shap"]
        )
    ],
    hovertemplate="%{text}<extra></extra>"
)


# ============================================================
# 8. Combined figure
# ============================================================

fig = make_subplots(
    rows=1,
    cols=2,
    column_widths=[0.74, 0.22],
    specs=[[{"type": "sankey"}, {"type": "scatter"}]],
    horizontal_spacing=0.045,
)

# -------- Panel A: Sankey --------
fig.add_trace(
    go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=38,
            line=dict(color="rgba(120,120,120,0.28)", width=0.7),
            label=labels,
            color=node_colors,
            customdata=labels,
            hovertemplate="<b>%{label}</b><extra></extra>",
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            label=link_label,
            color=link_color,
            customdata=link_customdata,
            hovertemplate="%{customdata}<extra></extra>",
        )
    ),
    row=1,
    col=1
)

# -------- Panel B: Bubble --------
fig.add_trace(
    bubble_trace,
    row=1,
    col=2
)


# ============================================================
# 9. Bubble-plot axes
# ============================================================

fig.update_yaxes(
    row=1,
    col=2,
    title_text="<b>Feature</b>",
    tickfont=dict(size=24, family="Arial Black, Arial"),
    showgrid=True,
    gridcolor="rgba(225,225,225,0.95)",
    gridwidth=1.4,

    # Hide the outer y-axis line and tick marks
    showline=False,
    ticks="",
    ticklen=0,

    categoryorder="array",
    categoryarray=feature_labels,
    autorange="reversed"
)

fig.update_xaxes(
    row=1,
    col=2,
    title_text="<b>Group</b>",
    tickmode="array",
    tickvals=[group_x_map[g] for g in groups],
    ticktext=groups,
    range=[-0.20, 1.70],

    tickfont=dict(size=24, family="Arial Black, Arial"),
    showgrid=True,
    gridcolor="rgba(225,225,225,0.95)",
    gridwidth=1.4,

    # Hide the outer x-axis line and tick marks
    showline=False,
    ticks="",
    ticklen=0,

    zeroline=False
)


# ============================================================
# 10. Color bar
# ============================================================

colorscale = [[0, "#5B8FF9"], [1, "#FF4B4B"]]

fig.add_trace(
    go.Scatter(
        x=[None, None],
        y=[None, None],
        mode="markers",
        marker=dict(
            colorscale=colorscale,
            cmin=shap_min,
            cmax=shap_max,
            color=[shap_min, shap_max],
            showscale=True,
            size=[0, 0],
            colorbar=dict(
                title="<b>Contribution value</b>",
                thickness=28,
                tickfont=dict(size=22, family="Arial Black, Arial"),
                x=1.065,
                y=0.80,
                len=0.43
            ),
        ),
        showlegend=False,
        hoverinfo="skip"
    ),
    row=1,
    col=2
)


# ============================================================
# 11. Panel labels and overall layout
# ============================================================

fig.add_annotation(
    x=0.005,
    y=1.04,
    xref="paper",
    yref="paper",
    text="<b>A</b>",
    showarrow=False,
    font=dict(size=52, family="Arial Black, Arial", color="black")
)

fig.add_annotation(
    x=0.795,
    y=1.04,
    xref="paper",
    yref="paper",
    text="<b>B</b>",
    showarrow=False,
    font=dict(size=52, family="Arial Black, Arial", color="black")
)

fig.update_layout(
    font=dict(
        size=29,
        color="#222222",
        family="Arial Black, Arial"
    ),
    margin=dict(t=65, l=20, r=70, b=35),
    height=1600,
    width=3000,
    hoverlabel=dict(
        bgcolor="#FFFFFF",
        font_size=20,
        font_family="Arial"
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    showlegend=False,
)


# ============================================================
# 12. Export
# ============================================================

fig.write_image(
    f"{OUTPUT_PREFIX}.svg",
    width=3000,
    height=1600,
    scale=2
)

fig.write_image(
    f"{OUTPUT_PREFIX}.png",
    width=3000,
    height=1600,
    scale=2
)

print(f"Figure files created: {OUTPUT_PREFIX}.svg / .png")
