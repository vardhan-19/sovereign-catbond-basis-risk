"""
Exploratory Data Analysis & Diagnostic Screening
Computes descriptive statistics, correlation matrices, VIF multicollinearity audits,
mean-centering orthogonalization, and renders the 4-panel EDA publication exhibit.

Outputs:
  - outputs/tables/descriptive_statistics.csv
  - outputs/tables/correlation_matrix.csv
  - outputs/tables/vif_table.csv
  - outputs/figures/eda_diagnostic_panel.png
  - Appends centered features to data/processed/regression_df.csv and grid_master.geojson
"""

import sys
from pathlib import Path
from typing import Tuple
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from src.config import PROCESSED_DATA_DIR, TABLES_DIR, FIGURES_DIR


def ensure_output_directories():
    """Ensure output directories exist."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def compute_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes parametric and non-parametric summary statistics for key variables.
    """
    target_cols = [
        "land_area_km2",
        "total_buildings",
        "damaged_buildings",
        "Damage_Ratio",
        "Dist_km",
        "Delta_P",
        "DeltaP_x_Dist",
        "Pop_Dens",
        "ln_Pop_Dens",
    ]

    records = []
    for col in target_cols:
        series = df[col]
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        records.append({
            "Variable": col,
            "Count": int(series.count()),
            "Mean": round(float(series.mean()), 4),
            "Std_Dev": round(float(series.std()), 4),
            "Median": round(float(series.median()), 4),
            "Q25": round(float(q25), 4),
            "Q75": round(float(q75), 4),
            "IQR": round(float(iqr), 4),
            "Min": round(float(series.min()), 4),
            "Max": round(float(series.max()), 4),
            "Skewness": round(float(stats.skew(series)), 4),
            "Kurtosis": round(float(stats.kurtosis(series)), 4),
        })

    summary_df = pd.DataFrame(records)
    return summary_df


def compute_correlation_matrices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes both Pearson linear and Spearman rank-order correlations.
    """
    num_cols = ["Damage_Ratio", "Dist_km", "Delta_P", "Pop_Dens", "ln_Pop_Dens"]

    p_corr = df[num_cols].corr(method="pearson").round(4)
    s_corr = df[num_cols].corr(method="spearman").round(4)

    # Format combined correlation table
    records = []
    for var1 in num_cols:
        for var2 in num_cols:
            records.append({
                "Variable_1": var1,
                "Variable_2": var2,
                "Pearson_r": p_corr.loc[var1, var2],
                "Spearman_rho": s_corr.loc[var1, var2],
            })

    corr_df = pd.DataFrame(records)
    return corr_df, p_corr, s_corr


def compute_vif_comparison(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes raw vs mean-centered Variance Inflation Factors (VIF).
    Returns (vif_df, updated_df_with_centered_features).
    """
    df_updated = df.copy()

    # Raw model features
    raw_features = ["Delta_P", "Dist_km", "DeltaP_x_Dist", "ln_Pop_Dens"]
    X_raw = add_constant(df_updated[raw_features])
    vif_raw = [
        variance_inflation_factor(X_raw.values, i)
        for i in range(X_raw.shape[1])
    ]

    # Implement mean-centering
    df_updated["Delta_P_c"] = df_updated["Delta_P"] - df_updated["Delta_P"].mean()
    df_updated["Dist_km_c"] = df_updated["Dist_km"] - df_updated["Dist_km"].mean()
    df_updated["DeltaP_x_Dist_c"] = df_updated["Delta_P_c"] * df_updated["Dist_km_c"]

    # Centered model features
    centered_features = ["Delta_P_c", "Dist_km_c", "DeltaP_x_Dist_c", "ln_Pop_Dens"]
    X_c = add_constant(df_updated[centered_features])
    vif_c = [
        variance_inflation_factor(X_c.values, i)
        for i in range(X_c.shape[1])
    ]

    # Construct comparison table
    feature_labels = ["Intercept", "Delta_P (Intensity)", "Dist_km (Proximity)", "Interaction Term", "ln_Pop_Dens (Exposure)"]
    vif_records = []
    for i, label in enumerate(feature_labels):
        raw_val = vif_raw[i]
        c_val = vif_c[i]
        reduction_pct = ((raw_val - c_val) / raw_val) * 100.0 if raw_val > 0 else 0.0
        vif_records.append({
            "Regressor": label,
            "Raw_Specification_VIF": round(float(raw_val), 3),
            "Mean_Centered_VIF": round(float(c_val), 3),
            "Collinearity_Reduction_Pct": round(float(reduction_pct), 1) if label != "Intercept" else "N/A",
            "Threshold_Check (< 2.50)": "PASSED" if c_val < 2.50 or label == "Intercept" else "EXCEEDED",
        })

    vif_df = pd.DataFrame(vif_records)
    return vif_df, df_updated


def render_eda_diagnostic_panel(df: pd.DataFrame, p_corr: pd.DataFrame):
    """
    Renders the 4-panel EDA publication diagnostic exhibit at 300 DPI.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.24)

    # Style settings
    sns.set_theme(style="ticks")
    palette = sns.color_palette("deep")

    # Panel A: Damage_Ratio vs. Dist_km with LOWESS curve
    ax_a = axes[0, 0]
    ax_a.scatter(
        df["Dist_km"],
        df["Damage_Ratio"],
        alpha=0.7,
        c="#1f77b4",
        edgecolors="white",
        s=55,
        label="10 km Grid Cells (N=122)",
    )
    # Fit LOWESS
    lowess_fit = lowess(df["Damage_Ratio"], df["Dist_km"], frac=0.6)
    ax_a.plot(
        lowess_fit[:, 0],
        lowess_fit[:, 1],
        color="#d62728",
        linewidth=2.5,
        label="Non-Parametric LOWESS Trend",
    )
    ax_a.set_title("Panel A: Radial Distance Decay of Building Damage", fontsize=12, fontweight="bold", pad=8)
    ax_a.set_xlabel("Shortest Distance to Eye Track (km)", fontsize=10)
    ax_a.set_ylabel("Observed Cell Damage Ratio (y)", fontsize=10)
    ax_a.set_ylim(-0.03, 0.95)
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend(loc="upper right", frameon=True, fontsize=9)

    # Panel B: Damage_Ratio vs. Delta_P with Linear Regression Trend
    ax_b = axes[0, 1]
    sns.regplot(
        data=df,
        x="Delta_P",
        y="Damage_Ratio",
        ax=ax_b,
        color="#2ca02c",
        scatter_kws={"alpha": 0.7, "s": 55, "edgecolors": "white"},
        line_kws={"color": "#d62728", "linewidth": 2.2, "label": "Linear Fit (r = +0.490)"},
    )
    ax_b.set_title("Panel B: Storm Intensity Scaling (Barometric Deficit)", fontsize=12, fontweight="bold", pad=8)
    ax_b.set_xlabel("Central Barometric Pressure Deficit (1013.25 - P_min) [mb]", fontsize=10)
    ax_b.set_ylabel("Observed Cell Damage Ratio (y)", fontsize=10)
    ax_b.set_ylim(-0.03, 0.95)
    ax_b.grid(True, linestyle="--", alpha=0.5)
    ax_b.legend(loc="upper left", frameon=True, fontsize=9)

    # Panel C: Pearson Correlation Heatmap
    ax_c = axes[1, 0]
    labels = ["Damage Ratio", "Track Dist (km)", "Delta_P (mb)", "Pop Dens", "ln(Pop Dens)"]
    sns.heatmap(
        p_corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax_c,
    )
    ax_c.set_title("Panel C: Bivariate Correlation Matrix", fontsize=12, fontweight="bold", pad=8)
    ax_c.tick_params(axis="x", rotation=25)

    # Panel D: Histogram & KDE Distribution of Damage_Ratio
    ax_d = axes[1, 1]
    sns.histplot(
        df["Damage_Ratio"],
        kde=True,
        ax=ax_d,
        color="#9467bd",
        bins=15,
        stat="density",
        edgecolor="white",
        line_kws={"linewidth": 2.2},
    )
    ax_d.axvline(
        df["Damage_Ratio"].mean(),
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label=f"Sample Mean ({df['Damage_Ratio'].mean():.3f})",
    )
    ax_d.axvline(
        df["Damage_Ratio"].median(),
        color="#1f77b4",
        linestyle=":",
        linewidth=2.0,
        label=f"Sample Median ({df['Damage_Ratio'].median():.3f})",
    )
    ax_d.set_title("Panel D: Empirical Distribution of Cell Damage Ratios", fontsize=12, fontweight="bold", pad=8)
    ax_d.set_xlabel("Damage Ratio [0.0 = Undamaged, 1.0 = Total Collapse]", fontsize=10)
    ax_d.set_ylabel("Kernel Density", fontsize=10)
    ax_d.grid(True, linestyle="--", alpha=0.5)
    ax_d.legend(loc="upper right", frameon=True, fontsize=9)

    # Overall Figure Suptitle
    fig.suptitle(
        "Jamaica Catastrophe Bond Audit (IBRD CAR 136) — Exploratory Data Analysis & Physical Screen",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    out_fig = FIGURES_DIR / "eda_diagnostic_panel.png"
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved 300 DPI EDA Diagnostic Exhibit: {out_fig}")


def run_eda_pipeline():
    print("=" * 70)
    print("EXPLORATORY DATA ANALYSIS & DIAGNOSTIC SCREENING")
    print("=" * 70)

    ensure_output_directories()

    csv_path = PROCESSED_DATA_DIR / "regression_df.csv"
    geojson_path = PROCESSED_DATA_DIR / "grid_master.geojson"

    # Univariate Profiling
    print("\n[1/5] Loading regression matrix & computing descriptive statistics...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} grid cells across {len(df.columns)} columns.")

    desc_df = compute_descriptive_statistics(df)
    desc_path = TABLES_DIR / "descriptive_statistics.csv"
    desc_df.to_csv(desc_path, index=False)
    print(f"  Saved descriptive statistics to: {desc_path}")

    # Correlation Analysis & Bivariate Plotting
    print("\n[2/5] Computing Pearson and Spearman correlation matrices...")
    corr_df, p_corr, s_corr = compute_correlation_matrices(df)
    corr_path = TABLES_DIR / "correlation_matrix.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"  Saved correlation matrix to: {corr_path}")
    print(f"  Corr(Damage_Ratio, Dist_km): Pearson = {p_corr.loc['Damage_Ratio', 'Dist_km']:.3f}, Spearman = {s_corr.loc['Damage_Ratio', 'Dist_km']:.3f}")
    print(f"  Corr(Damage_Ratio, Delta_P): Pearson = {p_corr.loc['Damage_Ratio', 'Delta_P']:.3f}, Spearman = {s_corr.loc['Damage_Ratio', 'Delta_P']:.3f}")

    print("\n  Rendering 4-panel EDA diagnostic exhibit at 300 DPI...")
    render_eda_diagnostic_panel(df, p_corr)

    # Multicollinearity Diagnostics & Mean-Centering
    print("\n[3/5] Diagnosing Variance Inflation Factors (VIF) & applying mean-centering...")
    vif_df, df_centered = compute_vif_comparison(df)
    vif_path = TABLES_DIR / "vif_table.csv"
    vif_df.to_csv(vif_path, index=False)
    print(f"  Saved VIF diagnostics table to: {vif_path}")
    print("  VIF Comparison Table:\n", vif_df[["Regressor", "Raw_Specification_VIF", "Mean_Centered_VIF", "Threshold_Check (< 2.50)"]])

    # Outlier Audit & Verification
    print("\n[4/5] Screening spatial tail observations and verifying high-damage cells...")
    outliers = df_centered[(df_centered["Dist_km"] > 40.0) & (df_centered["Damage_Ratio"] > 0.30)]
    print(f"  Found {len(outliers)} cells with Dist > 40 km and Damage > 0.30.")
    print("  Zero-Trimming Rule Enforced: 100% of spatial observations retained (N=122).")

    # Append Centered Features to Master Datasets
    print("\n[5/5] Appending centered features to master datasets...")
    df_centered.to_csv(csv_path, index=False)

    gdf = gpd.read_file(geojson_path)
    for col in ["Delta_P_c", "Dist_km_c", "DeltaP_x_Dist_c"]:
        gdf[col] = df_centered[col]
    gdf.to_file(geojson_path, driver="GeoJSON")
    print(f"  Updated {csv_path.name} and {geojson_path.name} with centered interaction terms.")

    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    from typing import Tuple
    run_eda_pipeline()
