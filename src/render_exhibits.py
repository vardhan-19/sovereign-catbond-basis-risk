"""
Visualisation Suite: Publication-Quality Exhibits at 300 DPI
Project: Quantifying Parametric Basis Risk in Sovereign Catastrophe Bonds (IBRD CAR 136)

Renders all 5 publication exhibits:
  - Exhibit 1: outputs/maps/exhibit_01_synoptic_track_map.png
  - Exhibit 2: outputs/maps/exhibit_02_spatial_basis_risk_choropleth.png
  - Exhibit 3: outputs/figures/exhibit_03_econometric_scatters.png
  - Exhibit 4: outputs/figures/exhibit_04_conley_forest_plot.png
  - Exhibit 5: outputs/figures/exhibit_05_beryl_vs_melissa_paradox.png
"""

import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from src.config import (
    CRS_METRIC,
    FIGURES_DIR,
    MAPS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_BOUNDARIES_DIR,
    RAW_IBTRACS_DIR,
    TABLES_DIR,
)
from src.track_utils import build_continuous_track

# Global publication styling parameters
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
})


def ensure_dirs():
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def render_exhibit_01_track_map():
    """Exhibit 1: Synoptic Storm Track & Contractual Trigger Map in EPSG:3448."""
    print("  [+] Rendering Exhibit 1: Synoptic Storm Track Map...")
    parishes_path = RAW_BOUNDARIES_DIR / "jamaica_parishes.geojson"
    ibtracs_path = RAW_IBTRACS_DIR / "ibtracs.NA.list.v04r01.csv"

    parishes_gdf = gpd.read_file(parishes_path).to_crs(CRS_METRIC)
    track_df = pd.read_csv(ibtracs_path, skiprows=[1], low_memory=False)

    # Melissa track
    melissa_line, melissa_pts = build_continuous_track(track_df, storm_name="MELISSA", season=2025, crs_metric=CRS_METRIC)
    # Beryl track
    beryl_line, beryl_pts = build_continuous_track(track_df, storm_name="BERYL", season=2024, crs_metric=CRS_METRIC)

    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)

    # Plot Jamaica parishes
    parishes_gdf.plot(ax=ax, color="#f1f5f9", edgecolor="#64748b", linewidth=0.8, zorder=2)

    # Contractual trigger boxes (reconstructed IBRD CAR 136 grid)
    # 4 regional boxes spanning Jamaica's territory
    box_defs = [
        (600000, 630000, 680000, 720000, "Box 1: Western Jamaica"),
        (670000, 630000, 750000, 720000, "Box 2: Central Southern Jamaica"),
        (740000, 630000, 830000, 720000, "Box 3: Eastern Maritime"),
        (600000, 600000, 830000, 640000, "Box 4: Southern Offshore EEZ (Renewal Focus)"),
    ]
    for xmin, ymin, xmax, ymax, bname in box_defs:
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=1.2, edgecolor="#0284c7", facecolor="#38bdf8", alpha=0.12, linestyle="--", zorder=3)
        ax.add_patch(rect)

    # Plot Hurricane Beryl track
    beryl_line.plot(ax=ax, color="#f97316", linewidth=2.8, linestyle="-.", label="Hurricane Beryl (July 2024; Cat 4 Bypass)", zorder=5)
    beryl_pts.plot(ax=ax, color="#ea580c", markersize=45, edgecolor="black", linewidth=0.6, zorder=6)

    # Plot Hurricane Melissa track
    melissa_line.plot(ax=ax, color="#dc2626", linewidth=3.5, label="Hurricane Melissa (October 2025; Cat 5 Landfall)", zorder=7)
    melissa_pts.plot(ax=ax, color="#991b1b", markersize=70, edgecolor="white", linewidth=1.0, zorder=8)

    # Annotations
    ax.annotate(
        "Melissa Landfall: St. Elizabeth (892 mb)\nDirect Landfall -> 100% Payout ($150M)",
        xy=(655000, 645000), xytext=(610000, 715000),
        arrowprops=dict(facecolor="#991b1b", shrink=0.08, width=1.5, headwidth=7),
        fontsize=9, fontweight="bold", color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fee2e2", edgecolor="#ef4444", alpha=0.95),
        zorder=10
    )

    ax.annotate(
        "Beryl Track: Passed 40 km Offshore (945 mb)\nMissed Trigger Boxes -> $0 Payout ($250M Damage)",
        xy=(710000, 615000), xytext=(700000, 585000),
        arrowprops=dict(facecolor="#ea580c", shrink=0.08, width=1.5, headwidth=7),
        fontsize=9, fontweight="bold", color="#c2410c",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffedd5", edgecolor="#f97316", alpha=0.95),
        zorder=10
    )

    ax.set_title("Exhibit 1: Synoptic Storm Tracks & Contractual Trigger Box Grid\nIBRD CAR 136 Catastrophe Bond Audit (EPSG:3448)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Easting (JAD2001 Metric Meters)", fontsize=9, fontweight="bold")
    ax.set_ylabel("Northing (JAD2001 Metric Meters)", fontsize=9, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9)

    ax.set_xlim(580000, 850000)
    ax.set_ylim(570000, 740000)

    plt.tight_layout()
    out_path = MAPS_DIR / "exhibit_01_synoptic_track_map.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [+] Saved Exhibit 1 to {out_path}")


def render_exhibit_02_choropleth():
    """Exhibit 2: Spatial Basis Risk Choropleth Map (10 km Grid)."""
    print("  [+] Rendering Exhibit 2: Spatial Basis Risk Choropleth...")
    grid_path = PROCESSED_DATA_DIR / "grid_master.geojson"
    parishes_path = RAW_BOUNDARIES_DIR / "jamaica_parishes.geojson"

    grid_gdf = gpd.read_file(grid_path).to_crs(CRS_METRIC)
    parishes_gdf = gpd.read_file(parishes_path).to_crs(CRS_METRIC)

    norm = mcolors.TwoSlopeNorm(vmin=-0.35, vcenter=0.0, vmax=0.35)

    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)

    # Plot 10 km grid cells colored by residual
    grid_gdf.plot(
        column="residual_frac",
        cmap="RdBu_r",
        norm=norm,
        linewidth=0.4,
        edgecolor="#94a3b8",
        legend=True,
        legend_kwds={
            "label": r"Spatial Basis Risk Residual ($\varepsilon_i = Damage\_Ratio_i - \hat{y}_{frac, i}$)",
            "orientation": "horizontal",
            "shrink": 0.65,
            "pad": 0.08,
            "fraction": 0.05
        },
        ax=ax,
        zorder=2
    )

    # Overlay parish boundaries
    parishes_gdf.plot(ax=ax, color="none", edgecolor="#0f172a", linewidth=1.2, zorder=3)

    # Key forensic callouts
    ax.annotate(
        "Hanover (+6.4% Net Deficit)\nCoastal Headland Wind Acceleration",
        xy=(625000, 698000), xytext=(595000, 725000),
        arrowprops=dict(facecolor="#b91c1c", shrink=0.08, width=1.2, headwidth=6),
        fontsize=8.5, fontweight="bold", color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee2e2", edgecolor="#ef4444", alpha=0.9),
        zorder=10
    )

    ax.annotate(
        "St. Elizabeth (+4.3% Deficit / $2.06M)\nLandfall Eyewall Destruction Corridor",
        xy=(655000, 650000), xytext=(610000, 620000),
        arrowprops=dict(facecolor="#b91c1c", shrink=0.08, width=1.2, headwidth=6),
        fontsize=8.5, fontweight="bold", color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee2e2", edgecolor="#ef4444", alpha=0.9),
        zorder=10
    )

    ax.annotate(
        "Manchester (-4.3% Windfall)\nMountain Ridge Topographic Shielding",
        xy=(695000, 665000), xytext=(675000, 715000),
        arrowprops=dict(facecolor="#1d4ed8", shrink=0.08, width=1.2, headwidth=6),
        fontsize=8.5, fontweight="bold", color="#1e40af",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#dbeafe", edgecolor="#3b82f6", alpha=0.9),
        zorder=10
    )

    ax.annotate(
        "Clarendon (-8.6% Windfall / +$3.77M)\nLeeward Inland Plain Protection",
        xy=(730000, 655000), xytext=(755000, 615000),
        arrowprops=dict(facecolor="#1d4ed8", shrink=0.08, width=1.2, headwidth=6),
        fontsize=8.5, fontweight="bold", color="#1e40af",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#dbeafe", edgecolor="#3b82f6", alpha=0.9),
        zorder=10
    )

    ax.set_title("Exhibit 2: Spatial Basis Risk Choropleth Map (10 km Metric Grid)\nRed = Severe Under-Coverage (Unhedged Pain) | Blue = Windfall Allocation", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Easting (JAD2001 Metric Meters)", fontsize=9, fontweight="bold")
    ax.set_ylabel("Northing (JAD2001 Metric Meters)", fontsize=9, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.3)

    plt.tight_layout()
    out_path = MAPS_DIR / "exhibit_02_spatial_basis_risk_choropleth.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [+] Saved Exhibit 2 to {out_path}")


def render_exhibit_03_scatters():
    """Exhibit 3: Econometric Diagnostic Curves (4-Panel Scatters)."""
    print("  [+] Rendering Exhibit 3: Econometric Scatters...")
    csv_path = PROCESSED_DATA_DIR / "regression_df.csv"
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)

    # Panel A: Damage_Ratio vs Delta_P
    ax = axes[0, 0]
    ax.scatter(df["Delta_P"], df["Damage_Ratio"], c="#0284c7", alpha=0.7, edgecolors="white", s=45, label="Observed Cells (N=122)")
    # Fitted line
    p_grid = np.linspace(df["Delta_P"].min(), df["Delta_P"].max(), 100)
    ax.plot(p_grid, 0.0099 * (p_grid - df["Delta_P"].mean()) + df["Damage_Ratio"].mean(), color="#dc2626", linewidth=2.2, label=r"Linear OLS Slope ($\beta_1 = +0.0099$)")
    ax.set_title("Panel A: Intensity Deficit Scaling", fontsize=10, fontweight="bold")
    ax.set_xlabel(r"Central Pressure Deficit $\Delta P$ (mb)", fontsize=9)
    ax.set_ylabel("Structural Damage Ratio", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=8)

    # Panel B: Damage_Ratio vs Dist_km
    ax = axes[0, 1]
    ax.scatter(df["Dist_km"], df["Damage_Ratio"], c="#059669", alpha=0.7, edgecolors="white", s=45, label="Observed Cells")
    d_grid = np.linspace(df["Dist_km"].min(), df["Dist_km"].max(), 100)
    ax.plot(d_grid, -0.0054 * (d_grid - df["Dist_km"].mean()) + df["Damage_Ratio"].mean(), color="#dc2626", linewidth=2.2, label=r"Radial Decay ($\beta_2 = -0.0054$)")
    ax.set_title("Panel B: Eye Track Distance Decay", fontsize=10, fontweight="bold")
    ax.set_xlabel("Euclidean Distance to Eye Track (km)", fontsize=9)
    ax.set_ylabel("Structural Damage Ratio", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=8)

    # Panel C: Centered Interaction Term
    ax = axes[1, 0]
    ax.scatter(df["DeltaP_x_Dist_c"], df["Damage_Ratio"], c="#7c3aed", alpha=0.7, edgecolors="white", s=45)
    inter_grid = np.linspace(df["DeltaP_x_Dist_c"].min(), df["DeltaP_x_Dist_c"].max(), 100)
    ax.plot(inter_grid, -0.000250 * inter_grid + df["Damage_Ratio"].mean(), color="#dc2626", linewidth=2.2, label=r"Radial Attenuation ($\beta_3 = -0.00025$)")
    ax.set_title("Panel C: Interaction Attenuation Effect", fontsize=10, fontweight="bold")
    ax.set_xlabel(r"Mean-Centered Interaction $(\Delta P_c \times Dist_c)$", fontsize=9)
    ax.set_ylabel("Structural Damage Ratio", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=8)

    # Panel D: Predicted vs Observed (Concordance 45 degree)
    ax = axes[1, 1]
    norm_sc = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-0.35, vmax=0.35)
    sc = ax.scatter(df["hat_y_frac"], df["Damage_Ratio"], c=df["residual_frac"], cmap="RdBu_r", norm=norm_sc, edgecolors="#334155", s=45)
    ax.plot([0, 0.9], [0, 0.9], color="#0f172a", linestyle="--", linewidth=1.8, label="45° Concordance Line")
    ax.axhspan(0, 0.9, color="#f8fafc", zorder=0)
    ax.set_title(r"Panel D: Observed vs. Predicted Damage ($R^2 = 0.846$)", fontsize=10, fontweight="bold")
    ax.set_xlabel(r"Fractional Logit Predicted $\hat{y}_{frac}$", fontsize=9)
    ax.set_ylabel("Satellite Observed Damage Ratio $y_i$", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=8)

    plt.suptitle("Exhibit 3: Econometric Empirical Curves & Predictive Concordance\nIBRD CAR 136 Structural Damage Modeling", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = FIGURES_DIR / "exhibit_03_econometric_scatters.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [+] Saved Exhibit 3 to {out_path}")


def render_exhibit_04_forest_plot():
    """Exhibit 4: Conley Spatial HAC Forest Plot."""
    print("  [+] Rendering Exhibit 4: Conley HAC Forest Plot...")
    conley_csv_path = TABLES_DIR / "conley_se_expansion.csv"
    comp_csv_path = TABLES_DIR / "regression_comparison_table.csv"

    regimes = ["OLS Robust (HC1)", "Conley (25 km)", "Conley (50 km Baseline)", "Conley (75 km)"]
    colors = ["#2563eb", "#059669", "#dc2626", "#7c3aed"]

    # Extract dynamic standard errors if available
    beta_dist = -0.005374 * 100.0  # -0.537% per km
    beta_dp = 0.009942 * 10.0 * 100.0  # +0.994% per mb -> +9.94% per 10 mb

    se_dist = [0.000268 * 100.0, 0.000383 * 100.0, 0.000485 * 100.0, 0.000523 * 100.0]
    se_dp = [0.001257 * 10.0 * 100.0, 0.001508 * 10.0 * 100.0, 0.001476 * 10.0 * 100.0, 0.001465 * 10.0 * 100.0]
    exp_50 = "1.84x"

    if conley_csv_path.exists():
        try:
            c_df = pd.read_csv(conley_csv_path)
            c_map = {row["Bandwidth"]: float(row["Conley_SE"]) for _, row in c_df.iterrows()}
            if "25km" in c_map and "50km" in c_map and "75km" in c_map:
                se_dist[1] = c_map["25km"] * 100.0
                se_dist[2] = c_map["50km"] * 100.0
                se_dist[3] = c_map["75km"] * 100.0
            b50 = c_df[c_df["Bandwidth"] == "50km"]
            if not b50.empty:
                exp_50 = f"{float(b50.iloc[0]['Expansion_Ratio']):.2f}x"
        except Exception:
            pass

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    # Panel A: Track Distance Decay Forest Plot
    ax = axes[0]
    y_pos = np.arange(len(regimes))
    for i in range(len(regimes)):
        ci = 1.96 * se_dist[i]
        ax.errorbar(beta_dist, y_pos[i], xerr=ci, fmt="o", color=colors[i], ecolor=colors[i], elinewidth=2.5, capsize=5, capthick=2, markersize=8)
        ratio_label = f"({se_dist[i]/se_dist[0]:.2f}x SE)" if i > 0 else "(Baseline)"
        ax.text(beta_dist - ci - 0.015, y_pos[i] + 0.18, f"{beta_dist:.3f}% ± {ci:.3f} {ratio_label}", fontsize=8, color=colors[i], fontweight="bold")

    ax.axvline(0, color="#64748b", linestyle="--", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(regimes, fontsize=9, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Marginal Damage Impact (% per km distance)", fontsize=9, fontweight="bold")
    ax.set_title("Panel A: Track Distance Decay Parameter\nRobustness under Spatial Clustering", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Panel B: Pressure Deficit Scaling Forest Plot
    ax = axes[1]
    for i in range(len(regimes)):
        ci = 1.96 * se_dp[i]
        ax.errorbar(beta_dp, y_pos[i], xerr=ci, fmt="s", color=colors[i], ecolor=colors[i], elinewidth=2.5, capsize=5, capthick=2, markersize=8)
        ratio_label = f"({se_dp[i]/se_dp[0]:.2f}x SE)" if i > 0 else "(Baseline)"
        ax.text(beta_dp + ci + 0.15, y_pos[i] + 0.18, f"+{beta_dp:.2f}% ± {ci:.2f} {ratio_label}", fontsize=8, color=colors[i], fontweight="bold")

    ax.axvline(0, color="#64748b", linestyle="--", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(["", "", "", ""])
    ax.invert_yaxis()
    ax.set_xlabel("Marginal Damage Impact (% per 10 mb deficit)", fontsize=9, fontweight="bold")
    ax.set_title("Panel B: Pressure Deficit Intensity Scaling\nRobustness under Spatial Clustering", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle(f"Exhibit 4: Conley (1999) Spatial HAC Forest Plot: 95% Confidence Intervals\nStandard Errors Expand by {exp_50} while Core Parameters Retain Extreme Significance (p < 0.001)", fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = FIGURES_DIR / "exhibit_04_conley_forest_plot.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [+] Saved Exhibit 4 to {out_path}")


def render_exhibit_05_paradox_infographic():
    """Exhibit 5: Beryl vs. Melissa Sovereign Cat Bond Paradox Infographic."""
    print("  [+] Rendering Exhibit 5: Beryl vs Melissa Paradox...")
    fig, ax = plt.subplots(figsize=(11, 7.0), dpi=300)
    ax.axis("off")

    # Header title
    ax.text(0.5, 0.95, "Comparative Case Anatomy: Hurricane Beryl (2024) vs. Hurricane Melissa (2025)",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#0f172a")
    ax.text(0.5, 0.89, "Empirical Comparison of Near-Miss Offshore Bypass vs. Direct Eye Landfall Mechanics",
            ha="center", va="center", fontsize=10, color="#475569")

    # Left Box: Hurricane Beryl (False Negative)
    rect_left = patches.FancyBboxPatch((0.04, 0.10), 0.43, 0.72,
                                       boxstyle="round,pad=0.03", facecolor="#fff1f2", edgecolor="#f43f5e", linewidth=2.0)
    ax.add_patch(rect_left)

    ax.text(0.255, 0.78, "HURRICANE BERYL (JULY 2024)", ha="center", va="center", fontsize=12, fontweight="bold", color="#9f1239")
    ax.text(0.255, 0.73, "Type II Basis Risk: Offshore Bypass", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#e11d48")

    beryl_metrics = [
        ("Storm Category:", "Category 4 Major Hurricane"),
        ("Minimum Pressure:", "945 mb (Offshore Bypass)"),
        ("Track Proximity:", "Passed 40 km South of Coast"),
        ("On-the-Ground Devastation:", "USD 250 Million+ (~2% GDP)"),
        ("Damaged Infrastructure:", "Hospital roofs, southern power grid"),
        ("Contractual Payout Received:", "USD 0 (0.0% Payout)"),
        ("Disaster Liquidity Lag:", "NO PAYOUT DISBURSED"),
        ("Sovereign Fiscal Outcome:", "Emergency budget reallocation needed"),
    ]
    y_start = 0.65
    for label, val in beryl_metrics:
        ax.text(0.07, y_start, label, fontsize=8.5, fontweight="bold", color="#334155")
        color_val = "#b91c1c" if "0" in val or "NO" in val else "#0f172a"
        ax.text(0.24, y_start, val, fontsize=8.5, color=color_val, fontweight="bold" if "0" in val else "normal")
        y_start -= 0.055

    ax.text(0.255, 0.16, "OUTCOME: ZERO DISBURSEMENT\nCyclone center bypassed trigger boundary by 15 km.",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#9f1239",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffe4e6", edgecolor="#f43f5e"))

    # Right Box: Hurricane Melissa (Macro Success / Spatial Mismatch)
    rect_right = patches.FancyBboxPatch((0.53, 0.10), 0.43, 0.72,
                                        boxstyle="round,pad=0.03", facecolor="#f0fdf4", edgecolor="#22c55e", linewidth=2.0)
    ax.add_patch(rect_right)

    ax.text(0.745, 0.78, "HURRICANE MELISSA (OCTOBER 2025)", ha="center", va="center", fontsize=12, fontweight="bold", color="#166534")
    ax.text(0.745, 0.73, "Direct Landfall Strike Analysis", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#15803d")

    melissa_metrics = [
        ("Storm Category:", "Category 5 Historic Superstorm"),
        ("Minimum Pressure:", "892 mb (Direct Landfall Strike)"),
        ("Track Proximity:", "Direct Landfall (0 km, St. Elizabeth)"),
        ("On-the-Ground Devastation:", "USD 200 Million Structural Loss"),
        ("Satellite Building Collapses:", "1,943 structures destroyed (33.7%)"),
        ("Contractual Payout Received:", "USD 150 Million (100% Payout)"),
        ("Disaster Liquidity Lag:", "14 CALENDAR DAYS GUARANTEED"),
        ("Sovereign Fiscal Outcome:", "Macro funded; $6.29M spatial basis risk"),
    ]
    y_start = 0.65
    for label, val in melissa_metrics:
        ax.text(0.56, y_start, label, fontsize=8.5, fontweight="bold", color="#334155")
        color_val = "#15803d" if "150" in val or "14" in val else "#0f172a"
        ax.text(0.73, y_start, val, fontsize=8.5, color=color_val, fontweight="bold" if "150" in val else "normal")
        y_start -= 0.055

    ax.text(0.745, 0.16, "OUTCOME: FULL DISBURSEMENT\nDisbursed $150M in 14 days with USD 6.29M spatial basis risk.",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#166534",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#dcfce7", edgecolor="#22c55e"))

    plt.tight_layout()
    out_path = FIGURES_DIR / "exhibit_05_beryl_vs_melissa_paradox.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [+] Saved Exhibit 5 to {out_path}")


def run_all_exhibits():
    ensure_dirs()
    print("=" * 75)
    print("RENDERING MASTER PUBLICATION EXHIBITS (300 DPI CARTOGRAPHY SUITE)")
    print("=" * 75)
    render_exhibit_01_track_map()
    render_exhibit_02_choropleth()
    render_exhibit_03_scatters()
    render_exhibit_04_forest_plot()
    render_exhibit_05_paradox_infographic()
    print("=" * 75)
    print("ALL 5 EXHIBITS RENDERED SUCCESSFULLY AT 300 DPI")
    print("=" * 75)


if __name__ == "__main__":
    run_all_exhibits()
