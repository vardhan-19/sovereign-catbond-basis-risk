"""
Econometric Modeling & Damage Prediction
Fits Weighted Linear OLS (HC1) and Papke & Wooldridge (1996) Fractional Logit (QMLE / HC0).
Computes Marginal Effects at the Mean (MEM), generates comparison tables, and appends fitted predictions.

Outputs:
  - outputs/tables/regression_comparison_table.csv
  - outputs/tables/regression_comparison_table.tex
  - Appends hat_y_ols and hat_y_frac to data/processed/regression_df.csv and grid_master.geojson
"""

import sys
from pathlib import Path
from typing import Dict, Tuple
import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, TABLES_DIR


def ensure_output_directories():
    """Ensure output directories exist."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def fit_weighted_ols(df: pd.DataFrame):
    """
    Fits Model 1: Weighted Least Squares with Huber-White HC1 robust standard errors.
    Weights: total_buildings.
    Regressors are fully mean-centered to eliminate multicollinearity and provide
    intuitive baseline intercept interpretation.
    """
    formula = "Damage_Ratio ~ Delta_P_c + Dist_km_c + DeltaP_x_Dist_c + ln_Pop_Dens"
    model = smf.wls(formula=formula, data=df, weights=df["total_buildings"])
    results = model.fit(cov_type="HC1")
    return results


def fit_fractional_logit(df: pd.DataFrame):
    """
    Fits Model 2: Papke & Wooldridge (1996) Fractional Response Logit via Quasi-GLM (QMLE).
    Weights: total_buildings, Covariance: HC0 sandwich.
    """
    y = df["Damage_Ratio"]
    X = sm.add_constant(df[["Delta_P_c", "Dist_km_c", "DeltaP_x_Dist_c", "ln_Pop_Dens"]])
    model = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(sm.families.links.Logit()),
        freq_weights=df["total_buildings"],
    )
    results = model.fit(cov_type="HC0")
    return results, X


def compute_marginal_effects_at_mean(model_results, X: pd.DataFrame) -> pd.Series:
    """
    Computes Marginal Effects at the Mean (MEM) for Fractional Logit:
      MEM_j = beta_j * p_hat * (1 - p_hat)
    where p_hat is the predicted probability evaluated at covariate means.
    """
    x_mean = X.mean()
    xb_mean = np.dot(x_mean, model_results.params)
    p_hat = 1.0 / (1.0 + np.exp(-xb_mean))
    deriv = p_hat * (1.0 - p_hat)
    mem = model_results.params * deriv
    return mem, p_hat


def export_regression_tables(ols_res, frac_res, mem_series: pd.Series, df: pd.DataFrame):
    """
    Generates structured CSV and publication LaTeX comparison tables.
    """
    variables = [
        ("const", "Intercept", "Intercept"),
        ("Delta_P_c", "Delta_P_c", "Delta_P_c (Intensity Deficit)"),
        ("Dist_km_c", "Dist_km_c", "Dist_km_c (Track Proximity)"),
        ("DeltaP_x_Dist_c", "DeltaP_x_Dist_c", "DeltaP_x_Dist_c (Radial Attenuation)"),
        ("ln_Pop_Dens", "ln_Pop_Dens", "ln_Pop_Dens (Exposure Control)"),
    ]

    records = []
    for var_key, ols_key, label in variables:
        # OLS terms
        ols_coef = ols_res.params[ols_key]
        ols_se = ols_res.bse[ols_key]
        ols_t = ols_res.tvalues[ols_key]
        ols_p = ols_res.pvalues[ols_key]

        # Frac logit terms
        frac_coef = frac_res.params[var_key]
        frac_se = frac_res.bse[var_key]
        frac_z = frac_res.tvalues[var_key]
        frac_p = frac_res.pvalues[var_key]
        frac_mem = mem_series.get(var_key, np.nan)

        records.append({
            "Regressor": label,
            "OLS_Coef": round(float(ols_coef), 4),
            "OLS_HC1_SE": round(float(ols_se), 4),
            "OLS_t_stat": round(float(ols_t), 3),
            "OLS_p_value": float(ols_p),
            "FracLogit_Coef": round(float(frac_coef), 4),
            "FracLogit_HC0_SE": round(float(frac_se), 4),
            "FracLogit_z_stat": round(float(frac_z), 3),
            "FracLogit_p_value": float(frac_p),
            "FracLogit_MEM": round(float(frac_mem), 6),
        })

    comp_df = pd.DataFrame(records)

    # Accuracy metrics
    y = df["Damage_Ratio"]
    hat_ols = ols_res.predict(df)
    X = sm.add_constant(df[["Delta_P_c", "Dist_km_c", "DeltaP_x_Dist_c", "ln_Pop_Dens"]])
    hat_frac = frac_res.predict(X)

    e_ols = y - hat_ols
    e_frac = y - hat_frac

    rmse_ols = np.sqrt(np.mean(e_ols**2))
    mae_ols = np.mean(np.abs(e_ols))
    r2_ols_raw = 1.0 - (np.sum(e_ols**2) / np.sum((y - y.mean())**2))

    rmse_frac = np.sqrt(np.mean(e_frac**2))
    mae_frac = np.mean(np.abs(e_frac))
    r2_frac_raw = 1.0 - (np.sum(e_frac**2) / np.sum((y - y.mean())**2))
    pseudo_r2_frac = np.corrcoef(y, hat_frac)[0, 1]**2

    # Export CSV
    csv_path = TABLES_DIR / "regression_comparison_table.csv"
    comp_df.to_csv(csv_path, index=False)
    print(f"  Saved regression comparison table to: {csv_path}")

    # Generate publication-grade LaTeX booktabs table
    tex_path = TABLES_DIR / "regression_comparison_table.tex"
    
    def stars(p):
        if p < 0.001:
            return "$^{***}$"
        elif p < 0.01:
            return "$^{**}$"
        elif p < 0.05:
            return "$^{*}$"
        return ""

    tex_lines = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\small",
        "\\caption{Econometric Damage Models: Weighted Linear OLS Baseline vs. Fractional Logit (QMLE)}",
        "\\label{tab:regression_comparison}",
        "\\begin{tabularx}{\\textwidth}{X r r r r}",
        "\\toprule",
        " & \\multicolumn{2}{c}{\\textbf{Model 1: Weighted OLS (HC1)}} & \\multicolumn{2}{c}{\\textbf{Model 2: Fractional Logit (HC0)}} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
        "\\textbf{Independent Regressors} & \\textbf{Coefficient (SE)} & \\textbf{$t$-stat} & \\textbf{Coefficient (SE)} & \\textbf{MEM} \\\\",
        "\\midrule"
    ]

    for _, row in comp_df.iterrows():
        name = row["Regressor"].replace("_", "\\_").replace("&", "\\&")
        o_c = f"{row['OLS_Coef']:.4f}{stars(row['OLS_p_value'])}"
        o_se = f"({row['OLS_HC1_SE']:.4f})"
        o_t = f"{row['OLS_t_stat']:.2f}"

        f_c = f"{row['FracLogit_Coef']:.4f}{stars(row['FracLogit_p_value'])}"
        f_se = f"({row['FracLogit_HC0_SE']:.4f})"
        f_mem = f"{row['FracLogit_MEM']:.4f}"

        tex_lines.append(f"{name} & {o_c} & {o_t} & {f_c} & {f_mem} \\\\")
        tex_lines.append(f" & \\multicolumn{{1}}{{c}}{{{o_se}}} & & \\multicolumn{{1}}{{c}}{{{f_se}}} & \\\\")
        tex_lines.append("\\addlinespace[2pt]")

    tex_lines.extend([
        "\\midrule",
        f"Observations ($N$) & \\multicolumn{{2}}{{c}}{{{len(df)}}} & \\multicolumn{{2}}{{c}}{{{len(df)}}} \\\\",
        "Sample Weights ($w_i$) & \\multicolumn{2}{c}{Total Surveyed Buildings} & \\multicolumn{2}{c}{Total Surveyed Buildings} \\\\",
        f"$R^2$ / Pseudo-$R^2$ & \\multicolumn{{2}}{{c}}{{{ols_res.rsquared:.3f} (Weighted) / {r2_ols_raw:.3f} (Raw)}} & \\multicolumn{{2}}{{c}}{{{pseudo_r2_frac:.3f} (Correlation Squared)}} \\\\",
        f"Root Mean Squared Error (RMSE) & \\multicolumn{{2}}{{c}}{{{rmse_ols:.4f}}} & \\multicolumn{{2}}{{c}}{{{rmse_frac:.4f}}} \\\\",
        f"Mean Absolute Error (MAE) & \\multicolumn{{2}}{{c}}{{{mae_ols:.4f}}} & \\multicolumn{{2}}{{c}}{{{mae_frac:.4f}}} \\\\",
        f"Negative Predictions ($\\hat{{y}} < 0$) & \\multicolumn{{2}}{{c}}{{{int((hat_ols < 0).sum())} cells}} & \\multicolumn{{2}}{{c}}{{{int((hat_frac < 0).sum())} cells (Strictly Bounded)}} \\\\",
        "\\bottomrule",
        "\\multicolumn{5}{l}{\\footnotesize $^{***}p < 0.001$, $^{**}p < 0.01$, $^{*}p < 0.05$. Robust standard errors in parentheses.} \\\\",
        "\\end{tabularx}",
        "\\end{table}"
    ])

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"  Saved LaTeX regression table to: {tex_path}")

    return {
        "rmse_ols": rmse_ols,
        "mae_ols": mae_ols,
        "r2_ols_weighted": ols_res.rsquared,
        "r2_ols_raw": r2_ols_raw,
        "rmse_frac": rmse_frac,
        "mae_frac": mae_frac,
        "pseudo_r2_frac": pseudo_r2_frac,
    }


def run_econometrics_pipeline():
    print("=" * 70)
    print("ECONOMETRIC MODELING & DAMAGE PREDICTION PIPELINE")
    print("=" * 70)

    ensure_output_directories()

    csv_path = PROCESSED_DATA_DIR / "regression_df.csv"
    geojson_path = PROCESSED_DATA_DIR / "grid_master.geojson"

    # Specification and Ingestion
    print("\n[1/4] Ingesting regression matrix & verifying specification...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} sample observations with exposure weights (min={df['total_buildings'].min()}, max={df['total_buildings'].max()}).")

    # Estimation
    print("\n[2/4] Fitting Model 1 (Weighted OLS with HC1 robust standard errors)...")
    ols_res = fit_weighted_ols(df)
    print(f"  OLS Delta_P_c Coef: {ols_res.params['Delta_P_c']:.4f} (p = {ols_res.pvalues['Delta_P_c']:.4e})")
    print(f"  OLS Dist_km_c Coef: {ols_res.params['Dist_km_c']:.4f} (p = {ols_res.pvalues['Dist_km_c']:.4e})")
    print(f"  OLS Interaction Coef: {ols_res.params['DeltaP_x_Dist_c']:.6f} (p = {ols_res.pvalues['DeltaP_x_Dist_c']:.4e})")
    print(f"  OLS Weighted R²: {ols_res.rsquared:.4f}")

    print("\n  Fitting Model 2 (Papke & Wooldridge Fractional Logit via QMLE / HC0)...")
    frac_res, X_mat = fit_fractional_logit(df)
    mem_series, p_hat_mean = compute_marginal_effects_at_mean(frac_res, X_mat)
    print(f"  Fractional Logit Predicted Probability at Mean: {p_hat_mean:.4f}")
    print(f"  Frac Logit Delta_P_c Coef: {frac_res.params['Delta_P_c']:.4f} (MEM = {mem_series['Delta_P_c']:+.4f})")
    print(f"  Frac Logit Dist_km_c Coef: {frac_res.params['Dist_km_c']:.4f} (MEM = {mem_series['Dist_km_c']:+.4f})")
    print(f"  Frac Logit Interaction Coef: {frac_res.params['DeltaP_x_Dist_c']:.6f} (MEM = {mem_series['DeltaP_x_Dist_c']:+.6f})")

    # Model Diagnostics & Comparison
    print("\n[3/4] Evaluating model accuracy and exporting comparison tables...")
    metrics = export_regression_tables(ols_res, frac_res, mem_series, df)
    print(f"  Weighted OLS R²: {metrics['r2_ols_weighted']:.4f}, RMSE: {metrics['rmse_ols']:.4f}, MAE: {metrics['mae_ols']:.4f}")
    print(f"  Fractional Logit Pseudo-R²: {metrics['pseudo_r2_frac']:.4f}, RMSE: {metrics['rmse_frac']:.4f}, MAE: {metrics['mae_frac']:.4f}")

    # Append Fitted Predictions & Residuals
    print("\n[4/4] Appending model fitted predictions and spatial residuals...")
    hat_y_ols = ols_res.predict(df)
    hat_y_frac = frac_res.predict(X_mat)

    df["hat_y_ols"] = hat_y_ols
    df["hat_y_frac"] = hat_y_frac
    df["residual_frac"] = df["Damage_Ratio"] - df["hat_y_frac"]
    df["abs_residual_frac"] = df["residual_frac"].abs()

    conditions = [
        df["residual_frac"] >= 0.10,
        df["residual_frac"] <= -0.10,
    ]
    choices = [
        "Severe Under-Coverage",
        "Windfall Over-Coverage",
    ]
    df["basis_risk_category"] = np.select(conditions, choices, default="Concordant Alignment")

    df.to_csv(csv_path, index=False)

    gdf = gpd.read_file(geojson_path)
    for col in ["hat_y_ols", "hat_y_frac", "residual_frac", "abs_residual_frac", "basis_risk_category"]:
        gdf[col] = df[col]
    gdf.to_file(geojson_path, driver="GeoJSON")

    print(f"  Appended hat_y_ols (span: [{hat_y_ols.min():.4f}, {hat_y_ols.max():.4f}])")
    print(f"  Appended hat_y_frac (span: [{hat_y_frac.min():.4f}, {hat_y_frac.max():.4f}])")
    print(f"  Appended residuals & basis risk categories (mean residual: {df['residual_frac'].mean():.4f})")
    print(f"  Updated {csv_path.name} and {geojson_path.name} (21 columns total).")

    print("\n" + "=" * 70)
    print("ECONOMETRIC MODELING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_econometrics_pipeline()
