# Spatial Econometric Audit of Parametric Basis Risk in Sovereign Catastrophe Bonds

Replication repository and empirical spatial audit of sovereign catastrophe bond trigger mechanics, evaluated on Jamaica's World Bank note (IBRD CAR 136).

---

## Overview

Parametric catastrophe bonds disburse liquidity within days based on pre-certified physical hazard parameters (such as storm track geometry and minimum central barometric pressure) certified by an independent calculation agent, bypassing protracted post-disaster loss adjustment. However, parametric designs introduce **basis risk**—the discrepancy between index-triggered payouts and actual physical damage on the ground.

This repository provides an empirical geospatial and econometric evaluation of sovereign basis risk across Jamaica's 14 parishes using:
1. **Physical hazard tracks:** NOAA IBTrACS tropical cyclone observations across near-miss offshore bypass and direct landfall strike geometries.
2. **Satellite structural damage assessments:** Copernicus Emergency Management Service building footprints (EMSR847), classified according to the European Macroseismic Scale (EMS-98 Grade 3+ structural failure).
3. **Spatial econometric modeling:** Weighted Linear OLS baseline and Papke & Wooldridge (1996) Fractional Response Logit (QMLE), coupled with Conley (1999) Spatial HAC covariance estimation and Global Moran's (1950) $I$ residual clustering diagnostics.

---

## Key Findings

- **Parametric Fit & Basis Risk Ceiling:** Track geometry and barometric pressure deficit explain 84.6% of cell-level damage variance ($R^2 = 0.846$), establishing a 15.4% unmodeled basis risk bound attributable to local terrain, rainfall, and structural exposure.
- **Physical Attenuation Gradients:** Observed damage decreases by 5.4 percentage points per 10 km from the cyclone track ($\beta = -0.0054, t = -20.02$) and intensifies by 9.9 percentage points per 10 mb of central barometric deficit ($\beta = +0.0099, t = +7.62$).
- **Spatial Error Clustering:** Regression residuals exhibit statistically significant spatial autocorrelation (Moran's $I = 0.2115, z = 4.16, p < 0.001$). Under non-parametric Conley Spatial HAC adjustment (75 km Bartlett kernel), standard errors expand by up to $1.99\times$ while core parameters retain strong statistical significance ($t = -10.27, p < 0.001$).
- **Subnational Allocation Discrepancies:** A 14-parish reconciliation identifies a USD 6.29 million spatial misallocation across local recovery allocations, characterized by severe under-coverage in western and interior parishes (St. Elizabeth, Hanover, St. Mary) and formula windfalls in sheltered southern plains (Clarendon).
- **Structuring Renewal Alternatives:** Extending the contractual grid 35 km into Jamaica's Exclusive Economic Zone, adopting a graduated 4-tier payout structure, and integrating NASA GPM IMERG compound precipitation triggers reduces basis risk variance by an estimated 64.0%.

---

## Repository Structure

```
sovereign-catbond-basis-risk/
├── data/
│   ├── raw/                 # Cyclone tracks, Copernicus footprints, parish boundaries, WorldPop
│   └── processed/           # 10 km master metric grid and analytical regression matrix
├── src/
│   ├── config.py            # Coordinate reference systems (EPSG:3448), physical constants, paths
│   ├── grid_utils.py        # 10 km metric grid tessellation and sliver filtering
│   ├── track_utils.py       # Cyclone interpolation and Holland vortex physics
│   ├── damage_utils.py      # Copernicus building damage grading and spatial aggregation
│   ├── spatial_stats.py     # OLS, Fractional Logit, Conley Spatial HAC, Moran's I
│   ├── build_features.py    # Master spatial join and feature engineering pipeline
│   ├── run_eda.py           # Descriptive statistics, correlation matrices, VIF audits
│   ├── run_econometrics.py  # Model calibration (OLS & Fractional Response GLM)
│   ├── run_spatial_diagnostics.py # Moran's I permutation tests and Conley HAC bandwidth analysis
│   └── render_exhibits.py   # Publication cartography and regression diagnostics
├── outputs/
│   ├── figures/             # Econometric scatter plots and Conley HAC forest plots
│   ├── maps/                # Synoptic storm track and spatial basis risk maps
│   └── tables/              # Regression tables, parish scorecards, diagnostic CSVs
├── requirements.txt         # Pinned Python package dependencies
└── README.md
```

---

## Replication Guide

### Environment Setup

```bash
git clone https://github.com/vardhan-19/sovereign-catbond-basis-risk.git
cd sovereign-catbond-basis-risk

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Running the End-to-End Pipeline

Execute the pipeline modules sequentially from the project root:

```bash
# 1. Construct 10 km metric grid and extract spatial features
python -m src.build_features

# 2. Compute descriptive statistics and collinearity diagnostics
python -m src.run_eda

# 3. Estimate Linear OLS and Fractional Logit models
python -m src.run_econometrics

# 4. Run spatial dependence diagnostics (Moran's I & Conley HAC)
python -m src.run_spatial_diagnostics

# 5. Render diagnostic figures and maps
python -m src.render_exhibits
```

---

## References

- Conley, T. G. (1999). *GMM Estimation with Cross Sectional Dependence*. Journal of Econometrics, 92(1), 1–45.
- Ghesquiere, F., & Mahul, O. (2010). *Financial Protection of the State against Natural Disasters: A Primer*. World Bank Policy Research Working Paper No. 5429.
- Grünthal, G. (1998). *European Macroseismic Scale 1998 (EMS-98)*. Cahiers du Centre Européen de Géodynamique et de Séismologie, Volume 15.
- Moran, P. A. P. (1950). *Notes on Continuous Stochastic Phenomena*. Biometrika, 37(1/2), 17–23.
- Papke, L. E., & Wooldridge, J. M. (1996). *Econometric Methods for Fractional Response Variables with an Application to 401(k) Plan Participation Rates*. Journal of Applied Econometrics, 11(6), 619–632.
- World Bank (2021). *Capital-at-Risk Notes Prospectus: International Bank for Reconstruction and Development USD 150,000,000 Catastrophe-Linked Notes (IBRD CAR 136)*.
