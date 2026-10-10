"""
Copernicus EMS remote sensing building damage classification and spatial aggregation.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from src.config import CRS_METRIC, MIN_BUILDING_COUNT


def classify_building_damage(buildings_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Standardize Copernicus EMS-98 damage grading into a binary damage indicator.

    EMS-98 Grade 1 (Negligible) & Grade 2 (Moderate) -> 0 (Undamaged / Habitable)
    EMS-98 Grade 3 (Heavy), Grade 4 (Very Heavy), Grade 5 (Destroyed) -> 1 (Damaged / Rebuild)

    Parameters
    ----------
    buildings_gdf : GeoDataFrame
        Raw building footprints or points from Copernicus EMSR847.

    Returns
    -------
    GeoDataFrame
        GeoDataFrame with validated 'is_damaged' integer series (0 or 1).
    """
    df = buildings_gdf.copy()

    # Detect damage column name
    grade_col = None
    for candidate in ["damage_grade", "grade", "DAMAGE", "dmg_grade", "grading"]:
        if candidate in df.columns:
            grade_col = candidate
            break

    if grade_col is None:
        raise ValueError("Could not find a valid damage grading column in building dataset.")

    # Convert numeric or string labels
    def map_grade(val):
        if pd.isna(val):
            return np.nan
        # If numeric 1-5
        try:
            num = int(float(val))
            if num in [3, 4, 5]:
                return 1
            elif num in [1, 2]:
                return 0
        except (ValueError, TypeError):
            pass

        # If string labels
        str_val = str(val).lower().strip()
        if any(term in str_val for term in ["destroyed", "heavy", "collapse", "severe", "grade 3", "grade 4", "grade 5"]):
            return 1
        elif any(term in str_val for term in ["negligible", "slight", "moderate", "minor", "grade 1", "grade 2"]):
            return 0
        return np.nan

    df["is_damaged"] = df[grade_col].apply(map_grade)
    # Drop records with invalid or missing damage assessments
    df = df.dropna(subset=["is_damaged"]).copy()
    df["is_damaged"] = df["is_damaged"].astype(int)

    return df


def aggregate_damage_to_grid(
    buildings_gdf: gpd.GeoDataFrame,
    grid_gdf: gpd.GeoDataFrame,
    crs_metric: str = CRS_METRIC,
    min_buildings: int = MIN_BUILDING_COUNT
) -> gpd.GeoDataFrame:
    """
    Spatially join building footprints to 10 km grid cells and compute bounded damage ratios.

    Parameters
    ----------
    buildings_gdf : GeoDataFrame
        Buildings with 'is_damaged' column.
    grid_gdf : GeoDataFrame
        Metric grid cells in EPSG:3448 with 'cell_id'.
    crs_metric : str
        Target metric CRS.
    min_buildings : int
        Minimum building count to enter primary regression sample.

    Returns
    -------
    GeoDataFrame
        Grid cells with total_buildings (T_i), damaged_buildings (D_i),
        Damage_Ratio (y_i), and exposure_weight (w_i).
    """
    b_proj = buildings_gdf.to_crs(crs_metric)
    g_proj = grid_gdf.to_crs(crs_metric)

    # Point-in-polygon spatial join
    joined = gpd.sjoin(b_proj, g_proj[["cell_id", "geometry"]], how="inner", predicate="within")

    # Groupby cell_id
    stats = joined.groupby("cell_id").agg(
        total_buildings=("is_damaged", "count"),
        damaged_buildings=("is_damaged", "sum")
    ).reset_index()

    stats["Damage_Ratio"] = stats["damaged_buildings"] / stats["total_buildings"]

    # Merge back to grid
    merged = g_proj.merge(stats, on="cell_id", how="left")
    merged["total_buildings"] = merged["total_buildings"].fillna(0).astype(int)
    merged["damaged_buildings"] = merged["damaged_buildings"].fillna(0).astype(int)

    # Exposure weight for Weighted Least Squares
    merged["exposure_weight"] = merged["total_buildings"]

    # Flag cells meeting minimum exposure threshold
    merged["valid_sample"] = (merged["total_buildings"] >= min_buildings)

    return merged
