"""
Spatial Engineering Pipeline & Master Feature Builder
Builds the 10 km metric grid in EPSG:3448, calculates storm proximity and pressure,
aggregates Copernicus building damage, extracts WorldPop density, and exports master datasets.

Outputs:
  - data/processed/grid_master.geojson
  - data/processed/regression_df.csv
"""

import os
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterstats
from shapely.geometry import Point

from src.config import (
    CRS_METRIC,
    CRS_GEOGRAPHIC,
    DEFAULT_GRID_SIZE_METERS,
    MIN_LAND_AREA_KM2,
    MIN_BUILDING_COUNT,
    STANDARD_SEA_LEVEL_PRESSURE_MB,
    RAW_BOUNDARIES_DIR,
    RAW_COPERNICUS_DIR,
    RAW_IBTRACS_DIR,
    RAW_POPULATION_DIR,
    PROCESSED_DATA_DIR,
)
from src.grid_utils import create_tessellation_grid
from src.damage_utils import classify_building_damage, aggregate_damage_to_grid
from src.track_utils import build_continuous_track


def run_spatial_pipeline():
    """
    Executes the spatial data engineering pipeline.
    """
    print("=" * 70)
    print("SPATIAL DATA ENGINEERING PIPELINE (10 KM MASTER GRID)")
    print("=" * 70)

    # Ensure processed data directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Geodetic Reprojection & Coordinate Standardization
    # -------------------------------------------------------------------------
    print("\n[1/5] Ingesting and standardizing vector layers into EPSG:3448...")
    parishes_path = RAW_BOUNDARIES_DIR / "jamaica_parishes.geojson"
    buildings_path = RAW_COPERNICUS_DIR / "EMSR847_buildings_damage.geojson"
    ibtracs_path = RAW_IBTRACS_DIR / "ibtracs.NA.list.v04r01.csv"
    pop_raster_path = RAW_POPULATION_DIR / "jam_pop_1km.tif"

    parishes_gdf = gpd.read_file(parishes_path).to_crs(CRS_METRIC)
    buildings_gdf = gpd.read_file(buildings_path).to_crs(CRS_METRIC)

    # Ingest IBTrACS and build continuous track
    track_df = pd.read_csv(ibtracs_path, skiprows=[1], low_memory=False)
    melissa_line_gdf, melissa_pts_gdf = build_continuous_track(
        track_df, storm_name="MELISSA", season=2025, crs_metric=CRS_METRIC
    )

    # Assert metric bounds in EPSG:3448
    xmin, ymin, xmax, ymax = parishes_gdf.total_bounds
    assert 600000.0 <= xmin and xmax <= 850000.0, f"X bounds outside range: {xmin}, {xmax}"
    assert 600000.0 <= ymin and ymax <= 750000.0, f"Y bounds outside range: {ymin}, {ymax}"
    print(f"  Parishes bounds validated in EPSG:3448: [{xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f}] m")
    print(f"  Melissa track length: {melissa_line_gdf.geometry.iloc[0].length / 1000.0:.2f} km")

    # -------------------------------------------------------------------------
    # 10 km Grid Tessellation & Coastal Sliver Filtering
    # -------------------------------------------------------------------------
    print("\n[2/5] Tessellating 10 km orthogonal grid & filtering coastal slivers...")
    raw_grid = create_tessellation_grid(
        parishes_gdf,
        cell_size_m=DEFAULT_GRID_SIZE_METERS,
        crs_metric=CRS_METRIC,
        min_land_area_km2=MIN_LAND_AREA_KM2,
    )
    print(f"  Initial land cells generated (Area >= {MIN_LAND_AREA_KM2} km²): {len(raw_grid)}")

    # -------------------------------------------------------------------------
    # Spatial Aggregation of Satellite Building Damage
    # -------------------------------------------------------------------------
    print("\n[3/5] Joining Copernicus building footprints & computing Damage_Ratio...")
    classified_buildings = classify_building_damage(buildings_gdf)
    grid_with_damage = aggregate_damage_to_grid(
        classified_buildings,
        raw_grid,
        crs_metric=CRS_METRIC,
        min_buildings=MIN_BUILDING_COUNT,
    )

    # Filter for valid sample (buildings >= 10)
    valid_sample_mask = grid_with_damage["valid_sample"]
    sample_grid = grid_with_damage[valid_sample_mask].copy().reset_index(drop=True)
    sample_grid["cell_id"] = sample_grid.index.astype(int)
    print(f"  Filtered sample cells (Total Buildings >= {MIN_BUILDING_COUNT}): {len(sample_grid)}")
    print(f"  Total buildings in sample: {sample_grid['total_buildings'].sum():,}")
    print(f"  Total damaged buildings in sample: {sample_grid['damaged_buildings'].sum():,}")
    print(f"  Mean cell Damage_Ratio: {sample_grid['Damage_Ratio'].mean():.4f}")

    # -------------------------------------------------------------------------
    # Storm Proximity & Pressure Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[4/5] Measuring Euclidean track distance & central pressure deficits...")
    melissa_line = melissa_line_gdf.geometry.iloc[0]

    # Re-extract metric centroids from final cell polygons
    centroids = sample_grid.geometry.centroid
    sample_grid["centroid_x"] = centroids.x
    sample_grid["centroid_y"] = centroids.y

    # Euclidean distance to track in kilometers
    sample_grid["Dist_km"] = centroids.distance(melissa_line) / 1000.0

    # Project centroids along track to interpolate central pressure at closest approach
    # Robust geodetic projection of points onto track line ensuring strict monotonicity and NaN safety
    valid_pts = melissa_pts_gdf.dropna(subset=["USA_PRES"]).copy()
    valid_pts["track_dist"] = [melissa_line.project(geom) for geom in valid_pts.geometry]
    valid_pts = (
        valid_pts.sort_values("track_dist")
        .drop_duplicates(subset=["track_dist"])
        .reset_index(drop=True)
    )

    pt_distances = valid_pts["track_dist"].values
    pt_pressures = valid_pts["USA_PRES"].values

    projected_locs = np.array([melissa_line.project(c) for c in centroids])
    interp_pressures = np.interp(projected_locs, pt_distances, pt_pressures)

    # Pressure deficit Delta_P = 1013.25 - P_min
    sample_grid["Delta_P"] = STANDARD_SEA_LEVEL_PRESSURE_MB - interp_pressures

    # Distance-decay interaction feature
    sample_grid["DeltaP_x_Dist"] = sample_grid["Delta_P"] * sample_grid["Dist_km"]

    print(f"  Dist_km span: [{sample_grid['Dist_km'].min():.2f}, {sample_grid['Dist_km'].max():.2f}] km (mean: {sample_grid['Dist_km'].mean():.2f} km)")
    print(f"  Delta_P span: [{sample_grid['Delta_P'].min():.2f}, {sample_grid['Delta_P'].max():.2f}] mb (mean: {sample_grid['Delta_P'].mean():.2f} mb)")

    # -------------------------------------------------------------------------
    # Exposure Controls, Parish Tagging & Dataset Export
    # -------------------------------------------------------------------------
    print("\n[5/5] Extracting WorldPop population density & assigning parishes...")

    # Extract demographic exposure control from WorldPop raster
    sample_grid_4326 = sample_grid.to_crs(CRS_GEOGRAPHIC)
    zonal_results = rasterstats.zonal_stats(
        sample_grid_4326, str(pop_raster_path), stats="mean"
    )
    pop_densities = [
        res["mean"] if (res["mean"] is not None and not np.isnan(res["mean"])) else 0.0
        for res in zonal_results
    ]
    sample_grid["Pop_Dens"] = pop_densities
    sample_grid["ln_Pop_Dens"] = np.log(sample_grid["Pop_Dens"] + 1.0)
    print(f"  Pop_Dens span: [{sample_grid['Pop_Dens'].min():.1f}, {sample_grid['Pop_Dens'].max():.1f}] people/km²")

    # Assign administrative parish by dominant area intersection
    inter = gpd.overlay(
        sample_grid[["cell_id", "geometry"]],
        parishes_gdf[["parish_name", "geometry"]],
        how="intersection",
    )
    inter["inter_area"] = inter.geometry.area
    dominant_parish = (
        inter.sort_values("inter_area", ascending=False)
        .groupby("cell_id")
        .first()
        .reset_index()
    )

    sample_grid = sample_grid.merge(
        dominant_parish[["cell_id", "parish_name"]], on="cell_id", how="left"
    )
    sample_grid = sample_grid.rename(columns={"parish_name": "parish"})

    # Check Kingston assignment: ensure cell intersecting downtown Kingston is tagged
    kingston_poly = parishes_gdf[parishes_gdf["parish_name"] == "Kingston"].geometry.iloc[0]
    k_inter = sample_grid[sample_grid.geometry.intersects(kingston_poly)]
    if not k_inter.empty:
        # Assign the cell with maximum Kingston overlap to Kingston
        best_k_cell_id = None
        max_k_area = 0.0
        for idx, row in k_inter.iterrows():
            area_k = row.geometry.intersection(kingston_poly).area
            if area_k > max_k_area:
                max_k_area = area_k
                best_k_cell_id = row["cell_id"]
        if best_k_cell_id is not None:
            sample_grid.loc[sample_grid["cell_id"] == best_k_cell_id, "parish"] = "Kingston"

    print(f"  Parishes represented: {sample_grid['parish'].nunique()} / 14 official parishes")
    print(f"  Parish value counts:\n{sample_grid['parish'].value_counts()}")

    # Format exact final columns
    final_cols = [
        "cell_id",
        "parish",
        "centroid_x",
        "centroid_y",
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

    # Tabular DataFrame
    regression_df = sample_grid[final_cols].copy()

    # Master GeoJSON
    master_geojson_gdf = sample_grid[final_cols + ["geometry"]].copy()

    # Export datasets
    geojson_out = PROCESSED_DATA_DIR / "grid_master.geojson"
    csv_out = PROCESSED_DATA_DIR / "regression_df.csv"

    master_geojson_gdf.to_file(geojson_out, driver="GeoJSON")
    regression_df.to_csv(csv_out, index=False)

    print("\n" + "=" * 70)
    print("SPATIAL PIPELINE EXECUTION SUCCESSFUL")
    print(f"  - Master Spatial Grid: {geojson_out} ({len(master_geojson_gdf)} features)")
    print(f"  - Master Regression CSV: {csv_out} ({len(regression_df)} rows, {len(regression_df.columns)} columns)")
    print("=" * 70)

    return {
        "n_cells": len(sample_grid),
        "total_buildings": int(sample_grid["total_buildings"].sum()),
        "damaged_buildings": int(sample_grid["damaged_buildings"].sum()),
        "mean_damage_ratio": float(sample_grid["Damage_Ratio"].mean()),
        "geojson_path": str(geojson_out),
        "csv_path": str(csv_out),
    }


if __name__ == "__main__":
    run_spatial_pipeline()
