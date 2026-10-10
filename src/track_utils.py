"""
Meteorological storm track processing and atmospheric vortex physics.
"""

from typing import Tuple
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from src.config import (
    AIR_DENSITY_KG_M3,
    CORIOLIS_F_AT_18N,
    CRS_GEOGRAPHIC,
    CRS_METRIC,
    STANDARD_SEA_LEVEL_PRESSURE_MB,
)


def build_continuous_track(
    track_df: pd.DataFrame,
    storm_name: str = "MELISSA",
    season: int = 2025,
    crs_metric: str = CRS_METRIC
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Filter and convert IBTrACS 6-hourly points into a continuous LineString in metric CRS.

    Parameters
    ----------
    track_df : pd.DataFrame
        Raw or cleaned IBTrACS DataFrame.
    storm_name : str
        Target storm identifier (e.g. 'MELISSA').
    season : int
        Cyclone season year.
    crs_metric : str
        Target projected metric CRS.

    Returns
    -------
    Tuple[GeoDataFrame, GeoDataFrame]
        (track_line_gdf, track_points_gdf) in metric CRS.
    """
    # Filter storm observations
    mask = (track_df["NAME"].str.upper() == storm_name.upper())
    if "SEASON" in track_df.columns:
        mask = mask & (track_df["SEASON"] == season)

    storm_obs = track_df[mask].copy()
    storm_obs = storm_obs.sort_values("ISO_TIME").reset_index(drop=True)

    # Ensure numeric types
    for col in ["LAT", "LON", "USA_PRES", "USA_WIND"]:
        storm_obs[col] = pd.to_numeric(storm_obs[col], errors="coerce")

    storm_obs = storm_obs.dropna(subset=["LAT", "LON"]).reset_index(drop=True)

    # Point geometries in WGS84
    points = [Point(lon, lat) for lon, lat in zip(storm_obs["LON"], storm_obs["LAT"])]
    track_points = gpd.GeoDataFrame(storm_obs, geometry=points, crs=CRS_GEOGRAPHIC)

    if len(points) < 2:
        raise ValueError(
            f"Cyclone track for storm '{storm_name}' (season {season}) contains only {len(points)} "
            "valid coordinate observation(s). A continuous LineString requires at least 2 distinct synoptic fixes."
        )

    # Continuous LineString
    track_line = LineString(points)
    track_line_gdf = gpd.GeoDataFrame(
        [{"storm_name": storm_name, "season": season, "geometry": track_line}],
        crs=CRS_GEOGRAPHIC
    )

    # Reproject to metric CRS (EPSG:3448)
    track_points_metric = track_points.to_crs(crs_metric)
    track_line_metric = track_line_gdf.to_crs(crs_metric)

    return track_line_metric, track_points_metric


def holland_gradient_wind(
    dist_km: float,
    delta_p_mb: float,
    r_max_km: float = 25.0,
    b_param: float = 1.8
) -> float:
    """
    Calculate theoretical Holland (1980) gradient wind speed in knots.

    Parameters
    ----------
    dist_km : float
        Radial distance from cyclone eye center in kilometers.
    delta_p_mb : float
        Central pressure deficit (1013.25 - P_min) in millibars.
    r_max_km : float
        Radius of maximum wind in kilometers (default: 25 km).
    b_param : float
        Holland shape parameter (default: 1.8).

    Returns
    -------
    float
        Gradient wind speed in knots.
    """
    if dist_km <= 0.1:
        dist_km = 0.1  # Prevent division by zero at exact vortex center

    r_m = dist_km * 1000.0
    r_max_m = r_max_km * 1000.0
    delta_p_pa = delta_p_mb * 100.0  # Convert mb (hPa) to Pascals

    ratio = (r_max_m / r_m) ** b_param
    term_p = (b_param / AIR_DENSITY_KG_M3) * ratio * delta_p_pa * np.exp(-ratio)
    coriolis_term = (r_m * CORIOLIS_F_AT_18N / 2.0) ** 2

    # Gradient wind in m/s
    v_mps = np.sqrt(term_p + coriolis_term) - (r_m * CORIOLIS_F_AT_18N / 2.0)
    # Convert m/s to knots (1 m/s = 1.94384 kts)
    return float(v_mps * 1.94384)


def extract_track_features(
    grid_gdf: gpd.GeoDataFrame,
    track_line_gdf: gpd.GeoDataFrame,
    track_points_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Calculate Euclidean distance (km), pressure deficit (mb), and interaction terms.

    Parameters
    ----------
    grid_gdf : GeoDataFrame
        Grid polygons with centroid coordinates in metric CRS.
    track_line_gdf : GeoDataFrame
        Storm track LineString in metric CRS.
    track_points_gdf : GeoDataFrame
        Discrete observation points in metric CRS.

    Returns
    -------
    GeoDataFrame
        Grid with Dist_km, Delta_P, Wind_kts, and DeltaP_x_Dist appended.
    """
    track_line = track_line_gdf.geometry.iloc[0]

    distances_km = []
    pressures_deficit = []
    max_winds_kts = []

    for _, row in grid_gdf.iterrows():
        centroid = row["centroid"] if ("centroid" in row and pd.notna(row["centroid"])) else row.geometry.centroid

        # Metric Euclidean distance to continuous eye track
        dist_meters = centroid.distance(track_line)
        dist_km = dist_meters / 1000.0
        distances_km.append(dist_km)

        # Nearest discrete observation along track
        point_on_track = track_line.interpolate(track_line.project(centroid))
        obs_dists = track_points_gdf.geometry.distance(point_on_track)
        nearest_idx = obs_dists.idxmin()
        nearest_obs = track_points_gdf.loc[nearest_idx]

        # Pressure deficit
        pres_min = nearest_obs.get("USA_PRES", np.nan)
        delta_p = STANDARD_SEA_LEVEL_PRESSURE_MB - pres_min if pd.notna(pres_min) else np.nan
        pressures_deficit.append(delta_p)

        # Wind speed
        wind = nearest_obs.get("USA_WIND", np.nan)
        max_winds_kts.append(wind)

    result_grid = grid_gdf.copy()
    result_grid["Dist_km"] = distances_km
    result_grid["Delta_P"] = pressures_deficit
    result_grid["Wind_kts"] = max_winds_kts
    result_grid["DeltaP_x_Dist"] = result_grid["Delta_P"] * result_grid["Dist_km"]

    return result_grid
