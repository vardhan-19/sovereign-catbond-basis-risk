"""
Spatial grid tessellation and geometry processing utilities in EPSG:3448.
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import box
from src.config import CRS_METRIC, DEFAULT_GRID_SIZE_METERS, MIN_LAND_AREA_KM2


def create_tessellation_grid(
    boundary_gdf: gpd.GeoDataFrame,
    cell_size_m: float = DEFAULT_GRID_SIZE_METERS,
    crs_metric: str = CRS_METRIC,
    min_land_area_km2: float = MIN_LAND_AREA_KM2
) -> gpd.GeoDataFrame:
    """
    Generate an orthogonal polygon grid over Jamaica's land territory.

    Parameters
    ----------
    boundary_gdf : GeoDataFrame
        Boundary polygon of Jamaica (e.g. dissolved 14 parishes).
    cell_size_m : float
        Grid cell dimension in meters (default: 10,000 m = 10 km).
    crs_metric : str
        Projected CRS preserving metric distances (default: EPSG:3448).
    min_land_area_km2 : float
        Threshold to filter out coastal slivers (default: 25 km²).

    Returns
    -------
    GeoDataFrame
        Clipped, sliver-filtered grid cells with unique cell_id, area, and centroids.
    """
    # Ensure boundary is in metric CRS
    boundary_proj = boundary_gdf.to_crs(crs_metric)
    dissolved_boundary = boundary_proj.dissolve()

    # Bounding envelope in metric meters
    xmin, ymin, xmax, ymax = dissolved_boundary.total_bounds

    # Construct regular square boxes
    x_coords = np.arange(xmin, xmax, cell_size_m)
    y_coords = np.arange(ymin, ymax, cell_size_m)

    grid_boxes = []
    raw_id = 0
    for x in x_coords:
        for y in y_coords:
            grid_boxes.append({
                "raw_id": raw_id,
                "geometry": box(x, y, x + cell_size_m, y + cell_size_m)
            })
            raw_id += 1

    raw_grid = gpd.GeoDataFrame(grid_boxes, crs=crs_metric)

    # Intersection clip to land territory
    clipped_grid = gpd.overlay(raw_grid, dissolved_boundary, how="intersection")

    # Calculate metric land area per cell in km²
    clipped_grid["land_area_km2"] = clipped_grid.geometry.area / 1_000_000.0

    # Filter out shoreline slivers
    valid_grid = clipped_grid[clipped_grid["land_area_km2"] >= min_land_area_km2].copy()
    valid_grid = valid_grid.reset_index(drop=True)
    valid_grid["cell_id"] = valid_grid.index

    # Centroid coordinates in meters for spatial distance & kernel calculations
    centroids = valid_grid.geometry.centroid
    valid_grid["centroid_x"] = centroids.x
    valid_grid["centroid_y"] = centroids.y

    return valid_grid[["cell_id", "land_area_km2", "centroid_x", "centroid_y", "geometry"]]
