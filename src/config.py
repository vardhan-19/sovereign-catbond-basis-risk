"""
Global configuration parameters and spatial constants for Jamaica Cat Bond Analysis.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_IBTRACS_DIR = RAW_DATA_DIR / "ibtracs"
RAW_COPERNICUS_DIR = RAW_DATA_DIR / "copernicus"
RAW_BOUNDARIES_DIR = RAW_DATA_DIR / "boundaries"
RAW_POPULATION_DIR = RAW_DATA_DIR / "population"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MAPS_DIR = OUTPUTS_DIR / "maps"

# Coordinate Reference Systems (CRS)
CRS_METRIC = "EPSG:3448"          # JAD2001 / Jamaica National Grid (Meters)
CRS_GEOGRAPHIC = "EPSG:4326"      # WGS84 (Decimal Degrees)
CRS_WEB_MERCATOR = "EPSG:3857"    # Web Mercator for Basemap Tiles

# Physical & Atmospheric Constants
STANDARD_SEA_LEVEL_PRESSURE_MB = 1013.25  # Reference standard atmospheric pressure (mb / hPa)
AIR_DENSITY_KG_M3 = 1.15                  # Tropical maritime air density
CORIOLIS_F_AT_18N = 4.5e-5                # Coriolis parameter f at 18 degrees North

# Grid & Sample Filtering Parameters
DEFAULT_GRID_SIZE_METERS = 10000.0        # 10 km x 10 km standard grid cell
MIN_LAND_AREA_KM2 = 25.0                  # Minimum land area to filter coastal slivers
MIN_BUILDING_COUNT = 10                   # Minimum surveyed buildings for regression sample

# Sovereign Financial Parameters
CAR_136_PRINCIPAL_USD = 150_000_000.0     # USD 150 million IBRD CAR 136 notional
CAR_RENEWAL_2026_USD = 200_000_000.0      # USD 200 million renewal target notional

# Spatial HAC Bandwidths (Conley Cutoffs in km)
CONLEY_CUTOFFS_KM = [25.0, 50.0, 75.0]
DEFAULT_CONLEY_CUTOFF_KM = 50.0
