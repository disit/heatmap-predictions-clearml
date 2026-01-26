''' Snap4city Computing HEATMAP - Configuration Module.
    Copyright (C) 2026 DISIT Lab http://www.disit.org - University of Florence

    This module centralizes all configurable parameters for the heatmap generation pipeline.
    It controls API endpoints, interpolation hyperparameters, data preprocessing rules,
    and IoT device registration metadata.

    Usage:
        import logic.config as config
        timeout = config.API_TIMEOUT
'''

# =============================================================================
# 1. ORCHESTRATION & GENERAL SETTINGS
# =============================================================================

# Default Coordinate Reference System (EPSG code) for UTM projection.
# 32632 = WGS 84 / UTM zone 32N (Central Italy/Europe).
DEFAULT_EPSG = 32632

# Standard Lat/Lon CRS used for input/output.
CRS_LATLON = "EPSG:4326"

# Flags controlling the generation workflow.
DEFAULT_CLUSTERED = 0      # 0: Generate standard heatmap, 1: Generate clustered view.
DEFAULT_FILE_FLAG = 0      # 0: Save to Database only, 1: Generate physical file on server.
DEFAULT_BROKER = "orionUNIFI"      # Default Context Broker tenant.

# Logic threshold: Minimum number of valid sensors required to attempt interpolation.
# If fewer sensors are found, the process aborts to avoid statistical errors.
MIN_SENSORS_REQUIRED = 3


# =============================================================================
# 2. API & DATA RETRIEVAL
# =============================================================================

# Base endpoint for Snap4City APIs.
SNAP4CITY_BASE_URL = "https://www.snap4city.org"

# Network timeout (in seconds) for HTTP requests to avoid hanging processes.
API_TIMEOUT = 30

# Constraints for the Sensor Discovery API (SuperServiceMap).
DISCOVERY_MAX_RESULTS = 100  # Max number of sensors to retrieve per query.
DISCOVERY_MAX_DIST = 5       # Max distance parameter (API specific).

# Date format used for API query strings.
DATE_FORMAT = '%Y-%m-%dT%H:%M:%S'


# =============================================================================
# 3. DATA PREPROCESSING & CLEANING
# =============================================================================

# Sensor categories where negative values are physically impossible.
# Any negative value found for these will be treated as NaN (Sensor Error).
PHYSICAL_POSITIVE_CATS = {
    "Air_quality_monitoring_station",
    "Traffic_sensor",
    "Weather_sensor"
}

# --- Aggregation Strategies ---
# Defines how to handle multiple sensors located at the exact same coordinate.

# Strategy: Arithmetic Mean
AGG_MEAN_CATS = {
    "Air_quality_monitoring_station", 
    "Weather_sensor"
}

# Strategy: Percentile (e.g., for Noise monitoring where peaks matter more)
AGG_PERCENTILE_CATS = {
    "Noise_monitoring_station"
}

# The percentile value to use if AGG_PERCENTILE_CATS strategy is active.
AGG_PERCENTILE_VALUE = 85

# Sentinel values indicating sensor hardware errors (to be converted to NaN).
SENTINEL_VALUES = [-9999, 9999, -999, 999]


# =============================================================================
# 4. SPATIAL INTERPOLATION (ALGORITHMS)
# =============================================================================

# --- General Grid Settings ---
# Maximum number of grid cells allowed. If the area requires more cells at
# base resolution, the resolution (step size) is automatically increased.
MAX_CELLS = 10000

# Base resolution target (in meters). The algorithm tries to respect this
# unless MAX_CELLS is exceeded.
BASE_CELL_SIZE = 10.0

# --- Inverse Distance Weighting (IDW) Hyperparameters ---
# Power parameter 'p'. Higher values assign greater influence to values 
# closest to the interpolated point.
IDW_POWER = 4

# Distance (meters) from the nearest sensor where the value starts fading to zero.
IDW_FADE_DISTANCE = 800

# Smoothing factor for the fade-out curve. Higher = smoother transition.
IDW_FADE_SMOOTHING = 5.0

# --- Akima Spline Hyperparameters ---
# Buffer multiplier for the Convex Hull.
# 3.0 means the interpolation area extends 3x the cell size beyond the hull.
AKIMA_HULL_BUFFER_FACTOR = 3.0

# Number of control points for the 1D spline interpolation rows/cols.
AKIMA_SPLINE_POINTS = 5

# Number of nearest neighbors (k) used in the cKDTree query.
AKIMA_K_NEIGHBORS = 6

# Inverse distance weight power used during the Akima pre-calculation step.
AKIMA_WEIGHT_POWER = 3.5


# =============================================================================
# 5. POST-PROCESSING
# =============================================================================

# Standard deviation (sigma) for the Gaussian Filter applied to the final grid.
# Used to smooth out sharp artifacts from the interpolation.
GAUSSIAN_SIGMA = 1.5


# =============================================================================
# 6. IOT DEVICE REGISTRATION & UPLOAD
# =============================================================================

# Orion Context Broker endpoints.
ORION_BASE_URL = "https://www.snap4city.org/orionfilter"

# Legacy API Keys for IoT Directory authentication.
IOT_K1 = "cdfc46e7-75fd-46c5-b11e-04e231b08f37"
IOT_K2 = "24f146b3-f2e8-43b8-b29f-0f9f1dd6cac5"

# --- Device Metadata Templates ---
DEVICE_PRODUCER = "DISIT"
DEVICE_MODEL_TYPE = "Heatmap"
DEVICE_MODEL_KIND = "sensor"
DEVICE_MODEL_FREQ = "600"    # Refresh rate in seconds
DEVICE_MODEL_FORMAT = "json"

# --- Internal Backend Endpoints ---
# These URLs point to internal cluster services for high-performance writing.
# They are typically accessible only from within the VPN/Cluster network.
HEATMAP_INSERT_URL = "http://192.168.0.59:8000/insertArray"
HEATMAP_SETMAP_URL = "http://192.168.0.59/setMap.php"

# Standard Headers
JSON_HEADER = {"Content-Type": "application/json"}