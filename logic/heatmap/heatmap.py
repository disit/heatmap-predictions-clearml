''' Snap4city Computing HEATMAP.
   Copyright (C) 2024 DISIT Lab http://www.disit.org - University of Florence

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as
   published by the Free Software Foundation, either version 3 of the
   License, or (at your option) any later version.
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.
   You should have received a copy of the GNU Affero General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import logging
import json
from datetime import datetime
import numpy as np
import pandas as pd
from pyproj import Transformer
import logic.config as config

# --- Internal Module Imports ---
from .data_upload import upload_heatmap_to_snap4city, save_interpolated_data, create_interpolated_heatmap
from .data_interpolation import Interpolator
from .data_preprocessing import preprocess_sensors_data
from .data_retrieval import load_sensors_data
from ..helper import checkParameters, convert_bbox_to_utm

# Optional: Silence downcasting warnings for newer Pandas versions
try:
    pd.set_option("future.no_silent_downcasting", True)
except (KeyError, AttributeError):
    pass

logger = logging.getLogger(__name__)

class HeatmapStatus:
    """
    Manages the structured response for the Snap4city Heatmap generation process.
    Encapsulates the status of device creation, data upload, and interpolation.
    """
    def __init__(self, name="default", date_time=None):
        self.heatmapName = name
        self.dateTime = date_time
        self.message = []
        self.device = {}
        self.device_data = {}
        self.interpolation = {}

    def add_message(self, msg):
        """Adds a log message to the main status list."""
        self.message.append(msg)
        logger.info(f"HeatmapStatus Message: {msg}")

    def set_error(self, section, status_msg, http_status=None, error_detail=None):
        """
        Standardizes error reporting for sub-sections.
        
        Args:
            section (str): The section to update ('device', 'device_data', or 'interpolation').
            status_msg (str): The value for 'POSTStatus'.
            http_status (int, optional): The HTTP response code.
            error_detail (str, optional): The detailed error message.
        """
        data = {
            "POSTStatus": status_msg,
            "http_status": http_status,
            "error": str(error_detail) if error_detail else None
        }
        if section == 'device': self.device = data
        elif section == 'device_data': self.device_data = data
        elif section == 'interpolation': self.interpolation = data

    def to_dict(self):
        """Exports the object to the dictionary format expected by Snap4City."""
        res = {
            "heatmapName": self.heatmapName,
            "dateTime": self.dateTime,
            "message": self.message
        }
        if self.device: res["device"] = self.device
        if self.device_data: res["device_data"] = self.device_data
        if self.interpolation: res["interpolation"] = self.interpolation
        return res

def generate_heatmap(params: dict):
    """
    Main orchestrator for heatmap generation with full logging and error handling.
    """
    
    # --- Start Parameter Extraction ---
    long_min = params.get("long_min")
    long_max = params.get("long_max")
    lat_min = params.get("lat_min")
    lat_max = params.get("lat_max")
    
    # Use config defaults for optional parameters
    epsg_projection = params.get("epsg_projection", config.DEFAULT_EPSG)
    clustered = params.get("clustered", config.DEFAULT_CLUSTERED)
    file_flag = params.get("file", config.DEFAULT_FILE_FLAG)
    broker = params.get("broker", config.DEFAULT_BROKER)
    max_cells = params.get("max_cells", config.MAX_CELLS)

    value_types = params.get("value_types")
    subnature = params.get("subnature")
    scenario = params.get("scenario")
    color_map = params.get("color_map")
    from_date_time = params.get("from_date_time")
    to_date_time = params.get("to_date_time")
    token = params.get("token")
    heat_map_model_name = params.get("heat_map_model_name")
    model_method = params.get("model")

    # --- Initial Logging & Validation ---
    logger.debug("--------- CHECK ON PARAMETERS START ---------")
    logger.debug(datetime.now())
    
    try:
        checkParameters(lat_min, long_min, lat_max, long_max, from_date_time, to_date_time)
    except ValueError as e:
        logger.error(f"Parameter validation failed: {e}")
        status = HeatmapStatus(name="Validation_Error", date_time=to_date_time)
        status.add_message(f"Validation Error: {str(e)}")
        return status.to_dict()

    logger.debug("--------- CHECK ON PARAMETERS END ---------")
    logger.debug(datetime.now())

    # --- Setup Heatmap Name & Status Object ---
    if isinstance(value_types, str):
        value_types = [vt.strip() for vt in value_types.split(",")]

    heatmap_name = f"{scenario}_" + "_".join(value_types or [])
    logger.info(f"Heatmap name: {heatmap_name}")
    logger.info(f"Value types: {value_types}")

    status = HeatmapStatus(name=heatmap_name, date_time=to_date_time)
    metric_name = color_map
    sensor_category = subnature

    # --- Data Retrieval ---
    logger.debug("--------- UPLOAD ALL SENSOR STATIONS IN THE AREA OF INTEREST - START ---------")
    logger.debug(datetime.now())
    try:
        sensors_data = load_sensors_data(
            lat_min, long_min, lat_max, long_max, 
            sensor_category, from_date_time, to_date_time, token
        )
    except Exception as e:
        logger.error(f"Data retrieval failed: {e}")
        status.add_message(f"Error loading sensors data: {str(e)}")
        return status.to_dict()
    
    logger.debug("--------- UPLOAD ALL SENSOR STATIONS IN THE AREA OF INTEREST - END ---------")
    logger.debug(datetime.now())

    # --- Preprocessing ---
    logger.debug("--------- PREPROCESSING -- START ---------")
    logger.debug(datetime.now())
    try:
        data = preprocess_sensors_data(sensors_data, value_types, sensor_category, status.to_dict())
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        status.add_message(f"Preprocessing Error: {str(e)}")
        return status.to_dict()
    
    logger.debug("--------- PREPROCESSING -- END ---------")
    logger.debug(datetime.now())

    # --- Coordinate Conversion (UTM) ---
    logger.debug("--------- LAT-LONG BBOX CONVERSION TO UTM - START ---------")
    logger.debug(datetime.now())
    city_bbox, _ = convert_bbox_to_utm(
        float(lat_min), float(lat_max), float(long_min), float(long_max), epsg_projection
    )
    logger.debug("--------- LAT-LONG BBOX CONVERSION TO UTM - END ---------")
    logger.debug(datetime.now())

    # --- Interpolation ---
    logger.debug("--------- DATA INTERPOLATION - START ---------")
    logger.debug(datetime.now())
    
    if len(data) >= config.MIN_SENSORS_REQUIRED:
        data = data.dropna().drop_duplicates().reset_index(drop=True)
        
        # Proiezione punti sensori
        transformer = Transformer.from_crs(config.CRS_LATLON, f"EPSG:{epsg_projection}", always_xy=True)
        data['X'], data['Y'] = transformer.transform(data['long'].values, data['lat'].values)

        xy_known = data[['X', 'Y']].values
        val_known = data['value'].astype(float).values
        
        # Esecuzione Interpolatore
        interpolator = Interpolator.build(model_method, xy_known, val_known, max_cells, bbox=city_bbox)
        grid_x, grid_y, grid_z, step_size = interpolator.run()
        
        logger.info(f"Grid shape: {grid_x.shape} -> total points: {grid_x.size}")

        mask_valid = ~np.isnan(grid_z.ravel())
        interpolated_df = pd.DataFrame({
            'X': grid_x.ravel()[mask_valid],
            'Y': grid_y.ravel()[mask_valid],
            'Z': grid_z.ravel()[mask_valid]
        })
        logger.info(f"interpolatedData dim: {len(interpolated_df)}")
    else:
        msg = f"Not enough data points for interpolation. At least {config.MIN_SENSORS_REQUIRED} valid data points are required."
        logger.error(msg)
        status.add_message(msg)
        return status.to_dict()

    logger.debug("--------- DATA INTERPOLATION - END ---------")
    logger.debug(datetime.now())

    # --- Upload & Persistence ---
    logger.debug("--------- INTERPOLATED DATA LIST CREATION - START ---------")
    logger.debug(datetime.now())
    
    interpolated_heatmap = create_interpolated_heatmap(
        interpolated_df, heatmap_name, metric_name, from_date_time, to_date_time, 
        clustered, step_size, epsg_projection, file_flag, model_method
    )
    
    logger.debug("--------- INTERPOLATED DATA LIST CREATION - END ---------")
    logger.debug(datetime.now())

    logger.debug("--------- UPLOAD HEATMAP DEVICE AND DATA - START ---------")
    logger.debug(datetime.now())
    
    upload_res = upload_heatmap_to_snap4city(
        token, heat_map_model_name, broker, sensor_category, heatmap_name, heatmap_name, color_map, 
        data[['long', 'lat']], interpolated_df, from_date_time, to_date_time, model_method
    )
    status.device = upload_res.get('device', {})
    status.device_data = upload_res.get('device_data', {})

    logger.debug("--------- UPLOAD HEATMAP DEVICE AND DATA - END ---------")
    logger.debug(datetime.now())

    # Log di campionamento (Sample)
    logger.debug("--------- INTERPOLATED HEATMAP SAMPLE ---------")
    if len(interpolated_heatmap['attributes']) > 0:
        sample_json = json.dumps(interpolated_heatmap['attributes'][:5], indent=2)
        for line in sample_json.splitlines():
            logger.debug(f"  {line}") 
    logger.debug("------------------------------------------------")

    logger.debug("--------- SAVING INTERPOLATED DATA LIST - START ---------")
    logger.debug(datetime.now())
    
    save_res = save_interpolated_data(interpolated_heatmap, heatmap_name, metric_name, to_date_time, token)
    status.interpolation = save_res.get('interpolation', {})

    logger.debug("--------- SAVING INTERPOLATED DATA LIST - END ---------")
    logger.debug(datetime.now())
    
    return status.to_dict()