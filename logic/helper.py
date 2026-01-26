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

import os
import re
import json
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pyproj import Transformer
from shapely.geometry import box
import logic.config as config

logger = logging.getLogger(__name__)

# --- Conversion and Utility Functions ---

def safe_float_conversion(x):
    """
    Safely extracts and converts a value to float from a dictionary structure.
    
    Args:
        x (dict): A dictionary expected to have a 'value' key.
        
    Returns:
        float: The converted value or np.nan if conversion fails or key is missing.
    """
    if isinstance(x, dict) and x.get('value') not in (None, ''):
        try:
            return float(x['value'])
        except ValueError:
            return np.nan
    return np.nan

def ISO_to_datetime(date_str):
    """
    Converts an ISO 8601 string to a datetime object, handling 'Z' suffix.
    
    Args:
        date_str (str): Date string in ISO format.
        
    Returns:
        datetime: Corresponding datetime object.
    """
    return datetime.fromisoformat(date_str.replace('Z', ''))

def date_time_to_ISO(dt_obj):
    """
    Converts a datetime object to an ISO 8601 formatted string.
    
    Args:
        dt_obj (datetime): Datetime object to convert.
        
    Returns:
        str: ISO formatted date string.
    """
    return dt_obj.strftime('%Y-%m-%dT%H:%M:%S%z')

def is_date(date_str, date_format=config.DATE_FORMAT):
    """
    Validates if a string matches the specified date format.
    
    Args:
        date_str (str): The string to validate.
        date_format (str): The expected format.
        
    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        datetime.strptime(date_str, date_format)
        return True
    except (ValueError, TypeError):
        return False

def round_time(time_str):
    """
    Rounds a time string (HH:MM) to the next 10-minute interval.
    
    Args:
        time_str (str): Time in HH:MM format.
        
    Returns:
        str: Rounded time in HH:MM format.
    """
    time_obj = datetime.strptime(time_str, '%H:%M')
    minute = (time_obj.minute // 10 + 1) * 10
    if minute == 60:
        time_obj += timedelta(hours=1)
        minute = 0
    return time_obj.replace(minute=minute).strftime('%H:%M')

# --- Geographic Logic ---

def convert_bbox_to_utm(lat_min, lat_max, long_min, long_max, epsg_projection):
    """
    Converts a bounding box from WGS84 (Lat/Long) to a specific UTM projection.
    
    This function uses a Transformer with always_xy=True to ensure consistent coordinate 
    ordering (Longitude first). It returns a geometric box and a DataFrame of bounds.
    
    Args:
        lat_min (float): Minimum latitude.
        lat_max (float): Maximum latitude.
        long_min (float): Minimum longitude.
        long_max (float): Maximum longitude.
        epsg_projection (int): EPSG code for the target UTM projection.
        
    Returns:
        tuple: (shapely.geometry.box, pandas.DataFrame containing X and Y bounds).
    """
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_projection}", always_xy=True)

    # Transform corners (Min/Max) to target CRS
    x_min, y_min = transformer.transform(long_min, lat_min)
    x_max, y_max = transformer.transform(long_max, lat_max)

    utm_bbox_coordinates = pd.DataFrame({
        'X': [x_min, x_max],
        'Y': [y_min, y_max]
    })

    # Create a solid geometry Polygon (box) representing the study area
    city_bbox = box(x_min, y_min, x_max, y_max)

    return city_bbox, utm_bbox_coordinates

# --- Temporal Logic ---

def compute_to_date(start_date, prevision_type, prevision_value):
    """
    Calculates a future date based on a start date and an offset.
    
    Args:
        start_date (str): Start date string in %Y-%m-%dT%H:%M:%S format.
        prevision_type (str): Unit of offset ('days', 'months', 'hours').
        prevision_value (int): Number of units to add.
        
    Returns:
        str: Resulting date string.
    """
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%dT%H:%M:%S')
    
    if prevision_type == 'days':
        new_date_obj = start_date_obj + timedelta(days=prevision_value)
    elif prevision_type == 'months':
        new_date_obj = start_date_obj + relativedelta(months=prevision_value)
    elif prevision_type == 'hours':
        new_date_obj = start_date_obj + timedelta(hours=prevision_value)
    else:
        raise ValueError(f"Invalid prevision type: {prevision_type}")
        
    return new_date_obj.strftime('%Y-%m-%dT%H:%M:%S')

def parse_from_date(from_date, to_date):
    """
    Parses relative date strings (e.g., '2-hours', '1-day') relative to a reference date.
    
    Args:
        from_date (str): Relative date string or specific date string.
        to_date (str): Reference end date.
        
    Returns:
        datetime: Calculated start datetime or None if format is not relative.
    """
    try:
        to_date_dt = datetime.strptime(to_date, config.DATE_FORMAT)
        match = re.match(r"(\d+)-(hours|day)", from_date)
        if not match:
            return None
            
        value, unit = int(match.group(1)), match.group(2)
        if unit == 'hours':
            return to_date_dt - timedelta(hours=value)
        return to_date_dt - timedelta(days=value)
    except Exception:
        return None

# --- API and Logging Services ---

def write_log(data):
    """
    Appends a JSON-formatted dictionary entry to a local log file with indentation.
    """
    file_name = 'log.txt'
    with open(file_name, 'a') as f:
        # Usiamo indent=4 per rendere leggibile anche il file log.txt esterno
        f.write(json.dumps(data, indent=4) + '\n')
        f.write("-" * 50 + "\n") # Separatore visivo nel file di testo

def get_sensor_real_time_data(params):
    """
    Fetches real-time sensor data from the Snap4City API.
    
    Args:
        params (dict): Query parameters for the API request.
        
    Returns:
        dict: JSON response from the API or an error dictionary.
    """
    base_url = os.getenv("BASE_URL", "https://www.snap4city.org") + "/superservicemap/api/v1"
    try:
        response = requests.get(base_url, params=params, timeout=30)
        write_log({'url': base_url, "params": params, "status": response.status_code})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        write_log({"url": base_url, "error": str(e)})
        return {"error": str(e)}

# --- Parameter Validation ---

def checkParameters(lat_min, long_min, lat_max, long_max, from_date_time, to_date_time):
    """
    Validates geographic boundaries and temporal consistency of input parameters.
    
    Checks that coordinates are within valid world ranges, that maximums exceed minimums,
    and that the timeframe is logically consistent.
    
    Args:
        lat_min, long_min, lat_max, long_max (float): Bounding box coordinates.
        from_date_time (str): Start time or relative duration.
        to_date_time (str): End time.
        
    Raises:
        ValueError: If any parameter violates geographic or logical constraints.
    """
    if not (-180 <= long_min <= 180 and -180 <= long_max <= 180):
        raise ValueError("Longitude must be between -180 and 180.")
    if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
        raise ValueError("Latitude must be between -90 and 90.")
    if long_max <= long_min or lat_max <= lat_min:
        raise ValueError("Max values must be greater than min values.")

    dt_from = parse_from_date(from_date_time, to_date_time)
    if dt_from is None:
        if not is_date(from_date_time):
            raise ValueError("Invalid from_date_time format.")
        dt_from = datetime.strptime(from_date_time, '%Y-%m-%dT%H:%M:%S')
            
    if not is_date(to_date_time):
        raise ValueError("Invalid to_date_time format.")
    dt_to = datetime.strptime(to_date_time, '%Y-%m-%dT%H:%M:%S')
        
    if dt_to <= dt_from:
        raise ValueError("to_date_time must be greater than from_date_time.")
