''' Snap4city Computing HEATMAP - Data Preprocessing Module.
    Copyright (C) 2024 DISIT Lab http://www.disit.org - University of Florence
'''

import pandas as pd
import numpy as np
import logging
from ..helper import safe_float_conversion
import logic.config as config

logger = logging.getLogger(__name__)

def preprocess_sensors_data(sensors_data, value_types, sensor_category, info_heatmap):
    """
    Main entry point for cleaning and preparing sensor data for interpolation.
    
    The workflow includes:
    1. Removing sensors with no data.
    2. Validating that requested value types exist in the dataset.
    3. Computing temporal averages per sensor.
    4. Aggregating multiple sensors sharing the same coordinates.
    5. Final sanitization of outlier or sentinel values.

    Args:
        sensors_data (list): List of sensor dictionaries retrieved from the API.
        value_types (list): List of strings representing the metric names to process.
        sensor_category (str): Subnature/Category of the sensors (e.g., 'Traffic_sensor').
        info_heatmap (dict): Dictionary to store metadata and status messages.

    Returns:
        pd.DataFrame: Cleaned DataFrame with ['lat', 'long', 'value'] columns.
    
    Raises:
        ValueError: If no valid data is found after any preprocessing step.
    """
    
    # 1. Remove sensors that are completely empty
    sensors_data = drop_empty_sensors(sensors_data)
    if not sensors_data:
        raise ValueError(f"No Available Data for {value_types}")

    # 2. Map requested value types to the available sensor columns
    names_matrix = validate_value_types(sensors_data, value_types)

    # 3. Compute the average value for each individual sensor over its time series
    data = compute_average_values(sensors_data, names_matrix)
    
    # 4. Handle domain-specific sanitization (e.g., no negative values for air quality)
    data = sanitize_values(data, sensor_category)

    # 5. Aggregate sensors that share exact spatial coordinates
    data = aggregate_sensors(data, sensor_category)

    # 6. Final cleaning of sentinel values and global NaN checks
    return clean_data_values(data, info_heatmap, value_types)


def drop_empty_sensors(sensors_data):
    """
    Filters out sensor entries where the realtime data is entirely null.
    """
    logger.info("Step 1: Dropping sensors with no data.")
    return [
        entry for entry in sensors_data
        if not entry['sensorRealtimeData'].isnull().all().all()
    ]


def validate_value_types(sensors_data, value_types):
    """
    Ensures that at least one of the requested value types exists in the sensor columns.
    Creates a mapping matrix for efficient retrieval.
    """
    logger.info("Step 2: Validating requested ValueTypes.")
    sensors_data_index, val_type_index, var_name_list = [], [], []
    
    for i, entry in enumerate(sensors_data):
        cols = entry['sensorRealtimeData'].columns
        for value in value_types:
            if value in cols:
                sensors_data_index.append(i)
                val_type_index.append(cols.get_loc(value))
                var_name_list.append(value)

    if not var_name_list:
        raise ValueError(f"Incorrect valueType. None of {value_types} were found.")

    return pd.DataFrame({
        'sensorDataIndex': sensors_data_index,
        'varNameList': var_name_list,
        'valTypeIndex': val_type_index
    })


def compute_average_values(sensors_data, names_matrix):
    """
    Calculates the temporal mean for each sensor's specific metric.
    """
    logger.info("Step 3: Computing temporal averages per sensor.")
    data = pd.DataFrame(columns=["lat", "long", "value"], index=range(len(sensors_data)))

    for j, entry in enumerate(sensors_data):
        # Extract the specific variable name for this sensor from the matrix
        matching_rows = names_matrix[names_matrix['sensorDataIndex'] == j]
        if not matching_rows.empty:
            var_name = matching_rows['varNameList'].values[0]
            series = entry['sensorRealtimeData'][var_name]

            # Convert to float using the helper utility
            clean_values = series.apply(safe_float_conversion)
            
            if clean_values.isna().all():
                data.at[j, "value"] = np.nan
            else:
                data.at[j, "value"] = clean_values.mean(skipna=True)

        # Assign coordinates (GeoJSON order is Longitude [0], Latitude [1])
        data.at[j, "lat"] = entry['sensorCoordinates'][1]
        data.at[j, "long"] = entry['sensorCoordinates'][0]

    return data


def sanitize_values(data, sensor_category):
    """
    Applies physical constraints based on the sensor type.
    Example: Air quality or Traffic counts cannot be negative.
    """
    data = data.copy()
    
    if sensor_category in config.PHYSICAL_POSITIVE_CATS:
        data.loc[data["value"] < 0, "value"] = np.nan

    return data


def aggregate_sensors(data, sensor_category):
    """
    Aggregates data points that share the exact same Latitude and Longitude.
    Different strategies are applied depending on the category.
    """
    if sensor_category in config.AGG_MEAN_CATS:
        agg_func = "mean"
    elif sensor_category in config.AGG_PERCENTILE_CATS:
        agg_func = lambda x: np.percentile(x, config.AGG_PERCENTILE_VALUE)
    else:
        # Default for Traffic and others: take the maximum observed impact
        agg_func = "max"

    grouped = (
        data
        .groupby(["lat", "long"], as_index=False)["value"]
        .agg(agg_func)
    )

    return grouped.dropna(subset=["value"])


def clean_data_values(data, info_heatmap, value_types):
    """
    Performs final cleanup: handles sentinel values (9999) and verifies 
    that data is not completely empty before returning.
    """
    logger.info("Step 4: Final data sanitization.")
    
    if data.empty or data['value'].isna().all():
        msg = f"No Available Data for {value_types}: Dataset is empty after processing."
        info_heatmap["message"].append(msg)
        logger.warning(msg)
    else:
        # Replace common sensor error/sentinel codes with NaN
        data['value'] = data['value'].replace(config.SENTINEL_VALUES, np.nan)
        data = data.dropna(subset=["value"])
        
    return data.infer_objects(copy=False)