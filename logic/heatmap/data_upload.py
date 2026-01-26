''' Snap4city Computing HEATMAP - Data Upload Module.
    Copyright (C) 2024 DISIT Lab http://www.disit.org - University of Florence
'''

import os
import time
import json
import logging
import requests
from urllib.parse import quote
from ..helper import write_log
import logic.config as config

logger = logging.getLogger(__name__)

# --- Data Formatting Functions ---

def create_interpolated_heatmap(interpolated_data, heatmap_name, metric_name, from_date_time, to_date_time, clustered, step_size, epsg_projection, file_flag):
    """
    Formats the interpolated grid results into a specific JSON structure required by the Snap4City backend.
    
    Args:
        interpolated_data (pd.DataFrame): Dataframe containing 'X', 'Y', and 'Z' (value).
        heatmap_name (str): Identifier for the heatmap.
        metric_name (str): The color map/metric identifier.
        from_date_time, to_date_time (str): Timeframe of the average.
        clustered (int): Clustering flag (0 or 1).
        step_size (float): The cell size of the grid.
        epsg_projection (int): EPSG code for coordinate projection.
        file_flag (int): Flag indicating if output is a file.

    Returns:
        dict: A dictionary containing a list of 'attributes' formatted for API ingestion.
    """
    heatmap_attributes = []
    
    for i, (_, row) in enumerate(interpolated_data.iterrows(), start=1):
        record = {
            'id': i,
            'mapName': heatmap_name,
            'metricName': metric_name,
            'description': f"Average from {from_date_time} to {to_date_time}",
            'clustered': clustered,
            'latitude': row['X'],  # UTM Easting
            'longitude': row['Y'], # UTM Northing
            'value': row['Z'],
            'date': f"{to_date_time}Z",
            'xLength': step_size,
            'yLength': step_size,
            'projection': int(epsg_projection),
            'file': file_flag,
            'org': config.DEVICE_PRODUCER
        }
        heatmap_attributes.append(record)

    logger.debug(f"Formatted {len(heatmap_attributes)} grid cells for upload.")
    return {'attributes': heatmap_attributes, 'saveStatus': None}

# --- Device Management Functions ---

def upload_heatmap_to_snap4city(token, heat_map_model_name, broker, subnature, device_name, heatmap_name, color_map, 
                                coordinates, interpolated_data, from_date_time, to_date_time):
    """
    Coordinates the creation of an IoT device and the subsequent upload of its metadata to Snap4City.
    """
    base_url = os.getenv("BASE_URL", config.SNAP4CITY_BASE_URL)
    orion_url = os.getenv("ORIONFILTER_BASE_URL", config.ORION_BASE_URL)
    broker = broker or os.getenv("DEFAULT_BROKER", config.DEFAULT_BROKER)

    # Construct the bounding box geometry
    lons, lats = coordinates['long'], coordinates['lat']
    wkt_poly = f"POLYGON(({lons.min()} {lats.min()},{lons.max()} {lats.min()},{lons.max()} {lats.max()},{lons.min()} {lats.max()},{lons.min()} {lats.min()}))"
    
    geo_json_poly = {
        "type": "Polygon",
        "coordinates": [[[lons.min(), lats.min()], [lons.max(), lats.min()], [lons.max(), lats.max()], [lons.min(), lats.max()], [lons.min(), lats.min()]]]
    }

    device_conf = {
        "token": token,
        "device_name": device_name,
        "heatmap_name": heatmap_name,
        "model": {
            "model_name": heat_map_model_name,
            "model_type": config.DEVICE_MODEL_TYPE,
            "model_kind": config.DEVICE_MODEL_KIND,
            "model_frequency": config.DEVICE_MODEL_FREQ,
            "model_contextbroker": broker,
            "model_format": config.DEVICE_MODEL_FORMAT,
            "model_subnature": subnature,
            "device_url": f"{base_url}/iot-directory/api/device.php?"
        },
        "producer": config.DEVICE_PRODUCER,
        "k1": config.IOT_K1,
        "k2": config.IOT_K2,
        "hlt": config.DEVICE_MODEL_TYPE,
        "lat": lats.values[0],
        "long": lons.values[0],
        "patch_url": f"{orion_url}/{broker}/v2/entities/",
        "color_map": color_map,
        "wkt": wkt_poly,
        "bounding_box": geo_json_poly,
        "size": len(interpolated_data),
        "date_observed": f"{to_date_time}Z",
        "description": f"Average from {from_date_time} to {to_date_time}",
        "maximum_date": to_date_time,
        "minimum_date": from_date_time
    }
    
    logger.debug("--------- HEATMAP DEVICE CONFIGURATION ---------")
    logger.debug(json.dumps(device_conf, indent=2))
    logger.debug("------------------------------------------------")

    # Execute Device Creation
    creation_res = create_heatmap_device(device_conf)
    results = {"device": {"POSTStatus": "Success" if creation_res.get("status") == "ok" else "Error"}}
    
    if creation_res.get("status") != "ok":
        results["device"]["error"] = creation_res.get("error_msg", "Unknown API error")
        return results

    # Pause to allow IOT Directory to propagate changes
    time.sleep(5)

    # Upload actual Heatmap Attributes (metadata)
    data_res = send_heatmap_device_data(device_conf)
    results["device_data"] = {
        "POSTStatus": "Success" if data_res.status_code in (200, 204) else "Failed",
        "http_status": data_res.status_code
    }

    return results

def create_heatmap_device(conf):
    """
    Registers a new Heatmap device in the IOT Directory using the tested legacy logic.
    Ensures all required query parameters (even empty ones) are sent to avoid 'Access Token not present' errors.
    """
    logger.info("Step: Creating IoT Device on Snap4City")
    token = conf['token']
    device_url = conf["model"]["device_url"]
    
    # Define the mandatory attributes for the Heatmap model
    # We keep the explicit list to ensure the backend receives exactly what it needs
    attr_list = [
        {"value_name": "dateObserved", "data_type": "string", "value_type": "timestamp", "editable": "0", "value_unit": "timestamp", "healthiness_criteria": "refresh_rate", "healthiness_value": "300", "real_time_flag": "false"},
        {"value_name": "mapName", "data_type": "string", "value_type": "Identifier", "editable": "0", "value_unit": "ID", "healthiness_criteria": "refresh_rate", "healthiness_value": "300", "real_time_flag": "false"},
        {"value_name": "colorMap", "data_type": "string", "value_type": "Identifier", "editable": "0", "value_unit": "ID", "healthiness_criteria": "refresh_rate", "healthiness_value": "300", "real_time_flag": "false"},
        {"value_name": "minimumDate", "data_type": "string", "value_type": "time", "editable": "0", "value_unit": "s", "healthiness_criteria": "refresh_rate", "healthiness_value": "300", "real_time_flag": "false"},
        {"value_name": "maximumDate", "data_type": "string", "value_type": "time", "editable": "0", "value_unit": "s", "healthiness_criteria": "refresh_rate", "healthiness_value": "300", "real_time_flag": "false"},
        {"value_name": "instances", "data_type": "integer", "value_type": "Count", "editable": "0", "value_unit": "#", "healthiness_criteria": "refresh_rate", "healthiness_value": "300", "real_time_flag": "false"},
        {"value_name": "description", "data_type": "string", "value_type": "status", "editable": "0", "value_unit": "status", "healthiness_criteria": "refresh_rate", "healthiness_value": "300", "real_time_flag": "false"},
        {"value_name": "boundingBox", "data_type": "json", "value_type": "Geometry", "editable": "0", "value_unit": "text", "healthiness_criteria": "refresh_rate", "healthiness_value": "300", "real_time_flag": "false"},
        {"value_name": "size", "data_type": "integer", "value_type": "status", "editable": "0", "value_unit": "status", "healthiness_criteria": "refresh_rate", "healthiness_value": "300", "real_time_flag": "false"}
    ]
    
    attributes = quote(json.dumps(attr_list))

    header = {
        "Content-Type": "application/json",
        "Accept": "application/x-www-form-urlencoded",
        "Authorization": f"Bearer {token}",
    }

    # Constructing the exact URL with all parameters, including empty ones (mac, visibility, etc.)
    # This specific order and completeness is what the legacy system expects.
    url = (
        f"{device_url}action=insert&attributes={attributes}&id={conf['device_name']}"
        f"&type={conf['model']['model_type']}&kind={conf['model']['model_kind']}"
        f"&contextbroker={conf['model']['model_contextbroker']}&format={conf['model']['model_format']}"
        f"&mac=&model={conf['model']['model_name']}&producer={conf['producer']}"
        f"&latitude={conf['lat']}&longitude={conf['long']}&visibility=&frequency={conf['model']['model_frequency']}"
        f"&accessToken={token}&k1={conf['k1']}&k2={conf['k2']}&edgegateway_type=&edgegateway_uri="
        f"&subnature={conf['model']['model_subnature']}&static_attributes=&servicePath=&nodered=false"
        f"&hlt={conf['hlt']}&wktGeometry={conf['wkt']}"
    )

    logger.debug(f"Device creation URL: {url}")

    try:
        response = requests.patch(url, headers=header, timeout=config.API_TIMEOUT)
        logger.debug(f"Create device HTTP status: {response.status_code}")
        
        if response.status_code >= 400:
            return {
                "status": "ko", 
                "http_status": response.status_code, 
                "error_msg": response.text
            }
        
        time.sleep(2) # Grace period for backend consistency
        return response.json() if response.status_code == 200 else {"status": "ok"}

    except Exception as e:
        logger.error(f"Error during device creation: {str(e)}")
        return {"status": "ko", "error_msg": str(e)}

def send_heatmap_device_data(conf):
    """
    Sends the metadata payload to the Orion Context Broker via OrionFilter.
    """
    payload = {
        "boundingBox": {"value": conf['bounding_box'], "type": "json"},
        "colorMap": {"value": conf['color_map'], "type": "string"},
        "dateObserved": {"value": conf['date_observed'], "type": "string"},
        "description": {"value": conf['description'], "type": "string"},
        "instances": {"value": 1, "type": "integer"},
        "mapName": {"value": conf['heatmap_name'], "type": "string"},
        "maximumDate": {"value": conf['maximum_date'], "type": "string"},
        "minimumDate": {"value": conf['minimum_date'], "type": "string"},
        "size": {"value": conf['size'], "type": "integer"}
    }
    
    url = f"{conf['patch_url']}{conf['device_name']}/attrs?elementid={conf['device_name']}&type={conf['model']['model_type']}"
    headers = {"Authorization": f"Bearer {conf['token']}", "Content-Type": "application/json"}
    
    response = requests.patch(url, data=json.dumps(payload), headers=headers, timeout=config.API_TIMEOUT)
    write_log({"endpoint": "Orion-Patch", "status": response.status_code})
    return response

# --- Grid Persistence Functions ---

def save_interpolated_data(interpolated_heatmap, heatmap_name, metric_name, to_date_time, token, info_heatmap=None):
    """
    Saves the heavy interpolated grid to the dedicated storage backend.
    
    Includes a 'Token Probe' check before attempting to send the full array.
    """
    info_heatmap = info_heatmap or {}
    insert_url = os.getenv("HEATMAP_INSERT_BASE_URL", config.HEATMAP_INSERT_URL)
    setmap_url = os.getenv("HEATMAP_SETMAP_URL", config.HEATMAP_SETMAP_URL)

    # --- TOKEN PROBE ---
    # Check if the session is still valid before sending large data
    probe_headers = {"Authorization": f"Bearer {token}"}
    try:
        # Pinging a lightweight directory endpoint as a probe
        base_probe = insert_url.rsplit('/', 1)[0]
        probe_res = requests.get(f"{base_probe}/probe", headers=probe_headers, timeout=5)
        if probe_res.status_code == 401:
            logger.error("Token expired during interpolation. Aborting upload.")
            info_heatmap['interpolation'] = {"POSTstatus": "Failed", "error": "Unauthorized/Token Expired"}
            return info_heatmap
    except Exception:
        pass # If probe endpoint doesn't exist, proceed with caution

    # --- MAIN UPLOAD ---
    try:
        response = requests.post(insert_url, json=interpolated_heatmap['attributes'], timeout=120)
        save_status = response.status_code
    except requests.RequestException as e:
        logger.error(f"Bulk insert failed: {e}")
        save_status = 500

    completed = 1 if save_status == 200 else 0
    
    # Notify the PHP controller to finalize the map registration
    final_query = f"{setmap_url}?mapName={heatmap_name}&metricName={metric_name}&date={to_date_time}Z&completed={completed}"
    try:
        final_res = requests.get(final_query, timeout=config.API_TIMEOUT)
        if final_res.status_code == 200 and completed == 1:
            info_heatmap['interpolation'] = {"POSTstatus": "Success"}
        else:
            info_heatmap['interpolation'] = {"POSTstatus": "Failed", "error": final_res.text}
    except Exception as e:
        info_heatmap['interpolation'] = {"POSTstatus": "Failed", "error": str(e)}

    return info_heatmap