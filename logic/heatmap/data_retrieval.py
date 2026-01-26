''' Snap4city Computing HEATMAP - Data Retrieval Module.
    Copyright (C) 2024 DISIT Lab http://www.disit.org - University of Florence
'''

import os
import requests
import pandas as pd
import logging
from ..helper import write_log, parse_from_date, round_time
import logic.config as config

logger = logging.getLogger(__name__)

# --- Data Retrieval Functions ---

def load_sensors_data(lat_min, long_min, lat_max, long_max, sensor_category, from_date_time, to_date_time, token):
    """
    Coordinates the discovery and data download for all sensors within a bounding box.

    Args:
        lat_min, long_min, lat_max, long_max (float): Bounding box coordinates.
        sensor_category (str): The category of sensors to retrieve.
        from_date_time (str): Start time for data history.
        to_date_time (str): End time for data history.
        token (str): Authentication token for API access.

    Returns:
        list: A list of dictionaries, each containing sensor metadata and a DataFrame of time-series data.
    """
    sensors_data = []
    
    # Step 1: Discover sensor IDs (ServiceUris) in the geographic area
    service_uris = get_sensors_in_area(lat_min, long_min, lat_max, long_max, sensor_category, token)

    # Step 2: Fetch detailed history for each discovered sensor
    for service_uri in service_uris:
        data = fetch_sensor_data(service_uri, from_date_time, to_date_time, token)
        if data:
            sensors_data.append(data)

    return sensors_data


def fetch_sensor_data(service_uri, from_date_time, to_date_time, access_token):
    """
    Fetches historical time-series data for a specific sensor.

    Args:
        service_uri (str): The unique identifier for the sensor service.
        from_date_time (str): Start time (ISO or relative like '2-hours').
        to_date_time (str): End time (ISO).
        access_token (str): Bearer token for authorization.

    Returns:
        dict: Processed sensor data including 'sensorCoordinates', 'sensorName', 
              and 'sensorRealtimeData' (DataFrame). Returns None if data is missing.
    """
    # Handle relative date parsing (e.g., "1-day" -> calculated ISO date)
    parsed_start = parse_from_date(from_date_time, to_date_time)
    start_time = parsed_start.strftime(config.DATE_FORMAT) if parsed_start else from_date_time

    base_url = os.getenv("BASE_URL", config.SNAP4CITY_BASE_URL)
    api_url = (
        f"{base_url}/superservicemap/api/v1/?serviceUri={service_uri}"
        f"&fromTime={start_time}&toTime={to_date_time}&accessToken={access_token}"
    )
    
    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        res = requests.get(api_url, headers=headers, timeout=config.API_TIMEOUT)
        res.raise_for_status()
        raw_data = res.json()

        write_log({"url": api_url, "status": res.status_code})

        # Validate existence of results in the JSON structure
        if 'realtime' not in raw_data or 'results' not in raw_data['realtime']:
            return None

        bindings = raw_data['realtime']['results'].get('bindings', [])
        if not bindings:
            return None

        # Extract Sensor Metadata (Coordinates and Name)
        service_info = raw_data.get('Service', {}).get('features', [{}])[0]
        coordinates = service_info.get('geometry', {}).get('coordinates', [0, 0])
        name = service_info.get('properties', {}).get('name') or f"{service_uri}_sensor"

        # Process Time-Series Data into a DataFrame
        df = pd.DataFrame(bindings)

        if 'measuredTime' not in df.columns:
            return None

        # Extract 'value' from the JSON-like dictionaries in cells
        df['measuredTime'] = df['measuredTime'].apply(
            lambda x: x['value'] if isinstance(x, dict) and 'value' in x else None
        )

        # Convert to datetime and drop invalid rows
        df['measuredTime'] = pd.to_datetime(df['measuredTime'], utc=True, errors='coerce')
        df = df.dropna(subset=['measuredTime'])

        # Generate readable time/date components for easier grouping later
        df['time'] = df['measuredTime'].dt.strftime('%H:%M')
        df['date'] = df['measuredTime'].dt.strftime('%Y-%m-%d')
        df['day'] = df['measuredTime'].dt.day_name()

        # Round time to the next 10-minute slot for grid alignment
        df['time'] = df['time'].apply(round_time)

        # Create standardized ISO dateTime string for consistent sorting
        df['dateTime'] = df.apply(lambda row: f"{row['date']}T{row['time']}:00", axis=1)
        df = df.sort_values(by='measuredTime')
        df['sensorName'] = name

        return {
            'sensorCoordinates': coordinates,
            'sensorName': name,
            'sensorRealtimeData': df
        }

    except Exception as e:
        logger.error(f"Failed to fetch data for {service_uri}: {e}")
        write_log({"exception": str(e), "serviceUri": service_uri})
        return None


def get_sensors_in_area(lat_min, long_min, lat_max, long_max, sensor_category, token):
    """
    Retrieves a list of available ServiceUris within a geographic bounding box.

    Args:
        lat_min, long_min, lat_max, long_max (float): Area coordinates.
        sensor_category (str): Type of sensors to filter for.
        token (str): Auth token for logging purposes.

    Returns:
        list: A list of serviceUri strings.

    Raises:
        RuntimeError: If the API call fails.
        ValueError: If no sensors are found in the area.
    """
    base_url = os.getenv("BASE_URL", config.SNAP4CITY_BASE_URL)
    query_url = (
        f"{base_url}/superservicemap/api/v1/?selection="
        f"{lat_min};{long_min};{lat_max};{long_max}"
        f"&categories={sensor_category}"
        f"&maxResults={config.DISCOVERY_MAX_RESULTS}&maxDists={config.DISCOVERY_MAX_DIST}&format=json"
    )

    response = requests.get(query_url, timeout=config.API_TIMEOUT)
    write_log({"url": query_url, "status": response.status_code})

    if response.status_code != 200:
        raise RuntimeError(f"Sensor discovery failed with status {response.status_code}")

    data_json = response.json()
    features = data_json.get('Services', {}).get('features', [])

    service_uris = [f['properties']['serviceUri'] for f in features if 'serviceUri' in f['properties']]

    if not service_uris:
        raise ValueError(f"No {sensor_category} stations found in the selected area.")

    return service_uris