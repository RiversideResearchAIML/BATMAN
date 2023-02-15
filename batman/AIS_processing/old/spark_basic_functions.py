import numpy as np
import databricks.koalas as ks
import pandas as pd

earth_radius_km = 6378
earth_radius_m = 6378E3


def extract_koalas_df_from_s3(s3_url: str, n_rows: int = None, is_ais=True):
    """
    TODO read multiple csv's into one df (shouldn't have to change much here)
    This function reads a csv file from s3 to a Koalas dataframe. The df will be formatted differently depending on
    whether the csv contains AIS messages or YOLO predictions.

    Parameters
    ----------
    s3_url: str
        URL of csv file formatted as s3a://<bucket-name>/<file-key>
    is_ais: bool
        True if the csv file contains AIS messages, otherwise the file will be expected to contain YOLO prediction info
    n_rows: int
        Number of rows to read from file (intended for debugging). Leave as None to read the entirety of the file
    header: bool
        True if the first row of the file contains column names.
    spark: SparkSession
        Existing SparkSession to create dataframe in. If left as None, parkSession.builder.getOrCreate() will be used
    app_name: str
        Name of the Spark Session to get or create. Only necessary if spark argument is included

    Returns
    -------
    Koalas dataframe containing the rows of the provided csv file, along with the SparkSession
    """
    df = ks.read_csv(s3_url, nrows=n_rows)
    if is_ais:
        df = df.rename(columns={'BaseDateTime': 'timestamp', 'MMSI': 'mmsi', 'LAT': 'latitude', 'LON': 'longitude'})
        df = df.dropna(subset=['mmsi'])
        df = df.astype({'mmsi': np.int32})
        df = df.dropna(subset=['latitude', 'longitude', 'timestamp'], how='any')
        df = df.drop_duplicates(subset=['mmsi', 'timestamp']).sort_values(['mmsi', 'timestamp']).reset_index()
    else:
        df = df.rename(columns={'datetime': 'timestamp'}).sort_values('timestamp')
        pdf = df.to_pandas()
        pdf['timestamp'] = pd.to_datetime(pdf['timestamp'])
        try:
            pdf['timestamp'] = pdf['timestamp'].dt.tz_convert(None)
        except:
            pdf['timestamp'] = pdf['timestamp'].dt.tz_localize(None)
        df = ks.from_pandas(pdf)
    return df


def extract_pandas_df_from_s3(s3_url: str, n_rows: int = None, is_ais=True):
    """
    TODO read multiple csv's into one df (shouldn't have to change much here)
    This function reads a csv file from s3 to a Pandas dataframe. The df will be formatted differently depending on
    whether the csv contains AIS messages or YOLO predictions.

    Parameters
    ----------
    s3_url: str
        URL of csv file formatted as s3a://<bucket-name>/<file-key>
    is_ais: bool
        True if the csv file contains AIS messages, otherwise the file will be expected to contain YOLO prediction info
    n_rows: int
        Number of rows to read from file (intended for debugging). Leave as None to read the entirety of the file
    header: bool
        True if the first row of the file contains column names.
    spark: SparkSession
        Existing SparkSession to create dataframe in. If left as None, parkSession.builder.getOrCreate() will be used
    app_name: str
        Name of the Spark Session to get or create. Only necessary if spark argument is included

    Returns
    -------
    Pandas dataframe containing the rows of the provided csv file, along with the SparkSession
    """
    df = pd.read_csv(s3_url, nrows=n_rows)
    if is_ais:
        df = df.rename(columns={'BaseDateTime': 'timestamp', 'MMSI': 'mmsi', 'LAT': 'latitude', 'LON': 'longitude'})
        df = df.dropna(subset=['mmsi'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.astype({'mmsi': np.int32})
        df = df.dropna(subset=['latitude', 'longitude', 'timestamp'], how='any')
        df = df.drop_duplicates(subset=['mmsi', 'timestamp']).sort_values(['mmsi', 'timestamp']).reset_index()
    else:
        df = df.rename(columns={'datetime': 'timestamp'}).sort_values('timestamp')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        try:
            df['timestamp'] = df['timestamp'].dt.tz_convert(None)
        except:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    return df


def haversine_km(lon1, lat1, lon2, lat2, bearing_too=False, meshgrid=False, convert2radians=False):
    """
    Calculate the great circle distance between two points
    on the earth (specified in radians!)

    All args must be of equal length.

    """
    # lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    if convert2radians:
        # lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)
    if meshgrid:
        lon1, lon2 = np.meshgrid(lon1, lon2)
        lat1, lat2 = np.meshgrid(lat1, lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = earth_radius_km * c
    if bearing_too:
        # https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
        bearing = np.arctan2(np.sin(dlon) * np.cos(lat2),
                             np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
        return km, np.degrees(bearing)
    else:
        return km


def haversine_m(lon1, lat1, lon2, lat2, bearing_too=False, meshgrid=False, convert2radians=False):
    if bearing_too:
        km, bearing = haversine_km(lon1, lat1, lon2, lat2, bearing_too, meshgrid, convert2radians)
        return km * 1000, bearing
    else:
        return haversine_km(lon1, lat1, lon2, lat2, bearing_too, meshgrid, convert2radians) * 1000
