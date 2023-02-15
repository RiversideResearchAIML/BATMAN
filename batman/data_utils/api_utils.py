#!/usr/bin/env python
# coding: utf-8
import warnings

import numpy as np
import pandas as pd
import requests
from timezonefinder import TimezoneFinder
from datetime import datetime, timedelta
from dark_ships.data_utils import s3_utils

pd.set_option('display.max_columns', 20, 'display.width', 1500)

DEFAULT_WEATHER_FILTER = ['maxtempC', 'mintempC', 'sunrise', 'sunset', 'moonrise', 'moonset', 'moon_phase',
                          'moon_illumination', 'tempC', 'windspeedKmph', 'winddirDegree', 'winddir16Point',
                          'weatherDesc', 'precipMM', 'humidity', 'visibility', 'pressure', 'cloudcover', 'sigHeight_m',
                          'swellHeight_m', 'swellDir', 'swellDir16Point', 'swellPeriod_secs', 'waterTemp_C', 'uvIndex',
                          'tideTime', 'tideHeight_mt', 'tide_type']
DEFAULT_POLLUTION_FILTER = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']


def api_query_s3_upload(lat_lon_list: list, dt_list: list, out_fname: str = None, bucket_name: str = "dark-ships",
                        contextual_data_dir: str = "contextual-data/dynamic"):
    """
    This function makes queries to the weather and pollution APIs based on the provided latitudes, longitudes, and
    dates. It then takes the responses of the APIs, compiles them into one dataframe per API, and uploads those
    dataframes as feather files to their respective directories within a provided s3.
    bucket.

    Parameters
    ----------
    lat_lon_list: list of tuples
        List containing tuples, where each tuple is populated by pairs of latitude and longitude as floats
    dt_list: list of datetimes
        List containing python datetimes for querying APIs
    out_fname: str
        Name of files to be uploaded to s3 in weather and pollution directories (do not include file extension). If
        left as None, the file names will be generated based on the provided date ranges.
    bucket_name: str
        Name of bucket in s3 in which feather files will be uploaded
    contextual_data_dir: str
        Path of the contextual dynamic data within the s3 bucket. This directory should contain both weather and
        pollution subdirectories.
    """
    w_df = pd.DataFrame()
    p_df = pd.DataFrame()

    for lat, lon in lat_lon_list:
        for dt in dt_list:
            u_start = int(dt.timestamp())
            u_end = int((dt + timedelta(days=1)).timestamp())
            w_df = w_df.append(query_weather_api(lat, lon, dt.strftime("%Y-%m-%d"), filter_data=DEFAULT_WEATHER_FILTER))
            p_df = p_df.append(query_pollution_api(lat, lon, u_start, u_end, filter_data=DEFAULT_POLLUTION_FILTER))

    if out_fname is None:
        out_fname = dt_list[0].strfime("%Y-%m-%d") + "_to_" + dt_list[-1].strfime("%Y-%m-%d") +\
                    "_latlon_" + lat_lon_list[0][0] + "_" + lat_lon_list[0][1]
    s3_utils.write_feather_to_s3(df=w_df, bucket_name=bucket_name,
                                 key_name=contextual_data_dir + "/weather_data/" + out_fname + ".feather")
    s3_utils.write_feather_to_s3(df=p_df, bucket_name=bucket_name,
                                 key_name=contextual_data_dir + "/pollution_data/" + out_fname + ".feather")


def api_query_to_feather(lat_lon_list: list, dt_list: list, out_fname: str = None,
                        root_dir: str = "contextual-data/dynamic"):
    """
    This function makes queries to the weather and pollution APIs based on the provided latitudes, longitudes, and
    dates. It then takes the responses of the APIs, compiles them into one dataframe per API, and uploads those
    dataframes as feather files to their respective directories within a provided s3.
    bucket.

    Parameters
    ----------
    lat_lon_list: list of tuples
        List containing tuples, where each tuple is populated by pairs of latitude and longitude as floats
    dt_list: list of datetimes
        List containing python datetimes for querying APIs
    out_fname: str
        Name of files to be uploaded to s3 in weather and pollution directories (do not include file extension). If
        left as None, the file names will be generated based on the provided date ranges.
    bucket_name: str
        Name of bucket in s3 in which feather files will be uploaded
    contextual_data_dir: str
        Path of the contextual dynamic data within the s3 bucket. This directory should contain both weather and
        pollution subdirectories.
    """
    w_df = pd.DataFrame()
    p_df = pd.DataFrame()

    for lat, lon, dt in lat_lon_list:
        u_start = int(dt.timestamp())
        u_end = int((dt + timedelta(days=1)).timestamp())
        w_df = w_df.append(query_weather_api(lat, lon, dt.strftime("%Y-%m-%d"), filter_data=DEFAULT_WEATHER_FILTER))
        p_df = p_df.append(query_pollution_api(lat, lon, u_start, u_end, filter_data=DEFAULT_POLLUTION_FILTER))

    if out_fname is None:
        out_fname = dt_list[0].strfime("%Y-%m-%d") + "_to_" + dt_list[-1].strfime("%Y-%m-%d") +\
                    "_latlon_" + lat_lon_list[0][0] + "_" + lat_lon_list[0][1]
    w_df.to_feather(root_dir + "/weather_data/" + out_fname + ".feather")
    p_df.to_feather(root_dir + "/weather_data/" + out_fname + ".feather")

        
def query_weather_api(lat: float, lon: float, dt: str, tp: int = 3, tides: str = "yes",
                      key: str = None, response_format: str = "json",
                      filter_data: list = None) -> pd.DataFrame:
    """
    This function queries WorldWeatherOnline for the weather, tidal and astronomical data at a specific latitude,
    longitude and date.
    Co-Authored by: Cody Freese & Ricky Dalrymple.

    Parameters
    ----------
    lat: float containing desired latitude
    lon: float containing desired longitude
    dt: str containing desired date, formatted as yyyy-MM-dd
    tp: int indicating weather forecast time interval in hours (options are 1, 3, 6, 12, 24)
    tides: str of "yes" or "no" indicating whether to include tide data
    key: string containing api key
    response_format: string requiring format to be either json or xml
    filter_data: list of strings containing fields to be included in response, if None all data will be included

    Returns
    -------
    pd.DataFrame with rows of weather data from API response
    """
    if tp not in [1, 3, 6, 12, 24]:
        warnings.warn("time period (tp) must be 1, 3, 6, 12, or 24... using default value of 3")
        tp = 3
    if tides != "yes" and tides != "no":
        warnings.warn("tides must be either 'yes' or 'no'... using default value of 'no'")
        tides = "no"

    response = requests.get("https://api.worldweatheronline.com/premium/v1/past-marine.ashx",
                            params={"key": key, "format": response_format, "q": str(lat) + "," + str(lon), "date": dt,
                                    "tp": tp, "tide": tides})

    # Get timezone info for provided latitude/longitude
    tf = TimezoneFinder()
    tz = tf.timezone_at(lat=lat, lng=lon)
    tz = int(tz[tz.find('GMT')+3:])

    tp = timedelta(hours=tp)
    response = response.json()
    data_list = []
    remove_list = []
    all_keys = response['data']['weather'][0]
    all_keys.update(response['data']['weather'][0]['hourly'][0])
    all_keys.update(response['data']['weather'][0]['astronomy'][0])
    breakpoint()
    all_keys.update(response['data']['weather'][0]['tides'][0]['tide_data'][0])
    all_keys = list(all_keys.keys())
    # won't need to check for these keys because the keys we are checking for are nested inside these dictionaries
    all_keys.remove('date'), all_keys.remove('astronomy'), all_keys.remove('tides'), all_keys.remove('hourly')
    if filter_data is not None:
        remove_list = [x for x in all_keys if x not in filter_data]

    for day in response['data']['weather']:
        tides_dict = {}
        for tide_event in day['tides'][0]['tide_data']:
            tide_dt_str = '{} UTC{:+03d}00'.format(tide_event['tideDateTime'], tz)
            tides_dict[datetime.strptime(tide_dt_str, '%Y-%m-%d %H:%M UTC%z')] = tide_event
        for hour_data in day['hourly']:
            time = int(hour_data['time'])
            dt_str = '{} {:04d} UTC{:+03d}00'.format(day['date'], time, tz)
            time = datetime.strptime(dt_str, '%Y-%m-%d %H%M UTC%z')
            hour_data['weatherDesc'] = hour_data['weatherDesc'][0]['value']
            hour_data.update(day['astronomy'][0])

            # Check for a tide event in this time period
            hour_data['tideTime'] = np.NAN
            hour_data['tideHeight_mt'] = np.NAN
            hour_data['tideDateTime'] = np.NAN
            hour_data['tide_type'] = np.NAN
            for tide_time, tide_data in tides_dict.items():
                if time.hour <= tide_time.hour <= (time + tp).hour:
                    hour_data.update(tide_data)
                    hour_data['tideDateTime'] = tide_time
                    hour_data['tideTime'] = tide_time.time()

            hour_data['maxtempC'] = day['maxtempC']
            hour_data['maxtempF'] = day['maxtempF']
            hour_data['mintempC'] = day['mintempC']
            hour_data['mintempF'] = day['mintempF']
            hour_data['uvIndex'] = day['uvIndex']
            for key in remove_list:
                hour_data.pop(key)
            hour_data['time'] = time
            hour_data['latitude'] = lat
            hour_data['longitude'] = lon
            data_list.append(hour_data)
    return pd.DataFrame(data=data_list).rename(columns={'time': 'datetime'})


def query_pollution_api(lat: float, lon: float, start: int, end: int,
                        api_key: str = None, filter_data: list = None) -> pd.DataFrame:
    """
    This function queries OpenWeatherMap.org for atmospheric particulates at a specific latitude & longitude.
    Author Cody Freese.
    Warning: Historical data is accessible from 27th November 2020.

    Parameters
    ----------
    lat: string containing desired latitude
    lon: string containing desired longitude
    start: integer containing desired start date (Unix, UTC)
    end: integer containing desired end date (Unix, UTC),
    api_key: string containing api key
    filter_data: list of strings containing fields to be included in response, if None all data will be included

    Returns
    -------
    pd.DataFrame with rows of weather data from API response
    """
    response = requests.get(
        'http://api.openweathermap.org/data/2.5/air_pollution/history',
        params={
            'lat': str(lat),
            'lon': str(lon),
            'start': start,
            'end': end,
            'appid': api_key,
        })

    response = response.json()
    data_list = []
    remove_list = []
    if filter_data is not None:
        remove_list = [x for x in response['list'][0]['components'].keys() if x not in filter_data]
    for data in response['list']:
        comp = data['components']
        comp['aqi'] = data['main']['aqi']
        for key in remove_list:
            comp.pop(key)
        comp['datetime'] = datetime.utcfromtimestamp(data['dt'])  # UTC timestamp
        comp['latitude'] = lat
        comp['longitude'] = lon
        data_list.append(comp)
    return pd.DataFrame(data=data_list)


if __name__ == '__main__':
    latitude = [30.553693, 40]
    longitude = [-80.440360, 20]
    lat_lon = [[26, -87, datetime(2021, 1, 1)],[29.50, -49.29, datetime(2021, 1, 1)]]
    date = [datetime(2021, 1, 1), datetime(2021, 1, 1)]
#     unix_start = int(datetime(2021, 1, 1).timestamp())
#     unix_end = int(datetime(2021, 1, 2).timestamp())
    
    api_query_to_feather(lat_lon_list = lat_lon, dt_list= date, out_fname = '2021-01-01',
                        root_dir = "contextual-data/dynamic")
        
    breakpoint()
#     weather_df = query_weather_api(latitude, longitude, date, filter_data=DEFAULT_WEATHER_FILTER)
#     pollution_df = query_pollution_api(latitude, longitude, unix_start, unix_end, filter_data=DEFAULT_POLLUTION_FILTER)
