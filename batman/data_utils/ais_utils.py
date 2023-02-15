import os
import numpy as np
import pandas as pd
from datetime import datetime
from operator import itemgetter

INPUT_DIR = os.path.join(os.getcwd(), "inputs")
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")


def import_ais(input_dir: str, num_files: int) -> pd.DataFrame:
    """
    Parameters
    ----------
    input_dir: str
            The directory containing AIS files to be read.
    num_files: int
            A limit to the number of files to be loaded.

    Returns
    -------
    ais_df: DataFrame
            Pandas DataFrame containing AIS messages read, indexed by MMSI and ordered
            by datetime.
    """
    files = np.sort([f for f in os.listdir(input_dir) if f.endswith('.csv')])
    if len(files) < 1:
        raise IOError("No csv files found in provided directory.")

    ais_df = pd.concat(map(lambda f: pd.read_csv(f, low_memory=False), files[:num_files]))
    ais_df.rename(columns={'speed': 'speed (knots/h)'}, inplace=True)
    ais_df.dropna(subset=['timestamp', 'mmsi'], inplace=True)  # without this the row is meaningless
    ais_df['datetime'] = pd.to_datetime(ais_df['timestamp'], errors='coerce')  # creating proper datetime
    ais_df.drop(labels='timestamp', axis=1, inplace=True)
    ais_df.drop_duplicates(subset=['mmsi', 'datetime'], inplace=True)
    ais_df.sort_values(['mmsi', 'datetime'], inplace=True)

    return ais_df


def get_basic_ais_info(ais_file: str = os.path.join(INPUT_DIR, 'AIS_2017_11_MiamiPort'
                                                               '/2017_11_01T21_20ships_sorted_datetime_mmsi.csv')):
    """
    This function takes a *.csv file containing AIS messages and returns a NumPy array containing just the MMSI,
    datetime, and lat/lon values of each AIS message.

    Parameters
    ----------
    ais_file: str containing file path to *.csv file with AIS data

    Returns
    -------
    NumPy array of shape (N, 4) where at each index of N there contains the MMSI, datetime, lat, and lon info
    """
    ais_df = pd.read_csv(ais_file)
    # TODO populate this column renaming as new AIS sources are added if they have different column names
    ais_df = ais_df.rename(columns={"MMSI": "mmsi", "BaseDateTime": "datetime", "LAT": "lat", "LON": "lon",
                                    "Latitude": "lat", "Longitude": "lon"})
    ais_df = ais_df.loc[:, ["mmsi", "datetime", "lat", "lon"]]
    return ais_df.to_numpy()


def ais_filter_by_lat_lon(input_file='./AIS_data/AIS_ASCII_by_UTM_Month/2017/AIS_2017_11_Zone17.csv',
                          # lat_bounds=[25.71, 25.81],
                          # lon_bounds=[-80.15, -80.08],
                          lat_bounds=[25.73698, 25.78061],
                          lon_bounds=[-80.14720, -80.08855],
                          output_file='./AIS_data/AIS_2017_11_MiamiPort/AIS_2017_11_MiamiPort.csv'):
    """
    Filter AIS messages to include only messages broadcast from within a user provided area
    defined in latitude and longitude values.
    NOTE: We are using files to hold data rather than taking a list of AIS messages as input and
              outputting a list of filtered AIS messages because files can contain 3+ GB of AIS messages.

    Parameters
    -----------
    input_file: string
            Path of input file containing AIS messages to be filtered

    lat_bounds: float list of length 2
            Latitude boundaries of the area of interest

    lon_bounds: float list of length 2
            Longitude boundaries of the area of interest

    output_file: string
            Path of output file, which will contain the filtered AIS messages
    """
    with open(output_file, 'w') as out_f:
        with open(input_file, 'r') as in_f:
            print('Filtering file by lat/lon: ' + input_file)
            headers = in_f.readline()
            out_f.write(headers)

            i = 1
            for line in in_f:
                if i % 10000000 == 0:
                    print('Reading line {}...'.format(i))
                i += 1
                # line = line.strip('\n')
                ais_message = line.split(',')
                lat = float(ais_message[2])
                if not (lat_bounds[0] <= lat <= lat_bounds[1]):
                    continue
                lon = float(ais_message[3])
                if not (lon_bounds[0] <= lon <= lon_bounds[1]):
                    continue

                out_f.write(line)


def ais_filter_by_num_ships(input_file='./AIS_data/AIS_2017_11_01T14_MiamiPort.csv',
                            n_ships=20,
                            use_avg_sog=False,
                            min_sog=None,
                            statuses=None,
                            output_file='./AIS_data/AIS_2017_11_01T14_MiamiPort_20ships.csv'):
    """
    Filter AIS messages to include only messages broadcast from a given number of ships. This is to limit
    the number of messages and ships so when the data is used in simulation, the simulation software is
    not overwhelmed and the videos of the AoI are not too cluttered. There are also options to only
    include ships with specific statuses and/or a minimum speed over ground.
    NOTE: We are using files to hold data rather than taking a list of AIS messages as input and
              outputting a list of filtered AIS messages because files can contain 3+ GB of AIS messages.

    Parameters
    -----------
    input_file: string
            Path of input file containing AIS messages to be filtered

    n_ships: int
            Maximum number of unique ships to include in AIS messages

    use_avg_sog: boolean
            If this parameter is True, then the n_ships chosen from the will be the 20 ships with
            the highest average SOG

    min_sog: float
            Minimum speed over ground for a ship to be included in AIS messages

    statuses: list of strings
            List of statuses that a ship must have to be included in AIS messages

    output_file: string
            Path of output file, which will contain the filtered AIS messages
    """
    ship_list = []
    ship_dict = {}
    with open(output_file, 'w') as out_f:
        with open(input_file, 'r') as in_f:
            print('Filtering file by number of ships: ' + input_file)
            headers = in_f.readline()
            out_f.write(headers)

            i = 1
            for line in in_f:
                if i % 1000000 == 0:
                    print('Reading line {}...'.format(i))
                i += 1

                ais_message = line.split(',')
                ship_mmsi = ais_message[0]
                if use_avg_sog:
                    if ship_mmsi not in ship_dict.keys():
                        ship_dict[ship_mmsi] = []
                    ship_dict[ship_mmsi].append(ais_message)
                    continue
                if ship_mmsi in ship_list:
                    out_f.write(line)
                    continue
                if len(ship_list) >= n_ships:
                    continue
                sog = float(ais_message[4])
                if min_sog is not None and sog < min_sog:
                    continue
                status = ais_message[11]
                if statuses is not None and status not in statuses:
                    continue
                ship_list.append(ship_mmsi)
                out_f.write(line)

    # Calculate avg SOG's
    if use_avg_sog:
        sog_avgs = {}
        for key in ship_dict.keys():
            cur_sum = 0.
            for ais_message in ship_dict[key]:
                cur_sum += float(ais_message[4])
            sog_avgs[key] = cur_sum / len(ship_dict[key])

        # Sort dictionary by average and write to output file
        sog_avgs = dict(sorted(sog_avgs.items(), key=itemgetter(1), reverse=True)[:20])
        print('Ships with top 20 average SOGs')
        for key, item in sog_avgs.items():
            print('{}: {}'.format(key, item))
        with open(output_file, 'w') as out_f:
            out_f.write(headers)
            for mmsi in sog_avgs.keys():
                for ais_message in ship_dict[mmsi]:
                    out_f.write(','.join(ais_message))


def ais_filter_by_datetime(input_file='./AIS_data/AIS_2017_11_01_MiamiPort.csv',
                           date_time='2017-11-01T14',
                           output_file='./AIS_data/AIS_2017_11_01T14_MiamiPort.csv'):
    """
    Filter AIS messages so that only those from a specific date are included in the resulting file.
    NOTE: We are using files to hold data rather than taking a list of AIS messages as input and
              outputting a list of filtered AIS messages because files can contain 3+ GB of AIS messages.

    Parameters
    -----------
    input_file: string
            Path of input file containing AIS messages to be filtered

    date_time: string
            String representing the desired date to take AIS messages from.
            Should be formatted as: YYYY-MM-DD

    output_file: string
            Path of output file, which will contain the filtered AIS messages
    """
    with open(output_file, 'w') as out_f:
        with open(input_file, 'r') as in_f:
            print('Filtering file by date: ' + input_file)
            headers = in_f.readline()
            out_f.write(headers)

            for line in in_f:
                if date_time in line:
                    out_f.write(line)


def sort_ais_messages(input_file='./AIS_data/AIS_2017_11_01_MiamiPort_20Ships_sorted_by_datetime.csv',
                      sort_by_col='MMSI',
                      output_file='./AIS_data/AIS_2017_11_01_MiamiPort_20Ships_sorted_by_datetime_MMSI.csv'):
    """
    Sort AIS messages by a specified column. This is particularly useful in using AIS information to
    generate paths for visually simulated ships.
    NOTE: We are using files to hold data rather than taking a list of AIS messages as input and
              outputting a list of filtered AIS messages because files can contain 3+ GB of AIS messages.

    Parameters
    -----------
    input_file: string
            Path of input file containing AIS messages to be filtered

    sort_by_col: string
            AIS column that messages are to be sorted by (i.e. "MMSI", "BaseDateTime", "SOG", ...)

    output_file: string
            Path of output file, which will contain the filtered AIS messages
    """
    lines_dict = {}
    with open(input_file, 'r') as in_f:
        headers = in_f.readline()
        headers_list = headers.split(',')
        col_index = headers_list.index(sort_by_col)

        for line in in_f:
            d = line.split(',')
            if len(d) > 0:
                d = d[col_index]
                if sort_by_col == 'BaseDateTime':
                    d = datetime.strptime(d, '%Y-%m-%dT%H:%M:%S')
                lines_dict[line] = d

    sorted_data = {k: v for k, v in sorted(lines_dict.items(), key=lambda item: item[1])}

    with open(output_file, 'w') as out_f:
        out_f.write(headers)
        for line in sorted_data.keys():
            out_f.write(line)


def find_best_hour(input_file='./AIS_data/AIS_2017_11_MiamiPort/AIS_2017_11_MiamiPort.csv'):
    """
    Finds hour within provided AIS data in which there is the most activity by ships. This is determined by average SOG
    Parameters
    ----------
    input_file: string containing path to file of AIS data to be evaluated

    Returns
    -------
    int indicating the hour 0-24 with the most ship activity by SOG
    """
    hours = {}
    for hour in range(0, 24):
        hours[hour] = []

    with open(input_file, 'r') as f:
        f.readline()
        for line in f:
            ais_message = line.split(',')
            basedatetime = datetime.strptime(ais_message[1], '%Y-%m-%dT%H:%M:%S')
            hours[basedatetime.hour].append(float(ais_message[4]))

    max_avg = 0
    best_hour = -1
    for key in hours.keys():
        cur_avg = np.average(hours[key])
        print('{}: {} knots'.format(key, cur_avg))
        if cur_avg > max_avg:
            max_avg = cur_avg
            best_hour = key

    print('\nBest hour: {} with average SOG of {} knots'.format(best_hour, max_avg))
    return best_hour
