########################################
#             UNCLASSIFIED             #
########################################
"""
This generate_mock_detections.py file is used to create mock YOLO detection
output from AIS messages. This will be used to test the functionality of
ship pairing code.
"""
import os

import numpy as np

from datetime import datetime

from data_utils.lon_lat_xy_utils import get_pixel_xy_from_lat_lon

INPUT_DIR = os.path.join(os.getcwd(), "inputs")
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
SETTINGS_NAME = "settings.json"


def generate_mock_detections(
        ais_file: str = INPUT_DIR + "/AIS_2017_11_01T21_MiamiPort_20ships_sorted_datetime_mmsi.csv",
        datetime_format: str = "%Y-%m-%dT%H:%M:%S",
        grid_size: np.ndarray = np.array([5, 5]),
        image_size: np.ndarray = np.array([1663, 771]),
        lat_lon_min_max: tuple = (np.array([25.7598, -80.1521]), np.array([25.7709, -80.1266]))) -> dict:

    # Pull in all messages from AIS file
    message_list = []
    with open(ais_file, 'r') as f:
        headers = f.readline()
        for line in f:
            message = line.split(',')
            message_list.append(message)

    # Sort AIS messages by time
    message_list = sorted(message_list, key=lambda x: x[1])

    # For every minute of data where there is at least one ship, build a grid cell matrix (gcm) containing "detections"
    gcm_list = []
    cell_dims = (image_size / grid_size).astype(np.uint16)
    for t in range(60): # taking advantage of the fact that we know the files contain one hour of data
        gcm = [[[] for _ in range(grid_size[0])] for _ in range(grid_size[1])]
        cur_messages = [i for i in message_list if datetime.strptime(message[1], datetime_format).hour == t]
        for m in cur_messages:
            lat_lon = np.array([float(m[2]), float(m[3])])
            pixel_xy = get_pixel_xy_from_lat_lon(lat_lon, image_size, lat_lon_min_max)
            if pixel_xy.shape[0] == 2:  # pixel_xy will be empty if the lat or lon are out of bounds
                column = (pixel_xy[0] / cell_dims[0]).astype(np.uint16)
                row = (pixel_xy[1] / cell_dims[1]).astype(np.uint16)
                m.insert(0, pixel_xy)
                gcm[row][column].append(m)
        gcm_list.append(gcm)
    return gcm_list
