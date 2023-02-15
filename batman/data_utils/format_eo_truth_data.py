"""
This file contains code that will read Google Earth annotation files from an s3 bucket and saves the contained
information in pandas dataframes as feather files.

@author rdalrymple
"""
import boto3
import pandas as pd
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup
from tqdm import tqdm
from io import BytesIO
from pyarrow.feather import write_feather

from dark_ships.data_utils.lat_lon_xy_utils import get_lat_lon_from_pixel_xy


pd.set_option('display.max_columns', 15, 'display.width', 1500)


def format_eo_truth_data(bucket_name: str = 'dark-ships',
                         ann_dir_path: str = 'EO-data/google-earth-eo-data/voc_annotations/',
                         out_f_path: str = 'EO-data/google-earth-eo-data/ship_truth_data.feather'):
    """
    This function reads all VOC formatted annotation files in the provided annotation directory and converts them to
    a dataframe. The dataframe is then saved to the provided output file path in s3.

    Parameters
    ----------
    bucket_name: str containing name of s3 bucket to connect to
    ann_dir_path: str containing directory path within the s3 bucket to VOC annotations
    out_f_path: str containing file path within bucket in which truth data will be saved
    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)

    datetime_list = []
    lat_list, lon_list = [], []
    ship_pixel_loc_list = []
    f_name_list = []
    im_w_list, im_h_list = [], []
    lat1_list, lon1_list, lat2_list, lon2_list = [], [], [], []
    m_per_pixel_list = []
    vessel_type_list = []
    x1_list, y1_list, x2_list, y2_list = [], [], [], []
    for obj in tqdm(bucket.objects.filter(Prefix=ann_dir_path), desc='Iterating through image annotations...'):
        xml_body = obj.get()['Body'].read().decode()
        bs_data = BeautifulSoup(xml_body, 'xml')
        ann = bs_data.find('annotation')
        im_w, im_h, _ = np.array(ann.find('size').get_text().strip().split('\n')).astype(np.uint16)
        f_name = ann.find('filename').get_text()
        datetime_str = ann.find('date').get_text() + ' ' + ann.find('time_estimate').get_text()
        datetime_estimate = datetime.strptime(datetime_str, '%Y-%m-%d %H%M UTC%z')
        m_per_pixel = float(ann.find('meters_per_pixel').get_text())
        lat1, lon1, lat2, lon2 = np.array(ann.find('lat_lon_bounds').get_text().strip().split('\n')).astype(np.float64)
        for ship in ann.find_all('object'):
            ship_class = int(ship.find('class').get_text())
            x1, y1, x2, y2 = np.array(ship.find('bndbox').get_text().strip().split('\n')).astype(np.uint16)
            ship_pixel_loc = (int(x1 + (x2-x1)/2), int(y1 + (y2-y1)/2))
            datetime_list.append(datetime_estimate)
            ship_pixel_loc_list.append(ship_pixel_loc)
            lat, lon = get_lat_lon_from_pixel_xy(x=ship_pixel_loc[0], y=ship_pixel_loc[1], image_size=(im_w, im_h),
                                                 lat_range=(lat1, lat2), lon_range=(lon1, lon2))
            lat_list.append(lat), lon_list.append(lon)
            f_name_list.append(f_name)
            im_w_list.append(im_w), im_h_list.append(im_h)
            lat1_list.append(lat1), lon1_list.append(lon1), lat2_list.append(lat2), lon2_list.append(lon2)
            m_per_pixel_list.append(m_per_pixel)
            vessel_type_list.append(ship_class)
            x1_list.append(x1), y1_list.append(y1), x2_list.append(x2), y2_list.append(y2)

            df = pd.DataFrame({
                'datetime': datetime_list,
                'latitude': lat_list,
                'longitude': lon_list,
                'filename': f_name_list,
                'image_w': im_w_list, 'image_h': im_h_list,
                'lat1': lat1_list, 'lon1': lon1_list, 'lat2': lat2_list, 'lon2': lon2_list,
                'meters_per_pixel': m_per_pixel_list,
                'vessel_type': vessel_type_list,
                'x1': x1_list, 'y1': y1_list, 'x2': x2_list, 'y2': y2_list,
                'ship_pixel_loc': ship_pixel_loc_list
            })

    # write df to feather file and upload to s3
    with BytesIO() as buf:
        write_feather(df, buf)
        s3.Object('dark-ships', key=out_f_path).put(Body=buf.getvalue())


if __name__ == '__main__':
    format_eo_truth_data()
