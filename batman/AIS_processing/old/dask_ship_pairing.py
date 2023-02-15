from dask.distributed import Client, progress
client = Client(processes=False, threads_per_worker=4,
                n_workers=10, memory_limit='16GB')
print(client)

import os
import boto3
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from scipy import interpolate

import dask.array as da
import dask.dataframe as dd

s3 = boto3.resource('s3',
                    aws_access_key_id=None,
                    aws_secret_access_key=None)


def infer_mmsi(df_no_mmsi: dd.DataFrame, threshold_sec: int = 1800, threshold_m: int = 10E3,
               clustering_speed: int = 100, max_depth: int = 12) -> dd.DataFrame:
    """
    This function takes a Pandas DataFrame of AIS messages and a Pandas DataFrame if YOLO predictions and attempts to
    pair ships between the two dataframes.

    Parameters
    ----------
    df_no_mmsi: Pandas DataFrame
        Rows of YOLO predictions
    threshold_sec: int
        Time difference (seconds) threshold between an AIS message and a YOLO predicted ship
    threshold_m: int
        Distance (meters) threshold between an AIS message and  a YOLO predicted ship
    clustering_speed: int in m/s
    max_depth: int
        Limit for number of permutations for searching over permutations of greedy

    Returns
    -------
    Pandas DataFrame of all rows with the following columns:
        ['mmsi', 'closest timestamp', 'interval sec', 'distance m']
    """
    # deal with stupid warning on np.sum( sep2[ rows[[order]], cols[[order]] ] )
    import warnings
    warnings.simplefilter(action='ignore',
                          category=FutureWarning)

    # used in permutation search
    max_permutations = np.math.factorial(max_depth)

    df_no_mmsi_shape = df_no_mmsi.shape
    df_no_mmsi['mmsi'] = -1
    df_no_mmsi['closest timestamp'] = dd.NaT
    df_no_mmsi['interval sec'] = np.NaN
    df_no_mmsi['distance m'] = np.NaN

    latitude_interp = dict()
    longitude_interp = dict()
    time_nearest_interp = dict()
    separation_w_mmsi_no_mmsi = dict()
    index_no_mmsi = dict()

    pbar = tqdm(df_no_mmsi.groupby('date'), desc='iterating over dates in df_no_mmsi')
    for date_, df_no_mmsi_ in pbar:
        pbar.set_postfix({'date': date_})
        index_no_mmsi[date_] = df_no_mmsi_.index.to_flat_index()

        try:  # if the parquet file is corrupted or empty, we will catch a TypeError and try rebuilding the file
            df_w_mmsi = import_ais(date=date_)
        except TypeError:
            copy_ais_data(dd.DataFrame({'date': [date_]}), overwrite=True)
            df_w_mmsi = import_ais(date=date_)
        a, b = np.meshgrid(df_w_mmsi['timestamp'].to_numpy(int), df_no_mmsi_['timestamp'].to_numpy(int))

        df_w_mmsi_close = df_w_mmsi[np.any(np.abs(a - b) / 1E9 < threshold_sec, axis=0)]
        df_w_mmsi_close = df_w_mmsi_close[np.any(
            haversine_m(df_w_mmsi_close['longitude'], df_w_mmsi_close['latitude'], df_no_mmsi_['longitude'],
                        df_no_mmsi_['latitude'], meshgrid=True, convert2radians=True) < threshold_m, axis=0)]

        for mmsi_, df_ in df_w_mmsi_close.groupby('mmsi'):
            if df_.shape[0] > 1:
                latitude_interp[mmsi_, date_] = interpolate.interp1d(df_['timestamp'].to_numpy(int),
                                                                     df_['latitude'].to_numpy(float), fill_value=np.NaN,
                                                                     bounds_error=False)(
                    df_no_mmsi_['timestamp'].to_numpy(int))
                longitude_interp[mmsi_, date_] = interpolate.interp1d(df_['timestamp'].to_numpy(int),
                                                                      df_['longitude'].to_numpy(float),
                                                                      fill_value=np.NaN,
                                                                      bounds_error=False)(
                    df_no_mmsi_['timestamp'].to_numpy(int))
                time_nearest_interp[mmsi_, date_] = interpolate.interp1d(df_['timestamp'].to_numpy(int),
                                                                         df_['timestamp'].to_numpy(int), kind='nearest',
                                                                         fill_value=np.NaN,
                                                                         bounds_error=False)(
                    df_no_mmsi_['timestamp'].to_numpy(int))
                separation_w_mmsi_no_mmsi[mmsi_, date_] = haversine_m(longitude_interp[mmsi_, date_],
                                                                      latitude_interp[mmsi_, date_],
                                                                      df_no_mmsi_['longitude'].to_numpy(float),
                                                                      df_no_mmsi_['latitude'].to_numpy(float),
                                                                      convert2radians=True)

    t0, t1 = np.meshgrid(df_no_mmsi['timestamp'].to_numpy(int), df_no_mmsi['timestamp'].to_numpy(int))
    delta_m = np.hypot((t0 - t1) / clustering_speed, haversine_m(df_no_mmsi['longitude'].to_numpy(float),
                                                                 df_no_mmsi['latitude'].to_numpy(float),
                                                                 df_no_mmsi['longitude'].to_numpy(float),
                                                                 df_no_mmsi['latitude'].to_numpy(float),
                                                                 meshgrid=True, convert2radians=True))

    clustering = DBSCAN(eps=threshold_m, min_samples=1, metric='precomputed').fit(delta_m)

    pbar = tqdm(np.unique(clustering.labels_), desc='clustering')

    for label in pbar:
        tf = clustering.labels_ == label

        dates = df_no_mmsi.loc[tf, 'date'].drop_duplicates().to_list()
        pbar.set_postfix({'cluster #': label, 'count': sum(tf), 'dates': dates})

        inds = np.array(list(itertools.chain(*[index_no_mmsi[date] for date in index_no_mmsi if date in dates])))
        mmsis = np.array([key[0] for key, val in separation_w_mmsi_no_mmsi.items() if key[1] in dates])
        if any(mmsis):
            time_nearest = np.array([val for key, val in time_nearest_interp.items() if key[1] in dates])
            sep = np.array([val for key, val in separation_w_mmsi_no_mmsi.items() if key[1] in dates])

            rows = []
            cols = []
            sep2 = np.power(sep, 2)
            # finding greedy pairing
            for argsort in np.argsort(sep, axis=None):
                if len(cols) == max([sep.shape[1], max_depth]):
                    break
                r, c = np.unravel_index(argsort, sep.shape)
                if not (c in cols):
                    rows.append(r)
                    cols.append(c)
                elif (len(cols) > sep.shape[1]) and not (r in rows):
                    rows.append(r)
                    cols.append(c)
            rows = np.array(rows)
            cols = np.array(cols)

            best_val = np.inf
            count = 0
            pbar.set_postfix(
                {'cluster #': label, 'dates': dates, 'count': sum(tf), 'factorial depth': min([len(cols), max_depth])})
            best_order = ()
            for i, order in enumerate(itertools.permutations(range(len(cols))), min([len(cols), max_depth])):
                if i > max_permutations:
                    pbar.set_postfix_str('%s, exceeded max_permutations!' % pbar.postfix)
                    break
                val = np.sum(sep2[(rows[[order]], cols[[order]])])
                if val < best_val:
                    best_order = order
                    best_val = val
                    count += 1
                    pbar.set_postfix(
                        {'cluster #': label, 'dates': dates, 'count': sum(tf), 'factorial depth': max_depth,
                         '# improvements': count})

            df_no_mmsi.loc[inds[cols[[best_order]]], 'mmsi'] = mmsis[rows[[best_order]]]
            df_no_mmsi.loc[inds[cols[[best_order]]], 'closest timestamp'] = dd.to_datetime(
                time_nearest[rows[[best_order]], cols[[best_order]]])
            df_no_mmsi.loc[inds[cols[[best_order]]], 'interval sec'] = (df_no_mmsi.loc[inds[cols[
                [best_order]]], 'timestamp'].to_numpy(int) - time_nearest[rows[[best_order]], cols[[best_order]]]) / 1E9
            df_no_mmsi.loc[inds[cols[[best_order]]], 'distance m'] = sep[rows[[best_order]], cols[[best_order]]]

    df = df_no_mmsi.iloc[:, df_no_mmsi_shape[1]:].sort_index().astype(
        {'mmsi': np.int32, 'interval sec': np.float32, 'distance m': np.float32}).reset_index()
    return df


# In[6]:


def copy_and_import_predictions(key: str, bucket_name: str = 'dark-ships', nrows: int = None,
                                npartitions: int = 10):
    obj = s3.Object(bucket_name=bucket_name, key=key)
    pdf = pd.read_csv(obj.get()['Body']).head(n=nrows)
    df = dd.from_pandas(pdf, npartitions=npartitions)
    df = df.rename(columns={'datetime': 'timestamp'}).sort_values('timestamp')
    df['timestamp'] = dd.to_datetime(df['timestamp'])
    try:
        df['timestamp'] = df['timestamp'].dt.tz_convert(None)
    except TypeError:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
    # df = df.filter(['timestamp', 'date', 'latitude', 'longitude', 'vessel_type'])
    return df


def copy_ais_data(df, bucket_name: str = 'dark-ships', s3_ais_dir: str = 'AIS-data/MarineCadastre/csv/',
                  local_ais_dir: str = '~/ais/', overwrite=False, npartitions: int = 10):
    local_ais_dir = local_ais_dir.replace('~', os.path.expanduser('~'))
    pbar = tqdm(df['date'].unique(), desc="copying AIS files from s3")
    for date in pbar:
        pbar.set_postfix(({'date': date}))
        date_str = date.replace('-', '_')
        if overwrite or "AIS_" + date_str + ".parquet" not in os.listdir(local_ais_dir):
            obj = s3.Object(bucket_name=bucket_name, key=s3_ais_dir + "AIS_" + date_str + ".csv")
            pdf = pd.read_csv(obj.get()['Body'], engine='c', on_bad_lines='skip')
            df = dd.from_pandas(pdf, npartitions=npartitions)
            df = df.rename(columns={'BaseDateTime': 'timestamp', 'MMSI': 'mmsi', 'LAT': 'latitude', 'LON': 'longitude'})
            df['timestamp'] = dd.to_datetime(df['timestamp'])
            try:
                df['timestamp'] = df['timestamp'].dt.tz_convert(None)
            except TypeError:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            if '\n' in ''.join(df.columns):
                df.columns = pd.MultiIndex.from_tuples([tuple(col.split('\n')) for col in df.columns])
            df = df.replace([np.inf, -np.inf], np.nan)

            df = df.dropna(subset=['latitude', 'longitude', 'mmsi', 'timestamp'], how='any').drop_duplicates(
                subset=['mmsi', 'timestamp']).sort_values(['mmsi'])
            df = df.sort_values(['timestamp']).reset_index().astype({'mmsi': np.int32})
            df.to_parquet(local_ais_dir + "AIS_" + date_str + ".parquet")


def import_ais(date: str, local_ais_dir: str = "~/ais/", npartitions: int = 10):
    """
    This function reads a csv file from s3 to a Koalas dataframe. The df will be formatted differently depending on
    whether the csv contains AIS messages or YOLO predictions.

    Parameters
    ----------
    date: str
        Date corresponding to AIS file to be read (will be formatted as `AIS_{date}.parquet`)
    local_ais_dir: str
        Local directory holding AIS files

    Returns
    -------
    Dataframe containing the rows of the provided csv file
    """
    local_ais_dir = local_ais_dir.replace('~', os.path.expanduser('~'))
    date_str = date.replace('-', '_')
    return dd.read_parquet(local_ais_dir + "AIS_" + date_str + ".parquet", npartitions=npartitions)


# In[7]:


EARTH_RADIUS_KM = 6378


def haversine_km(lon1, lat1, lon2, lat2, bearing_too=False, meshgrid=False, convert2radians=False):
    """
    Calculate the great circle distance between two points
    on the earth (specified in radians!)

    All args must be of equal length.

    """
    # lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    if convert2radians:
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    if meshgrid:
        lon1, lon2 = da.meshgrid(lon1, lon2)
        lat1, lat2 = da.meshgrid(lat1, lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = da.sin(dlat / 2.0) ** 2 + da.cos(lat1) * da.cos(lat2) * da.sin(dlon / 2.0) ** 2
    c = 2 * da.arcsin(da.sqrt(a))
    km = EARTH_RADIUS_KM * c
    if bearing_too:
        bearing = da.arctan2(da.sin(dlon) * da.cos(lat2),
                             da.cos(lat1) * da.sin(lat2) - da.sin(lat1) * da.cos(lat2) * da.cos(dlon))
        return km, np.degrees(bearing)
    else:
        return km


def haversine_m(lon1, lat1, lon2, lat2, bearing_too=False, meshgrid=False, convert2radians=False):
    if bearing_too:
        km, bearing = haversine_km(lon1, lat1, lon2, lat2, bearing_too, meshgrid, convert2radians)
        return km * 1000, bearing
    else:
        return haversine_km(lon1, lat1, lon2, lat2, bearing_too, meshgrid, convert2radians) * 1000


# In[8]:


def create_master_df(df_infer_mmsi: dd.DataFrame, df_no_mmsi: dd.DataFrame, out_fname: str, how: str = 'right',
                     local_results_dir: str = '~/output/') -> (dd.DataFrame, str):
    """
    This function takes the dataframe of predictions and combines it with the results of infer_mmsi() as well as the
    original AIS data into one master dataframe.

    Parameters
    ----------
    df_infer_mmsi: Pandas DataFrame resulting from infer_mmsi()
    df_no_mmsi: Pandas DataFrame of predictions from copy_and_import_predictions()
    out_fname: str containing file name to save master df as (must include ".parquet" suffix)
    how: str
        'right' for only waypoints from files_no_mmsi, will include waypoints for which no mmsi could be inferred
        'inner' for only waypoints from files_no_mmsi for which an mmsi could be inferred
        'left' for all waypoints from files_w_mmsi and waypoints from files_no_mmsi for which an mmsi could be inferred
        'outer' for all waypoints from files_w_mmsi and waypoints from files_no_mmsi
    local_results_dir: str containing name of local directory in which to save output
    """
    df_infer_mmsi = df_infer_mmsi.join(df_no_mmsi).drop(columns=['ship_pixel_loc'])
    df_infer_mmsi = df_infer_mmsi.astype({'vessel_type': 'category', 'filename': 'category',
                                          'meters_per_pixel': np.float16})

    if how in ['right', 'outer']:
        df_master = [df_infer_mmsi[df_infer_mmsi['mmsi'] == -1]]
    else:
        df_master = []

    pbar = tqdm(df_infer_mmsi.groupby(df_infer_mmsi['closest timestamp'].dt.strftime('%Y-%m-%d')),
                desc='assembling master df')
    for date_, df_infer_mmsi_ in pbar:
        df_ = import_ais(date=date_)
        pbar.set_postfix({'Y-m-d': date_, 'shape': df_.shape})
        df_ = df_.merge(df_infer_mmsi_, left_on=['mmsi', 'timestamp'], right_on=['mmsi', 'closest timestamp'], how=how)
        df_master.append(df_)

    df_master = dd.concat(df_master, axis=0).sort_values(['mmsi', 'timestamp']).reset_index(drop=True)
    full_output_path = local_results_dir.replace('~', os.path.expanduser('~')) + out_fname
    df_master.to_parquet(full_output_path)
    return df_master, full_output_path


def copy_to_s3(file_path: str, bucket_name: str = 'dark-ships',
               s3_output_dir: str = 'ship-pairing-output/ec2-results-parquet/'):
    print("Writing {} to s3 location {}/{}".format(file_path, bucket_name, s3_output_dir))
    s3.meta.client.upload_file(file_path, bucket_name, s3_output_dir + file_path[file_path.rfind('/') + 1:])


if __name__ == "__main__":
    predictions_key = "EO-data/google-earth-eo-data/ship_truth_data.csv"

    num_rows = 10
    max_perm_depth = 6
    num_partitions_ais = 10

    preds_df = copy_and_import_predictions(key=predictions_key, nrows=num_rows)
    copy_ais_data(preds_df, overwrite=False, npartitions=num_partitions_ais)
    print(preds_df.head())

    results_df = infer_mmsi(preds_df.copy(), max_depth=max_perm_depth)
    print(results_df.head())

    fname = args.key[predictions_key.rfind('/') + 1:].replace('.csv', '_paired.parquet')
    merged_df, output_path = create_master_df(results_df, preds_df, fname)
    print(output_path)
    print(merged_df.head())

    copy_to_s3(output_path)

