import pandas as pd
import numpy as np
import databricks.koalas as ks
import itertools

from tqdm import tqdm
from sklearn.cluster import DBSCAN
from scipy import interpolate
from pyspark.sql.utils import AnalysisException

from dark_ships.AIS_processing import spark_basic_functions as sbf

pd.set_option('display.max_columns', 20, 'display.width', 1500)


def koalas_infer_mmsi(df_no_mmsi: ks.DataFrame, threshold_sec: int = 1800, threshold_m: int = 10E3,
                      clustering_speed: int = 100, max_depth: int = 12, n_rows: int = None) -> ks.DataFrame:
    """
    This function takes a Koalas DataFrame of AIS messages and a Koalas DataFrame if YOLO predictions and attempts to
    pair ships between the two dataframes.

    Parameters
    ----------
    df_no_mmsi: Koalas DataFrame
        Rows of YOLO predictions
    threshold_sec: int
        Time difference (seconds) threshold between an AIS message and a YOLO predicted ship
    threshold_m: int
        Distance (meters) threshold between an AIS message and  a YOLO predicted ship
    clustering_speed: int in m/s
    max_depth: int
        Limit for number of permutations for searching over permutations of greedy
    n_rows: int
        Limit for number of rows to be read from a file into a dataframe, leave as None to read entire files
    spark: SparkSession
        Existing SparkSession to create dataframe in. If left as None, parkSession.builder.getOrCreate() will be used
    app_name: str
        Name of the Spark Session to get or create. Only necessary if spark argument is included

    Returns
    -------
    Koalas DataFrame of all rows with the following columns:
        ['mmsi', 'closest timestamp', 'interval sec', 'distance m']
    """
    # used in permutation search
    max_permutations = np.math.factorial(max_depth)

    df_no_mmsi['date'] = df_no_mmsi['timestamp'].dt.strftime('%Y-%m-%d')
    df_no_mmsi_shape = df_no_mmsi.shape
    df_no_mmsi['mmsi'] = -1
    df_no_mmsi['closest timestamp'] = np.NaN
    df_no_mmsi['interval sec'] = np.NaN
    df_no_mmsi['distance m'] = np.NaN

    latitude_interp = dict()
    longitude_interp = dict()
    time_nearest_interp = dict()
    separation_w_mmsi_no_mmsi = dict()
    index_no_mmsi = dict()

    pbar = tqdm(df_no_mmsi['date'].unique().to_numpy(), desc='Iterating through dates in ship predictions...')
    for date_ in pbar:
        pbar.set_postfix({'date': date_})
        df_no_mmsi_ = df_no_mmsi[df_no_mmsi['date'] == date_]
        index_no_mmsi[date_] = df_no_mmsi_.index.to_pandas()
        date_str = date_.replace('-', '_')
        # TODO incorporate bucket name and directory key for s3
        try:
            df_w_mmsi = sbf.extract_koalas_df_from_s3(
                None + date_str + ".csv", n_rows=n_rows)
        except AnalysisException:
            print('No AIS file found in .../MarineCadastre/csv directory for ' + date_str + '. Skipping this date...')
            continue

        a, b = np.meshgrid(df_w_mmsi['timestamp'].to_numpy().astype(np.int64),
                           df_no_mmsi_['timestamp'].to_numpy().astype(np.int64))

        df_time_cond = ks.DataFrame({'time_cond': np.any(np.abs(a - b) / 1E9 < threshold_sec, axis=0)})
        df_w_mmsi_close = df_w_mmsi.join(df_time_cond)
        df_w_mmsi_close = df_w_mmsi_close[df_w_mmsi_close['time_cond']]

        df_dist_cond = ks.DataFrame({'dist_cond': np.any(sbf.haversine_m(lon1=df_w_mmsi_close['longitude'].to_numpy(),
                                                                         lat1=df_w_mmsi_close['latitude'].to_numpy(),
                                                                         lon2=df_no_mmsi_['longitude'].to_numpy(),
                                                                         lat2=df_no_mmsi_['latitude'].to_numpy(),
                                                                         meshgrid=True,
                                                                         convert2radians=True) < threshold_m, axis=0)})
        df_w_mmsi_close = df_w_mmsi_close.join(df_dist_cond)
        df_w_mmsi_close = df_w_mmsi_close[df_w_mmsi_close['dist_cond']]

        for mmsi_ in df_w_mmsi_close['mmsi'].unique().to_numpy():
            df_ = df_w_mmsi_close[df_w_mmsi_close['mmsi'] == mmsi_]
            if df_.shape[0] > 1:
                latitude_interp[mmsi_, date_] = interpolate.interp1d(df_['timestamp'].to_numpy().astype(np.int64),
                                                                     df_['latitude'].to_numpy().astype(np.float64),
                                                                     fill_value=np.NaN,
                                                                     bounds_error=False)(
                    df_no_mmsi_['timestamp'].to_numpy().astype(np.int64))
                longitude_interp[mmsi_, date_] = interpolate.interp1d(df_['timestamp'].to_numpy().astype(np.int64),
                                                                      df_['longitude'].to_numpy().astype(np.float64),
                                                                      fill_value=np.NaN,
                                                                      bounds_error=False)(
                    df_no_mmsi_['timestamp'].to_numpy().astype(np.int64))
                time_nearest_interp[mmsi_, date_] = interpolate.interp1d(df_['timestamp'].to_numpy().astype(np.int64),
                                                                         df_['timestamp'].to_numpy().astype(np.int64),
                                                                         kind='nearest', fill_value=np.NaN,
                                                                         bounds_error=False)(
                    df_no_mmsi_['timestamp'].to_numpy().astype(np.int64))
                separation_w_mmsi_no_mmsi[mmsi_, date_] = sbf.haversine_m(longitude_interp[mmsi_, date_],
                                                                          latitude_interp[mmsi_, date_],
                                                                          df_no_mmsi_['longitude'].to_numpy().astype(
                                                                              np.float64),
                                                                          df_no_mmsi_['latitude'].to_numpy().astype(
                                                                              np.float64),
                                                                          convert2radians=True)

    t0, t1 = np.meshgrid(df_no_mmsi['timestamp'].to_numpy().astype(np.int64),
                         df_no_mmsi['timestamp'].to_numpy().astype(np.int64))
    delta_m = np.hypot((t0 - t1) / clustering_speed,
                       sbf.haversine_m(df_no_mmsi['longitude'].to_numpy().astype(np.float64),
                                       df_no_mmsi['latitude'].to_numpy().astype(np.float64),
                                       df_no_mmsi['longitude'].to_numpy().astype(np.float64),
                                       df_no_mmsi['latitude'].to_numpy().astype(np.float64),
                                       meshgrid=True, convert2radians=True))

    clustering = DBSCAN(eps=threshold_m, min_samples=1, metric='precomputed').fit(delta_m)

    pbar = tqdm(np.unique(clustering.labels_), desc='Clustering...')
    for label in pbar:
        tf = clustering.labels_ == label
        df_tf = ks.DataFrame({'tf': tf})
        if 'tf' in df_no_mmsi.columns.to_list():
            df_no_mmsi = df_no_mmsi.drop('tf')
        df_no_mmsi = df_no_mmsi.join(df_tf)
        dates = df_no_mmsi[df_no_mmsi['tf']]['date'].drop_duplicates().to_list()
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
            pbar.set_postfix({'cluster #': label, 'dates': dates, 'count': sum(tf),
                              'factorial depth': min([len(cols), max_depth])})
            for i, order in enumerate(itertools.permutations(range(len(cols))), min([len(cols), max_depth])):
                if i > max_permutations:
                    pbar.set_postfix_str('%s, exceeded max_permutations!' % pbar.postfix)
                    break
                val = np.sum(sep2[rows[[order]], cols[[order]]])
                if val < best_val:
                    best_order = order
                    best_val = val
                    count += 1
                    pbar.set_postfix(
                        {'cluster #': label, 'dates': dates, 'count': sum(tf), 'factorial depth': max_depth,
                         '# improvements': count})

            df_no_mmsi.loc[inds[cols[[best_order]]], 'mmsi'] = mmsis[rows[[best_order]]]
            df_no_mmsi.loc[inds[cols[[best_order]]], 'closest timestamp'] = pd.to_datetime(
                time_nearest[rows[[best_order]], cols[[best_order]]])
            df_no_mmsi.loc[inds[cols[[best_order]]], 'interval sec'] = (df_no_mmsi.loc[inds[cols[
                [best_order]]], 'timestamp'].to_numpy(int) - time_nearest[rows[[best_order]],
                                                                          cols[[best_order]]]) / 1E9
            df_no_mmsi.loc[inds[cols[[best_order]]], 'distance m'] = sep[rows[[best_order]], cols[[best_order]]]

    df_no_mmsi = df_no_mmsi.astype({'mmsi': np.int32, 'interval sec': np.float32, 'distance m': np.float32}).drop('tf')
    df = df_no_mmsi.iloc[:, df_no_mmsi_shape[1]:].sort_index()
    # TODO combine df into master df (see AIS_infer_mmsi.get_df())
    return df


def infer_mmsi(df_no_mmsi: pd.DataFrame, threshold_sec: int = 1800, threshold_m: int = 10E3,
               clustering_speed: int = 100, max_depth: int = 12, n_rows: int = None) -> pd.DataFrame:
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
    n_rows: int
        Limit for number of rows to be read from a file into a dataframe, leave as None to read entire files
    spark: SparkSession
        Existing SparkSession to create dataframe in. If left as None, parkSession.builder.getOrCreate() will be used
    app_name: str
        Name of the Spark Session to get or create. Only necessary if spark argument is included

    Returns
    -------
    Pandas DataFrame of all rows with the following columns:
        ['mmsi', 'closest timestamp', 'interval sec', 'distance m']
    """
    # used in permutation search
    max_permutations = np.math.factorial(max_depth)

    df_no_mmsi['date'] = df_no_mmsi['timestamp'].dt.strftime('%Y-%m-%d')
    df_no_mmsi_shape = df_no_mmsi.shape
    ##
    df_no_mmsi['mmsi'] = -1
    df_no_mmsi['closest timestamp'] = pd.NaT
    df_no_mmsi['interval sec'] = np.NaN
    df_no_mmsi['distance m'] = np.NaN

    latitude_interp = dict()
    longitude_interp = dict()
    time_nearest_interp = dict()
    separation_w_mmsi_no_mmsi = dict()
    index_no_mmsi = dict()

    pbar = tqdm(df_no_mmsi.groupby('date'), desc='Iterating through dates in ship predictions...')
    for date_, df_no_mmsi_ in pbar:
        pbar.set_postfix({'date': date_})
        index_no_mmsi[date_] = df_no_mmsi_.index.to_flat_index()
        date_str = date_.replace('-', '_')
        # TODO incorporate bucket name and directory key for s3
        try:
            df_w_mmsi = sbf.extract_pandas_df_from_s3(
                "s3a://dark-ships/AIS-data/MarineCadastre/csv/AIS_" + date_str + ".csv", n_rows=n_rows)
        except FileNotFoundError:
            print('No AIS file found in .../MarineCadastre/csv dir for ' + date_str + '. Skipping date...')
            continue

        a, b = np.meshgrid(df_w_mmsi['timestamp'].to_numpy(int), df_no_mmsi_['timestamp'].to_numpy(int))

        df_w_mmsi_close = df_w_mmsi[np.any(np.abs(a - b) / 1E9 < threshold_sec, axis=0)]
        df_w_mmsi_close = df_w_mmsi_close[np.any(
            sbf.haversine_m(df_w_mmsi_close['longitude'], df_w_mmsi_close['latitude'],
                            df_no_mmsi_['longitude'],
                            df_no_mmsi_['latitude'], meshgrid=True, convert2radians=True) < threshold_m,
            axis=0)]

        for mmsi_, df_ in df_w_mmsi_close.groupby('mmsi'):
            if df_.shape[0] > 1:
                latitude_interp[mmsi_, date_] = interpolate.interp1d(df_['timestamp'].to_numpy(int),
                                                                     df_['latitude'].to_numpy(float),
                                                                     fill_value=np.NaN,
                                                                     bounds_error=False)(
                    df_no_mmsi_['timestamp'].to_numpy(int))
                longitude_interp[mmsi_, date_] = interpolate.interp1d(df_['timestamp'].to_numpy(int),
                                                                      df_['longitude'].to_numpy(float),
                                                                      fill_value=np.NaN,
                                                                      bounds_error=False)(
                    df_no_mmsi_['timestamp'].to_numpy(int))
                time_nearest_interp[mmsi_, date_] = interpolate.interp1d(df_['timestamp'].to_numpy(int),
                                                                         df_['timestamp'].to_numpy(int),
                                                                         kind='nearest', fill_value=np.NaN,
                                                                         bounds_error=False)(
                    df_no_mmsi_['timestamp'].to_numpy(int))
                separation_w_mmsi_no_mmsi[mmsi_, date_] = sbf.haversine_m(longitude_interp[mmsi_, date_],
                                                                          latitude_interp[mmsi_, date_],
                                                                          df_no_mmsi_['longitude'].to_numpy(
                                                                              float),
                                                                          df_no_mmsi_['latitude'].to_numpy(
                                                                              float),
                                                                          convert2radians=True)

    t0, t1 = np.meshgrid(df_no_mmsi['timestamp'].to_numpy(int), df_no_mmsi['timestamp'].to_numpy(int))
    delta_m = np.hypot((t0 - t1) / clustering_speed, sbf.haversine_m(df_no_mmsi['longitude'].to_numpy(float),
                                                                     df_no_mmsi['latitude'].to_numpy(float),
                                                                     df_no_mmsi['longitude'].to_numpy(float),
                                                                     df_no_mmsi['latitude'].to_numpy(float),
                                                                     meshgrid=True, convert2radians=True))

    clustering = DBSCAN(eps=threshold_m, min_samples=1, metric='precomputed').fit(delta_m)

    pbar = tqdm(np.unique(clustering.labels_), desc='Clustering...')

    for label in pbar:
        tf = clustering.labels_ == label

        # print( label, tf.sum(), df_no_mmsi.loc[tf, 'timestamp'].mean(), df_no_mmsi.loc[tf, 'timestamp'].std())
        dates = df_no_mmsi.loc[tf, 'date'].drop_duplicates().to_list()
        pbar.set_postfix({'cluster #': label, 'count': sum(tf), 'dates': dates})

        inds = np.array(
            list(itertools.chain(*[index_no_mmsi[date] for date in index_no_mmsi if date in dates])))
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
            pbar.set_postfix({'cluster #': label, 'dates': dates, 'count': sum(tf),
                              'factorial depth': min([len(cols), max_depth])})
            for i, order in enumerate(itertools.permutations(range(len(cols))), min([len(cols), max_depth])):
                if i > max_permutations:
                    pbar.set_postfix_str('%s, exceeded max_permutations!' % pbar.postfix)
                    break
                val = np.sum(sep2[rows[[order]], cols[[order]]])
                if val < best_val:
                    best_order = order
                    best_val = val
                    count += 1
                    pbar.set_postfix(
                        {'cluster #': label, 'dates': dates, 'count': sum(tf), 'factorial depth': max_depth,
                         '# improvements': count})

            df_no_mmsi.loc[inds[cols[[best_order]]], 'mmsi'] = mmsis[rows[[best_order]]]
            df_no_mmsi.loc[inds[cols[[best_order]]], 'closest timestamp'] = pd.to_datetime(
                time_nearest[rows[[best_order]], cols[[best_order]]])
            df_no_mmsi.loc[inds[cols[[best_order]]], 'interval sec'] = (df_no_mmsi.loc[inds[cols[
                [best_order]]], 'timestamp'].to_numpy(int) - time_nearest[rows[[best_order]],
                                                                          cols[[best_order]]]) / 1E9
            df_no_mmsi.loc[inds[cols[[best_order]]], 'distance m'] = sep[rows[[best_order]], cols[[best_order]]]

    df_no_mmsi = df_no_mmsi.astype({'mmsi': np.int32, 'interval sec': np.float32, 'distance m': np.float32})
    df = df_no_mmsi.iloc[:, df_no_mmsi_shape[1]:].sort_index()

    return df
