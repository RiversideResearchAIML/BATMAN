"""
This file contains a function that can be called to perform ship pairing in a Spark cluster within AWS EMR.
"""
import pandas as pd

from dark_ships.AIS_processing.spark_basic_functions import extract_koalas_df_from_s3, extract_pandas_df_from_s3
from dark_ships.AIS_processing.spark_infer_mmsi import koalas_infer_mmsi, infer_mmsi


def ship_pairing_spark_driver(source_bucket_name: str = "dark-ships",
                              dest_bucket_name: str = "darkships-data-lake",
                              yolo_preds_key: str = "EO-data/google-earth-eo-data/ship_truth_data.csv",
                              output_key: str = "ship-pairing-output/test_output.parquet",
                              n_rows: int = None, with_koalas: bool = True):
    """
    This function acquires a DataFrame from the provided YOLO predictions csv file and calls the infer_mmsi() function
    from spark_infer_mmsi.py. The resulting DataFrame from infer_mmsi() is saved as a parquet file to s3.

    Parameters
    ----------
    source_bucket_name: str containing name of s3 bucket that predictions file is stored in
    dest_bucket_name: str containing name of s3 bucket that output will be written to
    yolo_preds_key: str containing location in s3 bucket of YOLO predictions file
    output_key: str containing location in s3 bucket in which output will be saved, must include '.parquet' extension
    n_rows: int limit of rows to be read from any csv file for debugging purposes, leave as None to read entire files
    with_koalas: bool indicating whether to use Spark or standard pandas
    """
    if with_koalas:
        df_no_mmsi = extract_koalas_df_from_s3("s3a://" + source_bucket_name + "/" + yolo_preds_key, n_rows=None, is_ais=False)
        paired_df = koalas_infer_mmsi(df_no_mmsi, n_rows=n_rows)
    else:
        df_no_mmsi = extract_pandas_df_from_s3("s3a://" + source_bucket_name + "/" + yolo_preds_key, n_rows=None, is_ais=False)
        paired_df = infer_mmsi(df_no_mmsi, n_rows=n_rows)

    print("-----------------------------------------------------------------------\n{}".format(paired_df.head()))
    print("-----------------------------------------------------------------------")
    paired_df.to_parquet("s3a://" + dest_bucket_name + "/" + output_key)


if __name__ == "__main__":
    ship_pairing_spark_driver(n_rows=1000, with_koalas=True, output_key="ship-pairing-output/test_output.parquet")
