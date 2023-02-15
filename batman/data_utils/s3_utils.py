"""
This file contains functions to interact with an s3 bucket. Specifically read a feather file, save a dataframe as a
feather file, and convert a json file to a feather on an s3 bucket.
Taken from https://gist.github.com/sminot/a82ad01eb28bd2114bf34fd6c7d99f6e
@author rdalrymple@riversideresearch.org
"""

import io
import json
import zipfile

# import boto3
import pandas as pd
from tqdm import tqdm
from pyarrow.feather import write_feather


def read_feather_file_from_s3(key_name: str, bucket_name: str = 'dark-ships') -> pd.DataFrame:
    """
    This function reads a feather file from s3 and returns its contents as a pandas dataframe.

    Parameters
    ----------
    key_name: str path to feather file to be read
    bucket_name: str name of bucket containing file to be read

    Returns
    -------
    df populated with the contents of the feather file that was read
    """
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=key_name)
    return pd.read_feather(io.BytesIO(obj['Body'].read()))


def write_feather_to_s3(df: pd.DataFrame, key_name: str, bucket_name: str = 'dark-ships', md_key: str = None,
                        md_val: str = None):
    """
    This function takes a dataframe, writes it to a feather file and uploads the file to an s3 bucket.

    Parameters
    ----------
    df: pd.DataFrame to be written to feather file
    key_name: str of the path and filename for the feather file to be written to
    bucket_name: str of the name of the bucket for the feather file to be written to
    md_key: str of key for metadata key:value pair, leave as None to omit metadata
    md_val: str of val for metadata key:value pair, leave as None to omit metadata
    """
    s3 = boto3.client("s3")
    with io.BytesIO() as buf:
        write_feather(df, buf)
        if md_key is None or md_val is None:
            print("INFO: metadata key and/or value are None... omitting metadata in s3 put operation.")
            s3.put_object(Bucket=bucket_name, Key=key_name, Body=buf.getvalue())
        else:
            s3.put_object(Bucket=bucket_name, Key=key_name, Body=buf.getvalue(), Metadata={md_key: md_val})


def convert_json_to_feather(key_name: str, bucket_name: str = 'dark-ships', md_key: str = None, md_val: str = None):
    """
    This function reads a json file from an s3 bucket and re-uploads it to the same location as a feather file.

    Parameters
    ----------
    key_name: str path of the json file to be read; this is also the path that the feather file will be saved to
    bucket_name: str of the name of the bucket containing the json file to be read
    md_key: str of key for metadata key:value pair, leave as None to omit metadata
    md_val: str of val for metadata key:value pair, leave as None to omit metadata
    """
    s3 = boto3.resource('s3')
    obj = s3.Bucket(bucket_name).Object(key_name)
    file = obj.get()['Body'].read().decode('utf-8')
    json_dict = json.loads(file)
    df = pd.DataFrame.from_dict(json_dict['features'])
    write_feather_to_s3(df, key_name[:-8] + '.feather', bucket_name, md_key, md_val)


def unpack_zips(source_bucket_name: str = "dark-ships", dest_bucket_name: str = "dark-ships",
                source_dir: str = "AIS-data/MarineCadastre/download/", dest_dir: str = "AIS-data/MarineCadastre/csv/"):
    """
    This function is provided an s3 directory containing zip files, unzips them and places the contents of the zip
    file in the provided destination directory.

    Parameters
    ----------
    source_bucket_name: str containing the name of the s3 bucket that holds source directory
    dest_bucket_name: str containing the name of the s3 bucket that holds the destination directory
    source_dir: str containing directory within source s3 bucket that contains zip files to be unzipped
    dest_dir: str containing directory within destination s3 bucket that zip files contents will be places
    """
    s3 = boto3.resource("s3")
    source_bucket = s3.Bucket(source_bucket_name)
    dest_bucket = s3.Bucket(dest_bucket_name)
    for obj in tqdm(source_bucket.objects.filter(Prefix=source_dir), desc='Downloading and unzipping files...'):
        check_str = obj.key[:-4].replace(source_dir, dest_dir)
        print(check_str)
        existing_csvs = list(dest_bucket.objects.filter(Prefix=check_str))
        if obj.key[-4:] == ".zip" and len(existing_csvs) == 0:
            with io.BytesIO(obj.get()["Body"].read()) as tf:
                # rewind the file
                tf.seek(0)

                # Read the file as a zipfile and process the members
                with zipfile.ZipFile(tf, mode='r') as zipf:
                    for subfile in zipf.namelist():
                        obj = s3.Object(dest_bucket_name, dest_dir + subfile)
                        obj.put(Body=zipf.read(subfile))
