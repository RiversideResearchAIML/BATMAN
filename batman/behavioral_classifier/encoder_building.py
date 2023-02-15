import pandas as pd
import pickle
import category_encoders as ce
import os

ENCODER_FOLDER = None


def base2_create(category: pd.DataFrame, save: str):
    encoder = ce.BaseNEncoder(base=2)
    encoder = encoder.fit(category)
    outpath = os.path.join(ENCODER_FOLDER, save)
    with open(outpath, 'wb') as f:
        pickle.dump(encoder, f)
        

        
        
        
ranges = 'BATMAN_behavioral_input_ranges.xlsx'
ports = "/wld_trs_ports_wfp trunc.feather"
noaa = "/NOAA_MPAI_v2020 trunc.feather"
pipelines = "/GEM_Oil_Gas_Pipelines_2022-10 trunc.feather"
status = "/VesselStatus.xlsx"

ship_classes = pd.read_excel(ranges, "AIS Ship Classes")['Type Code'] 
ports = pd.read_feather(ports)['portname']
noaa = pd.read_feather(noaa)['Fish_Rstr']
pipelines = pd.read_feather(pipelines)['PipelineName']
status = pd.read_excel(status)['code']
status.at[16] = 'None'



ship_classes.name = 'AIS_downsample_VesselType'
ports.name = 'wld_trs_ports_wfp.csv_lat_lon_portname'
noaa.name = 'NOAA_MPAI_v2020.gdb_geometry_Fish_Rstr'
pipelines.name = 'GEM_Oil_Gas_Pipelines_2022-10.geojson_geometry_PipelineName'
status.name = 'AIS_downsample_Status'


base2_create(ship_classes, "vessel_type.pkl")
base2_create(ports, 'port.pkl')
base2_create(noaa, "zone.pkl")
base2_create(pipelines, 'pipeline.pkl')
base2_create(status, 'status.pkl')


