import pandas as pd
import inspect
import os
import numpy as np
import tqdm
import AIS_parameters
import AIS_description
import AIS_geopandas_tagging
import AIS_interp
import datetime
import functions_basic_sak as funsak

fname = os.path.split(inspect.getfile(inspect.currentframe()))[1]

def get_df_old( date = '2022-01-01', pbar = None, join = False ):
    if isinstance(date, list):
        df_dict = {}
        if pbar is None:
            pbar = tqdm.tqdm( date, desc = '%s get_df'%fname )
            pbar_ = pbar
        else:
            pbar_ = date
        for date_ in pbar_:
            if isinstance( pbar, bool ) and ( pbar != False ):
                pbar.set_postfix( {'date': date_} )
            df_dict[date_] = get_df( date_, pbar = pbar, join = join )
        return df_dict
    else:
        df_desc = AIS_description.get_df(date=date, join = True, multicolumn = True)
        df_tag, _ = AIS_geopandas_tagging.get_df(date, tagging={'World_EEZ': {'tags': 'ISO_SOV1', 'geometry': 'geometry'}},
                                        overwrite=False, freq=None, join=False, pbar = pbar)
        ##
        def vessel_( type ):
            return np.any([df_desc[('AIS_description', 'VesselType')].cat.codes.to_numpy() == i
                   for i, val in enumerate(df_desc[('AIS_description', 'VesselType')].cat.categories) if type.lower() in val.lower()], axis = 0 )

        def sov_( type ):
            if type.upper() == 'EU':
                tmp = [df_tag[('World_EEZ_Intersect_IHO_v4_2020.geojson', 'geometry', 'ISO_SOV1')].cat.codes.to_numpy() == i
                     for i, val in enumerate(df_tag[('World_EEZ_Intersect_IHO_v4_2020.geojson', 'geometry', 'ISO_SOV1')].cat.categories) if
                    val.lower() in AIS_parameters.EU_COUNTRIES]
                if tmp == []:
                    return np.zeros( df_tag.shape[0], dtype= bool)
                else:
                    return np.any(tmp, axis=0)
            else:
                return np.array([df_tag[('World_EEZ_Intersect_IHO_v4_2020.geojson', 'geometry', 'ISO_SOV1')].cat.codes.to_numpy() == i
                     for i, val in enumerate(df_tag[('World_EEZ_Intersect_IHO_v4_2020.geojson', 'geometry', 'ISO_SOV1')].cat.categories) if type.lower() == val.lower()])

        ##
        AIS_required = dict()
        ##
        # tonnage requirement https://globalfishingwatch.org/faqs/what-vessels-are-required-to-use-ais-what-are-global-regulations-and-requirements-for-vessels-to-carry-ais/
        # Check if ship is passenger ship SOLAS
        AIS_required['passenger'] = vessel_('passenger')
        # TODO Figure out length and width corresponding to tonnage
        AIS_required['cargo'] = vessel_('cargo')
        # Check USA self-propelled vessel length requirement
        AIS_required['USA self-propelled vessel length'] = sov_('USA') & ( df['Length'] > 19.8 ) & ~vessel_('sailing (36)')
        # Check towing vessel
        AIS_required['towing'] = vessel_('towing')
        # Check tug vessel and length requirement
        AIS_required['tug vessel and length'] = vessel_('tug') & ( df['Length'] > 7.9 )
        # Check USA Class B Fishing requirements
        AIS_required['USA Class B Fishing'] = sov_('USA') & vessel_('fishing') & ( df['Length'] > 19.8 )
        # Check UK Fishing regulations UPPER LIMIT MAKES NO SENSE?!
        AIS_required['UK Fishing']= sov_('GBR') & vessel_('fishing') & (15 <= df['Length'] )  & ( df['Length'] <= 18)
        # EU Fishing Regulation
        AIS_required['EU Fishing'] = sov_('EU') & vessel_('fishing') & (12 <= df['Length'] )
        # Canadian AIS Requirements
        AIS_required['Canada'] = sov_('CAN') & (8 <= df['Length'])
        df = pd.DataFrame( AIS_required )
        df['required'] = df.to_numpy().any(axis = 1)
        if join:
            df.columns = pd.MultiIndex.from_product([[fname[:-3]], df.columns] )
            df = pd.concat( [df_desc.iloc[:, :-2], df_tag, df], axis = 1 )
            ##
        return df


def get_df( df = None, pbar = None, join = False ):
    df_desc = AIS_description.get_df( df = df, join = False, multicolumn = True, pbar = pbar)
    df_tag = AIS_geopandas_tagging.get_df( df, tagging={'World_EEZ': {'tags': 'ISO_SOV1', 'geometry': 'geometry'}},
                join=False, multicolumn = True, pbar = pbar)
    ##
    def vessel_( type ):
        return np.any([df_desc[('AIS_description', 'VesselType')].cat.codes.to_numpy() == i
               for i, val in enumerate(df_desc[('AIS_description', 'VesselType')].cat.categories) if type.lower() in val.lower()], axis = 0 )

    def sov_( type ):
        if type.upper() == 'EU':
            tmp = [df_tag[('World_EEZ_Intersect_IHO_v4_2020.geojson', 'geometry', 'ISO_SOV1')].cat.codes.to_numpy() == i
                 for i, val in enumerate(df_tag[('World_EEZ_Intersect_IHO_v4_2020.geojson', 'geometry', 'ISO_SOV1')].cat.categories) if
                val.lower() in AIS_parameters.EU_COUNTRIES]
            if tmp == []:
                return np.zeros( df_tag.shape[0], dtype= bool)
            else:
                return np.any(tmp, axis=0)
        else:
            return np.array([df_tag[('World_EEZ_Intersect_IHO_v4_2020.geojson', 'geometry', 'ISO_SOV1')].cat.codes.to_numpy() == i
                 for i, val in enumerate(df_tag[('World_EEZ_Intersect_IHO_v4_2020.geojson', 'geometry', 'ISO_SOV1')].cat.categories) if type.lower() == val.lower()])

    ##
    AIS_required = dict()
    ##
    # tonnage requirement https://globalfishingwatch.org/faqs/what-vessels-are-required-to-use-ais-what-are-global-regulations-and-requirements-for-vessels-to-carry-ais/
    # Check if ship is passenger ship SOLAS
    AIS_required['passenger'] = vessel_('passenger')
    # TODO Figure out length and width corresponding to tonnage
    AIS_required['cargo'] = vessel_('cargo')
    # Check USA self-propelled vessel length requirement
    ##
    AIS_required['USA self-propelled vessel length'] = sov_('USA') & ( df['Length'] > 19.8 ) & ~vessel_('sailing (36)')
    # Check towing vessel
    AIS_required['towing'] = vessel_('towing')
    ##
    # Check tug vessel and length requirement
    AIS_required['tug vessel and length'] = vessel_('tug') & ( df['Length'] > 7.9 )
    # Check USA Class B Fishing requirements
    AIS_required['USA Class B Fishing'] = sov_('USA') & vessel_('fishing') & ( df['Length'] > 19.8 )
    # Check UK Fishing regulations UPPER LIMIT MAKES NO SENSE?!
    AIS_required['UK Fishing']= sov_('GBR') & vessel_('fishing') & (15 <= df['Length'] )  & ( df['Length'] <= 18)
    # EU Fishing Regulation
    AIS_required['EU Fishing'] = sov_('EU') & vessel_('fishing') & (12 <= df['Length'] )
    # Canadian AIS Requirements
    AIS_required['Canada'] = sov_('CAN') & (8 <= df['Length'])
    df_out = pd.DataFrame( AIS_required )
    df_out['required'] = df_out.to_numpy().any(axis = 1)
    ##
    if join:
        df_out.columns = pd.MultiIndex.from_product([[fname[:-3]], df_out.columns] )
        df_out = pd.concat( [df, df_desc, df_tag, df_out], axis = 1 )
        ##
    return df_out

if __name__ == '__main__':
    if 'stephan' == AIS_parameters.user:
        DF = AIS_interp.resample_date_range( start_date= '2021-12-30', end_date='2022-02-03', date_range =  pd.date_range('2022-01-01', '2022-01-04', freq = 'H')[:-1], aux_columns = ['nearest'] )
        if isinstance( DF, dict):
            pbar = tqdm.tqdm( DF, desc = fname )
            for date in pbar:
                df = get_df( df = DF[date], join = True, pbar = pbar )
                if not os.path.isdir( os.path.join(AIS_parameters.dirs['root'], 'training' ) ):
                    os.mkdir( os.path.join(AIS_parameters.dirs['root'], 'training' ) )
                if not os.path.isdir( os.path.join(AIS_parameters.dirs['root'], 'training', str( datetime.datetime.now().date() ) ) ):
                    os.mkdir( os.path.join(AIS_parameters.dirs['root'], 'training', str( datetime.datetime.now().date() ) ) )
                funsak.MultiIndex2single( df ).to_feather(os.path.join(AIS_parameters.dirs['root'], 'training', str( datetime.datetime.now().date() ), '%s %s.feather' %( fname[:-3], str(date)) ) )
        else:
            df = get_df(df=DF, join=True)
        ##

        ##
