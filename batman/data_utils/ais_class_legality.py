import pandas as pd
import boto3
from shapely.geometry import shape, Point, MultiLineString, LineString
import json

EU_COUNTRIES = ["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia",
                "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania",
                "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia",
                "Spain", "Sweden"]


def region_check(lat: float, lon: float,
                 geojson_file: str = "contextual-data/static/geojsons/World_EEZ_Intersect_IHO_v4_2020.geojson") -> str:
    """
    This function checks what EEZ the latitude and longitude is in.
    Co-Authored: Cody Freese & Ricky Dalrymple

    Parameters
    ----------
    lat: float containing latitude
    lon: float containing longitude
    geojson_file: String containing pathway to EEZ geojson

    Returns
    -------
    String containing country code of EEZ that is True is lat_lon is within region
    """
    s3 = boto3.resource('s3')
    obj = s3.Object('dark-ships', geojson_file)
    data = json.load(obj.get()['Body'])

    point = Point(lon, lat)
    for feature in data["features"]:
        polygon = shape(feature["geometry"])
        if polygon.contains(point):
            return feature["properties"]["SOVEREIGN1"]

    return "INTERNATIONAL WATERS"


def ais_class_legality_check(ships: pd.DataFrame, ais_classes_file: str = "contextual-data/static/legal"
                                                                          "/ais-class/ais_ship_classes.csv"):
    """
    This function filters whether a ship is required to have AIS by the UN and regional laws.
    Co-Authored: Cody Freese & Ricky Dalrymple

    Parameters
    ----------
    ships: pandas DataFrame
        Dataframe containing the following columns: [lat,lon,length,width,ship_class].
    ais_classes_file: str Pathway to ship classes pickle file

    Returns
    -------
    Tuple of pandas DataFrames. The first of which contains ships that require AIS and the second contains ships that
     do not require AIS.
    """
    s3 = boto3.resource('s3')
    class_obj = s3.Bucket('dark-ships').Object(ais_classes_file).get()
    class_file = class_obj['Body']
    classes = pd.read_csv(class_file)
    required_ais = pd.DataFrame()
    not_required_ais = pd.DataFrame()
    for index, ship in ships.iterrows():
        ship["needs_ais"] = True  # Will be changed to False based on If statements
        ship["checked"] = True
        region = region_check(lat=ship["lat"], lon=ship["lon"])
        ship = ship.append(ratification_check(region))
        # SOLAS Substitute for 300 ton + International Voyage rule, based on length approx.
        if 8 <= ship["l"] <= 45 and region == "INTERNATIONAL WATERS":
            required_ais = required_ais.append(ship, ignore_index=True)
        # SOLAS Substitute for 500 ton, based on length approx.
        elif ship["l"] >= 45:
            required_ais = required_ais.append(ship, ignore_index=True)
        # Check if ship is passenger ship SOLAS
        elif "passenger" in classes.loc[ship["vessel_type"]]["Description"].lower():
            required_ais = required_ais.append(ship, ignore_index=True)
        elif "cargo" in classes.loc[ship["vessel_type"]]["Description"].lower():
            required_ais = required_ais.append(ship, ignore_index=True)
        # Check USA self-propelled vessel length requirement
        elif ship["l"] >= 19.8 and region == "United States":
            required_ais = required_ais.append(ship, ignore_index=True)
        # Check towing vessel and length requirement
        elif "towing" in classes.loc[ship["vessel_type"]]["Description"].lower() or \
                "tug" in classes.loc[ship["vessel_type"]]["Description"].lower() and ship["l"] >= 7.9:
            required_ais = required_ais.append(ship, ignore_index=True)
        # Check USA Class B Fishing requirements
        elif ship["l"] >= 19.8 and "fishing" in classes.loc[ship["vessel_type"]]["Description"].lower() and \
                region == "United States":
            required_ais = required_ais.append(ship, ignore_index=True)
        # Check UK Fishing regulations
        elif 15 <= ship["l"] <= 18 and "fishing" in classes.loc[ship["vessel_type"]]["Description"].lower() and \
                region == "United Kingdom":
            required_ais = required_ais.append(ship, ignore_index=True)
        # EU Fishing Regulation
        elif ship["l"] >= 12 and "fishing" in classes.loc[ship["vessel_type"]]["Description"].lower() and \
                region in EU_COUNTRIES:
            required_ais = required_ais.append(ship, ignore_index=True)
        # Canadian AIS Requirements
        elif ship["l"] >= 8 and region == "Canada":
            required_ais = required_ais.append(ship, ignore_index=True)
        else:
            ship["needs_ais"] = False
            not_required_ais = not_required_ais.append(ship, ignore_index=True)
    return required_ais, not_required_ais


def ratification_check(region: str,
                       rat_file: str = "contextual-data/static/legal/ais-class/ratification_status.pkl"):
    """
    This function checks the ratification status of UN Legislation in the subsequent EEZ
    Author: Cody Freese

    Parameters
    ----------
    region: string containing EEZ
    rat_file: pathway to ratification pickle file

    Returns
    -------
    None
    """
    s3 = boto3.resource('s3')
    ratification_obj = s3.Bucket('dark-ships').Object(rat_file).get()
    ratification_file = ratification_obj['Body']
    df = pd.read_pickle(ratification_file)
    if region == "United States":
        region = "United States of America"
    res = df.loc[df["Country"] == region]
    if res.shape[0] == 1:
        return res.iloc[0]
    return None


def underwater_cable_check(lat: float, lon: float):
    """
    This function checks if a coordinate is within proximity of underwater telecommunication cables
    Author: Cody Freese

    Parameters
    ----------
    lat: float containing latitude
    lon: float containing longitude

    Returns
    -------
    String containing feature_id of submarine cable that is True is lat_lon is within region
    """
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket='dark-ships', Key='contextual-data/static/geojsons/submarine_cable_width_update.geojson')
    cable_file = json.loads(obj['Body'].read().decode('utf-8'))
    point = Point(lon, lat)
    for feature in cable_file["features"]:
        linestring = MultiLineString(feature["geometry"]["coordinates"])
        if linestring.distance(point) < 0.05:
            return feature["properties"]["feature_id"]
    return "Not Within 5km of Undersea Cable"


def petroleum_pipeline_check(lat: float, lon: float):
    """
    This function checks if a coordinate is within the proximity of underwater oil and gas pipelines.
    Author: Cody Freese

    Parameters
    ----------
    lat: float containing latitude
    lon: float containing longitude

    Returns
    -------
    String containing PipelineName of oil and gas pipelines that are True is lat_lon is within region
    """
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket='dark-ships',
                        Key='contextual-data/static/geojsons/GEM_Oil_Gas_Pipelines_2022-10.geojson')
    petrol_file = json.loads(obj['Body'].read().decode('utf-8'))
    point = Point(lon, lat)
    for feature in petrol_file["features"]:
        if not feature["geometry"]:
            continue
        try:
            linestring = LineString(feature["geometry"]["coordinates"])
        except:
            linestring = MultiLineString(feature["geometry"]["coordinates"])
        if linestring.distance(point) < 0.05:
            return feature["properties"]["PipelineName"]
    return "Not Within 5km of Undersea Pipeline"


def marine_protected_area_check(lat: float, lon: float):
    """
    This function checks if coordinate is within proximity of marine protected area
    Author: Cody Freese

    Parameters
    ----------
    lat: float containing latitude
    lon: float containing longitude

    Returns
    -------
    String containing mpa_name that is True is lat_lon is within region
    """
    s3 = boto3.client('s3')
    mpa_obj = s3.get_object(Bucket='dark-ships', Key='contextual-data/static/geojsons/NOAA.Fisheries'
                                                     '.MPAS_East_Coast_Atlantic.geojson')
    mpa_file = json.loads(mpa_obj['Body'].read().decode('utf-8'))
    point = Point(lon, lat)
    for feature in mpa_file['features']:
        polygon = shape(feature['geometry'])
        if polygon.contains(point):
            return "Inside " + feature['properties']['MPA_Name']
        if polygon.distance(point) < 0.05:
            return "Within 5km of " + feature['properties']['MPA_Name']
    return "Not Within 5km Of A Marine Protected Area"


def unesco_protected_area_check(lat: float, lon: float):
    """
    This function checks if coordinate is within proximity of UNESCO Protected Area
    Author: Cody Freese

    Parameters
    ----------
    lat: float containing latitude
    lon: float containing longitude

    Returns
    -------
    String containing Full_Name that is True is lat_lon is within region
    """
    s3 = boto3.client('s3')
    unesco_obj = s3.get_object(Bucket='dark-ships', Key="contextual-data/static/geojsons"
                                                        "/World_Heritage_Marine_Program_UNESCO.geojson")
    unesco_file = json.loads(unesco_obj['Body'].read().decode('utf-8'))
    point = Point(lon, lat)
    for feature in unesco_file['features']:
        polygon = shape(feature['geometry'])
        if polygon.contains(point):
            return "Inside " + feature['properties']['Full_Name']
        if polygon.distance(point) < 0.05:
            return "Within 5km of " + feature['properties']['Full_Name']
    return "Not Within 5km of UNESCO World Site"


if __name__ == "__main__":
    # Coordinate 1 = NordStream Pipeline Leak Location
    # Coordinate 2 = Off coast of Miami Harbor
    ships_df = pd.DataFrame({"lat": [55.535, 25.749965],
                             "lon": [15.698, -80.086614],
                             "l": [20.0, 12.0],
                             "w": [8.0, 12.0],
                             "vessel_type": [60, 70],
                             "checked": [False, False],
                             "needs_ais": [True, False]})

    required, not_required = ais_class_legality_check(ships_df)
