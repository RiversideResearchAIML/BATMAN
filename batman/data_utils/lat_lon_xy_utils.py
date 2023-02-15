########################################
#             UNCLASSIFIED             #
########################################
"""
This lat_lon_xy_utils.py file is used to get a pixel location within an image based on latitude  and longitude
coordinates.
"""
import warnings
import numpy as np
from haversine import haversine

# Following values taken from https://rechneronline.de/earth-radius/
R_EQUATOR = 6378137  # Earth's radius at equator in meters
R_POLES = 6356752  # Earth's radius at poles in meters


def get_pixel_xy_from_lat_lon(lat: float,
                              lon: float,
                              image_size: tuple = (1663, 771),
                              lat_range: tuple = (25.7598, 25.7709),
                              lon_range: tuple = (-80.1521, -80.1266),
                              image_width_m: float = None) -> tuple:
    """
    This function takes a latitude and longitude coordinate pair and converts it to a pixel coordinate, according to
    the provided latitude and longitude boundaries and the size of the image.

    NOTE: Pixel y locations start with zero at the top. For this reason after converting the longitude to be on the
    scale of the pixel height, we subtract the value from the images pixel height.

    Parameters
    ----------
    lat: float
        Latitude value to be converted to a pixel y value.
    lon: float
        Longitude value to be converted to a pixel x value.
    image_size: tuple of ints
        Tuple containing the width and height (respectively) of the image.
    lat_range: tuple of floats
        Tuple containing the minimum and maximum latitude values (respectively) of the image.
    lon_range: tuple of floats
        Tuple containing the minimum and maximum longitude values (respectively) of the image.
    image_width_m: float
        Meter width of image. Will be used for a meter/pixel value during conversion. If omitted, this value will be
        calculated using the Haversine formula across the center of the image (horizontally).

    Returns
    -------
    pixel_xy: tuple of ints
        Tuple containing the x and y (respectively) pixel coordinates corresponding to the provided
        latitude and longitude coordinates.
    """
    if not lat_range[0] <= lat <= lat_range[1] or not lon_range[0] <= lon <= lon_range[1]:
        warnings.warn("lat & lon values must be within the bounds of lat_range and lon_range... Returning empty tuple")
        return ()
    lat_delta = np.abs(lat_range[1] - lat_range[0])
    lon_delta = np.abs(lon_range[1] - lon_range[0])
    if image_width_m is None:  # use Haversine formula to calculate distance across center of image horizontally
        image_width_m = haversine((lat_range[0] + lat_delta / 2, lon_range[0]),
                                  (lat_range[0] + lat_delta / 2, lon_range[1]), unit='m')

        # Equation taken from https://rechneronline.de/earth-radius/
    r_s = np.sqrt(((R_EQUATOR ** 2 * np.cos(np.radians(lat))) ** 2 + (R_POLES ** 2 * np.sin(np.radians(lat))) ** 2) /
                  ((R_EQUATOR * np.cos(np.radians(lat))) ** 2 + (R_POLES * np.sin(np.radians(lat))) ** 2))
    x_m = r_s * np.sin(np.radians(lon - lon_range[0]))
    x = int(x_m * (image_size[0] / image_width_m))  # convert x from meters to pixels

    h_scale_factor = image_size[1] / lat_delta
    y = lat - lat_range[0]
    y *= h_scale_factor
    y = image_size[1] - y  # flip y scale because pixels start at top of image and measures down
    return int(x), int(y)


def get_lat_lon_from_pixel_xy(x: int,
                              y: int,
                              image_size: tuple = (1663, 771),
                              lat_range: tuple = (25.7598, 25.7709),
                              lon_range: tuple = (-80.1521, -80.1266),
                              image_width_m: float = None) -> tuple:
    """
    This function takes a pixel coordinate and converts it to a longitude and latitude coordinate, according to
    the provided longitude and latitude boundaries and the size of the image.

    NOTE: Pixel y locations start with zero at the top. For this reason after converting the longitude to be on the
    scale of the pixel height, we subtract the value from the images pixel height.

    Parameters
    ----------
    x: int
        The x pixel coordinate to be converted to longitude.
    y: int
        The y pixel coordinate to be converted to latitude.
    image_size: tuple of ints
        Tuple containing the width and height (respectively) of the image.
    lat_range: tuple of floats
        Tuple containing the minimum and maximum latitude values (respectively) of the image.
    lon_range: tuple of floats
        Tuple containing the minimum and maximum longitude values (respectively) of the image.
    image_width_m: float
        Meter width of image. Will be used for a meter/pixel value during conversion. If omitted, this value will be
        calculated using the Haversine formula across the center of the image (horizontally).

    Returns
    -------
        Tuple containing float values of the latitude and longitude (respectively) which were converted from the pixel
        x, y coordinates.
    """
    if not 0 <= x <= image_size[0] or not 0 <= y <= image_size[1]:
        print('x: {}, y: {}'.format(x, y))
        print('image_size: {}'.format(image_size))
        warnings.warn("Pixel xy values must be between zero and image dimensions... Returning empty tuple")
        return ()
    y = image_size[1] - y  # flip y scale because pixels start at top of image and measures down
    lat_delta = np.abs(lat_range[1] - lat_range[0])
    lon_delta = np.abs(lon_range[1] - lon_range[0])

    h_scale_factor = image_size[1] / lat_delta
    lat = (y / h_scale_factor) + lat_range[0]

    if image_width_m is None:  # use Haversine formula to calculate distance across center of image horizontally
        image_width_m = haversine((lat_range[0] + lat_delta / 2, lon_range[0]),
                                  (lat_range[0] + lat_delta / 2, lon_range[1]), unit='m')

    # Equation taken from https://rechneronline.de/earth-radius/
    r_s = np.sqrt(((R_EQUATOR**2 * np.cos(np.radians(lat)))**2 + (R_POLES**2 * np.sin(np.radians(lat)))**2) /
                  ((R_EQUATOR * np.cos(np.radians(lat)))**2 + (R_POLES * np.sin(np.radians(lat)))**2))
    x_m = x * (image_width_m / image_size[0])  # convert x from pixels to meters
    lon = np.degrees(np.arcsin(x_m / r_s))
    lon += lon_range[0]
    return lat, lon
