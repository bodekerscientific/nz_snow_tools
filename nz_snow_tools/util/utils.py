"""
a collection of utilities to handle dates / grids etc to support snow model
"""

import datetime
import numpy as np
import shapefile
from PIL import Image
from matplotlib.path import Path
from pyproj import Proj, transform


### Date utilities

def convert_datetime_julian_day(dt_list):
    """ returns the day number for each date object in d_list
    :param d_list: list of dateobjects to evaluate
    :return: list of the julian day for each date
    """
    day = [(d - datetime.datetime(d.year, 1, 1)).days + 1 for d in dt_list]
    return day


def convert_unix_time_seconds_to_dt(uts_list):
    """ converts list of timestamps of seconds since the start of unix time to datetime objects
    :param uts_list: a list of times in seconds since the start of unix time
    :return: dt_list: list of datetime objects
    """
    dt_list = []
    for i in range(len(uts_list)):
        dt_list.append(datetime.datetime.utcfromtimestamp(uts_list[i]))
    return dt_list


def make_regular_timeseries(start_dt, stop_dt, num_secs):
    """
    makes a regular timeseries between two points. The difference between the start and end points must be a multiple of the num_secs
    :param start_dt: first datetime required
    :param stop_dt: last datetime required
    :param num_secs: number of seconds in timestep required
    :return: list of datetime objects between
    """
    epoch = datetime.datetime.utcfromtimestamp(0)
    st = (start_dt - epoch).total_seconds()
    et = (stop_dt - epoch).total_seconds()
    new_timestamp = np.linspace(st, et, int((et - st) / num_secs + 1))

    return convert_unix_time_seconds_to_dt(new_timestamp)


def convert_dt_to_hourdec(dt_list):
    """
    convert datetime to decimal hour
    :param dt_list:
    :return:
    """

    decimal_hour = [dt1.hour + dt1.minute / 60. + dt1.second / 3600. + dt1.microsecond / 3600000. for dt1 in dt_list]
    return np.asarray(decimal_hour)


def convert_datetime_decimal_day(dt_list):
    """ returns the decimal day number for each datetime object in dt_list
    :param dt_list: list of datetime objects to evaluate
    :return: numpy array of the decimal day for each datetime
    """
    timestamp = np.zeros(len(dt_list))
    for i in range(len(dt_list)):
        timestamp[i] = (dt_list[i] - datetime.datetime(dt_list[i].year, 1, 1)).total_seconds() / 86400. + 1
    return timestamp


def convert_date_hydro_DOY(d_list, hemis='south'):
    """ returns the day of the hydrological year for each date object in d_list
    :param d_list: list of dateobjects to evaluate
    :return: array containing the  day of the hydological year for each date
    """
    if hemis == 'south':
        end_month = 3
    elif hemis == 'north':
        end_month = 9
    h_DOY = []
    for d in d_list:
        if d.month <= end_month:  #
            h_DOY.append((d.date() - datetime.date(d.year - 1, end_month + 1, 1)).days + 1)
        else:
            h_DOY.append((d.date() - datetime.date(d.year, end_month + 1, 1)).days + 1)
    return np.asarray(h_DOY)


def convert_hydro_DOY_to_date(h_DOY, year, hemis='south'):
    """ converts day of the hydrological year into a datetime object
    :param d_list: array containing day of the hydological year
    :param year: the hydrological year the data is from . hydrological years are denoted by the year of the last day
    of the HY i.e. 2011 is HY ending 31 March 2011 in the SH
    :return: array containing datetime objects for a given hydrological year
    """
    if hemis == 'south':
        epoch = datetime.date(year - 1, 4, 1)
    elif hemis == 'north':
        epoch = datetime.date(year - 1, 10, 1)
    d = []
    for doy in h_DOY:
        d.append(epoch + datetime.timedelta(days=doy - 1))
    return np.asarray(d)


### temporal interpolation utilities

def process_precip(precip_daily, one_day=False):
    """
    Generate hourly precip fields using a multiplicative cascade model

    Described in Rupp et al 2009 (http://onlinelibrary.wiley.com/doi/10.1029/2008WR007321/pdf)

    :param model:
    :return:
    """

    if one_day == True:  # assume is 2d and add a time dimension on the start
        precip_daily = precip_daily.reshape([1, precip_daily.shape[0], precip_daily.shape[1]])

    if precip_daily.ndim == 2:
        hourly_data = np.zeros((precip_daily.shape[0] * 24, precip_daily.shape[1]), dtype=np.float32)
    elif precip_daily.ndim == 3:
        hourly_data = np.zeros((precip_daily.shape[0] * 24, precip_daily.shape[1], precip_daily.shape[2]), dtype=np.float32)

    store_day_weights = []
    for idx in range(precip_daily.shape[0]):
        # Generate an new multiplicative cascade function for the day
        day_weights = random_cascade(24)
        store_day_weights.append(day_weights)
        # multiply to create hourly data
        if precip_daily.ndim == 2:
            hourly_data[idx * 24: (idx + 1) * 24] = day_weights[:, np.newaxis] * precip_daily[idx]
        elif precip_daily.ndim == 3:
            hourly_data[idx * 24: (idx + 1) * 24] = day_weights[:, np.newaxis, np.newaxis] * precip_daily[idx]
    return hourly_data, store_day_weights


def random_cascade(l):
    """
    A recursive implementation of a Multiplicative Random Cascade (MRC)

    :param l: The size of the resulting output. Must be divisible by 2
    :return: Weights that sum to 1.0
    """
    res = np.ones(l)

    if l <= 3:
        res = np.random.random(l)
        res /= res.sum()
        return res

    weights = np.random.random(2)
    weights /= weights.sum()

    res[:l / 2] = weights[0] * random_cascade(l // 2)
    res[l / 2:] *= weights[1] * random_cascade(l // 2)
    return res


def process_temp(max_temp_daily, min_temp_daily):
    """
    Generate hourly fields

    Sine curve through max/min. 2pm/8am for max, min as a first gues
    :param model:
    :return:
    """

    def hour_func(dec_hours):
        """
        Piecewise fit to daily tmax and tmin using sine curves
        """
        f = np.piecewise(dec_hours, [dec_hours < 8, (dec_hours >= 8) & (dec_hours < 14), dec_hours >= 14], [
            lambda x: np.cos(2. * np.pi / 36. * (x + 10.)),  # 36 hour period starting 10 hours through
            lambda x: np.cos(2. * np.pi / 12. * (x - 14.)),  # 12 hour period (only using rising 6 hours between 8 am and 2pm)
            lambda x: np.cos(2. * np.pi / 36. * (x - 14.))  # 36 hour period starting at 2pm
        ])
        return (f + 1.) / 2.  # Set the range to be 0 - 1 0 is tmin and 1 being tmax

    scaling_factors = hour_func(np.arange(24.))

    hourly_data = np.zeros((max_temp_daily.shape[0] * 24, max_temp_daily.shape[1]), dtype=np.float32)
    hours = np.array(range(24) * max_temp_daily.shape[0])

    # Calculate each piecewise element seperately as some need the previous days data
    mask = hours < 8
    max_temp_prev_day = np.concatenate((max_temp_daily[0][np.newaxis, :], max_temp_daily[:-1]))
    hourly_data[mask] = _interp_temp(scaling_factors[:8], max_temp_prev_day, min_temp_daily)

    mask = (hours >= 8) & (hours < 14)
    hourly_data[mask] = _interp_temp(scaling_factors[8:14], max_temp_daily, min_temp_daily)

    mask = hours >= 14
    min_temp_next_day = np.concatenate((min_temp_daily[1:], min_temp_daily[-1][np.newaxis, :]))
    hourly_data[mask] = _interp_temp(scaling_factors[14:], max_temp_daily, min_temp_next_day)

    return hourly_data


def _interp_temp(scaling, max_temp, min_temp):
    res = np.zeros((len(scaling) * len(max_temp),) + max_temp.shape[1:])

    for i in range(len(max_temp)):
        res[i * len(scaling): (i + 1) * len(scaling)] = scaling[:, np.newaxis] * max_temp[i] + (1 - scaling[:, np.newaxis]) * min_temp[i]
    return res


# grid utilities

def create_mask_from_shpfile(lat, lon, shp_path, idx=0):
    """
    Creates a mask for numpy array

    creates a boolean array on the same grid as lat,long where all cells,
    whose centroid is inside a line shapefile, are true. The shapefile must be
    a line, in the same CRS as the data, and have only 1 feature (only the first
    feature will be used as a mask)

    :param lat: a 1D array of latitiude positions on a regular grid
    :param long: a 1D array of longitute position on a regular grid
    :param shp_path: the path to a line shape file (.shp) only the first feature will be used to mask.
    :return: boolean array on the same grid as lat,long
    """

    lat = np.asarray(lat)
    lon = np.asarray(lon)

    # load shapefile
    shp = shapefile.Reader(shp_path)
    shapes2 = shp.shapes()
    shppath = Path(shapes2[idx].points)

    if lat.ndim == 1 and lon.ndim == 1:  # create array of lat and lon
        nx, ny = len(lon), len(lat)
        longarray, latarray = np.meshgrid(lon, lat)
    elif lat.ndim == 2 and lon.ndim == 2:
        assert lat.shape == lon.shape
        nx, ny = lat.shape[1], lat.shape[0]
        longarray = lon
        latarray = lat
    else:
        raise ValueError("lat and lon are not the same shape")

    # Create vertex coordinates for each grid cell
    x, y = longarray.flatten(), latarray.flatten()
    points = np.vstack((x, y)).T

    # create the mask array
    grid = shppath.contains_points(points)
    grid = grid.reshape((ny, nx))
    return grid


def nztm_to_wgs84(in_y, in_x):
    """converts from NZTM to WGS84  Inputs and outputs can be arrays.
    """
    inProj = Proj(init='epsg:2193')
    outProj = Proj(init='epsg:4326')
    out_x, out_y = transform(inProj, outProj, in_x, in_y)
    return out_y, out_x


def trim_lat_lon_bounds(mask, lat_array, lon_array, nztm_dem, y_centres, x_centres):
    # Trim down the number of latitudes requested so it all stays in memory
    valid_lat_bounds = np.nonzero(mask.sum(axis=1))[0]
    lat_min_idx = valid_lat_bounds.min()
    lat_max_idx = valid_lat_bounds.max()
    valid_lon_bounds = np.nonzero(mask.sum(axis=0))[0]
    lon_min_idx = valid_lon_bounds.min()
    lon_max_idx = valid_lon_bounds.max()

    lats = lat_array[lat_min_idx:lat_max_idx + 1, lon_min_idx:lon_max_idx + 1]
    lons = lon_array[lat_min_idx:lat_max_idx + 1, lon_min_idx:lon_max_idx + 1]
    elev = nztm_dem[lat_min_idx:lat_max_idx + 1, lon_min_idx:lon_max_idx + 1]
    northings = y_centres[lat_min_idx:lat_max_idx + 1]
    eastings = x_centres[lon_min_idx:lon_max_idx + 1]

    return lats, lons, elev, northings, eastings


# misc

def calc_toa(lat_ref, lon_ref, hourly_dt):
    """
    calculate top of atmopshere radiation for given lat, lon and datetime
    :param lat_ref:
    :param lon_ref:
    :param hourly_dt:
    :return:
    """
    dtstep = (hourly_dt[1] - hourly_dt[0]).total_seconds()
    # compute at midpoint between timestep and previous timestep
    jd = convert_datetime_decimal_day(hourly_dt) - 0.5 * (hourly_dt[1] - hourly_dt[0]).total_seconds() / 86400.
    hourdec = convert_dt_to_hourdec(hourly_dt) - 0.5 * (hourly_dt[1] - hourly_dt[0]).total_seconds() / 3600.

    latitude = lat_ref  # deg
    longitude = lon_ref  # deg
    timezone = -12
    # Calculating    correction    factor    for direct beam radiation
    d0_rad = 2 * np.pi * (jd - 1) / 365.  # day    angle,
    # solar        declination        Iqbal, 1983
    Declination_rad = np.arcsin(
        0.006918 - 0.399912 * np.cos(d0_rad) + 0.070257 * np.sin(d0_rad) - 0.006758 * np.cos(2 * d0_rad)
        + 0.000907 * np.sin(2 * d0_rad) - 0.002697 * np.cos(3 * d0_rad) + 0.00148 * np.sin(3 * d0_rad))
    HourAngle_rad = (-1 * (180 + (dtstep / 3600 * 15 / 2) - hourdec * 15) + (longitude - 15 * timezone)) * np.pi / 180.
    ZenithAngle_rad = np.arccos(np.cos(latitude * np.pi / 180) * np.cos(Declination_rad) * np.cos(HourAngle_rad)
                                + np.sin(latitude * np.pi / 180) * np.sin(Declination_rad))
    ZenithAngle_deg = ZenithAngle_rad * 180 / np.pi
    sundown = 0 * ZenithAngle_deg
    sundown[ZenithAngle_deg > 90] = 1
    E0 = 1.000110 + 0.034221 * np.cos(d0_rad) + 0.00128 * np.sin(d0_rad) + 0.000719 * np.cos(
        2 * d0_rad) + 0.000077 * np.sin(2 * d0_rad)
    SRtoa = 1372 * E0 * np.cos(ZenithAngle_rad)  # SRin    at    the    top    of    the    atmosphere
    SRtoa[sundown == 1] = 0
    return SRtoa


def setup_nztm_dem(dem_file, extent_w=1.2e6, extent_e=1.4e6, extent_n=5.13e6, extent_s=4.82e6, resolution=250):
    """
    load dem tif file. defaults to clutha 250 dem.
    :param dem_file: string specifying path to dem
    :param extent_w: extent in nztm
    :param extent_e: extent in nztm
    :param extent_n: extent in nztm
    :param extent_s: extent in nztm
    :param resolution: resolution in m
    :return:
    """
    nztm_dem = Image.open(dem_file)
    # np.array(nztm_dem).shape is (1240L, 800L) but origin is in NW corner. Move to SW to align with increasing Easting and northing.
    nztm_dem = np.flipud(np.array(nztm_dem))
    # extent_w = 1.2e6
    # extent_e = 1.4e6
    # extent_n = 5.13e6
    # extent_s = 4.82e6
    # resolution = 250
    # create coordinates
    x_centres = np.arange(extent_w + resolution / 2, extent_e, resolution)
    y_centres = np.arange(extent_s + resolution / 2, extent_n, resolution)
    y_array, x_array = np.meshgrid(y_centres, x_centres, indexing='ij')
    lat_array, lon_array = nztm_to_wgs84(y_array, x_array)
    # plot to check the dem
    # plt.imshow(nztm_dem, origin=0, interpolation='none', cmap='terrain')
    # plt.colorbar(ticks=np.arange(0, 3000, 100))
    # plt.show()
    return nztm_dem, x_centres, y_centres, lat_array, lon_array


def trim_data_to_mask(data, mask):
    """
    # trim data to minimum box needed for mask
    :param data: 2D (x,y) or 3D (time,x,y) array
    :param mask: 2D boolean with same x,y dimensions as data
    :return: data trimmed
    """
    valid_lat_bounds = np.nonzero(mask.sum(axis=1))[0]
    lat_min_idx = valid_lat_bounds.min()
    lat_max_idx = valid_lat_bounds.max()
    valid_lon_bounds = np.nonzero(mask.sum(axis=0))[0]
    lon_min_idx = valid_lon_bounds.min()
    lon_max_idx = valid_lon_bounds.max()

    if data.ndim == 2:
        trimmed_data = data[lat_min_idx:lat_max_idx + 1, lon_min_idx:lon_max_idx + 1]
    elif data.ndim == 3:
        trimmed_data = data[:, lat_min_idx:lat_max_idx + 1, lon_min_idx:lon_max_idx + 1]
    else:
        print 'data does not have correct dimensions'

    return trimmed_data


def nash_sut(y_sim, y_obs):
    """
    calculate the nash_sutcliffe efficiency criterion (taken from Ayala, 2017, WRR)

    :param y_sim: series of simulated values
    :param y_obs: series of observed values
    :return:
    """
    assert y_sim.ndim == 1 and y_obs.ndim == 1

    ns = 1 - np.sum((y_sim - y_obs) ** 2) / np.sum((y_obs - np.mean(y_obs)) ** 2)

    return ns


def mean_bias(y_sim, y_obs):
    """
    calculate the mean bias difference (taken from Ayala, 2017, WRR)

    :param y_sim: series of simulated values
    :param y_obs: series of observed values
    :return:
    """
    assert y_sim.ndim == 1 and y_obs.ndim == 1 and len(y_sim) == len(y_obs)

    mbd = np.sum(y_sim - y_obs) / len(y_sim)

    return mbd


def rmsd(y_sim, y_obs):
    """
    calculate the mean bias difference (taken from Ayala, 2017, WRR)

    :param y_sim: series of simulated values
    :param y_obs: series of observed values
    :return:
    """
    assert y_sim.ndim == 1 and y_obs.ndim == 1 and len(y_sim) == len(y_obs)

    rs = np.sqrt(np.mean((y_sim - y_obs) ** 2))

    return rs
