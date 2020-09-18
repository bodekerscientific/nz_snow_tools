"""
code to run snow model on large grid for multiple years. calculates meteorology on the fly and outputs to netCDF
"""
from __future__ import division

import netCDF4 as nc
import pickle
import numpy as np
import datetime as dt
import matplotlib.pylab as plt
import json
import os

from nz_snow_tools.snow.clark2009_snow_model import calc_dswe
from nz_snow_tools.util.utils import create_mask_from_shpfile, make_regular_timeseries, convert_datetime_julian_day
from nz_snow_tools.met.interp_met_data_hourly_vcsn_data import load_new_vscn, interpolate_met, process_precip, daily_to_hourly_swin_grids, \
    daily_to_hourly_temp_grids, setup_nztm_dem, setup_nztm_grid_netcdf, trim_lat_lon_bounds


run_id = 'dsc_default'
met_inp = 'vcsn_norton' #identifier for input meteorology
# model options
which_model = 'dsc_snow'  # string identifying the model to be run. options include 'clark2009', 'dsc_snow' # future will include 'fsm'

# time and grid extent options
years_to_take = np.arange(2000, 2019 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
catchment = 'SI'  # string identifying the catchment to run. must match the naming of the catchment shapefile
output_dem = 'si_dem_250m'
mask_dem = False  # boolean to set whether or not to mask the output dem
mask_created = True  # boolean to set whether or not the mask has already been created
mask_folder = None  # location of numpy catchment mask. must be writeable if mask_created == False
# mask_shpfile = 'Z:/GIS_DATA/Hydrology/Catchments/{}.shp'.format(
#     catchment)  # shapefile containing polyline or polygon of catchment in WGS84. Not needed if mask_created==True
# catchment_shp_folder = 'Z:/GIS_DATA/Hydrology/Catchments'

# output options

dem_folder = '/nesi/project/niwa00004/jonoconway'  # dem used for output #'C:/Users/conwayjp/OneDrive - NIWA/Data/GIS_DATA/Topography/DEM_NZSOS'#
output_folder = '/nesi/nobackup/niwa00004/jonoconway/snow_sims_nz/vcsn'  # snow model output
data_folder = '/nesi/nobackup/niwa00004/jonoconway'  # input meteorology #'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2101/tn_file_compilation'#

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# paths to input met data
nc_file_rain = nc.Dataset('/nesi/nobackup/niwa00026/VCSN/Rain/rain_vclim_clidb_1972010100_2020080200_south-island_p05_daily_netcdf4.nc', 'r')
#nc_file_tmax = nc.Dataset('T:/newVCSN/tmax_vclim_clidb_1972010100_2017102000_south-island_p05_daily.nc', 'r')
nc_file_tmax = nc.Dataset('/nesi/nobackup/niwa00026/VCSN/TMax_Norton/tmax_vclim_clidb_Norton_1972010100_2020080200_south-island_p05_daily_netcdf4.nc', 'r')
#nc_file_tmin = nc.Dataset('T:/newVCSN/tmin_vclim_clidb_1972010100_2017102000_south-island_p05_daily.nc', 'r')
nc_file_tmin = nc.Dataset('/nesi/nobackup/niwa00026/VCSN/TMin_Norton/tmin_vclim_clidb_Norton_1972010100_2020080200_south-island_p05_daily_netcdf4.nc', 'r')
nc_file_srad = nc.Dataset('/nesi/nobackup/niwa00026/VCSN/SRad/srad_vclim_clidb_1972010100_2020080200_south-island_p05_daily_netcdf4.nc', 'r')


# configuration dictionary containing model parameters.
config = {}
config['tacc'] = 274.16
config['tmelt'] = 273.16
# clark2009 melt parameters
config['mf_mean'] = 5.0
config['mf_amp'] = 5.0
config['mf_alb'] = 2.5
config['mf_alb_decay'] = 5.0
config['mf_ros'] = 2.5
config['mf_doy_max_ddf'] = 356
# dsc_snow melt parameters
config['tf'] = 0.05 * 24  # hamish 0.13
config['rf'] = 0.0108 * 24  # hamish 0.0075
# albedo parameters
config['dc'] = 11.0
config['tc'] = 10.0
config['a_ice'] = 0.42
config['a_freshsnow'] = 0.90
config['a_firn'] = 0.62
config['alb_swe_thres'] = 10
config['ros'] = False
config['ta_m_tt'] = False

# load met grid (assume same for all input data)
vcsn_elev = np.flipud(nc_file_rain.variables['elevation'][:])
vcsn_elev_interp = np.ma.fix_invalid(vcsn_elev).data
vcsn_lats = nc_file_rain.variables['latitude'][::-1]
vcsn_lons = nc_file_rain.variables['longitude'][:]
vcsn_dt = nc.num2date(nc_file_rain.variables['time'][:], nc_file_rain.variables['time'].units)
vcsn_dt2 = nc.num2date(nc_file_tmax.variables['time'][:], nc_file_tmax.variables['time'].units)
vcsn_dt3 = nc.num2date(nc_file_tmin.variables['time'][:], nc_file_tmin.variables['time'].units)
vcsn_dt4 = nc.num2date(nc_file_srad.variables['time'][:], nc_file_srad.variables['time'].units)

# check for the same timestamp
# assert vcsn_dt[0] == vcsn_dt2[0] == vcsn_dt3 == vcsn_dt4
# vcsn_dt2 = None
# vcsn_dt3 = None
# vcsn_dt4 = None

# calculate model grid etc:
# output DEM
dem_file = dem_folder + '/' + output_dem + '.tif'
if output_dem == 'clutha_dem_250m':
    nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(dem_file)

if output_dem == 'si_dem_250m':
    nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(dem_file, extent_w=1.08e6, extent_e=1.72e6, extent_n=5.52e6, extent_s=4.82e6,
                                                                          resolution=250)

if mask_dem == True:
    # Get the masks for the individual regions of interest
    if mask_created == True:  # load precalculated mask
        mask = np.load(mask_folder + '/{}_{}.npy'.format(catchment, output_dem))
    else:  # create mask and save to npy file
        print('need to precalculate mask file')
    #     mask = create_mask_from_shpfile(lat_array, lon_array, mask_shpfile)
    #     np.save(mask_folder + '/{}_{}.npy'.format(catchment, output_dem), mask)
    # Trim down the number of latitudes requested so it all stays in memory
    lats, lons, elev, northings, eastings = trim_lat_lon_bounds(mask, lat_array, lon_array, nztm_dem, y_centres, x_centres)
    _, _, trimmed_mask, _, _ = trim_lat_lon_bounds(mask, lat_array, lon_array, mask.copy(), y_centres, x_centres)
else:
    mask = None
    lats = lat_array
    lons = lon_array
    elev = nztm_dem
    northings = y_centres
    eastings = x_centres
    # set mask to all land points
    mask = elev > 0
    trimmed_mask = mask

# set up time to run, paths to input files etc
for year_to_take in years_to_take:
    # specify the days to run (output is at the end of each day)
    # out_dt = np.asarray(make_regular_timeseries(dt.datetime(year_to_take, 7, 1), dt.datetime(year_to_take, 7, 2), 86400))
    out_dt = np.asarray(make_regular_timeseries(dt.datetime(year_to_take-1, 4, 1), dt.datetime(year_to_take, 4, 1), 86400))

    # set up output netCDF:
    out_nc_file = setup_nztm_grid_netcdf(output_folder + '/snow_out_{}_{}_{}_{}_{}_{}.nc'.format(met_inp, which_model, catchment, output_dem, run_id, year_to_take),
                                         None, ['swe', 'acc', 'melt'],
                                         out_dt, northings, eastings, lats, lons, elev)

    # set up initial states of prognostic variables
    init_swe = np.zeros(elev.shape)  # default to no snow
    init_d_snow = np.ones(elev.shape) * 30  # default to a month since snowfall
    swe = init_swe
    d_snow = init_d_snow
    # set up daily buckets for melt and accumulation
    bucket_melt = swe * 0
    bucket_acc = swe * 0
    swe_day_before = swe * 0

    # store initial swe value
    out_nc_file.variables['swe'][0, :, :] = init_swe
    out_nc_file.variables['acc'][0, :, :] = 0
    out_nc_file.variables['melt'][0, :, :] = 0

    # storage array for random precip weightings
    day_weightings = []
    # for each day:
    for ii, dt_t in enumerate(out_dt[:-1]):
        # load one day of precip and shortwave rad data
        precip_daily = load_new_vscn('rain', dt_t, nc_file_rain, nc_opt=True, single_dt=True, nc_datetimes=vcsn_dt)
        # load 3 days of temperature data (1 either side of day of interest - needed for interpolation of temperature)
        dts_to_take = make_regular_timeseries(dt_t - dt.timedelta(days=1), dt_t + dt.timedelta(days=1), 86400)
        max_temp_daily = load_new_vscn('tmax', dts_to_take, nc_file_tmax, nc_opt=True, nc_datetimes=vcsn_dt2)
        min_temp_daily = load_new_vscn('tmin', dts_to_take, nc_file_tmin, nc_opt=True, nc_datetimes=vcsn_dt3)

        # interpolate data to fine grid
        hi_res_precip = interpolate_met(precip_daily.filled(np.nan), 'rain', vcsn_lons, vcsn_lats, vcsn_elev_interp, lons, lats, elev, single_dt=True)
        hi_res_max_temp = interpolate_met(max_temp_daily.filled(np.nan), 'tmax', vcsn_lons, vcsn_lats, vcsn_elev_interp, lons, lats, elev)
        hi_res_min_temp = interpolate_met(min_temp_daily.filled(np.nan), 'tmin', vcsn_lons, vcsn_lats, vcsn_elev_interp, lons, lats, elev)
        # mask out areas we don't want/need
        if mask is not None:
            hi_res_precip[trimmed_mask == 0] = np.nan
            hi_res_max_temp[:, trimmed_mask == 0] = np.nan
            hi_res_min_temp[:, trimmed_mask == 0] = np.nan
        # interpolate to hourly step
        hourly_precip, day_weightings_1 = process_precip(hi_res_precip,
                                                         one_day=True)  # TODO: align to 9am-9am - currently pretends is midnight-midnight

        hourly_dt = np.asarray(make_regular_timeseries(dt_t, dt_t + dt.timedelta(hours=23), 3600))
        # air temperature is three part sinusoidal between min at 8am and max at 2pm. NOTE original VCSN data has correct timestamp - ie. minimum to 9am, maximum from 9am.
        # use three days but only keep middle day
        hourly_temp = daily_to_hourly_temp_grids(hi_res_max_temp, hi_res_min_temp)
        hourly_temp = hourly_temp[24:48]

        # store precip day weights.
        day_weightings.extend(day_weightings_1)
        hourly_doy = convert_datetime_julian_day(hourly_dt)

        if which_model == 'dsc_snow':
            sw_rad_daily = load_new_vscn('srad', dt_t, nc_file_srad, nc_opt=True, single_dt=True, nc_datetimes=vcsn_dt4)
            hi_res_sw_rad = interpolate_met(sw_rad_daily.filled(np.nan), 'srad', vcsn_lons, vcsn_lats, vcsn_elev_interp, lons, lats, elev, single_dt=True)
            if mask is not None:
                hi_res_sw_rad[trimmed_mask == 0] = np.nan
            hourly_swin = daily_to_hourly_swin_grids(hi_res_sw_rad, lats, lons, hourly_dt, single_dt=True)

        # calculate snow and output to netcdf
        for i in range(len(hourly_dt)):
            # d_snow += dtstep / 86400.0
            if which_model == 'dsc_snow':
                swe, d_snow, melt, acc = calc_dswe(swe, d_snow, hourly_temp[i], hourly_precip[i], hourly_doy[i], 3600, sw=hourly_swin[i], which_melt=which_model,
                                                   **config)
            else:
                swe, d_snow, melt, acc = calc_dswe(swe, d_snow, hourly_temp[i], hourly_precip[i], hourly_doy[i], 3600, which_melt=which_model,
                                               **config)
            # print swe[0]
            bucket_melt = bucket_melt + melt
            bucket_acc = bucket_acc + acc

        # output at the end of each day,
        for var, data in zip(['swe', 'acc', 'melt'], [swe, bucket_acc, bucket_melt]):
            # data[(np.isnan(data))] = -9999.
            out_nc_file.variables[var][ii + 1, :, :] = data

        # decide if albedo is reset
        d_snow += 1
        swe_alb = swe - swe_day_before
        d_snow[(swe_alb > config['alb_swe_thres'])] = 0
        swe_day_before = swe * 1.0
        # reset buckets
        bucket_melt = bucket_melt * 0
        bucket_acc = bucket_acc * 0

    out_nc_file.close()

    json.dump(config, open(output_folder + '/config_{}_{}_{}_{}_{}.json'.format(met_inp,which_model, catchment, output_dem, run_id, year_to_take), 'w'))
    pickle.dump(config, open(output_folder + '/config_{}_{}_{}_{}_{}_{}.pkl'.format(met_inp,which_model, catchment, output_dem, run_id, year_to_take), 'wb'), protocol=3)
    pickle.dump(day_weightings, open(output_folder + '/met_inp_{}_{}_{}_{}_{}_{}_daywts.pkl'.format(met_inp,which_model, catchment, output_dem, run_id, year_to_take), 'wb'), protocol=3)
