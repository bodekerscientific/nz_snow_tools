"""
code to run snow model on large grid for multiple years. calculates meteorology on the fly and outputs to netCDF
"""
from __future__ import division

import json
import netCDF4 as nc
import pickle
import numpy as np
import cartopy.crs as ccrs
import datetime as dt
import os
# import matplotlib.pylab as plt

from nz_snow_tools.snow.clark2009_snow_model import calc_dswe
from nz_snow_tools.util.utils import create_mask_from_shpfile, make_regular_timeseries, convert_datetime_julian_day
from nz_snow_tools.met.interp_met_data_hourly_vcsn_data import load_new_vscn, interpolate_met, process_precip, daily_to_hourly_swin_grids, \
    daily_to_hourly_temp_grids, setup_nztm_dem, setup_nztm_grid_netcdf, trim_lat_lon_bounds
import sys


def run_snow_otf_nzcsm_main(hydro_years_to_take, run_id, met_inp, which_model, catchment, output_dem, mask_dem, mask_folder, dem_folder, output_folder, data_folder,
                      orog_infile, precip_infile, air_temp_infile, solar_rad_infile, config):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # open input met data files
    nc_file_orog = nc.Dataset(data_folder + '/' + orog_infile, 'r')

    nc_file_rain = nc.Dataset(data_folder + '/' + precip_infile, 'r')
    nc_file_temp = nc.Dataset(data_folder + '/' + air_temp_infile, 'r')

    nc_rain = nc_file_rain.variables['sum_total_precip']
    nc_temp = nc_file_temp.variables['sfc_temp']

    if which_model == 'dsc_snow':
        nc_file_srad = nc.Dataset(data_folder + '/' + solar_rad_infile, 'r')
        nc_srad = nc_file_srad.variables['sfc_dw_sw_flux']
        vcsn_dt4 = nc.num2date(nc_file_srad.variables['time1'][:], nc_file_srad.variables['time1'].units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)

    # load met grid (assume same for all input data)
    vcsn_elev = nc_file_orog.variables['orog_model'][:]
    vcsn_elev_interp = vcsn_elev.copy()
    vcsn_lats = nc_file_orog.variables['rlat'][:]
    vcsn_lons = nc_file_orog.variables['rlon'][:]
    rot_pole = nc_file_orog.variables['rotated_pole']
    rot_pole_crs = ccrs.RotatedPole(rot_pole.grid_north_pole_longitude, rot_pole.grid_north_pole_latitude, rot_pole.north_pole_grid_longitude)

    vcsn_dt = nc.num2date(nc_file_rain.variables['time2'][:], nc_file_rain.variables['time2'].units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)
    vcsn_dt2 = nc.num2date(nc_file_temp.variables['time0'][:], nc_file_temp.variables['time0'].units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)


    # calculate model grid etc:
    # output DEM
    dem_file = dem_folder + '/' + output_dem + '.tif'
    if output_dem == 'si_dem_250m':
        nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(dem_file, extent_w=1.08e6, extent_e=1.72e6, extent_n=5.52e6, extent_s=4.82e6,
                                                                              resolution=250)
    elif output_dem == 'nz_dem_250m':
        nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(dem_file, extent_w=1.05e6, extent_e=2.10e6, extent_n=6.275e6, extent_s=4.70e6,
                                                                              resolution=250, origin='bottomleft')
    else:
        print('incorrect dem chosen')

    if mask_dem == True:
        # Get the masks for the individual regions of interest
        mask = np.load(mask_folder + '/{}_{}.npy'.format(catchment, output_dem))
        # Trim down the number of latitudes requested so it all stays in memory
        wgs84_lats, wgs84_lons, elev, northings, eastings = trim_lat_lon_bounds(mask, lat_array, lon_array, nztm_dem, y_centres, x_centres)
        _, _, trimmed_mask, _, _ = trim_lat_lon_bounds(mask, lat_array, lon_array, mask.copy(), y_centres, x_centres)
    else:
        wgs84_lats = lat_array
        wgs84_lons = lon_array
        elev = nztm_dem
        northings = y_centres
        eastings = x_centres
        # set mask to all land points
        mask = elev > 0
        trimmed_mask = mask
    # calculate lat/lon on rotated grid of input
    yy, xx = np.meshgrid(northings, eastings, indexing='ij')
    rotated_coords = rot_pole_crs.transform_points(ccrs.epsg(2193), xx, yy)
    rlats = rotated_coords[:, :, 1]
    rlons = rotated_coords[:, :, 0]
    rlons[rlons < 0] = rlons[rlons < 0] + 360

    # set up time to run, paths to input files etc
    for year_to_take in hydro_years_to_take:
        print(year_to_take)
        # specify the days to run (output is at the end of each day)
        # out_dt = np.asarray(make_regular_timeseries(dt.datetime(year_to_take, 7, 1), dt.datetime(year_to_take, 7, 2), 86400))
        out_dt = np.asarray(make_regular_timeseries(dt.datetime(year_to_take - 1, 4, 1), dt.datetime(year_to_take, 4, 1), 86400))

        # set up output netCDF:
        out_nc_file = setup_nztm_grid_netcdf(
            output_folder + '/snow_out_{}_{}_{}_{}_{}_{}.nc'.format(met_inp, which_model, catchment, output_dem, run_id, year_to_take),
            None, ['swe', 'acc', 'melt', 'rain', 'ros', 'ros_melt'],
            out_dt, northings, eastings, wgs84_lats, wgs84_lons, elev)

        # set up initial states of prognostic variables
        init_swe = np.zeros(elev.shape)  # default to no snow
        init_d_snow = np.ones(elev.shape) * 30  # default to a month since snowfall
        swe = init_swe
        d_snow = init_d_snow
        # set up daily buckets for melt and accumulation
        bucket_melt = swe * 0
        bucket_acc = swe * 0
        swe_day_before = swe * 0
        bucket_rain = swe * 0
        bucket_ros = swe * 0
        bucket_ros_melt = swe * 0

        # store initial swe value
        out_nc_file.variables['swe'][0, :, :] = init_swe
        out_nc_file.variables['acc'][0, :, :] = 0
        out_nc_file.variables['melt'][0, :, :] = 0
        out_nc_file.variables['rain'][0, :, :] = 0
        out_nc_file.variables['ros'][0, :, :] = 0
        out_nc_file.variables['ros_melt'][0, :, :] = 0

        # for each day:
        for ii, dt_t in enumerate(out_dt[:-1]):
            print('processing', dt_t)
            # load one day of precip and shortwave rad data
            precip_hourly = nc_rain[int(np.where(vcsn_dt == dt_t)[0]):int(int(np.where(vcsn_dt == dt_t)[0]) + 24)]
            temp_hourly = nc_temp[int(np.where(vcsn_dt2 == dt_t)[0]):int(np.where(vcsn_dt2 == dt_t)[0]) + 24]

            # interpolate data to fine grid
            hi_res_precip = interpolate_met(precip_hourly.filled(np.nan), 'rain', vcsn_lons, vcsn_lats, vcsn_elev_interp, rlons, rlats, elev)
            hi_res_temp = interpolate_met(temp_hourly.filled(np.nan), 'tmax', vcsn_lons, vcsn_lats, vcsn_elev_interp, rlons, rlats, elev)

            # mask out areas we don't want/need
            if mask is not None:
                hi_res_precip[:, trimmed_mask == 0] = np.nan
                hi_res_temp[:, trimmed_mask == 0] = np.nan

            hourly_dt = np.asarray(make_regular_timeseries(dt_t, dt_t + dt.timedelta(hours=23), 3600))
            hourly_doy = convert_datetime_julian_day(hourly_dt)
            hourly_temp = hi_res_temp
            hourly_precip = hi_res_precip

            if which_model == 'dsc_snow':
                sw_rad_hourly = nc_srad[int(np.where(vcsn_dt4 == dt_t)[0]):int(np.where(vcsn_dt4 == dt_t)[0]) + 24]
                hi_res_sw_rad = interpolate_met(sw_rad_hourly.filled(np.nan), 'srad', vcsn_lons, vcsn_lats, vcsn_elev_interp, rlons, rlats, elev)
                if mask is not None:
                    hi_res_sw_rad[:, trimmed_mask == 0] = np.nan
                hourly_swin = hi_res_sw_rad

            # calculate snow and output to netcdf
            for i in range(len(hourly_dt)):
                # d_snow += dtstep / 86400.0
                if which_model == 'dsc_snow':
                    swe, d_snow, melt, acc = calc_dswe(swe, d_snow, hourly_temp[i], hourly_precip[i], hourly_doy[i], 3600, which_melt=which_model,
                                                       sw=hourly_swin[i],
                                                       **config)  #
                else:
                    swe, d_snow, melt, acc = calc_dswe(swe, d_snow, hourly_temp[i], hourly_precip[i], hourly_doy[i], 3600, which_melt=which_model,
                                                       **config)
                # print swe[0]
                bucket_melt = bucket_melt + melt
                bucket_acc = bucket_acc + acc
                rain = hourly_precip[i] - acc
                bucket_rain = bucket_rain + rain
                bucket_ros = bucket_ros + rain * (swe > 0).astype(np.int)  # creates binary snow cover then multiples by rain (0 or 1)

                # first calculate the energy availble for melting due to rainfall (Wm^-2) over snowcovered cells only
                qprc = (swe > 0).astype(np.int) * 4220. * rain / 3600. * (hourly_temp[i] - 273.16)
                # then calculate potential melt per timestep . don't limit to available swe as could have contributed to intial snow melt (accounted for by degree-day model)
                ros_melt = qprc / 334000. * 3600.
                ros_melt[(ros_melt < 0)] = 0  # only take positive portion (could be some rain at air temperature < 0)
                bucket_ros_melt = bucket_ros_melt + ros_melt
            # output at the end of each day,
            for var, data in zip(['swe', 'acc', 'melt', 'rain', 'ros', 'ros_melt'], [swe, bucket_acc, bucket_melt, bucket_rain, bucket_ros, bucket_ros_melt]):
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
            bucket_rain = bucket_rain * 0
            bucket_ros = bucket_ros * 0
            bucket_ros_melt = bucket_ros_melt * 0
        out_nc_file.close()

        json.dump(config, open(output_folder + '/config_{}_{}_{}_{}_{}_{}.json'.format(met_inp, which_model, catchment, output_dem, run_id, year_to_take), 'w'))
        pickle.dump(config,
                    open(output_folder + '/config_{}_{}_{}_{}_{}_{}.pkl'.format(met_inp, which_model, catchment, output_dem, run_id, year_to_take), 'wb'),
                    protocol=3)


if __name__ == '__main__':

    if len(sys.argv) == 2:
        hydro_years_to_take = [int(sys.argv[1])]
        print('taking year from user input')
    else:
        print('taking year from script')
        hydro_years_to_take = [2021]  # np.arange(2018, 2018 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)

    # naming conventions
    run_id = 'cl09_default_ros'
    met_inp = 'nzcsm7-12'  # identifier for input meteorology
    # model options
    which_model = 'clark2009'  # 'clark2009'  # 'clark2009'  # 'dsc_snow'  # string identifying the model to be run. options include 'clark2009', 'dsc_snow' # future will include 'fsm'

    # time and grid extent options
    catchment = 'NZ'  # 'MtCook'  #  # # string identifying the catchment to run. must match the naming of the catchment mask file
    output_dem = 'nz_dem_250m'  # 'si_dem_250m'  #  #  string identifying output DEM
    mask_dem = True

    # folder location
    mask_folder = '/nesi/project/niwa00004/jonoconway'
    dem_folder = '/nesi/project/niwa00004/jonoconway'  # dem used for output #'C:/Users/conwayjp/OneDrive - NIWA/Data/GIS_DATA/Topography/DEM_NZSOS'#
    output_folder = '/nesi/nobackup/niwa00004/jonoconway/snow_sims_nz/nzcsm'  # snow model output
    data_folder = '/nesi/nobackup/niwa00004/jonoconway'  # input meteorology #'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2101/tn_file_compilation'#
    # input filenames
    orog_infile = 'tn_2020083000-utc_nzcsm_coords.nc'
    precip_infile = 'total_precip_nzcsm_2015043010_2020112306_national_hourly_FR7-12.nc'
    air_temp_infile = 'air_temperature_nzcsm_2015043010_2020112306_national_hourly_FR7-12.nc'
    solar_rad_infile = 'solar_radiation_nzcsm_2015043010_2020061706_national_hourly_FR7-12.nc'

    # configuration dictionary containing model parameters.
    config = {}
    config['which_model'] = which_model
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

    run_snow_otf_nzcsm_main(hydro_years_to_take, run_id, met_inp, which_model, catchment, output_dem, mask_dem, mask_folder, dem_folder, output_folder, data_folder,
                      orog_infile, precip_infile, air_temp_infile, solar_rad_infile, config)
