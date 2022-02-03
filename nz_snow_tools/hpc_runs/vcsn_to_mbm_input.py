"""
 use environment  pvlib38-x
"""

import sys
import pickle
import numpy as np
import datetime as dt
import pandas as pd

import matplotlib.pylab as plt
import xarray as xr
from nz_snow_tools.util.utils import make_regular_timeseries, process_precip, process_temp_flex, calc_toa, daily_to_hourly_temp_grids_new, \
    daily_to_hourly_swin_grids_new

# sys.path.append('C:/Users/conwayjp/Documents/code/git_niwa_local/cloudglacier')
# from obj1.process_cloud_vcsn import process_daily_swin_to_cloud, ea_from_tc_rh

# if len(sys.argv) == 2:
#     config_file = sys.argv[1]
#     config = yaml.load(open(config_file), Loader=yaml.FullLoader)
#     print('reading configuration file')
# else:
#     print('incorrect number of commandline inputs')
#

# set up output times
# first_time = parser.parse(config['output_file']['first_timestamp'])
# last_time = parser.parse(config['output_file']['last_timestamp'])

met_out_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Nariefa/vcsn'
data_id = 'brewster_test_mbm'

# find lats and lons of closest points
lat_to_take = [-44.075, -44.125]
lon_to_take = [169.425, 169.475]

first_time = dt.datetime(2009, 4, 1, 1, 0, tzinfo=dt.timezone(dt.timedelta(hours=12)))
last_time = dt.datetime(2021, 4, 1, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=12)))
out_dt = pd.date_range(first_time, last_time, freq='1H').to_pydatetime()

first_time_lt = dt.datetime(2009, 4, 1, 1, 0)
last_time_lt = dt.datetime(2021, 4, 1, 0, 0)
out_dt_lt = pd.date_range(first_time_lt, last_time_lt, freq='1H').to_pydatetime()

outfile = met_out_folder + '/met_inp_{}_{}_{}.dat'.format(data_id, first_time_lt.strftime('%Y%m%d%H%M'), last_time_lt.strftime('%Y%m%d%H%M'))

# out_dt = np.asarray(make_regular_timeseries(first_time, last_time, 3600))
print('time output from {} to {}'.format(first_time.isoformat(), last_time.isoformat()))

# define location of input files
# vcsn_folder = '/scale_wlg_persistent/filesets/project/niwa00026/VCSN_grid'
# vcsn_tmax_file = vcsn_folder + '/TMax_Norton/tmax_vclim_clidb_Norton_1972010200_2021090200_south-island_p05_daily_netcdf4.nc'# max air temp (C) over 24 hours FROM 9am on local day
# vcsn_tmin_file = vcsn_folder + '/TMin_Norton/tmin_vclim_clidb_Norton_1972010200_2021090200_south-island_p05_daily_netcdf4.nc'# min air temp (C) over 24 hours TO 9 am on local day
# vcsn_rh_file = vcsn_folder + '/RH/rh_vclim_clidb_1972010200_2021090200_south-island_p05_daily_netcdf4.nc'# rh (%) AT 9am on local day
# vcsn_mslp_file = vcsn_folder + '/MSLP/mslp_vclim_clidb_1972010200_2021090200_south-island_p05_daily_netcdf4.nc'# mslp (hPa) AT 9am on local day
# vcsn_swin_file = vcsn_folder + '/SRad/srad_vclim_clidb_1972010200_2021090200_south-island_p05_daily_netcdf4.nc'# Total global solar radiation (MJ/m2) over 24 hours FROM midnight on local day
# vcsn_ws_file = vcsn_folder + '/Wind/wind_vclim_clidb_1972010200_2021090200_south-island_p05_daily_netcdf4.nc'# Mean 10m wind speed (m/s) over 24 hours FROM midnight local day
# vcsn_precip_file = vcsn_folder + '/Rain/rain_vclim_clidb_1972010200_2021090200_south-island_p05_daily_netcdf4.nc'# Total precip (mm) over 24 hours FROM 9am local day

# ds_rain = xr.open_dataset('Rain/rain_vclim_clidb_1972010200_2021090200_south-island_p05_daily_netcdf4.nc')
# ds_rain = ds_rain.assign_coords(longitude=ds_rain['longitude'].round(3)).assign_coords(latitude=ds_rain['latitude'].round(3))
# ds_rain.sel(longitude=[169.425,169.475],latitude=[-44.075,-44.125]).to_netcdf('/nesi/project/niwa03098/rain.nc')

vcsn_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Nariefa/vcsn/to share'
vcsn_tmax_file = vcsn_folder + '/tmax.nc'  # max air temp (C) over 24 hours TO 9am on local day #  NOTE this differs from cliflo version
vcsn_tmin_file = vcsn_folder + '/tmin.nc'  # min air temp (C) over 24 hours TO 9 am on local day
vcsn_rh_file = vcsn_folder + '/rh.nc'  # rh (%) AT 9am on local day
vcsn_mslp_file = vcsn_folder + '/mslp.nc'  # mslp (hPa) AT 9am on local day
vcsn_swin_file = vcsn_folder + '/srad.nc'  # Total global solar radiation (MJ/m2) over 24 hours FROM midnight on local day # this is different to the description in netCDF
vcsn_ws_file = vcsn_folder + '/ws.nc'  # Mean 10m wind speed (m/s) over 24 hours FROM midnight local day # this is different the description in netCDF
vcsn_precip_file = vcsn_folder + '/rain.nc'  # Total precip (mm) over 24 hours TO 9am local day # NOTE this differs from cliflo version
vcsn_tmax_original = vcsn_folder + '/tmax_original.nc'  # old version of air temperature
vcsn_tmin_original = vcsn_folder + '/tmin_original.nc'  # old version of air temperature

# define times to take (include offset to)
ds_tmax = xr.open_dataset(vcsn_tmax_file)
ds_tmin = xr.open_dataset(vcsn_tmin_file)
ds_rh = xr.open_dataset(vcsn_rh_file)
ds_mslp = xr.open_dataset(vcsn_mslp_file)
ds_swin = xr.open_dataset(vcsn_swin_file)
ds_ws = xr.open_dataset(vcsn_ws_file)
ds_precip = xr.open_dataset(vcsn_precip_file)
# ds_tmax_orig = xr.open_dataset(vcsn_tmax_original)
# ds_tmin_orig = xr.open_dataset(vcsn_tmin_original)

# load variables
# take day either side for tmax,tmin,rh,mslp, as well as offsetting tmax forward one day and also taking one extra day for precip. SW and ws are already midnight-midnight
inp_precip = ds_precip['rain'].sel(time=slice(first_time, last_time + dt.timedelta(days=1)), longitude=lon_to_take,
                                   latitude=lat_to_take)  # also take value from next day as is total over previous 24 hours
inp_tmax = ds_tmax['tmax'].sel(time=slice(first_time, last_time + dt.timedelta(days=2)), longitude=lon_to_take,
                               latitude=lat_to_take)  # take value from next day as is the max value over previous 24 hours.
inp_tmin = ds_tmin['tmin'].sel(time=slice(first_time - dt.timedelta(days=1), last_time + dt.timedelta(days=1)), longitude=lon_to_take,
                               latitude=lat_to_take)  # take day either side
inp_rh = ds_rh['rh'].sel(time=slice(first_time - dt.timedelta(days=1), last_time + dt.timedelta(days=1)), longitude=lon_to_take,
                         latitude=lat_to_take)  # take a day either side to interpolate
inp_mslp = ds_mslp['mslp'].sel(time=slice(first_time - dt.timedelta(days=1), last_time + dt.timedelta(days=1)), longitude=lon_to_take,
                               latitude=lat_to_take)  # take a day either side to interpolate
inp_swin = ds_swin['srad'].sel(time=slice(first_time, last_time), longitude=lon_to_take,
                               latitude=lat_to_take)  # data actually averages from midnight to midnight, so modify timestamp
updated_time = inp_swin.time + np.timedelta64(15, 'h')
inp_swin = inp_swin.assign_coords(time=('time', updated_time.data))
inp_ws = ds_ws['wind'].sel(time=slice(first_time - dt.timedelta(days=1), last_time), longitude=lon_to_take,
                           latitude=lat_to_take)  # take extra day on start to make interpolation easier. data actually averages from midnight to midnight, so modify timestamp
updated_time = inp_ws.time + np.timedelta64(15, 'h')  # move timestamp to midnight at end of day.
inp_ws = inp_ws.assign_coords(time=('time', updated_time.data))

# inp_tmax_orig = ds_tmax_orig['tmax'].sel(time=slice(first_time, last_time), longitude=lon_to_take, latitude=lat_to_take) # TODO need to take value from next day as the variable is the max value over previous 24 hours.
# inp_tmin_orig = ds_tmin_orig['tmin'].sel(time=slice(first_time, last_time), longitude=lon_to_take, latitude=lat_to_take)

# spatial interpolation and bias correction
hi_res_precip = inp_precip  #
hi_res_tmax = inp_tmax  # TODO introduce bias correction
hi_res_tmin = inp_tmin  # TODO introduce bias correction
hi_res_rh = inp_rh  #
hi_res_pres = inp_mslp  # TODO lapse to correct elevation
hi_res_swin = inp_swin  #
hi_res_ws = inp_ws  #
hi_res_lats = ds_rh['latitude'].data
hi_res_lons = ds_rh['longitude'].data

# convert to hourly

# send in data for start day to end date+1, then cut the first 15 hours and last 9 hours.
hourly_precip_full, day_weightings_full = process_precip(hi_res_precip.data)
hourly_precip = hourly_precip_full[
                15:-9]  # remove 15 hours data from day before and cut 9 hours from end (to align 24 hrs to 9am totals, to hourly totals midnight-midnight

# air temperature is three part sinusoidal curve
# brewster glacier AWS shows hour to 0600 and 1500 as mode of morning Tmin and afternoon Tmax, respectively.
# assumes daily input data are aligned so that min temp from previous 24 hours and max temp for next 24 hours are given in same index.
# TODO assert that tmax and tmin timebounds are 24 hours offset.
# send in day either side
hourly_temp = daily_to_hourly_temp_grids_new(hi_res_tmax.data, hi_res_tmin.data, time_min=6, time_max=15)
hourly_temp = hourly_temp[24:-24]# remove day either side

hourly_rh = hi_res_rh.resample(time="1H").interpolate(kind='cubic')
hourly_rh = hourly_rh[16:-9]  # trim to 1am on first day to midnight on last

hourly_mslp = hi_res_pres.resample(time="1H").interpolate(kind='cubic')
hourly_mslp = hourly_mslp[16:-9]  # trim to 1am on first day to midnight on last

hourly_vp = ea_from_tc_rh(hourly_temp - 273.16, hourly_rh)

# just compute cloudiness for one grid point
daily_vp = hourly_vp.data.reshape((-1, 24, hourly_temp.shape[1], hourly_temp.shape[2])).sum(axis=1)
daily_tk = hourly_temp.reshape((-1, 24, hourly_temp.shape[1], hourly_temp.shape[2])).sum(axis=1)
daily_neff, daily_trc = process_daily_swin_to_cloud(hi_res_swin[:, 0, 0].to_pandas(), daily_vp[:, 0, 0], daily_tk[:, 0, 0], 1500, -45, 179)

hourly_neff = (np.expand_dims(daily_neff,axis=1) * np.ones(24)).reshape(daily_neff.shape[0]*24)
hourly_neff = xr.DataArray(hourly_neff,hourly_rh[:,0,0].coords,name='neff')
hourly_swin = daily_to_hourly_swin_grids_new(hi_res_swin.data, hi_res_lats, hi_res_lons, out_dt_lt)  # TODO output cloud cover not solar radiation

hourly_ws = hi_res_ws.resample(time="1H").bfill()[1:]  # fill in average value and remove first timestamp

# merge into one dataset
hourly_precip = xr.DataArray(hourly_precip,hourly_rh.coords,name='precip')
hourly_temp = xr.DataArray(hourly_temp,hourly_rh.coords,name='tempK')
hourly_swin = xr.DataArray(hourly_swin,hourly_rh.coords,name='swin')

ds = xr.merge([hourly_precip,hourly_temp,hourly_rh,hourly_mslp,hourly_ws,hourly_neff,hourly_swin])
# select one point and output to csv
df = ds.sel(latitude=-44.075,longitude=169.425,method='nearest').to_pandas()
# tidy up
df = df.drop(labels='inplace',axis='columns')

#first add the timezone info, then convert to NZST
df['NZST'] = df.index.tz_localize(tz='UTC').tz_convert(tz=dt.timezone(dt.timedelta(hours=12)))
df = df.set_index('NZST')
df['doy'] = df.index.day_of_year
df['hour'] = df.index.hour
df['year'] = df.index.year

df.to_csv(outfile)

pickle.dump(day_weightings_full,
            open(met_out_folder + '/met_inp_{}_{}_{}_daywts.pkl'.format(data_id, first_time.strftime('%Y%m%d%H%M'), last_time.strftime('%Y%m%d%H%M')),
                 'wb'), protocol=3)

print()
