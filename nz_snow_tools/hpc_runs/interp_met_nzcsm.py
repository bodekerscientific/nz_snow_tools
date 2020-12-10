"""
interpolate input data to model grid adjusting temperature-dependent fields for elevation changes and output to netCDF

assumes input air temp in K for calculation of LW rad

requires
- dem
- mask created corresponding
by default create
"""

import yaml
import os
import sys
import netCDF4 as nc
import numpy as np
import cartopy.crs as ccrs
import datetime as dt
from dateutil import parser
import matplotlib.pylab as plt

from nz_snow_tools.util.utils import make_regular_timeseries, u_v_from_ws_wd,ws_wd_from_u_v
from nz_snow_tools.met.interp_met_data_hourly_vcsn_data import interpolate_met, setup_nztm_dem, setup_nztm_grid_netcdf, trim_lat_lon_bounds

if len(sys.argv) == 2:
    config_file = sys.argv[1]
    config = yaml.load(open(config_file), Loader=yaml.FullLoader)
    print('reading configuration file')
else:
    print('incorrect number of commandline inputs')
    config = yaml.load(open(r'C:\Users\conwayjp\Documents\code\GitHub\nz_snow_tools\nz_snow_tools\hpc_runs\nzcsm_local.yaml'), Loader=yaml.FullLoader)

# open input met grid (assume is the same for all variables)
print('processing input orogrpahy')
nc_file_orog = nc.Dataset(config['input_grid']['dem_file'], 'r')
input_elev = nc_file_orog.variables['orog_model'][:]
inp_elev_interp = input_elev.copy()
# inp_elev_interp = np.ma.fix_invalid(input_elev).data
inp_lats = nc_file_orog.variables['rlat'][:]
inp_lons = nc_file_orog.variables['rlon'][:]
rot_pole = nc_file_orog.variables['rotated_pole']
rot_pole_crs = ccrs.RotatedPole(rot_pole.grid_north_pole_longitude, rot_pole.grid_north_pole_latitude, rot_pole.north_pole_grid_longitude)

# create dem of model output grid:
print('processing output orogrpahy')
if config['output_grid']['dem_name'] == 'si_dem_250m':
    nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(config['output_grid']['dem_file'], extent_w=1.08e6, extent_e=1.72e6, extent_n=5.52e6,
                                                                          extent_s=4.82e6, resolution=250, origin='bottomleft')

elif config['output_grid']['dem_name'] == 'nz_dem_250m':
    nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(config['output_grid']['dem_file'], extent_w=1.05e6, extent_e=2.10e6, extent_n=6.275e6,
                                                                          extent_s=4.70e6, resolution=250, origin='bottomleft')

if config['output_grid']['catchment_mask'] == "elev":  # just set mask to all land points
    wgs84_lats = lat_array
    wgs84_lons = lon_array
    elev = nztm_dem
    northings = y_centres
    eastings = x_centres
    mask = elev > 0
    trimmed_mask = mask
else:  # Get the mask for the region of interest
    mask = np.load(config['output_grid']['catchment_mask'])
    # Trim down the number of latitudes requested so it all stays in memory
    wgs84_lats, wgs84_lons, elev, northings, eastings = trim_lat_lon_bounds(mask, lat_array, lon_array, nztm_dem, y_centres, x_centres)
    _, _, trimmed_mask, _, _ = trim_lat_lon_bounds(mask, lat_array, lon_array, mask.copy(), y_centres, x_centres)

# calculate rotated grid lat/lon of output grid
yy, xx = np.meshgrid(northings, eastings, indexing='ij')
rotated_coords = rot_pole_crs.transform_points(ccrs.epsg(2193), xx, yy)
rlats = rotated_coords[:, :, 1]
rlons = rotated_coords[:, :, 0]
rlons[rlons < 0] = rlons[rlons < 0] + 360

# set up output times
first_time = parser.parse(config['output_file']['first_timestamp'])
last_time = parser.parse(config['output_file']['last_timestamp'])
out_dt = np.asarray(make_regular_timeseries(first_time, last_time, config['output_file']['timestep']))
print('time output from {} to {}'.format(first_time.strftime('%Y-%m-%d %H:%M'), last_time.strftime('%Y-%m-%d %H:%M')))

# set up output netCDF without variables
if not os.path.exists(config['output_file']['output_folder']):
    os.makedirs(config['output_file']['output_folder'])
output_file = config['output_file']['file_name_template'].format(first_time.strftime('%Y%m%d%H%M'), last_time.strftime('%Y%m%d%H%M'))
out_nc_file = setup_nztm_grid_netcdf(config['output_file']['output_folder'] + output_file, None, [], out_dt, northings, eastings, wgs84_lats, wgs84_lons, elev)
# run through each variable
for var in config['variables'].keys():
    print('processing {}'.format(var))
    t = out_nc_file.createVariable(config['variables'][var]['output_name'], 'f4', ('time', 'northing', 'easting',), zlib=True)  # ,chunksizes=(1, 100, 100)
    t.setncatts(config['variables'][var]['output_meta'])
    inp_nc_file = nc.Dataset(config['variables'][var]['input_file'], 'r')
    inp_dt = nc.num2date(inp_nc_file.variables[config['variables'][var]['input_time_var']][:],
                         inp_nc_file.variables[config['variables'][var]['input_time_var']].units,
                         only_use_cftime_datetimes=False)  # only_use_python_datetimes=True
    if 'round_time' in config['variables'][var].keys():
        if config['variables'][var]['round_time']:
            inp_hours = nc.date2num(inp_dt, 'hours since 1900-01-01 00:00')
            inp_dt = nc.num2date(np.round(inp_hours, 0), 'hours since 1900-01-01 00:00')
    # load variables relevant for interpolation
    if var == 'lw_rad':
        inp_nc_var = inp_nc_file.variables[config['variables'][var]['input_var_name']]
        # load air temperature
        inp_nc_file_t = nc.Dataset(config['variables']['air_temp']['input_file'], 'r')
        inp_dt_t = nc.num2date(inp_nc_file_t.variables[config['variables']['air_temp']['input_time_var']][:],
                               inp_nc_file_t.variables[config['variables']['air_temp']['input_time_var']].units, only_use_cftime_datetimes=False)
        inp_nc_var_t = inp_nc_file_t.variables[config['variables']['air_temp']['input_var_name']]
    elif var == 'wind_speed' or var == 'wind_direction':
        if 'convert_uv' in config['variables'][var].keys():
            if config['variables'][var]['convert_uv']:
                inp_nc_var_u = inp_nc_file.variables[config['variables'][var]['input_var_name_u']]
                inp_nc_var_v = inp_nc_file.variables[config['variables'][var]['input_var_name_v']]
        else:
            inp_nc_var = inp_nc_file.variables[config['variables'][var]['input_var_name']]
            if var == 'wind_direction':  # load wind speed to enable conversion to u/v for interpolation
                inp_nc_var_ws = inp_nc_file.variables[config['variables']['wind_speed']['input_var_name']]
    else:
        inp_nc_var = inp_nc_file.variables[config['variables'][var]['input_var_name']]

    if var == 'total_precip':  # load temp (and optionally rh) to calculate rain/snow rate if needed
        if 'calc_rain_snow_rate' in config['variables'][var].keys():
            if config['variables'][var]['calc_rain_snow_rate']:
                # set up additional outputs
                sfr = out_nc_file.createVariable('snowfall_rate', 'f4', ('time', 'northing', 'easting',), zlib=True)
                sfr.setncatts(config['variables'][var]['snow_rate_output_meta'])
                rfr = out_nc_file.createVariable('rainfall_rate', 'f4', ('time', 'northing', 'easting',), zlib=True)
                rfr.setncatts(config['variables'][var]['rain_rate_output_meta'])
                # load air temperature
                inp_nc_file_t = nc.Dataset(config['variables']['air_temp']['input_file'], 'r')
                inp_dt_t = nc.num2date(inp_nc_file_t.variables[config['variables']['air_temp']['input_time_var']][:],
                                       inp_nc_file_t.variables[config['variables']['air_temp']['input_time_var']].units, only_use_cftime_datetimes=False)
                inp_nc_var_t = inp_nc_file_t.variables[config['variables']['air_temp']['input_var_name']]
                if config['variables'][var]['rain_snow_method'] == 'harder':
                    # load rh #TODO arange variable keys so that t and rh are calculated before rain/snow rate and lw_rad
                    inp_nc_file_rh = nc.Dataset(config['variables']['rh']['input_file'], 'r')
                    inp_dt_rh = nc.num2date(inp_nc_file_rh.variables[config['variables']['rh']['input_time_var']][:],
                                            inp_nc_file_rh.variables[config['variables']['rh']['input_time_var']].units, only_use_cftime_datetimes=False)
                    inp_nc_var_rh = inp_nc_file_rh.variables[config['variables']['rh']['input_var_name']]
                    import pickle
                    from scipy import interpolate

                    dict = pickle.load(open('C:/Users/conwayjp/OneDrive - NIWA/projects/SIN_density_SIP/input_met/hydrometeor_temp_lookup.pkl', 'rb'))
                    th_interp = interpolate.interp2d(dict['rh'], dict['tc'], dict['th'], kind='linear')

    # run though each timestep output interpolate data to fine grid
    for ii, dt_t in enumerate(out_dt):
        if var == 'lw_rad':
            # calculate effective emissivity, interpolate that, then recreate lw rad with lapsed air temperature.
            input_hourly = inp_nc_var[int(np.where(inp_dt == dt_t)[0]), :, :]
            input_hourly = input_hourly / (5.67e-8 * inp_nc_var_t[int(np.where(inp_dt_t == dt_t)[0]), :, :] ** 4)
            hi_res_out = interpolate_met(input_hourly.filled(np.nan), var, inp_lons, inp_lats, inp_elev_interp, rlons, rlats, elev, single_dt=True)
            hi_res_tk = out_nc_file[config['variables']['air_temp']['output_name']][ii, :, :]
            hi_res_out = hi_res_out * (5.67e-8 * hi_res_tk ** 4)
        elif var == 'air_pres':
            # reduce to sea-level - interpolate then raise to new grid.
            input_hourly = inp_nc_var[int(np.where(inp_dt == dt_t)[0]), :, :]
            input_hourly = input_hourly + 101325 * (1 - (1 - input_elev / 44307.69231) ** 5.253283)  # taken from campbell logger program from Athabasca Glacier
            hi_res_out = interpolate_met(input_hourly.filled(np.nan), var, inp_lons, inp_lats, inp_elev_interp, rlons, rlats, elev, single_dt=True)
            hi_res_out = hi_res_out - 101325 * (1 - (1 - elev / 44307.69231) ** 5.253283)
        elif var == 'rh':
            # need to ignore mask for rh data and has incorrect limit of 1.
            input_hourly = inp_nc_var[int(np.where(inp_dt == dt_t)[0]), :, :].data
            input_hourly = input_hourly * 100  # convert to %
            hi_res_out = interpolate_met(input_hourly, var, inp_lons, inp_lats, inp_elev_interp, rlons, rlats, elev, single_dt=True)
        elif var == 'wind_speed':
            if 'convert_uv' in config['variables'][var].keys():
                if config['variables'][var]['convert_uv']:
                    input_hourly = np.sqrt(inp_nc_var_u[int(np.where(inp_dt == dt_t)[0]), :, :] ** 2 +
                                           inp_nc_var_v[int(np.where(inp_dt == dt_t)[0]), :, :] ** 2)
            else:
                input_hourly = inp_nc_var[int(np.where(inp_dt == dt_t)[0]), :, :]
            hi_res_out = interpolate_met(input_hourly.filled(np.nan), var, inp_lons, inp_lats, inp_elev_interp, rlons, rlats, elev, single_dt=True)

        elif var == 'wind_direction':
            if 'convert_uv' in config['variables'][var].keys():
                if config['variables'][var]['convert_uv']:
                    input_hourly_u = inp_nc_var_u[int(np.where(inp_dt == dt_t)[0]), :, :]
                    input_hourly_v = inp_nc_var_v[int(np.where(inp_dt == dt_t)[0]), :, :]
            else:
                input_hourly_wd = inp_nc_var[int(np.where(inp_dt == dt_t)[0]), :, :]
                input_hourly_ws = inp_nc_var_ws[int(np.where(inp_dt == dt_t)[0]), :, :]
                input_hourly_u, input_hourly_v = u_v_from_ws_wd(input_hourly_ws, input_hourly_wd)

            hi_res_out_u = interpolate_met(input_hourly_u.filled(np.nan), var, inp_lons, inp_lats, inp_elev_interp, rlons, rlats, elev, single_dt=True)
            hi_res_out_v = interpolate_met(input_hourly_v.filled(np.nan), var, inp_lons, inp_lats, inp_elev_interp, rlons, rlats, elev, single_dt=True)
            hi_res_out = np.rad2deg(np.arctan2(-hi_res_out_u, -hi_res_out_v))
        else:
            input_hourly = inp_nc_var[int(np.where(inp_dt == dt_t)[0]), :, :]
            hi_res_out = interpolate_met(input_hourly.filled(np.nan), var, inp_lons, inp_lats, inp_elev_interp, rlons, rlats, elev, single_dt=True)

        hi_res_out[trimmed_mask == 0] = np.nan

        # plt.figure()
        # plt.imshow(hi_res_out, origin='lower')
        # plt.colorbar()
        # plt.show()
        # add climate chnage offsets (temperature change will also affect the longwave radiation and rain/snow partioning, but not RH, or SW rad)
        if 'climate_change_offsets' in config.keys():
            if var in config['climate_change_offsets'].keys():
                if 'percentage_change' in config['climate_change_offsets'][var].keys():
                    hi_res_out = hi_res_out * (100. + config['climate_change_offsets'][var]['percentage_change']) / 100.
                elif 'absolute_change' in config['climate_change_offsets'][var].keys():
                    hi_res_out = hi_res_out + config['climate_change_offsets'][var]['absolute_change']

        t[ii, :, :] = hi_res_out
        if var == 'total_precip':  # load temp (and optionally rh) to calculate rain/snow rate if needed
            if 'calc_rain_snow_rate' in config['variables'][var].keys():
                if config['variables'][var]['calc_rain_snow_rate']:
                    if config['variables'][var]['rain_snow_method'] == 'harder':
                        hi_res_tk = out_nc_file[config['variables']['air_temp']['output_name']][ii, :, :]
                        hi_res_rh = out_nc_file[config['variables']['rh']['output_name']][ii, :, :]
                        hi_res_tc = hi_res_tk - 273.15
                        th = np.asarray([th_interp(r, t) for r, t in zip(hi_res_rh.ravel(), hi_res_tc.ravel())]).squeeze().reshape(hi_res_rh.shape)
                        b = 2.6300006
                        c = 0.09336
                        hi_res_frs = 1 - (1. / (1 + b * c ** th))  # fraction of snowfall
                        hi_res_frs[hi_res_frs < 0.01] = 0
                        hi_res_frs[hi_res_frs > 0.99] = 1

                    else:
                        hi_res_tk = out_nc_file[config['variables']['air_temp']['output_name']][ii, :, :]
                        hi_res_frs = (hi_res_tk < config['variables'][var]['rain_snow_method']).astype('float')
                    hi_res_rain_rate = hi_res_out * (1 - hi_res_frs) / config['output_file']['timestep']
                    hi_res_snow_rate = hi_res_out * hi_res_frs / config['output_file']['timestep']
                    rfr[ii, :, :] = hi_res_rain_rate
                    sfr[ii, :, :] = hi_res_snow_rate
                    # plt.figure()
                    # plt.imshow(hi_res_rain_rate, origin='lower')
                    # plt.colorbar()
                    # plt.figure()
                    # plt.imshow(hi_res_snow_rate, origin='lower')
                    # plt.colorbar()
                    # plt.show()

out_nc_file.close()
