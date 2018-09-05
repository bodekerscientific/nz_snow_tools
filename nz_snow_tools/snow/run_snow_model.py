"""
code to run snow model on large grid for multiple years. outputs to pickle files to be easily read into python later
"""
from __future__ import division

import netCDF4 as nc
import pickle
import numpy as np
import datetime as dt
import matplotlib.pylab as plt

from nz_snow_tools.snow.clark2009_snow_model import snow_main
from nz_snow_tools.util.utils import create_mask_from_shpfile, make_regular_timeseries

which_model = 'dsc_snow'  # string identifying the model to be run. options include 'clark2009', 'dsc_snow' # future will include 'fsm'
catchment = 'Clutha'  # string identifying the catchment to run. must match the naming of the catchment shapefile
output_dem = 'nztm250m'  # identifier for output dem
hydro_years_to_take = range(2001, 2016 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
catchment_shp_folder = 'Z:/GIS_DATA/Hydrology/Catchments'
output_folder = 'T:/DSC-Snow/nz_snow_runs/baseline_clutha1'
met_data_folder = 'T:/DSC-Snow/input_data_hourly'

#configuration dictionary containing model parameters.
config = {}
config['tacc'] = 274.16


# clark2009 melt parameters
config['mf_mean'] = 5.0
config['mf_amp'] = 5.0
config['mf_alb'] = 2.5
config['mf_alb_decay'] = 5.0
config['mf_ros'] = 0
config['mf_doy_max_ddf'] = 35

# dsc_snow melt parameters
config['tf'] = 0.2 * 24  # hamish 0.13
config['rf'] = 0.005 * 24  # hamish 0.0075
# albedo parameters
config['dc'] = 11.0
config['tc'] = 10.0
config['a_ice'] = 0.42
config['a_freshsnow'] = 0.90
config['a_firn'] = 0.62
config['alb_swe_thres'] = 20

for which_model in ['clark2009', 'dsc_snow']:
    if which_model == 'clark2009':
        config['tmelt'] = 273.16
    if which_model == 'dsc_snow':
        config['tmelt'] = 274.16
    for hydro_year_to_take in hydro_years_to_take:
        # run model and return timeseries of daily swe, acc and melt.
        met_infile = met_data_folder + '/met_inp_{}_{}_hy{}.nc'.format(catchment, output_dem, hydro_year_to_take)
        st_swe, st_melt, st_acc = snow_main(met_infile, which_melt=which_model,**config)
        # create the extra variables required by catchment_evaluation
        inp_met = nc.Dataset(met_infile, 'r')
        inp_dt = nc.num2date(inp_met.variables['time'][:], inp_met.variables['time'].units)
        out_dt = np.asarray(make_regular_timeseries(inp_dt[0], inp_dt[-1], 86400))
        mask = create_mask_from_shpfile(inp_met.variables['lat'][:], inp_met.variables['lon'][:], catchment_shp_folder + '/{}.shp'.format(catchment))
        pickle.dump([st_swe.astype(dtype=np.float32), st_melt.astype(dtype=np.float32), st_acc.astype(dtype=np.float32), out_dt, mask, config], open(
            output_folder + '/{}_{}_hy{}_{}.pkl'.format(catchment, output_dem, hydro_year_to_take, which_model), 'wb'), -1)
