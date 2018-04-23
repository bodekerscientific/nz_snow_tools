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

which_model = 'clark2009'  # string identifying the model to be run. options include 'clark2009', 'dsc_snow', or 'all' # future will include 'fsm'
catchment = 'Nevis' # note need to add ' full WGS84" to mask file path for Clutha catchment
output_dem = 'nztm250m'  # identifier for output dem
hydro_years_to_take = range(2013, 2017 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
catchment_shp_folder = 'Z:/GIS_DATA/Hydrology/Catchments'
output_folder = 'P:/Projects/DSC-Snow/nz_snow_runs'
met_data_folder = 'Y:/DSC-Snow/input_data_hourly'

for hydro_year_to_take in hydro_years_to_take:
    # run model and return timeseries of daily swe, acc and melt.
    met_infile = met_data_folder + '/met_inp_{}_{}_hy{}.nc'.format(catchment, output_dem, hydro_year_to_take)
    st_swe, st_melt, st_acc = snow_main(met_infile, which_melt=which_model)
    # create the extra variables required by catchment_evaluation
    inp_met = nc.Dataset(met_infile, 'r')
    inp_dt = nc.num2date(inp_met.variables['time'][:], inp_met.variables['time'].units)
    out_dt = np.asarray(make_regular_timeseries(inp_dt[0], inp_dt[-1] + dt.timedelta(days=1), 86400))
    mask = create_mask_from_shpfile(inp_met.variables['lat'][:], inp_met.variables['lon'][:], catchment_shp_folder + '/{}.shp'.format(catchment))
    pickle.dump([st_swe.astype(dtype=np.float32), st_melt.astype(dtype=np.float32), st_acc.astype(dtype=np.float32), out_dt, mask], open(
        output_folder + '/{}_{}_hy{}_{}.pkl'.format(catchment, output_dem, hydro_year_to_take, which_model), 'wb'), -1)
