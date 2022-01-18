"""
code to call the snow model for a simple test case using mueller hut data

modified to
"""

import numpy as np
import pickle as pkl
import pandas as pd
from nz_snow_tools.snow.clark2009_snow_model import snow_main_simple
from nz_snow_tools.util.utils import make_regular_timeseries, nash_sut, mean_bias, rmsd
import matplotlib.pylab as plt
import datetime as dt
import matplotlib.dates as mdates

outfolder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2201/snow_model_ensembles/clark_output'
# ros = True
# ta_m_tt = False
# load input data for mueller hut
infile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/SIN_density_SIP/input_met/mueller_hut_met_20170501_20200401_with_hs_swe_rain_withcloud_precip_harder.pkl'
aws_df = pkl.load(open(infile, 'rb'))
start_t = 1 # drop first timestamp as model uses initial value
end_t = None # take the end point.
inp_dt = [i.to_pydatetime() for i in aws_df.index[start_t-1:end_t]]
inp_doy = [i.dayofyear for i in aws_df.index[start_t:end_t]]
inp_hourdec = [i.hour for i in aws_df.index[start_t:end_t]]

# make grids of input data
grid_size = 1
grid_id = np.arange(grid_size)
inp_ta = aws_df.tc[start_t:end_t][:, np.newaxis] * np.ones(grid_size) + 273.16  # 2 metre air temp in C
inp_precip = aws_df.precip[start_t:end_t][:, np.newaxis] * np.ones(grid_size)  # precip in mm
inp_sw = aws_df.srad[start_t:end_t][:, np.newaxis] * np.ones(grid_size)

# set up
init_swe = np.ones(inp_ta.shape[1:]) * 0  # give initial value of swe
init_d_snow = np.ones(inp_ta.shape[1:]) * 30  # give initial value of days since snow

# configuration dictionary containing model parameters.
config = {}
config['num_secs_output'] = 3600
config['tacc'] = 274.16
# config['tmelt'] = 274.16

# dsc_snow melt parameters
# # config['tf'] = 0.05*24  # hamish 0.13
# config['rf'] = 0.0108 * 24  # hamish 0.0075
# # albedo parameters
# config['dc'] = 11.0
# config['tc'] = 10
# config['a_ice'] = 0.42
# config['a_freshsnow'] = 0.90
# config['a_firn'] = 0.62
# config['alb_swe_thres'] = 20
# config['ros'] = ros
# config['ta_m_tt'] = ta_m_tt  # use tmelt as baseline when calculating degree days

# use measured albedo
# tacc_list = np.linspace(-1,3,5) + 273.16
tmelt_list = np.linspace(-3, 6, 10) + 273.16 #23
tf_list = np.linspace(1, 10, 10)

# clark2009 melt parameters
config['mf_mean'] = 5.0
config['mf_amp'] = 5.0
config['mf_alb'] = 2.5
config['mf_alb_decay'] = 1
config['mf_ros'] = 0  # default 2.5
config['mf_doy_max_ddf'] = 356  # default 356
config['mf_doy_min_ddf'] = 210


# rf_list = np.linspace(0,100e-4,11)

# for tmelt in tmelt_list:
#     config['tmelt'] = tmelt
# loop to call range of parameters

stor_dict = {}

ii = 0
for i, tf in enumerate(tf_list):
    for j, tt in enumerate(tmelt_list):
        stor_dict[ii] = {}
        config['mf_mean'] = tf
        config['tmelt'] = tt
        # call main function once hourly/sub-hourly temp and precip data available.
        st_swe, st_melt, st_acc, st_alb = snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep=3600, init_swe=init_swe,
                                                           init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='clark2009', **config)

        run_dict = {
            'timestamp': inp_dt,
            'swe': np.squeeze(st_swe),
            'melt': np.squeeze(st_melt),
            'acc': np.squeeze(st_acc),
            'alb': np.squeeze(st_alb)
        }
        run_df = pd.DataFrame.from_dict(run_dict)
        run_df.set_index('timestamp',inplace=True)
        stor_dict[ii]['states_output'] = run_df
        # stor_dict[ii]['inp_dt'] = inp_dt
        # stor_dict[ii]['st_swe'] = st_swe
        # stor_dict[ii]['st_melt'] = st_melt
        # stor_dict[ii]['st_acc'] = st_acc
        # stor_dict[ii]['st_alb'] = st_alb
        stor_dict[ii]['config'] = config.copy() # need to copy to avoid being changed with config dictionary is updated

        ii += 1
        print(ii)
print()

pkl.dump(stor_dict, open(outfolder + '/collated_output_testA.pkl','wb'),-1)


