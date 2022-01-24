"""
code to call the snow model for a simple test case using mueller hut data

modified to include random generation of parameters
"""

import numpy as np
import pickle as pkl
import pandas as pd
from nz_snow_tools.snow.clark2009_snow_model import snow_main_simple
from nz_snow_tools.util.utils import make_regular_timeseries, nash_sut, mean_bias, rmsd
import matplotlib.pylab as plt
import datetime as dt
import matplotlib.dates as mdates

np.random.seed(1)# seed the same random numbers to make reproducable

model = 'clark'# 'eti' or 'clark'
n_runs = 2000
ensemble_id = 'test_randomD'
outfolder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2201/snow_model_ensembles/clark_output'
# ros = True
# ta_m_tt = False
# load input data for mueller hut
infile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/SIN_density_SIP/input_met/mueller_hut_met_20170501_20200401_with_hs_swe_rain_withcloud_precip_harder_update1.pkl'
aws_df = pkl.load(open(infile, 'rb'))
start_t = 1  # drop first timestamp as model uses initial value
end_t = None  # take the end point.
inp_dt = [i.to_pydatetime() for i in aws_df.index[start_t - 1:end_t]]
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
config['num_secs_output'] = 3600 # (s) overwrites default value of 1 day
# config['tacc'] = 273.16
# config['tmelt'] = 274.16

if model == 'clark':
    # clark2009 melt parameters
    # config['mf_mean'] = 5.0
    # config['mf_amp'] = 5.0
    # config['mf_alb'] = 2.5
    # config['mf_alb_decay'] = 1
    # config['mf_ros'] = 0  # default 2.5
    config['mf_doy_max_ddf'] = 356  # default 356
    # config['mf_doy_min_ddf'] = 210

    # just set range for now #TODO give option to set where is the range or mean and std deviation
    random_params = {
        'tacc': [272.16,278.16],
        'tmelt': [270.16, 279.16],
        'mf_mean': [1, 10],
        'mf_amp':[2.5,7.5],
        'mf_alb':[0,5],
        'mf_alb_decay':[1,10],
        'mf_ros':[0, 5],
        # 'mf_doy_max_ddf':[],
        'mf_doy_min_ddf':[180,240]
    }

elif model == 'eti':

    # dsc_snow melt parameters
    # config['tf'] = 0.05*24  # hamish 0.13
    # config['rf'] = 0.0108 * 24  # hamish 0.0075
    # albedo parameters
    config['dc'] = 11.0 # only affects albedo of snow over ice
    # config['tc'] = 10
    config['a_ice'] = 0.42 # not used by model for seasonal snow
    # config['a_freshsnow'] = 0.90
    # config['a_firn'] = 0.62
    config['alb_swe_thres'] = 20 # keep constant and equal to FSM2 value
    config['ros'] = False # include rain on snow melt
    config['ta_m_tt'] = False  # use tmelt as baseline when calculating degree days

    random_params = {
        'tacc': [272.16,278.16],
        'tmelt': [270.16, 279.16],
        'tf': [1, 10], # range the same as clark model. default somewhere around 1
        'rf':[0, 0.259], # range from 0 (temperature only model) to theoretical value 0.0108 * 24
        # albedo parameters
        'tc': [10,21], # between values in Conway (2016) and (Oerlemans and Knap 1998)
        'a_freshsnow':[0.8, 0.95],
        'a_firn':[0.5, 0.65]

    }

def set_random_params(random_params, config):
    """
    set selected parameters in config based on parameters in random params dictionary
    :param random_params:
    :param config:
    :return: config dictionary
    """

    for key in random_params.keys():
        config[key] = random_params[key][0] + np.random.random(1) * (random_params[key][1] - random_params[key][0])
    return config

# loop to call range of parameters
stor_dict = {}
for ii in range(n_runs):
    config = set_random_params(random_params, config)

    stor_dict[ii] = {}

    # call main function once hourly/sub-hourly temp and precip data available.
    if model == 'clark':

        st_swe, st_melt, st_acc, st_alb = snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep=3600, init_swe=init_swe,
                                                       init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='clark2009', **config)
    elif model == 'eti':
        st_swe, st_melt, st_acc, st_alb = snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep=3600, init_swe=init_swe,
                                                       init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='dsc_snow', **config)
    run_dict = {
        'timestamp': inp_dt,
        'swe': np.squeeze(st_swe),
        'melt': np.squeeze(st_melt),
        'acc': np.squeeze(st_acc),
        'alb': np.squeeze(st_alb)
    }
    run_df = pd.DataFrame.from_dict(run_dict)
    run_df.set_index('timestamp', inplace=True)
    stor_dict[ii]['states_output'] = run_df
    # stor_dict[ii]['inp_dt'] = inp_dt
    # stor_dict[ii]['st_swe'] = st_swe
    # stor_dict[ii]['st_melt'] = st_melt
    # stor_dict[ii]['st_acc'] = st_acc
    # stor_dict[ii]['st_alb'] = st_alb
    stor_dict[ii]['config'] = config.copy()  # need to copy to avoid being changed with config dictionary is updated

    ii += 1
    print(ii)
print()

pkl.dump(stor_dict, open(outfolder + '/collated_output_{}.pkl'.format(ensemble_id), 'wb'), -1)
