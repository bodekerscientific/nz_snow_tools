"""
code to call the snow model for a simple test case using brewster glacier data
"""

import numpy as np
from nz_snow_tools.snow.clark2009_snow_model import snow_main_simple
import matplotlib.pylab as plt



# configuration dictionary containing model parameters.
config = {}
config['tacc'] = 274.16
config['tmelt'] = 273.16

# clark2009 melt parameters
config['mf_mean'] = 5.0
config['mf_amp'] = 5.0
config['mf_alb'] = 2.5
config['mf_alb_decay'] = 5.0
config['mf_ros'] = 0
config['mf_doy_max_ddf'] = 356

# dsc_snow melt parameters
config['tf'] = 0.13*24  # hamish 0.13
config['rf'] = 0.0075*24 # hamish 0.0075
# albedo parameters
config['dc'] = 11.0
config['tc'] = 10.0
config['a_ice'] = 0.42
config['a_freshsnow'] = 0.95
config['a_firn'] = 0.65

# load brewster glacier data
inp_dat = np.genfromtxt(
    'S:\Scratch\Jono\Final Brewster Datasets\updated_met_data\BrewsterGlacier_Oct10_Sep12_mod3.dat')
inp_doy = inp_dat[:, 2]
inp_hourdec = inp_dat[:, 3]
# make grids of input data
grid_size = 1
grid_id = np.arange(grid_size)
inp_ta = inp_dat[:, 7][:, np.newaxis] * np.ones(grid_size) + 273.16  # 2 metre air temp in C
inp_precip = inp_dat[:, 21][:, np.newaxis] * np.ones(grid_size)  # precip in mm
inp_sw = inp_dat[:, 15][:, np.newaxis] * np.ones(grid_size)

config['inp_alb'] = inp_dat[:, 15][:, np.newaxis] * np.ones(grid_size) # TODO check column number

init_swe = np.ones(inp_ta.shape[1:]) * 10000  # give initial value of swe as starts in spring
init_d_snow = np.ones(inp_ta.shape[1:]) * 10  # give initial value of days since snow

# call main function once hourly/sub-hourly temp and precip data available.
st_swe, st_melt, st_acc = snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep=1800, init_swe=init_swe,
                                           init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='dsc_snow', **config)

st_swe1, st_melt1, st_acc1 = snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep=1800, init_swe=init_swe,
                                           init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='clark2009', **config)


# additional tests with set values for ta and precip
# 0 degrees and 1 mm per 30-mins precip
st_swe1, st_melt1, st_acc1 = snow_main_simple(inp_ta*0 + 273.16 , inp_precip*0 + 1, inp_doy, inp_hourdec, dtstep=1800, init_swe=init_swe,
                                           init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='dsc_snow', **config)
# 0.5 degree with 1 mm per 30-mins precip
st_swe2, st_melt2, st_acc2 = snow_main_simple(inp_ta*0 + 273.66 , inp_precip*0 + 1, inp_doy, inp_hourdec, dtstep=1800, init_swe=init_swe,
                                           init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='dsc_snow', **config)
# 1 degree with 1 mm per 30-mins precip
st_swe3, st_melt3, st_acc3 = snow_main_simple(inp_ta*0 + 274.16 , inp_precip*0 + 1, inp_doy, inp_hourdec, dtstep=1800, init_swe=init_swe,
                                           init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='dsc_snow', **config)
# 2 degrees with 1 mm per 30-mins precip
st_swe4, st_melt4, st_acc4 = snow_main_simple(inp_ta*0 + 275.16 , inp_precip*0 + 1, inp_doy, inp_hourdec, dtstep=1800, init_swe=init_swe,
                                           init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='dsc_snow', **config)
# 2 degrees without 1 mm per 30-mins precip
st_swe5, st_melt5, st_acc5 = snow_main_simple(inp_ta*0 + 275.16 , inp_precip*0, inp_doy, inp_hourdec, dtstep=1800, init_swe=init_swe,
                                           init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='dsc_snow', **config)
# 1 degree without 1 mm per 30-mins precip
st_swe6, st_melt6, st_acc6 = snow_main_simple(inp_ta*0 + 274.16 , inp_precip*0, inp_doy, inp_hourdec, dtstep=1800, init_swe=init_swe,
                                           init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='dsc_snow', **config)

print('done')


# # create fake input data
grid_size = 1
inp_ta = np.zeros((365 * 24, grid_size)) + 273.16
inp_precip = np.zeros((365 * 24, grid_size))
inp_doy = np.linspace(0, 365, 365 * 24 + 1)
st_swe1 = snow_main_simple(inp_ta, inp_precip + 1, inp_doy)  # 0 degrees with precip
st_swe2 = snow_main_simple(inp_ta + 0.5, inp_precip + 1, inp_doy)  # 0.5 degree with
st_swe3 = snow_main_simple(inp_ta + 1, inp_precip + 1, inp_doy)  # 1 degree with rain
st_swe4 = snow_main_simple(inp_ta + 2, inp_precip + 1, inp_doy)  # 2 degrees with rain
st_swe5 = snow_main_simple(inp_ta + 2, inp_precip, inp_doy)  # 2 degrees without rain
st_swe6 = snow_main_simple(inp_ta + 1, inp_precip, inp_doy)  # 1 degree without rain


plt.plot(st_swe1[:, 0])
plt.plot(st_swe2[:, 0])
plt.plot(st_swe3[:, 0])
plt.plot(st_swe4[:, 0])
plt.plot(st_swe5[:, 0])
plt.plot(st_swe6[:, 0])
plt.legend(range(3, 7))
