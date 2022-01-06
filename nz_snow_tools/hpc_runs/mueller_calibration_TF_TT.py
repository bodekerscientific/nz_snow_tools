"""
code to call the snow model for a simple test case using mueller hut data

modified to
"""

import numpy as np
import pickle as pkl
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
start_t = 1
end_t = -1
inp_dt = [i.to_pydatetime() for i in aws_df.index[start_t:end_t]]
inp_doy = [i.dayofyear for i in aws_df.index[start_t:end_t]]
inp_hourdec = [i.hour for i in aws_df.index[start_t:end_t]]

# make grids of input data
grid_size = 1
grid_id = np.arange(grid_size)
inp_ta = aws_df.tc[start_t:end_t][:, np.newaxis] * np.ones(grid_size) + 273.16  # 2 metre air temp in C
inp_precip = aws_df.precip[start_t:end_t][:, np.newaxis] * np.ones(grid_size)  # precip in mm
inp_sw = aws_df.srad[start_t:end_t][:, np.newaxis] * np.ones(grid_size)

# validation data
obs_swe = aws_df.swe[start_t:end_t] * 1000  # measured swe - convert to mm w.e.

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
tmelt_list = np.linspace(-5, 6, 23) + 273.16
tf_list = np.linspace(1, 10, 19)

# clark2009 melt parameters
config['mf_mean'] = 5.0
config['mf_amp'] = 5.0
config['mf_alb'] = 2.5
config['mf_alb_decay'] = 1
config['mf_ros'] = 0  # default 2.5
config['mf_doy_max_ddf'] = 356  # default 356
config['mf_doy_min_ddf'] = 210


# rf_list = np.linspace(0,100e-4,11)

ns_array = np.zeros((len(tf_list), len(tmelt_list)))
mbd_array = np.zeros((len(tf_list), len(tmelt_list)))
rmsd_array = np.zeros((len(tf_list), len(tmelt_list)))
h_ns_array = np.zeros((len(tf_list), len(tmelt_list)))
h_mbd_array = np.zeros((len(tf_list), len(tmelt_list)))
h_rmsd_array = np.zeros((len(tf_list), len(tmelt_list)))

# for tmelt in tmelt_list:
#     config['tmelt'] = tmelt
# loop to call range of parameters
for i, tf in enumerate(tf_list):
    for j, tt in enumerate(tmelt_list):
        config['mf_mean'] = tf
        config['tmelt'] = tt
        # call main function once hourly/sub-hourly temp and precip data available.
        st_swe, st_melt, st_acc, st_alb = snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep=3600, init_swe=init_swe,
                                                           init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='clark2009', **config)
        #TODO save output and config to dictionary similar to FSM2 collated output
        # stor_dict[outname]['states_output'] = hourly timeseries
        # stor_dict[outname]['namelist'] = nml

        # compute daily melt
        daily_swe3 = []
        obs_swe_daily = []
        for k in range(47, len(st_swe[:, 0]), 48):
            daily_swe3.append(st_swe[k, 0])
            obs_swe_daily.append(obs_swe[k])

        # compute validation metrics
        mb_sim = -1 * np.diff(np.asarray(daily_swe3))
        dSWE_daily_obs = -1 * np.diff(np.asarray(obs_swe_daily))
        ns_array[i, j] = nash_sut(mb_sim, dSWE_daily_obs)
        mbd_array[i, j] = mean_bias(mb_sim, dSWE_daily_obs)
        rmsd_array[i, j] = rmsd(mb_sim, dSWE_daily_obs)

        h_ns_array[i, j] = nash_sut(st_swe[1:, 0], obs_swe)
        h_mbd_array[i, j] = mean_bias(st_swe[1:, 0], obs_swe)
        h_rmsd_array[i, j] = rmsd(st_swe[1:, 0], obs_swe)

np.savetxt(outfolder + '/daily_sfc_ns_2010_TF_TT_ROS{}_ta_m_tt{}.txt'.format(config['ros'], config['ta_m_tt']), ns_array)
np.savetxt(outfolder + '/daily_sfc_mbd_2010_TF_TT_ROS{}_ta_m_tt{}.txt'.format(config['ros'], config['ta_m_tt']), mbd_array)
np.savetxt(outfolder + '/daily_sfc_rmsd_2010_TF_TT_ROS{}_ta_m_tt{}.txt'.format(config['ros'], config['ta_m_tt']), rmsd_array)
np.savetxt(outfolder + '/hourly_sfc_ns_2010_TF_TT_ROS{}_ta_m_tt{}.txt'.format(config['ros'], config['ta_m_tt']), h_ns_array)
np.savetxt(outfolder + '/hourly_sfc_mbd_2010_TF_TT_ROS{}_ta_m_tt{}.txt'.format(config['ros'], config['ta_m_tt']), h_mbd_array)
np.savetxt(outfolder + '/hourly_sfc_rmsd_2010_TF_TT_ROS{}_ta_m_tt{}.txt'.format(config['ros'], config['ta_m_tt']), h_rmsd_array)

plt.figure(figsize=[8, 12])

plt.subplot(4, 3, 1)
CS = plt.contour(tmelt_list, tf_list, ns_array, levels=np.linspace(-1, 1, 21), cmap=plt.cm.winter, vmax=1, vmin=-1)
plt.clabel(CS, inline=1, fontsize=10)
plt.ylabel('TF (mm w.e. per hour)'), plt.xlabel('T melt threshold (C)')
plt.title('NS daily 2010')

plt.subplot(4, 3, 2)
CS = plt.contour(tmelt_list, tf_list, mbd_array, levels=np.linspace(-20, 20, 21), cmap=plt.cm.copper, vmax=20, vmin=-20)
plt.clabel(CS, inline=1, fontsize=10)
plt.ylabel('TF (mm w.e. per hour)'), plt.xlabel('T melt threshold (C)')
plt.title('MBD daily 2010')

plt.subplot(4, 3, 3)
CS = plt.contour(tmelt_list, tf_list, rmsd_array, levels=np.linspace(10, 30, 21), cmap=plt.cm.winter_r, vmax=30, vmin=10)
plt.clabel(CS, inline=1, fontsize=10)
plt.ylabel('TF (mm w.e. per hour)'), plt.xlabel('T melt threshold (C)')
plt.title('RMSD daily 2010')

plt.tight_layout()
# plt.savefig(outfolder + '/daily_sfc_fit_metrics2010_TF_TT_ROS{}_ta_m_tt{}.png'.format(config['ros'],config['ta_m_tt']))

# plt.close()
# plt.figure(figsize=[8,3])

plt.subplot(4, 3, 7)
CS = plt.contour(tmelt_list, tf_list, h_ns_array, levels=np.linspace(-1, 1, 21), cmap=plt.cm.winter, vmax=1, vmin=-1)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('NS hourly 2010')
plt.ylabel('TF (mm w.e. per hour)'), plt.xlabel('T melt threshold (C)')
plt.tight_layout()

plt.subplot(4, 3, 8)
CS = plt.contour(tmelt_list, tf_list, h_mbd_array, levels=np.linspace(-0.4, 0.4, 17), cmap=plt.cm.copper, vmax=0.4, vmin=-0.4)
plt.clabel(CS, inline=1, fontsize=10)
plt.ylabel('TF (mm w.e. per hour)'), plt.xlabel('T melt threshold (C)')
plt.title('MBD hourly 2010')

plt.subplot(4, 3, 9)
CS = plt.contour(tmelt_list, tf_list, h_rmsd_array, levels=np.linspace(0.2, 1, 17), cmap=plt.cm.winter_r, vmax=1, vmin=0.2)
plt.clabel(CS, inline=1, fontsize=10)
plt.ylabel('TF (mm w.e. per hour)'), plt.xlabel('T melt threshold (C)')
plt.title('RMSD hourly 2010')
plt.tight_layout()
# plt.tight_layout()
# plt.savefig(outfolder + '/hourly_SEB_fit_metrics2010_TF_TT_ROS{}_ta_m_tt{}.png'.format(config['ros'],config['ta_m_tt']))
# plt.close()
