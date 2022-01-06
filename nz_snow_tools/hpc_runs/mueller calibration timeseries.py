"""
code to call the snow model for a simple test case using brewster glacier data
"""

import numpy as np
from nz_snow_tools.snow.clark2009_snow_model import snow_main_simple
from nz_snow_tools.util.utils import make_regular_timeseries
import matplotlib.pylab as plt
import datetime as dt
import matplotlib.dates as mdates
import pickle as pkl

# configuration dictionary containing model parameters.
config = {}
config['num_secs_output'] = 3600
config['tacc'] = 274.16
config['tmelt'] = 273.16

# clark2009 melt parameters
config['mf_mean'] = 5.0
config['mf_amp'] = 5.0
config['mf_alb'] = 2.5
config['mf_alb_decay'] = 1
config['mf_ros'] = 0  # default 2.5
config['mf_doy_max_ddf'] = 356  # default 356
config['mf_doy_min_ddf'] = 210

# dsc_snow melt parameters
config['tf'] = 0.05 * 24  # hamish 0.13. ruschle 0.04, pelliciotti 0.05
config['rf'] = 0.0108 * 24  # hamish 0.0075,ruschle 0.009, pelliciotti 0.0094
# dsc_snow albedo parameters
config['tc'] = 10
config['a_ice'] = 0.42
config['a_freshsnow'] = 0.90
config['a_firn'] = 0.62

config['dc'] = 11.0
config['alb_swe_thres'] = 10
config['ros'] = True
config['ta_m_tt'] = False

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

init_swe = np.ones(inp_ta.shape[1:]) * 0  # give initial value of swe
init_d_snow = np.ones(inp_ta.shape[1:]) * 30  # give initial value of days since snow

# call main function once hourly/sub-hourly temp and precip data available.
st_swe, st_melt, st_acc, st_alb = snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep=3600, init_swe=init_swe,
                                                   init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='clark2009', **config)

st_swe1, st_melt1, st_acc1, st_alb1 = snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep=3600, init_swe=init_swe,
                                                       init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='dsc_snow', **config)

# observed SWE
obs_swe = aws_df.swe[start_t:end_t] * 1000  # measured swe - convert to mm w.e.
plot_dt = inp_dt  # model stores initial state
plt.plot(plot_dt, st_swe[1:, 0], label='clark2009')
plt.plot(plot_dt, st_swe1[1:, 0], label='dsc_snow-param albedo')
plt.plot(plot_dt, obs_swe, label='obs')
# plt.xticks(range(0,len(st_swe[:, 0]),48*30),np.linspace(inp_doy[0],inp_doy[-1]+365+1,len(st_swe[:, 0])/(48*30.)+1,dtype=int))
plt.gcf().autofmt_xdate()
months = mdates.MonthLocator()  # every month
# days = mdates.DayLocator(interval=1)  # every day interval = 7 every week
monthsFmt = mdates.DateFormatter('%b')
ax = plt.gca()
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)

plt.xlabel('month')
plt.ylabel('SWE mm w.e.')
plt.legend()
plt.title('cumulative mass balance TF:{:2.3f}, RF: {:2.4f}, Tmelt:{:3.2f}'.format(config['tf'], config['rf'], config['tmelt']))

plt.savefig('C:/Users/conwayjp/OneDrive - NIWA/projects/SIN_density_SIP/tuning_dsc_snow/Mueller topnet TF{:2.3f}RF{:2.4f}Tmelt{:3.2f}_ros{}.png'.format(config['tf'],
                                                                                                                                                 config['rf'],
                                                                                                                                                 config[
                                                                                                                                                     'tmelt'],
                                                                                                                                                 config['ros']))
plt.show()
plt.close()
