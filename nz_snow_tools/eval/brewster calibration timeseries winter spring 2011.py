"""
code to call the snow model for a simple test case using brewster glacier data
"""

import numpy as np
from nz_snow_tools.snow.clark2009_snow_model import snow_main_simple
from nz_snow_tools.util.utils import make_regular_timeseries
import matplotlib.pylab as plt
import datetime as dt
import matplotlib.dates as mdates

# configuration dictionary containing model parameters.
config = {}
config['num_secs_output']=1800
config['tacc'] = 274.16
config['tmelt'] = 276.16

# clark2009 melt parameters
config['mf_mean'] = 5.0
config['mf_amp'] = 5.0
config['mf_alb'] = 2.5
config['mf_alb_decay'] = 5.0
config['mf_ros'] = 2.5 # default 2.5
config['mf_doy_max_ddf'] = 356 # default 356
config['mf_doy_min_ddf'] = 210 # default 210

# dsc_snow melt parameters
config['tf'] = 0.05*24  # hamish 0.13. ruschle 0.04, pelliciotti 0.05
config['rf'] = 0.0108*24 # hamish 0.0075,ruschle 0.009, pelliciotti 0.0094

# albedo parameters
config['dc'] = 11.0
config['tc'] = 10
config['a_ice'] = 0.42
config['a_freshsnow'] = 0.90
config['a_firn'] = 0.62
config['alb_swe_thres'] = 10
config['ros'] = True
config['ta_m_tt'] = False

# load brewster glacier data
inp_dat = np.genfromtxt(
    'C:/Users/conwayjp/OneDrive - NIWA/projects/SIN/BrewsterGlacier_Oct10_Sep12_mod3.dat')
start_t = 9600 -1# 9456 = start of doy 130 10th May 2011 9600 = end of 13th May,18432 = start of 11th Nov 2013,19296 = 1st december 2011
end_t = 21360  # 20783 = end of doy 365, 21264 = end of 10th January 2012, 21360 = end of 12th Jan
inp_dt = make_regular_timeseries(dt.datetime(2010,10,25,00,30),dt.datetime(2012,9,2,00,00),1800)

inp_doy = inp_dat[start_t:end_t, 2]
inp_hourdec = inp_dat[start_t:end_t, 3]
# make grids of input data
grid_size = 1
grid_id = np.arange(grid_size)
inp_ta = inp_dat[start_t:end_t, 7][:, np.newaxis] * np.ones(grid_size) + 273.16  # 2 metre air temp in C
inp_precip = inp_dat[start_t:end_t, 21][:, np.newaxis] * np.ones(grid_size)  # precip in mm
inp_sw = inp_dat[start_t:end_t, 15][:, np.newaxis] * np.ones(grid_size)
inp_sfc = inp_dat[start_t-1:end_t, 19] # surface height change
inp_sfc -= inp_sfc[0]# reset to 0 at beginning of period

# validation data
seb_dat = np.genfromtxt(
    'C:/Users/conwayjp/OneDrive - NIWA/projects/SIN/modelOUT_br1_headings.txt',skip_header=3)
seb_mb = seb_dat[start_t-1:end_t, -1]
seb_mb -= seb_mb[0] # reset to 0

# read in measured daily SEB change
mb_dat = np.genfromtxt(
    'C:/Users/conwayjp/OneDrive - NIWA/projects/SIN/mchange.dat')
# note that the measured MB interprets surface height loss in the winter as mass loss, rather than compaction.
mb_dt = make_regular_timeseries(dt.datetime(2010,10,26,00,00),dt.datetime(2012,9,2,00,00),86400)
ts_mb = plt.cumsum(mb_dat[:,0])
np.where(np.asarray(mb_dt)==dt.datetime(2011,5,13,00,00))
ts_mb -= ts_mb[199]
#

init_swe = np.ones(inp_ta.shape[1:]) * 0  # give initial value of swe as starts in spring
init_d_snow = np.ones(inp_ta.shape[1:]) * 30  # give initial value of days since snow

# call main function once hourly/sub-hourly temp and precip data available.
st_swe, st_melt, st_acc, st_alb = snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep=1800, init_swe=init_swe,
                                           init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='clark2009', **config)

st_swe1, st_melt1, st_acc1, st_alb1 = snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep=1800, init_swe=init_swe,
                                           init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='dsc_snow', **config)

config['inp_alb'] = inp_dat[start_t:end_t, 16][:, np.newaxis] * np.ones(grid_size)
st_swe3, st_melt3, st_acc3, st_alb3 = snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep=1800, init_swe=init_swe,
                                           init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='dsc_snow', **config)

plot_dt = inp_dt[start_t-1:end_t] # model stores initial state
# plt.plot(plot_dt,st_swe[:, 0],label='clark2009')
plt.plot(plot_dt,st_swe1[:, 0],label='dsc_snow-param albedo')
plt.plot(plot_dt,st_swe3[:, 0],label='dsc_snow-obs albedo')
plt.plot(plot_dt,seb_mb, label='SEB')
plt.plot(plot_dt,inp_sfc*492,label='sfc*492')
plt.plot([dt.datetime(2011,7,18),dt.datetime(2011,10,27),dt.datetime(2011,11,13)],[577,1448,1291],'o',label='stake_mb') # measured accumualation to 27th November 2011
#plt.xticks(range(0,len(st_swe[:, 0]),48*30),np.linspace(inp_doy[0],inp_doy[-1]+365+1,len(st_swe[:, 0])/(48*30.)+1,dtype=int))
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
plt.title('cumulative mass balance TF:{}, RF: {}, Tmelt:{}'.format(config['tf'],config['rf'],config['tmelt']))
plt.savefig('C:/Users/conwayjp/OneDrive - NIWA/projects/SIN/Winter Spring 2011 TF{}RF{}Tmelt{}_ros.png'.format(config['tf'],config['rf'],config['tmelt']))
plt.close()

