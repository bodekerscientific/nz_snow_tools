"""
code to call the snow model for a simple test case using brewster glacier data
"""

import numpy as np
from nz_snow_tools.snow.clark2009_snow_model import snow_main_simple
from nz_snow_tools.util.utils import make_regular_timeseries,convert_datetime_julian_day,convert_dt_to_hourdec,nash_sut, mean_bias, rmsd, mean_absolute_error
import matplotlib.pylab as plt
import datetime as dt
import matplotlib.dates as mdates

# configuration dictionary containing model parameters.
config = {}
config['num_secs_output']=3600
config['tacc'] = 274.16
config['tmelt'] = 274.16

# clark2009 melt parameters
config['mf_mean'] = 5.0
config['mf_amp'] = 5.0
config['mf_alb'] = 2.5
config['mf_alb_decay'] = 5.0
config['mf_ros'] = 2.5 # default 2.5
config['mf_doy_max_ddf'] = 356 # default 356

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

# load data
inp_dat = np.load("C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Mueller/Mueller_2016.npy",allow_pickle=True)


inp_doy = np.asarray(convert_datetime_julian_day(inp_dat[:, 0]))
inp_hourdec = convert_dt_to_hourdec(inp_dat[:, 0])
plot_dt = inp_dat[:, 0] # model stores initial state
# make grids of input data
grid_size = 1
grid_id = np.arange(grid_size)
inp_ta = np.asarray(inp_dat[:,2],dtype=np.float)[:, np.newaxis] * np.ones(grid_size) + 273.16  # 2 metre air temp in C
inp_precip = np.asarray(inp_dat[:,4],dtype=np.float)[:, np.newaxis] * np.ones(grid_size)  # precip in mm
inp_sw = np.asarray(inp_dat[:,3],dtype=np.float)[:, np.newaxis] * np.ones(grid_size)


init_swe = np.ones(inp_ta.shape[1:],dtype=np.float) * 0  # give initial value of swe as starts in spring
init_d_snow = np.ones(inp_ta.shape[1:],dtype=np.float) * 30  # give initial value of days since snow

# call main function once hourly/sub-hourly temp and precip data available.
st_swe, st_melt, st_acc, st_alb = snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep=3600,init_swe=init_swe,
                                           init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='clark2009', **config)

st_swe1, st_melt1, st_acc1, st_alb1 = snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep=3600, init_swe=init_swe,
                                           init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='dsc_snow', **config)
# load observed data
inp_datobs = np.genfromtxt("C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Mueller SWE.csv", delimiter=',',usecols=(1),
                        skip_header=4)*1000
inp_timeobs = np.genfromtxt("C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Mueller SWE.csv", usecols=(0),
                         dtype=(str), delimiter=',', skip_header=4)
inp_dtobs = np.asarray([dt.datetime.strptime(t, '%d/%m/%Y %H:%M') for t in inp_timeobs])

ind = np.logical_and(inp_dtobs >= plot_dt[0],inp_dtobs <= plot_dt[-1])

# print('rmsd = {}'.format(rmsd(mod,obs)))
plt.plot(inp_dtobs[ind],inp_datobs[ind],"o", label = "Observed SWE")

plt.plot(plot_dt,st_swe[1:, 0],label='clark2009')
plt.plot(plot_dt,st_swe1[1:, 0],label='dsc_snow-param albedo')
plt.legend()
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(plot_dt,np.cumsum(inp_precip), label = "Precipitation")
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
plt.legend(loc = 'upper right')
plt.title('cumulative mass balance TF:{:2.4f}, RF: {:2.4f}, Tmelt:{:3.2f}, year : 2016'.format(config['tf'],config['rf'],config['tmelt']))
plt.savefig('C:/Users/Bonnamourar/OneDrive - NIWA/for Ambre/SIN calibration daily TF{:2.4f}RF{:2.4f}Tmelt{:3.2f}_ros.png'.format(config['tf'],config['rf'],config['tmelt']))
plt.show()
plt.close()

