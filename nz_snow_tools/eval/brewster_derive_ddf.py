import numpy as np
from nz_snow_tools.snow.clark2009_snow_model import snow_main_simple
from nz_snow_tools.util.utils import make_regular_timeseries
import matplotlib.pylab as plt
import datetime as dt
import matplotlib.dates as mdates
from nz_snow_tools.util.utils import blockfun

# set temperature threshold with which to calculate degree days
tmelt = 275.15
thres_melt = 1 # threshold (mm / day) to filter out days with small melt
thres_dd = 0.5 # threshold (K / day) to filter out days with small dd

# load brewster glacier data
inp_dat = np.genfromtxt(
    'C:/Users/conwayjp/OneDrive - NIWA/projects/SIN/BrewsterGlacier_Oct10_Sep12_mod3.dat')
start_t = 0# 9456 = start of doy 130 10th May 2011 9600 = end of 13th May,18432 = start of 11th Nov 2013,19296 = 1st december 2011
end_t = 32544  # 20783 = end of doy 365, 21264 = end of 10th January 2012, 21360 = end of 12th Jan
inp_dt = make_regular_timeseries(dt.datetime(2010,10,25,00,30),dt.datetime(2012,9,2,00,00),1800)

inp_doy = inp_dat[start_t:end_t, 2]
inp_hourdec = inp_dat[start_t:end_t, 3]

seb_dat = np.genfromtxt('C:/Users/conwayjp/OneDrive - NIWA/projects/SIN/modelOUT_br1_headings.txt',skip_header=3)
seb_mb = seb_dat[:,-1] # mass balance is last column
seb_melt = np.diff(seb_mb) * -1 # seb melt from change in mass balance
seb_melt[seb_melt<=0] = 0
seb_melt = np.hstack([np.asarray([0]),seb_melt])

seb_qm = seb_dat[:,24]
seb_surface_melt = seb_qm / 334000 * 1800 # seb melt at surface in mass units

inp_ta = inp_dat[start_t:end_t, 7] + 273.15

dd = inp_ta.squeeze() - tmelt
dd[dd<0] = 0 # limit to positive degree days

daily_melt = blockfun(seb_melt,48,method='sum')
daily_dd = blockfun(dd,48,method='mean')
daily_surface_melt = blockfun(seb_surface_melt,48,method='sum')
daily_doy = blockfun(inp_doy,48,method='mean')

precip = inp_dat[:,21]
rain = precip.copy()
rain[inp_ta[:]<274.16] = 0
daily_rain = blockfun(rain,48,method='sum')

ind = np.logical_and(daily_melt > thres_melt,daily_dd>thres_dd)
ind2 = np.logical_and(ind,daily_rain==0)
plt.figure()
plt.scatter(daily_doy[ind2],daily_melt[ind2]/daily_dd[ind2],c='k',label='dry days')
# plt.plot(daily_doy[ind],daily_surface_melt[ind]/daily_dd[ind],'ok')
plt.scatter(daily_doy[ind],daily_melt[ind]/daily_dd[ind],c=np.log(daily_rain[ind]),label='rain days')
plt.legend()
plt.ylabel('ddf (mm w.e. C^-1 day^-1)')
plt.xlabel('day of year')
plt.title('ddf derived from Brewster SEB/SMB model - melt thres={}K'.format(tmelt))
plt.savefig(r'C:\Users\conwayjp\OneDrive - NIWA\projects\CARH2201\snow_model_ensembles\deriving ddf from SEB output\brewster_ddf_tmelt{}.png'.format(tmelt))