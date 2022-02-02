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
import pandas as pd


def convert_decimal_hours_to_hours_min_secs(dec_hour):
    """
    convert array or list of decimal hours to arrays of datetime components
    :param dec_hour:
    :return:
    """
    hours = np.asarray([int(h) for h in dec_hour])
    minutes = np.asarray([int((h * 60) % 60) for h in dec_hour])
    seconds = np.asarray([((((h * 60) % 60) * 60) % 60) for h in dec_hour])

    return hours, minutes, seconds


def arrays_to_datetimes(years, months, days, hours):
    """

    converts lists or arrays of time components into timestamp
    """
    import datetime
    timestamp = [datetime.datetime(y, m, d, h) for y, m, d, h in zip(years, months, days, hours)]

    return np.asarray(timestamp)



plot_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2201/snow_model_ensembles/fsm2_output'
data_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2201/snow_model_ensembles/fsm2_output'
output_filename = 'Muel_defaultstat.txt'
run_id = 'fsm2 default'

# load input data for mueller hut
infile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/SIN_density_SIP/input_met/mueller_hut_met_20170501_20200401_with_hs_swe_rain_withcloud_precip_harder_update1.pkl'
aws_df = pkl.load(open(infile, 'rb'))
start_t = 0
end_t = None
inp_dt = [i.to_pydatetime() for i in aws_df.index[start_t:end_t]]

# read fsm2 output
headers = ['year','month','day','dechour','snowdepth','swe','sveg','tsoil','tsrf','tveg']
fsm_df = pd.read_fwf(plot_folder+'/'+output_filename,names = headers, widths=[4,4,4,8,14,14,14,14,14,14])
hours, minutes, seconds = convert_decimal_hours_to_hours_min_secs(fsm_df.dechour.values)
timestamp = [dt.datetime(y,m,d,h) for y,m,d,h in zip(fsm_df.year.values,fsm_df.month.values,fsm_df.day.values,hours)]
fsm_df.index = timestamp

# observed SWE
obs_swe = aws_df.swe[start_t:end_t] * 1000  # measured swe - convert to mm w.e.
plot_dt = inp_dt  # model stores initial state

plt.figure(figsize=(4,3))
plt.plot(plot_dt, fsm_df.swe, label='mod')
# plt.plot(plot_dt, st_swe1[1:, 0], label='dsc_snow-param albedo')
plt.plot(plot_dt, obs_swe, label='obs')
# plt.xticks(range(0,len(st_swe[:, 0]),48*30),np.linspace(inp_doy[0],inp_doy[-1]+365+1,len(st_swe[:, 0])/(48*30.)+1,dtype=int))
plt.gcf().autofmt_xdate()
months = mdates.MonthLocator(interval=3)  # every month
# days = mdates.DayLocator(interval=1)  # every day interval = 7 every week
monthsFmt = mdates.DateFormatter('%b')
ax = plt.gca()
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)

plt.xlabel('Month')
plt.ylabel('SWE (mm w.e.)')
plt.legend()
plt.tight_layout()
plt.savefig(plot_folder + '/SWE {}.png'.format(run_id),dpi=300)
# plt.title('cumulative mass balance TF:{:2.3f}, RF: {:2.4f}, Tmelt:{:3.2f}'.format(config['tf'], config['rf'], config['tmelt']))


plt.figure(figsize=(4,3))
plt.plot(plot_dt, fsm_df.snowdepth, label='mod')
# plt.plot(plot_dt, st_swe1[1:, 0], label='dsc_snow-param albedo')
plt.plot(plot_dt, aws_df.hs, label='obs')
# plt.xticks(range(0,len(st_swe[:, 0]),48*30),np.linspace(inp_doy[0],inp_doy[-1]+365+1,len(st_swe[:, 0])/(48*30.)+1,dtype=int))
plt.gcf().autofmt_xdate()
months = mdates.MonthLocator(interval=3)  # every month
# days = mdates.DayLocator(interval=1)  # every day interval = 7 every week
monthsFmt = mdates.DateFormatter('%b')
ax = plt.gca()
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)

plt.xlabel('Month')
plt.ylabel('HS (m)')
plt.legend()
plt.tight_layout()
plt.savefig(plot_folder + '/HS {}.png'.format(run_id),dpi=300)

# plt.savefig('C:/Users/conwayjp/OneDrive - NIWA/projects/SIN_density_SIP/tuning_dsc_snow/Mueller clark210 TF{:2.3f}RF{:2.4f}Tmelt{:3.2f}_ros{}.png'.format(config['tf'],
#                                                                                                                                                  config['rf'],
#                                                                                                                                                  config[
#                                                                                                                                                      'tmelt'],
#                                                                                                                                                  config['ros']))
plt.show()
plt.close()


print(fsm_df.swe.mean())
print(np.sum(fsm_df.swe > 10)/24/3)

# read in second +1K file
# read fsm2 output
output_filename1 = 'Muel_default_plus1Kstat.txt'

headers = ['year','month','day','dechour','snowdepth','swe','sveg','tsoil','tsrf','tveg']
fsm_df1 = pd.read_fwf(plot_folder+'/'+output_filename1,names = headers, widths=[4,4,4,8,14,14,14,14,14,14])
hours, minutes, seconds = convert_decimal_hours_to_hours_min_secs(fsm_df1.dechour.values)
timestamp = [dt.datetime(y,m,d,h) for y,m,d,h in zip(fsm_df1.year.values,fsm_df1.month.values,fsm_df1.day.values,hours)]
fsm_df1.index = timestamp

plt.figure(figsize=(4,3))
plt.plot(plot_dt, fsm_df.swe, label='current')
# plt.plot(plot_dt, st_swe1[1:, 0], label='dsc_snow-param albedo')
plt.plot(plot_dt, fsm_df1.swe, label='+1K')
# plt.xticks(range(0,len(st_swe[:, 0]),48*30),np.linspace(inp_doy[0],inp_doy[-1]+365+1,len(st_swe[:, 0])/(48*30.)+1,dtype=int))
plt.gcf().autofmt_xdate()
months = mdates.MonthLocator(interval=3)  # every month
# days = mdates.DayLocator(interval=1)  # every day interval = 7 every week
monthsFmt = mdates.DateFormatter('%b')
ax = plt.gca()
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)

plt.xlabel('Month')
plt.ylabel('SWE (mm w.e.)')
plt.legend()
plt.tight_layout()
plt.savefig(plot_folder + '/SWE +1K {}.png'.format(run_id),dpi=300)
# plt.title('cumulative mass balance TF:{:2.3f}, RF: {:2.4f}, Tmelt:{:3.2f}'.format(config['tf'], config['rf'], config['tmelt']))

