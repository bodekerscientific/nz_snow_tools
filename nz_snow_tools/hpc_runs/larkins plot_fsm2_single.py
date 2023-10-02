"""
code to plot output from FSM2
"""

import numpy as np
import matplotlib.pylab as plt
import datetime as dt
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



plot_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2201/snow_model_ensembles/fsm2_output'
data_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2201/snow_model_ensembles/fsm2_output'
output_filename = 'Lark_coldstartstat.txt'
run_id = 'lark fsm2 default'

# load input data for mueller hut
infile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/FWWR_SIN/data processing/precip_larkins/larkins_met_20170501_20191204_with_hs_swe_rain_precip_harder_cloud_lw.csv'
aws_df = pd.read_csv(infile, parse_dates=True, index_col='Timestamp')
start_t = 0
end_t = None
inp_dt = [i.to_pydatetime() for i in aws_df.index[start_t:end_t]]

# read fsm2 output
headers = ['year', 'month', 'day', 'dechour', 'snowdepth', 'swe', 'sveg', 'tsoil', 'tsrf', 'tveg']
fsm_df = pd.read_fwf(plot_folder + '/' + output_filename, names=headers, widths=[4, 4, 4, 8, 14, 14, 14, 14, 14, 14])
hours, minutes, seconds = convert_decimal_hours_to_hours_min_secs(fsm_df.dechour.values)
timestamp = [dt.datetime(y, m, d, h) for y, m, d, h in zip(fsm_df.year.values, fsm_df.month.values, fsm_df.day.values, hours)]
fsm_df.index = timestamp

# observed SWE
obs_swe = aws_df.swe[start_t:end_t] * 1000  # measured swe - convert to mm w.e.
plot_dt = inp_dt  # model stores initial state

plt.figure(figsize=(12, 3))
plt.plot(plot_dt, fsm_df.swe, label='mod')
# plt.plot(plot_dt, st_swe1[1:, 0], label='dsc_snow-param albedo')
plt.plot(plot_dt, obs_swe, label='obs')
# plt.xticks(range(0,len(st_swe[:, 0]),48*30),np.linspace(inp_doy[0],inp_doy[-1]+365+1,len(st_swe[:, 0])/(48*30.)+1,dtype=int))
plt.gcf().autofmt_xdate()
# months = mdates.MonthLocator(interval=3)  # every month
# # days = mdates.DayLocator(interval=1)  # every day interval = 7 every week
# monthsFmt = mdates.DateFormatter('%b')
# ax = plt.gca()
# ax.xaxis.set_major_locator(months)
# ax.xaxis.set_major_formatter(monthsFmt)

plt.xlabel('Month')
plt.ylabel('SWE (mm w.e.)')
plt.legend()
plt.tight_layout()
plt.savefig(plot_folder + '/SWE long {}.png'.format(run_id), dpi=300)
# plt.title('cumulative mass balance TF:{:2.3f}, RF: {:2.4f}, Tmelt:{:3.2f}'.format(config['tf'], config['rf'], config['tmelt']))

obs_hs = aws_df.hs[start_t:end_t]
plt.figure(figsize=(12, 3))
plt.plot(plot_dt, fsm_df.snowdepth, label='mod')
# plt.plot(plot_dt, st_swe1[1:, 0], label='dsc_snow-param albedo')
plt.plot(plot_dt, obs_hs, label='obs')
# plt.xticks(range(0,len(st_swe[:, 0]),48*30),np.linspace(inp_doy[0],inp_doy[-1]+365+1,len(st_swe[:, 0])/(48*30.)+1,dtype=int))
plt.gcf().autofmt_xdate()
# months = mdates.MonthLocator(interval=3)  # every month
# # days = mdates.DayLocator(interval=1)  # every day interval = 7 every week
# monthsFmt = mdates.DateFormatter('%b')
# ax = plt.gca()
# ax.xaxis.set_major_locator(months)
# ax.xaxis.set_major_formatter(monthsFmt)

plt.xlabel('Month')
plt.ylabel('HS (m)')
plt.legend()
plt.tight_layout()
plt.savefig(plot_folder + '/HS long {}.png'.format(run_id), dpi=300)

# plt.savefig('C:/Users/conwayjp/OneDrive - NIWA/projects/SIN_density_SIP/tuning_dsc_snow/Mueller clark210 TF{:2.3f}RF{:2.4f}Tmelt{:3.2f}_ros{}.png'.format(config['tf'],
#                                                                                                                                                  config['rf'],
#                                                                                                                                                  config[
#                                                                                                                                                      'tmelt'],
#                                                                                                                                                  config['ros']))
plt.show()
plt.close()

plt.plot(aws_df.sw_cs_ghi.index, aws_df.sw_cs_ghi.values)
plt.plot(aws_df.swin.index, aws_df.swin.values)
plt.gcf().autofmt_xdate()

fsm_inp = pd.read_csv('C:/Users/conwayjp/OneDrive - NIWA/projects/FWWR_SIN/data processing/precip_larkins/larkins_met_20170501_20191204_FSM2.txt', delimiter='\t',
                      names=['year', 'month', 'day', 'hour', 'swin', 'lwmod', 'snowfall_rate_harder', 'rainfall_rate_harder', 'tk', 'rh', 'ws', 'pressure'])
timestamp = [dt.datetime(y, m, d, h) for y, m, d, h in zip(fsm_inp.year.values, fsm_inp.month.values, fsm_inp.day.values,fsm_inp.hour)]
fsm_inp.index = timestamp

fsm_inp[['swin', 'lwmod','snowfall_rate_harder', 'rainfall_rate_harder','tk', 'rh', 'ws', 'pressure']].plot(subplots=True)


# print(fsm_df.swe.mean())
# print(np.sum(fsm_df.swe > 10)/24/3)
#
# # read in second +1K file
# # read fsm2 output
# output_filename1 = 'Muel_default_plus1Kstat.txt'
#
# headers = ['year','month','day','dechour','snowdepth','swe','sveg','tsoil','tsrf','tveg']
# fsm_df1 = pd.read_fwf(plot_folder+'/'+output_filename1,names = headers, widths=[4,4,4,8,14,14,14,14,14,14])
# hours, minutes, seconds = convert_decimal_hours_to_hours_min_secs(fsm_df1.dechour.values)
# timestamp = [dt.datetime(y,m,d,h) for y,m,d,h in zip(fsm_df1.year.values,fsm_df1.month.values,fsm_df1.day.values,hours)]
# fsm_df1.index = timestamp
#
# plt.figure(figsize=(4,3))
# plt.plot(plot_dt, fsm_df.swe, label='current')
# # plt.plot(plot_dt, st_swe1[1:, 0], label='dsc_snow-param albedo')
# plt.plot(plot_dt, fsm_df1.swe, label='+1K')
# # plt.xticks(range(0,len(st_swe[:, 0]),48*30),np.linspace(inp_doy[0],inp_doy[-1]+365+1,len(st_swe[:, 0])/(48*30.)+1,dtype=int))
# plt.gcf().autofmt_xdate()
# months = mdates.MonthLocator(interval=3)  # every month
# # days = mdates.DayLocator(interval=1)  # every day interval = 7 every week
# monthsFmt = mdates.DateFormatter('%b')
# ax = plt.gca()
# ax.xaxis.set_major_locator(months)
# ax.xaxis.set_major_formatter(monthsFmt)
#
# plt.xlabel('Month')
# plt.ylabel('SWE (mm w.e.)')
# plt.legend()
# plt.tight_layout()
# plt.savefig(plot_folder + '/SWE +1K {}.png'.format(run_id),dpi=300)
# # plt.title('cumulative mass balance TF:{:2.3f}, RF: {:2.4f}, Tmelt:{:3.2f}'.format(config['tf'], config['rf'], config['tmelt']))
#
