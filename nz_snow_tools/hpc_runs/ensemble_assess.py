# run through ensemble output, compute statistics and visualise

import pandas as pd
import numpy as np
import pickle as pkl
from nz_snow_tools.util.utils import make_regular_timeseries, nash_sut, mean_bias, rmsd, mean_absolute_error
import matplotlib.pylab as plt
import datetime as dt
import matplotlib.dates as mdates


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


ensemble_id = 'test_randomC'
model = 'eti'#'clark'  # 'eti' 'fsm2'
hy = '2017-18'
# hy = '2019-20'

if model == 'clark':
    outfolder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2201/snow_model_ensembles/clark_output'
    dict_in = pkl.load(
        open('C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2201/snow_model_ensembles/clark_output/collated_output_{}.pkl'.format(ensemble_id), 'rb'))

if model == 'eti':
    outfolder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2201/snow_model_ensembles/ETI_output'
    dict_in = pkl.load(
        open('C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2201/snow_model_ensembles/ETI_output/collated_output_{}.pkl'.format(ensemble_id), 'rb'))

elif model == 'fsm2':
    outfolder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2201/snow_model_ensembles/FSM2_output'

    dict_in = pkl.load(
        open('C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2201/snow_model_ensembles/FSM2_output/collated_output_{}.pkl'.format(ensemble_id), 'rb'))

# dict_in = pkl.load(open('C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2201/snow_model_ensembles/clark_output/clark_test_output.pkl', 'rb'))

obs_file = 'C:/Users/conwayjp/OneDrive - NIWA/projects/SIN_density_SIP/input_met/mueller_hut_met_20170501_20200401_with_hs_swe_rain_withcloud_precip_harder.pkl'
dfin = pkl.load(open(obs_file, 'rb'))
dfin.insert(2, 'Meas_SWE', 1000 * dfin['swe'].values)
dfin.insert(3, 'Meas_snowdepth', dfin['hs'].values)
#
# obs_file = r"C:/Users/conwayjp/OneDrive - NIWA/projects/SIN_density_SIP/FSM2-master/FSM2-master/mueller_in_hs_swe_measured.csv"
# dfin = pd.read_csv(obs_file, names=["Meas_year", "Meas_month", "Meas_day", "Meas_hour", "Meas_snowdepth", "Meas_SWE"])
# hours, minutes, seconds = convert_decimal_hours_to_hours_min_secs(dfin.Meas_hour.values)
# timestamp = [dt.datetime(y, m, d, h) for y, m, d, h in zip(dfin.Meas_year.values, dfin.Meas_month.values, dfin.Meas_day.values, hours)]
# dfin.index = timestamp
# dfin["Meas_SWE"] = 1000 * dfin["Meas_SWE"]

# met_file = r"C:/Users/conwayjp/OneDrive - NIWA/projects/SIN_density_SIP/FSM2-master/FSM2-master/mueller_hut_met_20170501_20200101_withcloud_precip_harder.csv"
# met_in = pd.read_csv(met_file, index_col=0, parse_dates=True)

swe_threshold = 250  # mm
snow_depth_threshold = 0.5  # in metres
density_threshold = 800
dfin.insert(6, "Meas_density", 40)
dfin["Meas_density"] = dfin["Meas_SWE"] / dfin["Meas_snowdepth"]
dfin['Meas_density'] = np.where(dfin.Meas_snowdepth <= snow_depth_threshold, np.nan, dfin.Meas_density)
dfin['Meas_density'] = np.where(dfin.Meas_SWE <= swe_threshold, np.nan, dfin.Meas_density)
dfin['Meas_density'] = np.where((dfin.Meas_density > density_threshold) | (dfin.Meas_density < 0), np.nan, dfin.Meas_density)

n_runs = len(dict_in.keys())
stats_store = {}  # dictionary to store stats
stats_store['run_id'] = np.full(n_runs, 'missing', dtype=object)
stats_store['ns'] = np.full(n_runs, np.nan)
stats_store['bias'] = np.full(n_runs, np.nan)
stats_store['rmsd'] = np.full(n_runs, np.nan)
stats_store['mae'] = np.full(n_runs, np.nan)

# construct store of model parameters
param_store = {}
if model == 'fsm2':
    for key in dict_in[list(dict_in.keys())[0]]['namelist']['params'].keys():
        param_store[key] = np.full(n_runs, np.nan)
    # param_store['exe'] = np.full(n_runs, 'missing', dtype=object)
    param_store['density'] = np.full(n_runs, np.nan)
    param_store['exchng'] = np.full(n_runs, np.nan)
    param_store['hydrol'] = np.full(n_runs, np.nan)
elif model == 'clark' or model == 'eti':
    for key in dict_in[list(dict_in.keys())[0]]['config'].keys():
        param_store[key] = np.full(n_runs, np.nan)


if hy == '2017-18':
    obs_swe = dfin["Meas_SWE"].copy().truncate(after='2017-12-31 23:00:00')
    # ["Meas_snowdepth"]
if hy == '2019-20':
    obs_swe = dfin["Meas_SWE"].copy().truncate(before='2019-05-01 00:00:00', after='2019-11-12 00:00:00')

obs_ind = ~np.isnan(obs_swe.values)
obs_swe = obs_swe[~np.isnan(obs_swe)]
for i, run_id in enumerate(dict_in.keys()):
    if hy == '2017-18':
        sim_swe = dict_in[run_id]['states_output'].swe.copy().truncate(after='2017-12-31 23:00:00')
    if hy == '2019-20':
        sim_swe = dict_in[run_id]['states_output'].swe.copy().truncate(before='2019-05-01 00:00:00', after='2019-11-12 00:00:00')
    # assert np.all(sim_swe.index == obs_swe.index) #TODO fix index so both timezone aware

    # cut to just period with observed swe
    sim_swe = sim_swe[obs_ind]
    if model == 'fsm2':
        sim_swe = sim_swe.values

    stats_store['ns'][i] = nash_sut(sim_swe, obs_swe)
    stats_store['bias'][i] = mean_bias(sim_swe, obs_swe)
    stats_store['rmsd'][i] = rmsd(sim_swe, obs_swe)
    stats_store['mae'][i] = mean_absolute_error(sim_swe, obs_swe)
    stats_store['run_id'][i] = run_id

    if model == 'fsm2':
        sim_nml = dict_in[run_id]['namelist']
        for key in sim_nml['params'].keys():
            param_store[key][i] = sim_nml['params'][key]
        # param_store['exe'][i] = dict_in[run_id]['exe'][1] #
        param_store['density'][i] = dict_in[run_id]['exe'][0]
        param_store['exchng'][i] = dict_in[run_id]['exe'][1]
        param_store['hydrol'][i] = dict_in[run_id]['exe'][2]
    elif model == 'clark' or model =='eti':
        for key in dict_in[run_id]['config'].keys():
            param_store[key][i] = dict_in[run_id]['config'][key]

# plotting

if model == 'fsm2' or model == 'eti':
    fig, axs = plt.subplots(4, 4)
    axs = axs.ravel()
    # axs[8].semilogx()
elif model == 'clark' :
    fig, axs = plt.subplots(3, 4)
    axs = axs.ravel()

fig.set_size_inches(8,8)
for j, key in enumerate(param_store.keys()):
    axs[j].hist(param_store[key])
# choose cut off for reducing population
ind = stats_store['ns'] > 0.7
for j, key in enumerate(param_store.keys()):
    axs[j].hist(param_store[key][ind])

ind = stats_store['ns'] > 0.8
for j, key in enumerate(param_store.keys()):
    axs[j].hist(param_store[key][ind])

ind = stats_store['ns'] > 0.9
for j, key in enumerate(param_store.keys()):
    axs[j].hist(param_store[key][ind])

for j, key in enumerate(param_store.keys()):
    axs[j].set_xlabel(key)
fig.tight_layout()

fig.savefig(outfolder + "/Params_optimised_{}.png".format(ensemble_id), format="png", )

print('done collating')
ind = stats_store['ns'] > 0.8
# plt.scatter(param_store['mf_mean'][ind],param_store['tmelt'][ind])

fig, (ax1) = plt.subplots(1, 1)
# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
fig.set_size_inches(8, 3)
# add titles and measured data
ax1.set_ylabel('SWE (mm)', fontsize=8)

for outname in stats_store['run_id'][ind]:
    # if model == 'fsm2':
    d = dict_in[outname]['states_output']
    ax1.plot(d.index, d.swe, linewidth=0.5, color='grey')
    # elif model == 'clark' or model == 'eti':
    #     d = dict_in[outname]
    #     ax1.plot(dfin['Meas_SWE'][:-1].index, d['st_swe'], linewidth=0.5, color='grey')
    # ax1.plot(dict_in[outname]['states_output'].index, dict_in[outname]['states_output'].swe, linewidth=0.5, color='grey')

# plot measured values last to show up best on figure
ax1.plot(dfin.index, dfin.Meas_SWE, color='black', label='Observed')
# plot empty dataset for legend
ax1.plot(dfin.index, [np.nan] * len(dfin.index), color='grey', label='Modelled')
plt.legend()
plt.tight_layout()
fig.savefig(outfolder + "/Mod_vs_Meas_SWE_optimised_{}.png".format(ensemble_id), format="png", )

if model == 'fsm2':
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    fig.set_size_inches(8, 8)
    # add titles and measured data
    ax1.set_ylabel("Snow depth (m)", fontsize=8)
    ax2.set_ylabel('SWE (mm)', fontsize=8)
    ax1.xaxis.set_visible(False)
    ax2.xaxis.set_visible(False)
    ax3.set_xlabel("Date", fontsize=8)
    ax3.set_ylabel(r'Snow density kg m$^{-3}$', fontsize=8)
    for outname in stats_store['run_id'][ind]:
        ax1.plot(dict_in[outname]['states_output'].index, dict_in[outname]['states_output'].snowdepth, linewidth=0.5, color='grey')
        ax2.plot(dict_in[outname]['states_output'].index, dict_in[outname]['states_output'].swe, linewidth=0.5, color='grey')
        ax3.plot(dict_in[outname]['states_output'].index, dict_in[outname]['states_output'].density, linewidth=0.5, color='grey')

    # plot measured values last to show up best on figure
    ax1.plot(dfin.index, dfin.Meas_snowdepth, color='black', label='Measured')
    ax2.plot(dfin.index, dfin.Meas_SWE, color='black')
    ax3.plot(dfin.index, dfin.Meas_density, color='black')
    # plot empty dataset for legend
    ax1.plot(dfin.index, [np.nan] * len(dfin.index), color='grey', label='Modelled')
    plt.tight_layout()
    fig.savefig(outfolder + "/Mod_vs_Meas_line_optimised_{}.png".format(ensemble_id), format="png", )

# # compute daily melt
# daily_swe3 = []
# obs_swe_daily = []
# for k in range(47, len(st_swe[:, 0]), 48):
#     daily_swe3.append(st_swe[k, 0])
#     obs_swe_daily.append(obs_swe[k])
#
# # compute validation metrics
# mb_sim = -1 * np.diff(np.asarray(daily_swe3))
# dSWE_daily_obs = -1 * np.diff(np.asarray(obs_swe_daily))
# ns_array[i, j] = nash_sut(mb_sim, dSWE_daily_obs)
# mbd_array[i, j] = mean_bias(mb_sim, dSWE_daily_obs)
# rmsd_array[i, j] = rmsd(mb_sim, dSWE_daily_obs)
