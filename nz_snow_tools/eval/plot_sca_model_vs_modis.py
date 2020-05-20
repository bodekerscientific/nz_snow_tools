import numpy as np
import pickle
import matplotlib.pylab as plt
import matplotlib.dates as mdates
import datetime as dt
from nz_snow_tools.util.utils import convert_datetime_julian_day


def plot_ens_area_ts(mod_streamq, dt_mod, ax=None, al=0.5, smooth_period=11):
    # plot shaded ensemble timeseries - assumes input dimensions are (time,ens) if ax is specified will plot in this axis
    if ax == None:
        _, ax = plt.subplots(1, 1)
    for i in range(mod_streamq.shape[0]):
        mod_streamq[i,smooth_period/2:-(smooth_period//2)] = np.convolve(mod_streamq[i,:], np.ones((smooth_period,)) / smooth_period, mode='valid')
    # dt_mod = dt_mod[smooth_period/2:-(smooth_period//2)]

    p0 = mod_streamq_ens_percentiles_1(mod_streamq, 0)
    p5 = mod_streamq_ens_percentiles_1(mod_streamq, 5)
    p25 = mod_streamq_ens_percentiles_1(mod_streamq, 25)
    p50 = mod_streamq_ens_percentiles_1(mod_streamq, 50)
    p75 = mod_streamq_ens_percentiles_1(mod_streamq, 75)
    p95 = mod_streamq_ens_percentiles_1(mod_streamq, 95)
    p100 = mod_streamq_ens_percentiles_1(mod_streamq, 100)

    # ax1.plot(nc_times,p0,'--k')0.92,0.95,0.98 [0.73,0.84,0.92]
    # ax.fill_between(dt_mod, p0, p5, facecolor=[0.92, 0.95, 0.98], alpha=al, label='min-max')
    ax.fill_between(dt_mod, p0, p25, facecolor=[0.42, 0.68, 0.84], alpha=al, label='0-100%')
    ax.fill_between(dt_mod, p25, p75, facecolor=[0.15, 0.47, 0.72], alpha=al, label='25-75%')
    ax.plot(dt_mod, p50, color=[0.03, 0.20, 0.44], label='median')
    ax.fill_between(dt_mod, p75, p100, facecolor=[0.42, 0.68, 0.84], alpha=al)
    # ax.fill_between(dt_mod, p95, p100, facecolor=[0.92, 0.95, 0.98], alpha=al)
    # ax1.plot(nc_times,p100,'--k')
    # ax.set_ylim([0, 0.5])
    ax.set_ylabel('Snow Covered Area (square km)')
    ax.set_xlabel('Month')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # to set tick format
    ax.fmt_xdata = mdates.DateFormatter('%d-%b')  # to set pointer display
    if ax == None:
        plt.autofmt_xdate()
        plt.legend()


def mod_streamq_ens_percentiles_1(mod_streamq, p):
    """
    calculate percentile of flow for each timestep from ensemble model output
    :param mod_streamq: modelled streamflow output from topnet. dimensions [time,nens]
    :param p: percentile to return. 0 to 100
    :param ind_rch: index of rch requested. 0 to nrch-1
    :return: timeseries of percentile. dimensions [time]
    """
    assert len(mod_streamq.shape) == 2
    percent = np.percentile(np.ma.filled(mod_streamq[:, :], fill_value=np.nan), p, axis=0)
    return percent


plot_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/DSC Snow/SouthIsland_results/modis_comparison'
model_analysis_area = 145378 # sq km.
catchment = 'SI'  # string identifying catchment modelled
modis_dem = 'nztm250m'  # identifier for output dem
years_to_take = np.arange(2000, 2016 + 1)  # range(2016, 2016 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
modis_sc_threshold = 50  # value of fsca (in percent) that is counted as being snow covered
output_folder = r'C:\Users\conwayjp\OneDrive - NIWA\projects\DSC Snow\MODIS'

[ann_ts_av_sca_m, ann_ts_av_sca_thres_m, ann_dt_m, ann_scd_m] = pickle.load(open(
    output_folder + '/summary_MODIS_{}_{}_{}_{}_thres{}.pkl'.format(years_to_take[0], years_to_take[-1], catchment, modis_dem,
                                                                    modis_sc_threshold), 'rb'))

run_id = 'norton_5_topleft'
catchment = 'SouthIsland'
years_to_take = range(2000, 2016 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
model_swe_sc_threshold = 20  # threshold for treating a grid cell as snow covered (mm w.e)
model_output_folder = 'C:/Users/conwayjp/Documents/Temp'


[ann_ts_av_swe, ann_ts_av_sca_thres, ann_dt, ann_scd, ann_swe] = pickle.load(open(
    model_output_folder + '/summary_MODEL_{}_{}_{}_{}_thres{}.pkl'.format(years_to_take[0], years_to_take[-1], catchment, run_id,
                                                                          model_swe_sc_threshold), 'rb'))

# np.sum(~np.isnan(ann_scd[0]))/16 = 145378
ann_scd = np.asarray(ann_scd) / 100.  # convert to fraction from %
ann_ts_av_sca_thres = np.asarray(ann_ts_av_sca_thres) / 100

sca_model_2001_2012 = np.full((11, 365), np.nan)
swe_model_2001_2012 = np.full((11, 365), np.nan)
sca_modis_2001_2012 = np.full((11, 365), np.nan)

for i in np.arange(11):
    sca_model_2001_2012[i, :] = ann_ts_av_sca_thres[i+1][:365] * model_analysis_area# skip first year
    swe_model_2001_2012[i, :] = ann_ts_av_swe[i + 1][:365] * model_analysis_area / 1e6 # convert to cubic km? need to convert from mm w.e. to km w.e. / 1e6
    sca_modis_2001_2012[i, :] = ann_ts_av_sca_thres_m[i+1][:365] * model_analysis_area


plot_ens_area_ts(sca_model_2001_2012, ann_dt_m[1])
plt.legend()
ax = plt.gca()
ax.set_ylim([0,7.5e4])
plt.savefig(plot_folder + '/SCA model 2001 to 2011.png'.format(run_id), dpi=600)

plot_ens_area_ts(swe_model_2001_2012, ann_dt_m[1])
plt.legend()
ax = plt.gca()
ax.set_ylabel('Snow water storage (cubic km)')
ax.set_ylim([0,15])
plt.savefig(plot_folder + '/SWE model 2001 to 2011.png', dpi=600)

plot_ens_area_ts(sca_modis_2001_2012, ann_dt_m[1])
plt.legend()
ax = plt.gca()
ax.set_ylim([0,7.5e4])
plt.savefig(plot_folder + '/SCA modis 2001 to 2011.png', dpi=600)

# plt.show()
