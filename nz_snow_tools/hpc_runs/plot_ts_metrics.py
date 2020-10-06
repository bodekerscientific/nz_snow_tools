import numpy as np
import pickle
import matplotlib.pylab as plt
import matplotlib.dates as mdates
import datetime as dt
# from nz_snow_tools.util.utils import convert_datetime_julian_day


def plot_ens_area_ts(mod_streamq, dt_mod, ax=None, al=0.5, smooth_period=11):
    # plot shaded ensemble timeseries - assumes input dimensions are (time,ens) if ax is specified will plot in this axis
    if ax == None:
        _, ax = plt.subplots(1, 1)
    for i in range(mod_streamq.shape[0]):
        mod_streamq[i, smooth_period // 2:-(smooth_period // 2)] = np.convolve(mod_streamq[i, :], np.ones((smooth_period,)) / smooth_period, mode='valid')
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


hydro_years_to_take = np.arange(2017, 2020 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
plot_folder = '/nesi/nobackup/niwa00004/jonoconway/snow_sims_nz/vcsn'
# model_analysis_area = 145378  # sq km.
catchment = 'SI'  # string identifying catchment modelled

# # modis options
modis_dem = 'nztm250m'  # identifier for output dem
modis_sc_threshold = 35  # value of fsca (in percent) that is counted as being snow covered
modis_output_folder = '/nesi/nobackup/niwa00004/jonoconway/snow_sims_nz/'

[ann_ts_av_sca_m, ann_ts_av_sca_thres_m, ann_dt_m, ann_scd_m] = pickle.load(open(
    modis_output_folder + '/summary_MODIS_{}_{}_{}_{}_thres{}.pkl'.format(hydro_years_to_take[0], hydro_years_to_take[-1], catchment, modis_dem,
                                                                          modis_sc_threshold), 'rb'))
# model options

# run_id = 'cl09_default' #
# which_model ='clark2009' #
run_id = 'dsc_default' # 'cl09_default'
which_model = 'dsc_snow' # 'clark2009'
met_inp = 'nzcsm7-12'#'vcsn_norton' #   # identifier for input meteorology

output_dem = 'si_dem_250m'
model_swe_sc_threshold = 30  # threshold for treating a grid cell as snow covered (mm w.e)
model_output_folder = '/nesi/nobackup/niwa00004/jonoconway/snow_sims_nz/vcsn'

[ann_ts_av_swe, ann_ts_av_sca_thres, ann_dt, ann_scd, ann_av_swe, ann_max_swe, ann_metadata] = pickle.load(open(
    model_output_folder + '/summary_MODEL_{}_{}_{}_{}_{}_{}_{}_thres{}.pkl'.format(hydro_years_to_take[0], hydro_years_to_take[-1], met_inp, which_model,
                                                                                   catchment, output_dem, run_id, model_swe_sc_threshold), 'rb'))

model_analysis_area = ann_metadata['area_domain']
# np.sum(~np.isnan(ann_scd[0]))/16 = 145378
plot_dt = ann_dt[0][:365]
n_years = len(hydro_years_to_take)
sca_model_ts = np.full((n_years, 365), np.nan)
swe_model_ts = np.full((n_years, 365), np.nan)
sca_modis_ts = np.full((n_years, 365), np.nan)

for i in np.arange(n_years):
    sca_model_ts[i, :] = ann_ts_av_sca_thres[i][:365] * model_analysis_area
    swe_model_ts[i, :] = ann_ts_av_swe[i][:365] / 1e6 * model_analysis_area  # convert to km^3 from mm w.e. and km^2
    sca_modis_ts[i, :] = ann_ts_av_sca_thres_m[i][:365] * model_analysis_area

plot_ens_area_ts(sca_model_ts, plot_dt)
plt.legend()
ax = plt.gca()
ax.set_ylabel('Snow Covered Area (square km)')
ax.set_ylim(bottom=0)
plt.savefig(plot_folder + '/SCA model {}_{}_{}_{}_{}_{}_{}_thres{}.png'.format(hydro_years_to_take[0], hydro_years_to_take[-1], met_inp, which_model, catchment,
                                                                               output_dem, run_id, model_swe_sc_threshold), dpi=600)
plot_ens_area_ts(swe_model_ts, plot_dt)
plt.legend()
ax = plt.gca()
ax.set_ylabel('Snow water storage (cubic km)')
ax.set_ylim(bottom=0)
# ax.set_ylim([0,15])
plt.savefig(plot_folder + '/SWE model {}_{}_{}_{}_{}_{}_{}_thres{}.png'.format(hydro_years_to_take[0], hydro_years_to_take[-1], met_inp, which_model, catchment,
                                                                               output_dem, run_id, model_swe_sc_threshold), dpi=600)

plot_ens_area_ts(sca_modis_ts, plot_dt)
plt.legend()
ax = plt.gca()
# ax.set_ylim([0,7.5e4])
plt.savefig(plot_folder + '/SCA modis {}_{}_{}_{}_modis_thres{}.png'.format(hydro_years_to_take[0], hydro_years_to_take[-1], catchment, output_dem,
                                                                            modis_sc_threshold), dpi=600)
plt.show()
