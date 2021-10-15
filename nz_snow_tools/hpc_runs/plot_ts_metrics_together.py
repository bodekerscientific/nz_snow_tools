import numpy as np
import pickle
import matplotlib.pylab as plt
import matplotlib.dates as mdates
import datetime as dt
# import nc_time_axis
import cftime

# from nz_snow_tools.util.utils import convert_datetime_julian_day

def smooth(mod_streamq,smooth_period=5):
    if len(mod_streamq.shape)>1:
        for i in range(mod_streamq.shape[0]):
            mod_streamq[i, smooth_period // 2:-(smooth_period // 2)] = np.convolve(mod_streamq[i, :], np.ones((smooth_period,)) / smooth_period, mode='valid')
    else:
        mod_streamq[smooth_period // 2:-(smooth_period // 2)] = np.convolve(mod_streamq[:], np.ones((smooth_period,)) / smooth_period, mode='valid')
    return mod_streamq.copy()

def plot_ens_area_ts(mod_streamq, dt_mod, ax=None, al=0.5, smooth_period=11):
    # plot shaded ensemble timeseries - assumes input dimensions are (time,ens) if ax is specified will plot in this axis
    if ax == None:
        _, ax = plt.subplots(1, 1)

    #

    if smooth_period > 1:
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


    if mod_streamq.shape[0]<=3:
        ax.fill_between(dt_mod, p0, p100, facecolor=[0.42, 0.68, 0.84], alpha=al)#, label='0-100%')
        # ax.plot(dt_mod, p50, color=[0.03, 0.20, 0.44], label='median')
    else:
        ax.fill_between(dt_mod, p0, p100, facecolor=[0.42, 0.68, 0.84], alpha=al, label='0-100%')
        ax.fill_between(dt_mod, p25, p75, facecolor=[0.15, 0.47, 0.72], alpha=al, label='25-75%')
        ax.plot(dt_mod, p50, color=[0.03, 0.20, 0.44], label='median')
        ax.fill_between(dt_mod, p75, p100, facecolor=[0.42, 0.68, 0.84], alpha=al)

    # ax.fill_between(dt_mod, p0, p5, facecolor=[0.92, 0.95, 0.98], alpha=al, label='min-max')
    # ax.fill_between(dt_mod, p95, p100, facecolor=[0.92, 0.95, 0.98], alpha=al)
    # ax.set_ylim([0, 0.5])
    ax.set_xlabel('Month')
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[4,6,8,10,12,2,4])) # tick every 2 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # to set tick format
    ax.fmt_xdata = mdates.DateFormatter('%d-%b')  # to set pointer display
    # if ax == None:
    #     plt.autofmt_xdate()
    #     plt.legend()


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

if __name__ == '__main__':

    hydro_years_to_take = np.arange(2018,2020+1)  # [2013 + 1]  # range(2001, 2013 + 1)
    plot_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2101/snow reanalysis/SI/ts_plots'
    # model_analysis_area = 145378  # sq km.
    catchment = 'SI'  # string identifying catchment modelled
    smooth_period = 11
    # # modis options
    output_dem = 'si_dem_250m'  # identifier for output dem
    modis_sc_threshold = 50  # value of fsca (in percent) that is counted as being snow covered
    modis_output_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2101/snow reanalysis'

    [ann_ts_av_sca_m, ann_ts_av_sca_thres_m, ann_dt_m, ann_scd_m] = pickle.load(open(
        modis_output_folder + '/summary_MODIS_{}_{}_{}_{}_thres{}.pkl'.format(hydro_years_to_take[0], hydro_years_to_take[-1], catchment, output_dem,
                                                                              modis_sc_threshold), 'rb'))
    # model options

    # run_id = 'cl09_default' #
    # which_model ='clark2009' #
    run_id = 'cl09_default'#'dsc_default'  #
    which_model = 'clark2009'# 'dsc_snow'  #
    met_inp = 'nzcsm7-12'  # 'vcsn_norton' #   # identifier for input meteorology

    output_dem = 'si_dem_250m'
    model_swe_sc_threshold = 5#30  # threshold for treating a grid cell as snow covered (mm w.e)
    model_output_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2101/snow reanalysis'

    [ann_ts_av_swe, ann_ts_av_sca_thres, ann_dt, ann_scd, ann_av_swe, ann_max_swe, ann_metadata] = pickle.load(open(
        model_output_folder + '/summary_MODEL_{}_{}_{}_{}_{}_{}_{}_thres{}.pkl'.format(hydro_years_to_take[0], hydro_years_to_take[-1], met_inp, which_model,
                                                                                       catchment, output_dem, run_id, model_swe_sc_threshold), 'rb'))

    model_analysis_area = ann_metadata['area_domain']
    # np.sum(~np.isnan(ann_scd[0]))/16 = 145378
    plot_dt = ann_dt[0][:365]
    n_years = len(hydro_years_to_take)
    sca_model_ts_5nzcsm = np.full((n_years, 365), np.nan)
    swe_model_ts_5nzcsm = np.full((n_years, 365), np.nan)
    sca_modis_ts = np.full((n_years, 365), np.nan)

    for i in np.arange(n_years):
        sca_model_ts_5nzcsm[i, :] = ann_ts_av_sca_thres[i][:365] * model_analysis_area
        swe_model_ts_5nzcsm[i, :] = ann_ts_av_swe[i][:365] / 1e6 * model_analysis_area  # convert to km^3 from mm w.e. and km^2
        sca_modis_ts[i, :] = ann_ts_av_sca_thres_m[i][:365] * model_analysis_area

    # convert to regular datetime to enable fill between
    if isinstance(plot_dt[0], cftime.real_datetime) or isinstance(plot_dt[0], cftime.DatetimeGregorian):
         plot_dt = cftime.num2date(cftime.date2num(plot_dt,'days since 1900-01-01 00:00'),'days since 1900-01-01 00:00')

    #load run 2
    met_inp = 'nzcsm7-12'  # 'vcsn_norton' #   # identifier for input meteorology
    model_swe_sc_threshold = 30  # threshold for treating a grid cell as snow covered (mm w.e)
    [ann_ts_av_swe, ann_ts_av_sca_thres, ann_dt, ann_scd, ann_av_swe, ann_max_swe, ann_metadata] = pickle.load(open(
        model_output_folder + '/summary_MODEL_{}_{}_{}_{}_{}_{}_{}_thres{}.pkl'.format(hydro_years_to_take[0], hydro_years_to_take[-1], met_inp, which_model,
                                                                                       catchment, output_dem, run_id, model_swe_sc_threshold), 'rb'))
    model_analysis_area = ann_metadata['area_domain']
    # np.sum(~np.isnan(ann_scd[0]))/16 = 145378
    n_years = len(hydro_years_to_take)
    sca_model_ts_30nzcsm = np.full((n_years, 365), np.nan)
    swe_model_ts_30nzcsm = np.full((n_years, 365), np.nan)
    for i in np.arange(n_years):
        sca_model_ts_30nzcsm[i, :] = ann_ts_av_sca_thres[i][:365] * model_analysis_area
        swe_model_ts_30nzcsm[i, :] = ann_ts_av_swe[i][:365] / 1e6 * model_analysis_area  # convert to km^3 from mm w.e. and km^2

    #load run 3
    met_inp = 'vcsn_norton' #   # identifier for input meteorology
    model_swe_sc_threshold = 5  # threshold for treating a grid cell as snow covered (mm w.e)
    [ann_ts_av_swe, ann_ts_av_sca_thres, ann_dt, ann_scd, ann_av_swe, ann_max_swe, ann_metadata] = pickle.load(open(
        model_output_folder + '/summary_MODEL_{}_{}_{}_{}_{}_{}_{}_thres{}.pkl'.format(hydro_years_to_take[0], hydro_years_to_take[-1], met_inp, which_model,
                                                                                       catchment, output_dem, run_id, model_swe_sc_threshold), 'rb'))
    model_analysis_area = ann_metadata['area_domain']
    # np.sum(~np.isnan(ann_scd[0]))/16 = 145378
    n_years = len(hydro_years_to_take)
    sca_model_ts_5vcsn = np.full((n_years, 365), np.nan)
    swe_model_ts_5vcsn = np.full((n_years, 365), np.nan)
    for i in np.arange(n_years):
        sca_model_ts_5vcsn[i, :] = ann_ts_av_sca_thres[i][:365] * model_analysis_area
        swe_model_ts_5vcsn[i, :] = ann_ts_av_swe[i][:365] / 1e6 * model_analysis_area  # convert to km^3 from mm w.e. and km^2

    #load run 4
    met_inp = 'vcsn_norton' #   # identifier for input meteorology
    model_swe_sc_threshold = 30  # threshold for treating a grid cell as snow covered (mm w.e)
    [ann_ts_av_swe, ann_ts_av_sca_thres, ann_dt, ann_scd, ann_av_swe, ann_max_swe, ann_metadata] = pickle.load(open(
        model_output_folder + '/summary_MODEL_{}_{}_{}_{}_{}_{}_{}_thres{}.pkl'.format(hydro_years_to_take[0], hydro_years_to_take[-1], met_inp, which_model,
                                                                                       catchment, output_dem, run_id, model_swe_sc_threshold), 'rb'))
    model_analysis_area = ann_metadata['area_domain']
    # np.sum(~np.isnan(ann_scd[0]))/16 = 145378
    n_years = len(hydro_years_to_take)
    sca_model_ts_30vcsn = np.full((n_years, 365), np.nan)
    swe_model_ts_30vcsn = np.full((n_years, 365), np.nan)
    for i in np.arange(n_years):
        sca_model_ts_30vcsn[i, :] = ann_ts_av_sca_thres[i][:365] * model_analysis_area
        swe_model_ts_30vcsn[i, :] = ann_ts_av_swe[i][:365] / 1e6 * model_analysis_area  # convert to km^3 from mm w.e. and km^2



####### smooth####
    sca_modis_ts = smooth(sca_modis_ts,smooth_period=smooth_period)
    sca_model_ts_5nzcsm = smooth(sca_model_ts_5nzcsm,smooth_period=smooth_period)
    sca_model_ts_30nzcsm = smooth(sca_model_ts_30nzcsm,smooth_period=smooth_period)
    sca_model_ts_5vcsn = smooth(sca_model_ts_5vcsn,smooth_period=smooth_period)
    sca_model_ts_30vcsn = smooth(sca_model_ts_30vcsn,smooth_period=smooth_period)
############## start plotting ############

    plt.rcParams.update({'font.size': 8})
    plt.rcParams.update({'axes.titlesize': 8})

    style = ['-', '--', '-.']
    fig,[ax1,ax2] = plt.subplots(nrows=1,ncols=2,figsize = [10,4])
    plt.sca(ax1)
    for i, t in enumerate(sca_modis_ts):
        plt.plot(plot_dt, t, style[i], color='k',
             label='MODIS {}-{}'.format(hydro_years_to_take[i] - 1, hydro_years_to_take[i] - 2000))
    for i, t in enumerate(sca_model_ts_30nzcsm):
        plt.plot(plot_dt, t, style[i], color='b',
             label='MODEL {}-{}'.format(hydro_years_to_take[i] - 1, hydro_years_to_take[i] - 2000))
    plt.legend()
    ax = plt.gca()
    ax.set_ylabel('Snow Covered Area (square km)')
    ax.set_ylim([0, 6.4e4])
    # plt.grid(True)
    ax.set_xlabel('Month')
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[4, 6, 8, 10, 12, 2, 4]))  # tick every 2 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # to set tick format
    ax.fmt_xdata = mdates.DateFormatter('%d-%b')  # to set pointer display
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[4,5,6,7,8,9,10,11,12,1,2,3,4])) # tick every 2 months
    plt.title('(a) Model threshold 30 mm w.e', fontweight='bold',fontsize=10)
    plt.sca(ax2)
    for i, t in enumerate(sca_modis_ts):
        plt.plot(plot_dt, t, style[i], color='k',
             label='MODIS {}-{}'.format(hydro_years_to_take[i] - 1, hydro_years_to_take[i] - 2000))
    for i, t in enumerate(sca_model_ts_5nzcsm):
        plt.plot(plot_dt, t, style[i], color='b',
             label='MODEL {}-{}'.format(hydro_years_to_take[i] - 1, hydro_years_to_take[i] - 2000))
    plt.title('(b) Model threshold 5 mm w.e', fontweight='bold',fontsize=10)
    plt.legend()
    ax = plt.gca()
    ax.set_ylabel('Snow Covered Area (square km)')
    ax.set_ylim([0, 6.4e4])
    # plt.grid(True)
    ax.set_xlabel('Month')
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[4, 6, 8, 10, 12, 2, 4]))  # tick every 2 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # to set tick format
    ax.fmt_xdata = mdates.DateFormatter('%d-%b')  # to set pointer display
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[4,5,6,7,8,9,10,11,12,1,2,3,4])) # tick every 2 months
    plt.tight_layout()
    plt.savefig(plot_folder + '/SCA nzcsm both 5 and 30 {}_{}_{}_{}_smoothed{}.png'.format(hydro_years_to_take[0], hydro_years_to_take[-1], catchment, output_dem,
                                                                                smooth_period), dpi=600)

    #VCSN plot
    fig,[ax1,ax2] = plt.subplots(nrows=1,ncols=2,figsize = [10,4])
    plt.sca(ax1)
    for i, t in enumerate(sca_modis_ts):
        plt.plot(plot_dt, t, style[i], color='k',
             label='MODIS {}-{}'.format(hydro_years_to_take[i] - 1, hydro_years_to_take[i] - 2000))
    for i, t in enumerate(sca_model_ts_30vcsn):
        plt.plot(plot_dt, t, style[i], color='b',
             label='MODEL {}-{}'.format(hydro_years_to_take[i] - 1, hydro_years_to_take[i] - 2000))
    plt.legend()
    ax = plt.gca()
    ax.set_ylabel('Snow Covered Area (square km)')
    ax.set_ylim([0, 6.4e4])
    # plt.grid(True)
    ax.set_xlabel('Month')
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[4, 6, 8, 10, 12, 2, 4]))  # tick every 2 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # to set tick format
    ax.fmt_xdata = mdates.DateFormatter('%d-%b')  # to set pointer display
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[4,5,6,7,8,9,10,11,12,1,2,3,4])) # tick every 2 months
    plt.title('(a) Model threshold 30 mm w.e.',fontweight='bold',fontsize=10)
    plt.sca(ax2)
    for i, t in enumerate(sca_modis_ts):
        plt.plot(plot_dt, t, style[i], color='k',
             label='MODIS {}-{}'.format(hydro_years_to_take[i] - 1, hydro_years_to_take[i] - 2000))
    for i, t in enumerate(sca_model_ts_5vcsn):
        plt.plot(plot_dt, t, style[i], color='b',
             label='MODEL {}-{}'.format(hydro_years_to_take[i] - 1, hydro_years_to_take[i] - 2000))
    plt.title('(b) Model threshold 5 mm w.e', fontweight='bold',fontsize=10)
    plt.legend()
    ax = plt.gca()
    ax.set_ylabel('Snow Covered Area (square km)')
    ax.set_ylim([0, 6.4e4])
    # plt.grid(True)
    ax.set_xlabel('Month')
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[4, 6, 8, 10, 12, 2, 4]))  # tick every 2 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # to set tick format
    ax.fmt_xdata = mdates.DateFormatter('%d-%b')  # to set pointer display
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[4,5,6,7,8,9,10,11,12,1,2,3,4])) # tick every 2 months
    plt.tight_layout()
    plt.savefig(plot_folder + '/SCA vcsn both 5 and 30 {}_{}_{}_{}_smoothed{}.png'.format(hydro_years_to_take[0], hydro_years_to_take[-1], catchment, output_dem,
                                                                                smooth_period), dpi=600)


    plt.figure(figsize = [5,4])
    for i, t in enumerate(swe_model_ts_30nzcsm):
        plt.plot(plot_dt, t, style[i], color='b',
             label='{}-{}'.format(hydro_years_to_take[i] - 1, hydro_years_to_take[i] - 2000))
    plt.legend()
    ax = plt.gca()
    ax.set_ylabel('Snow water storage (cubic km)')
    ax.set_ylim(bottom=0)
    # plt.grid(True)
    ax.set_xlabel('Month')
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[4, 6, 8, 10, 12, 2, 4]))  # tick every 2 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # to set tick format
    ax.fmt_xdata = mdates.DateFormatter('%d-%b')  # to set pointer display
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[4,5,6,7,8,9,10,11,12,1,2,3,4])) # tick every 2 months
    plt.tight_layout()
    plt.grid()
    plt.savefig(plot_folder + '/SWE nzcsm {}_{}_{}_{}_smoothed{}.png'.format(hydro_years_to_take[0], hydro_years_to_take[-1], catchment, output_dem,
                                                                                smooth_period), dpi=600)



    plt.figure(figsize = [5,4])
    for i, t in enumerate(swe_model_ts_30vcsn):
        plt.plot(plot_dt, t, style[i], color='b',
             label='{}-{}'.format(hydro_years_to_take[i] - 1, hydro_years_to_take[i] - 2000))
    plt.legend()
    ax = plt.gca()
    ax.set_ylabel('Snow water storage (cubic km)')
    ax.set_ylim(bottom=0)
    # plt.grid(True)
    ax.set_xlabel('Month')
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[4, 6, 8, 10, 12, 2, 4]))  # tick every 2 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # to set tick format
    ax.fmt_xdata = mdates.DateFormatter('%d-%b')  # to set pointer display
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[4,5,6,7,8,9,10,11,12,1,2,3,4])) # tick every 2 months
    plt.tight_layout()
    plt.grid()
    plt.savefig(plot_folder + '/SWE vcsn {}_{}_{}_{}_smoothed{}.png'.format(hydro_years_to_take[0], hydro_years_to_take[-1], catchment, output_dem,
                                                                                smooth_period), dpi=600)