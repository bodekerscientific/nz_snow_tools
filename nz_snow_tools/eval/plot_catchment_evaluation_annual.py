"""
code to plot results of catchment_evalutation - which evaluates snow models at catchment scale (i.e. Nevis or Clutha river)

Jono Conway
"""
from __future__ import division

import numpy as np
import pickle
import matplotlib.dates as mdates
import matplotlib.pylab as plt
import datetime as dt

from nz_snow_tools.util.utils import convert_date_hydro_DOY, convert_hydro_DOY_to_date

if __name__ == '__main__':

    which_model = 'dsc_snow'  # string identifying the model to be run. options include 'clark2009', 'dsc_snow', or 'all' # future will include 'fsm'
    smooth_period = 5  # number of days to smooth data
    clark2009run = True  # boolean specifying if the run already exists
    dsc_snow_opt = 'fortran'  # string identifying which version of the dsc snow model to use output from 'python' or 'fortran'
    run_id = 'jobst_ucc_4'  # string identifying fortran dsc_snow run. everything after the year
    catchment = 'Clutha'
    output_dem = 'nztm250m'  # identifier for output dem
    years_to_take = [2011,2011]#range(2000, 2016 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
    modis_sc_thresholds = [50]  # value of fsca (in percent) that is counted as being snow covered 30,40,50,60,70,80
    model_swe_sc_threshold = 5  # threshold for treating a grid cell as snow covered
    model_output_folder = 'P:/Projects/DSC-Snow/runs/output/clutha_nztm250m_erebus'
    plot_folder = 'P:/Projects/DSC-Snow/runs/output/clutha_nztm250m_erebus'

    plt.figure(figsize=[8, 6])
    if which_model == 'all':
        plt.subplot(3, 1, 1)
    else:
        plt.subplot(2, 1, 1)
    # load the first file to get model + basic modis info
    ann = pickle.load(open(
        model_output_folder + '/summary_{}_{}_thres{}_swe{}_{}_{}.pkl'.format(catchment, output_dem, modis_sc_thresholds[0], model_swe_sc_threshold,
                                                                              which_model, run_id), 'rb'))

    # [ann_ts_av_sca_m, ann_hydro_days_m, ann_dt_m, ann_scd_m, ann_ts_av_sca, ann_ts_av_swe, ann_doy, ann_dt,
    #  ann_scd, ann_ts_av_sca2, ann_ts_av_swe2, ann_hydro_days2, ann_dt2, ann_scd2] = ann
    ann_ts_av_sca_m = ann[0]
    ann_dt_m = ann[2]
    ann_ts_av_sca = ann[4]
    ann_ts_av_swe = ann[5]
    ann_doy = ann[6]
    ann_ts_av_sca2 = ann[9]
    ann_ts_av_swe2 = ann[10]
    ann_doy2 = ann[11]
    # ann_ts_av_sca_m_thres = ann[14]
    if smooth_period > 1:
        for i in range(len(ann_ts_av_sca)):
            ann_ts_av_sca[i] = np.convolve(ann_ts_av_sca[i], np.ones((smooth_period,)) / smooth_period, mode='same')
            if which_model == 'all':
                ann_ts_av_sca2[i] = np.convolve(ann_ts_av_sca2[i], np.ones((smooth_period,)) / smooth_period, mode='same')

    # put data into an array
    ann_ts_av_swe_array = np.zeros((len(ann_ts_av_swe), 365), dtype=np.float32)
    ann_ts_av_sca_array = np.zeros((len(ann_ts_av_swe), 365), dtype=np.float32)
    ann_ts_av_sca_m_array = np.zeros((len(ann_ts_av_swe), 365), dtype=np.float32)
    if which_model == 'all':
        ann_ts_av_swe2_array = np.zeros((len(ann_ts_av_swe), 365), dtype=np.float32)
        ann_ts_av_sca2_array = np.zeros((len(ann_ts_av_swe), 365), dtype=np.float32)
        # ann_ts_av_sca_m_thres_array = np.zeros((len(ann_ts_av_swe), 365))
    # ann_ts_av_sca_m[5][:] = np.nan  # years 2006 and 2011 have bad modis data
    # ann_ts_av_sca_m[10][:] = np.nan
    # ann_ts_av_sca_m_thres[5][:] = np.nan # years 2006 and 2011 have bad modis data
    # ann_ts_av_sca_m_thres[10][:] = np.nan
    for i in range(len(ann_ts_av_swe)):
        for j in range(365):
            ann_ts_av_swe_array[i, j] = ann_ts_av_swe[i][j]
            ann_ts_av_sca_m_array[i, j] = ann_ts_av_sca_m[i][j]
            # ann_ts_av_sca_m_thres_array[i, j] = ann_ts_av_sca_m_thres[i][j]
            ann_ts_av_sca_array[i, j] = ann_ts_av_sca[i][j]
            if which_model == 'all':
                ann_ts_av_sca2_array[i, j] = ann_ts_av_sca2[i][j]
                ann_ts_av_swe2_array[i, j] = ann_ts_av_swe2[i][j]
    # snow covered area plot

    # plot average fsca for each year
    for ts in ann_ts_av_sca_m:
        dt_index = [dt.datetime(2000, 1, 1) + dt.timedelta(days=d - 1) for d in range(1, len(ts)+1)]  # need to plot all in the same year
        plt.plot(dt_index[:-1], ts[:-1], c=[0.4, 0.4, 0.4], linewidth=.5)

    # plot average for different modis fsca thresholds
    for modis_sc_threshold in modis_sc_thresholds:
        ann = pickle.load(
            open(model_output_folder + '/summary_{}_{}_thres{}_swe{}_{}_{}.pkl'.format(catchment, output_dem, modis_sc_thresholds[0], model_swe_sc_threshold,
                                                                              which_model, run_id), 'rb'))
        ann_ts_av_sca_m_thres = ann[14]
        # ann_ts_av_sca_m_thres[5][:] = np.nan  # years 2006 and 2011 have bad modis data
        # ann_ts_av_sca_m_thres[10][:] = np.nan
        ann_ts_av_sca_m_thres_array = np.zeros((len(ann_ts_av_swe), 365))
        for i in range(len(ann_ts_av_swe)):
            for j in range(365):
                ann_ts_av_sca_m_thres_array[i, j] = ann_ts_av_sca_m_thres[i][j]
        plt.plot(dt_index[:365], np.nanmean(ann_ts_av_sca_m_thres_array, axis=0), label='average: modis >= {}% fsca'.format(modis_sc_threshold))

    plt.plot(dt_index[:365], np.nanmean(ann_ts_av_sca_m_array, axis=0), label='average: modis fsca')
    ax = plt.gca()
    ax.set_ylim([0, 1])
    ax.set_ylabel('SCA')
    months = mdates.MonthLocator()
    ax.xaxis.set_major_locator(months)
    monthsFmt = mdates.DateFormatter('%b')
    ax.xaxis.set_major_formatter(monthsFmt)
    plt.legend()
    plt.tight_layout()
    # common x axis
    if which_model == 'all':
        plt.subplot(3, 1, 2)
    else:
        plt.subplot(2, 1, 2)

    for ts, doy in zip(ann_ts_av_sca, ann_doy):
        dt_index = [dt.datetime(2000, 1, 1) + dt.timedelta(days=d - 1) for d in range(1, len(doy)+1)]  # need to plot all in the same year
        plt.plot(dt_index[:-1], ts[:-1], c=[0.4, 0.4, 0.4], linewidth=.5)  # remove last point as is 1st april in next year
    if which_model != 'all':
        label_text = 'average:' + which_model
    else:
        label_text = 'average: clark2009'
    plt.plot(dt_index[:365], np.mean(ann_ts_av_sca_array, axis=0), 'r', label=label_text)
    ax = plt.gca()
    ax.set_ylim([0, 1])
    ax.set_ylabel('SCA')
    months = mdates.MonthLocator()
    ax.xaxis.set_major_locator(months)
    if which_model != 'all':
        monthsFmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_formatter(monthsFmt)
    plt.legend()
    plt.tight_layout()

    if which_model == 'all':
        plt.subplot(3, 1, 3)
        for ts, doy in zip(ann_ts_av_sca2, ann_doy2):
            dt_index = [dt.datetime(2000, 1, 1) + dt.timedelta(days=d - 1) for d in range(1, len(doy)+1)] # need to plot all in the same year
            plt.plot(dt_index[:-1], ts[:-1], c=[0.4, 0.4, 0.4], linewidth=.5)
        plt.plot(dt_index[:365], np.mean(ann_ts_av_sca2_array, axis=0), 'r', label='average: dsc_snow')

        plt.gcf().autofmt_xdate()
        years = mdates.YearLocator()  # every year
        months = mdates.MonthLocator()  # every month
        # days = mdates.DayLocator(interval=1)  # every day interval = 7 every week
        monthsFmt = mdates.DateFormatter('%b')
        ax = plt.gca()
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(monthsFmt)
        # ax.xaxis.set_minor_locator(days)
        ax.set_ylim([0, 1])
        ax.set_ylabel('SCA')
        plt.legend()
        plt.tight_layout()

    plt.savefig(plot_folder + '/av_SCA_ts_{}_{}_swe{}_{}_{}.png'.format(catchment, output_dem, model_swe_sc_threshold, which_model, run_id), dpi=300)

    # plot timeseries of SWE
    plt.figure()
    # common x axis
    if which_model == 'all':
        plt.subplot(2, 1, 1)

    for ts, doy in zip(ann_ts_av_swe, ann_doy):
        dt_index = [dt.datetime(2000,1,1)+dt.timedelta(days=d-1) for d in range(1,len(doy)+1)]# need to plot all in the same year
        plt.plot(dt_index[:-1], ts[:-1], c=[0.4, 0.4, 0.4], linewidth=.5)
    if which_model != 'all':
        label_text = 'average:' + which_model
    else:
        label_text = 'average: clark2009'
    plt.plot(dt_index[:365], np.mean(ann_ts_av_swe_array, axis=0), 'r', label=label_text)
    ax = plt.gca()
    ax.set_ylim([0, 60])
    ax.set_ylabel('SWE (mm w.e.)')
    months = mdates.MonthLocator()
    ax.xaxis.set_major_locator(months)
    if which_model != 'all':
        monthsFmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_formatter(monthsFmt)

    plt.legend()
    if which_model == 'all':
        plt.subplot(2, 1, 2)
        for ts, doy in zip(ann_ts_av_swe2, ann_doy2):
            dt_index = [dt.datetime(2000, 1, 1) + dt.timedelta(days=d - 1) for d in range(1, len(doy)+1)] # need to plot all in the same year
            plt.plot(dt_index[:-1], ts[:-1], c=[0.4, 0.4, 0.4], linewidth=.5)
        plt.plot(dt_index[:365], np.mean(ann_ts_av_swe2_array, axis=0), 'r', label='average: dsc_snow')

        plt.gcf().autofmt_xdate()
        years = mdates.YearLocator()  # every year
        months = mdates.MonthLocator()  # every month
        # days = mdates.DayLocator(interval=1)  # every day interval = 7 every week
        monthsFmt = mdates.DateFormatter('%b')
        ax = plt.gca()
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(monthsFmt)
        # ax.xaxis.set_minor_locator(days)
        ax.set_ylim([0, 60])
        ax.set_ylabel('SWE (mm w.e.)')
        plt.legend()
        plt.tight_layout()
    plt.savefig(plot_folder + '/av_SWE_ts_{}_{}_swe{}_{}_{}.png'.format(catchment, output_dem, model_swe_sc_threshold,which_model, run_id), dpi=300)

    plt.figure()
    plt.plot(np.mean(ann_ts_av_sca_m_array, axis=1), label='modis')
    plt.plot(np.mean(ann_ts_av_sca_array, axis=1), label='model')
    if which_model == 'all':
        plt.plot(np.mean(ann_ts_av_sca2_array, axis=1), label='dsc_snow')
    plt.title('average snow covered area HY {}-{}'.format(years_to_take[0], years_to_take[-1]))
    plt.xticks(range(len(years_to_take)), years_to_take, rotation=45)
    plt.ylabel('SCA')
    plt.legend()
    plt.savefig(plot_folder + '/average snow covered area HY {}-{} swe{} {} {}.png'.format(years_to_take[0], years_to_take[-1], model_swe_sc_threshold,which_model, run_id))
