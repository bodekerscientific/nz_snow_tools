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
    catchment = 'Clutha'
    output_dem = 'nztm250m'  # identifier for output dem
    years_to_take = range(2000, 2016 + 1)  # range(2000, 2016 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
    modis_sc_thresholds = [50]  # value of fsca (in percent) that is counted as being snow covered 30,40,50,60,70,80
    model_swe_sc_threshold = 5  # threshold for treating a grid cell as snow covered
    model_output_folder = 'P:/Projects/DSC-Snow/runs/output/clutha_nztm250m_erebus'
    plot_folder = 'P:/Projects/DSC-Snow/runs/output/clutha_nztm250m_erebus'

    # run_id = 'jobst_ucc_4'  # string identifying fortran dsc_snow run. everything after the year
    run_ids = ['vcsn_4','jobst_ucc_4']  # ,'norton_4'

    for run_id in run_ids:
        plt.figure(figsize=[8, 6])
        plt.subplot(2, 1, 1)
        # load the first file to get model + basic modis info
        ann = pickle.load(open(
            model_output_folder + '/summary_{}_{}_thres{}_swe{}_{}_{}.pkl'.format(catchment, output_dem, modis_sc_thresholds[0], model_swe_sc_threshold,
                                                                                  which_model, run_id), 'rb'))
        # ann = [ann_ts_av_sca_m, ann_ts_av_sca_thres_m, ann_dt_m, ann_scd_m, ann_ts_av_sca, ann_ts_av_swe, ann_dt, ann_scd, configs] # from catchment_evaluation_annual

        ann_ts_av_sca_m = ann[0]
        ann_dt_m = ann[2]
        ann_ts_av_sca = ann[4]
        ann_ts_av_swe = ann[5]
        ann_dt = ann[6]

        if smooth_period > 1:
            for i in range(len(ann_ts_av_sca)):
                ann_ts_av_sca[i] = np.convolve(ann_ts_av_sca[i], np.ones((smooth_period,)) / smooth_period, mode='same')

        # put data into an array
        ann_ts_av_swe_array = np.zeros((len(ann_ts_av_swe), 365), dtype=np.float32)
        ann_ts_av_sca_array = np.zeros((len(ann_ts_av_swe), 365), dtype=np.float32)
        ann_ts_av_sca_m_array = np.zeros((len(ann_ts_av_swe), 365), dtype=np.float32)

            # ann_ts_av_sca_m_thres_array = np.zeros((len(ann_ts_av_swe), 365))
        # ann_ts_av_sca_m[5][:] = np.nan  # years 2006 and 2011 have bad modis data
        # ann_ts_av_sca_m[10][:] = np.nan
        # ann_ts_av_sca_m_thres[5][:] = np.nan # years 2006 and 2011 have bad modis data
        # ann_ts_av_sca_m_thres[10][:] = np.nan
        for i in range(len(ann_ts_av_swe)):
            for j in range(365):
                ann_ts_av_swe_array[i, j] = ann_ts_av_swe[i][j]
                if years_to_take[i]==2000: # hack for missing data at start of year 2000
                    if j>=53:
                        ann_ts_av_sca_m_array[i, j] = ann_ts_av_sca_m[i][j-53]
                else:
                    ann_ts_av_sca_m_array[i, j] = ann_ts_av_sca_m[i][j]

                # ann_ts_av_sca_m_thres_array[i, j] = ann_ts_av_sca_m_thres[i][j]
                ann_ts_av_sca_array[i, j] = ann_ts_av_sca[i][j]

        # snow covered area plot

        # plot average fsca for each year
        for i, ts in enumerate(ann_ts_av_sca_m):
            if ann_dt_m[i][0].year  == 2000:
                dt_index = [dt.datetime(2000, 1, 1) + dt.timedelta(days=d - 1) for d in range(53, len(ts) + 53)]  # need to plot all in the same year
            else:
                dt_index = [dt.datetime(2000, 1, 1) + dt.timedelta(days=d - 1) for d in range(1, len(ts) + 1)]  # need to plot all in the same year
            plt.plot(dt_index[:-1], ts[:-1], c=[0.4, 0.4, 0.4], linewidth=.5)

        # plot average for different modis fsca thresholds
        for modis_sc_threshold in modis_sc_thresholds:
            ann = pickle.load(
                open(model_output_folder + '/summary_{}_{}_thres{}_swe{}_{}_{}.pkl'.format(catchment, output_dem, modis_sc_thresholds[0], model_swe_sc_threshold,
                                                                                           which_model, run_id), 'rb'))
            ann_ts_av_sca_m_thres = ann[1]
            ann_ts_av_sca_m_thres_array = np.zeros((len(ann_ts_av_swe), 365))
            for i in range(len(ann_ts_av_swe)):
                for j in range(365):
                    if years_to_take[i] == 2000:  # hack for missing data at start of year 2000
                        if j >= 53:
                            ann_ts_av_sca_m_thres_array[i, j] = ann_ts_av_sca_m_thres[i][j - 53]
                    else:
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

        # common x axis
        plt.subplot(2, 1, 2)
        for ts in ann_ts_av_sca:
            dt_index = [dt.datetime(2000, 1, 1) + dt.timedelta(days=d - 1) for d in range(1, len(ts) + 1)]  # need to plot all in the same year
            plt.plot(dt_index[:-1], ts[:-1], c=[0.4, 0.4, 0.4], linewidth=.5)  # remove last point as is 1st april in next year
        label_text = 'average: {} {}'.format(which_model, run_id)
        plt.plot(dt_index[:365], np.mean(ann_ts_av_sca_array, axis=0), 'r', label=label_text)
        ax = plt.gca()
        ax.set_ylim([0, 1])
        ax.set_ylabel('SCA')
        months = mdates.MonthLocator()
        ax.xaxis.set_major_locator(months)
        monthsFmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_formatter(monthsFmt)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_folder + '/av_SCA_ts_{}_{}_swe{}_{}_{}.png'.format(catchment, output_dem, model_swe_sc_threshold, which_model, run_id), dpi=300)
        plt.close()

        # plot timeseries of SWE
        plt.figure()
        # common x axis
        for ts in ann_ts_av_swe:
            dt_index = [dt.datetime(2000, 1, 1) + dt.timedelta(days=d - 1) for d in range(1, len(ts) + 1)]  # need to plot all in the same year
            plt.plot(dt_index[:-1], ts[:-1], c=[0.4, 0.4, 0.4], linewidth=.5)
        label_text = 'average: {} {}'.format(which_model, run_id)
        plt.plot(dt_index[:365], np.mean(ann_ts_av_swe_array, axis=0), 'r', label=label_text)
        ax = plt.gca()
        ax.set_ylim([0, 60])
        ax.set_ylabel('SWE (mm w.e.)')
        months = mdates.MonthLocator()
        ax.xaxis.set_major_locator(months)
        monthsFmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_formatter(monthsFmt)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_folder + '/av_SWE_ts_{}_{}_swe{}_{}_{}.png'.format(catchment, output_dem, model_swe_sc_threshold, which_model, run_id), dpi=300)
        plt.close()

        plt.figure()
        plt.plot(np.mean(ann_ts_av_sca_m_array, axis=1), label='modis')
        plt.plot(np.mean(ann_ts_av_sca_array, axis=1), label='{} {}'.format(which_model,run_id))
        plt.title('average snow covered area {}-{}'.format(years_to_take[0], years_to_take[-1]))
        plt.xticks(range(len(years_to_take)), years_to_take, rotation=45)
        plt.ylabel('SCA')
        plt.ylim([0, 0.2])
        plt.legend(loc=1)
        plt.savefig(
            plot_folder + '/average snow covered area HY {}-{} swe{} {} {}.png'.format(years_to_take[0], years_to_take[-1], model_swe_sc_threshold, which_model,
                                                                                       run_id))
        plt.close()