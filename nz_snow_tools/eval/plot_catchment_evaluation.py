"""
code to plot results of catchment_evalutation - which evaluates snow models at catchment scale (i.e. Nevis or Clutha river)

Jono Conway
"""
from __future__ import division

import numpy as np
import pickle
import matplotlib.dates as mdates
import matplotlib.pylab as plt

from nz_snow_tools.util.utils import convert_date_hydro_DOY, convert_hydro_DOY_to_date


if __name__ == '__main__':

    which_model = 'clark2009'  # string identifying the model to be run. options include 'clark2009', 'dsc_snow', or 'all' # future will include 'fsm'
    clark2009run = True  # boolean specifying if the run already exists
    dsc_snow_opt = 'python'  # string identifying which version of the dsc snow model to use output from 'python' or 'fortran'
    catchment = 'Nevis'
    output_dem = 'nztm250m'  # identifier for output dem
    hydro_years_to_take = range(2001, 2016 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
    modis_sc_threshold = 50  # value of fsca (in percent) that is counted as being snow covered
    model_output_folder = 'P:/Projects/DSC-Snow/nz_snow_runs/test'
    plot_folder = 'P:/Projects/DSC-Snow/nz_snow_runs/test'

    ann = pickle.load(open(model_output_folder + '/summary_{}_{}.pkl'.format(catchment, output_dem), 'rb'))

    # [ann_ts_av_sca_m, ann_hydro_days_m, ann_dt_m, ann_scd_m, ann_ts_av_sca, ann_ts_av_swe, ann_hydro_days, ann_dt,
    #  ann_scd, ann_ts_av_sca2, ann_ts_av_swe2, ann_hydro_days2, ann_dt2, ann_scd2] = ann
    ann_ts_av_sca_m = ann[0]
    ann_ts_av_sca = ann[4]
    ann_ts_av_swe = ann[5]
    ann_hydro_days = ann[6]
    ann_ts_av_sca2 = ann[9]
    ann_ts_av_swe2 = ann[10]
    ann_hydro_days2 = ann[11]

    # put data into an array
    ann_ts_av_swe_array = np.zeros((len(ann_ts_av_swe), 365))
    ann_ts_av_swe2_array = np.zeros((len(ann_ts_av_swe), 365))
    ann_ts_av_sca_array = np.zeros((len(ann_ts_av_swe), 365))
    ann_ts_av_sca2_array = np.zeros((len(ann_ts_av_swe), 365))
    ann_ts_av_sca_m_array = np.zeros((len(ann_ts_av_swe), 365))

    for i in range(len(ann_ts_av_swe)):
        for j in range(365):
            ann_ts_av_swe_array[i,j] = ann_ts_av_swe[i][j]
            ann_ts_av_swe2_array[i, j] = ann_ts_av_swe2[i][j]
            ann_ts_av_sca_m_array[i, j] = ann_ts_av_sca_m[i][j]
            ann_ts_av_sca_array[i, j] = ann_ts_av_sca[i][j]
            ann_ts_av_sca2_array[i, j] = ann_ts_av_sca2[i][j]
    # compute average of annual series

    # plot timeseries
    plt.figure()
     # common x axis
    plt.subplot(2, 1, 1)
    for ts,hd in zip(ann_ts_av_swe,ann_hydro_days):
        dt_index = convert_hydro_DOY_to_date(hd, 2010)
        plt.plot(dt_index[:-1],ts[:-1], c=[0.4, 0.4, 0.4])
    plt.plot(dt_index[:365],np.mean(ann_ts_av_swe_array, axis=0),'r')
    ax = plt.gca()
    ax.set_ylim([0, 100])
    ax.set_ylabel('SWE (mm w.e.)')
    plt.subplot(2, 1, 2)
    for ts,hd in zip(ann_ts_av_swe2,ann_hydro_days2):
        dt_index = convert_hydro_DOY_to_date(hd, 2010)
        plt.plot(dt_index[:-1], ts[:-1], c=[0.4, 0.4, 0.4])
    plt.plot(dt_index[:365],np.mean(ann_ts_av_swe2_array, axis=0),'r')

    plt.gcf().autofmt_xdate()
    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    # days = mdates.DayLocator(interval=1)  # every day interval = 7 every week
    monthsFmt = mdates.DateFormatter('%b')
    ax = plt.gca()
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    # ax.xaxis.set_minor_locator(days)
    ax.set_ylim([0,100])
    ax.set_ylabel('SWE (mm w.e.)')
    plt.tight_layout()
    plt.savefig(plot_folder + '/av SWE ts.png',dpi=300)

    # snow covered area plot
    plt.figure()
    plt.subplot(3, 1, 1)
    for ts, hd in zip(ann_ts_av_sca_m, ann_hydro_days):
        dt_index = convert_hydro_DOY_to_date(hd, 2010)
        plt.plot(dt_index[:-1], ts, c=[0.4, 0.4, 0.4])
    plt.plot(dt_index[:365], np.mean(ann_ts_av_sca_m_array, axis=0), 'r')
    ax = plt.gca()
    ax.set_ylim([0, 1])
    ax.set_ylabel('SCA')
    # common x axis
    plt.subplot(3, 1, 2)
    for ts, hd in zip(ann_ts_av_sca, ann_hydro_days):
        dt_index = convert_hydro_DOY_to_date(hd, 2010)
        plt.plot(dt_index[:-1], ts[:-1], c=[0.4, 0.4, 0.4]) # remove last point as is 1st april in next year
    plt.plot(dt_index[:365], np.mean(ann_ts_av_sca_array, axis=0), 'r')
    ax = plt.gca()
    ax.set_ylim([0, 1])
    ax.set_ylabel('SCA')
    plt.subplot(3, 1, 3)
    for ts, hd in zip(ann_ts_av_sca2, ann_hydro_days2):
        dt_index = convert_hydro_DOY_to_date(hd, 2010)
        plt.plot(dt_index[:-1], ts[:-1], c=[0.4, 0.4, 0.4])
    plt.plot(dt_index[:365], np.mean(ann_ts_av_sca2_array, axis=0), 'r')

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
    plt.tight_layout()
    plt.savefig(plot_folder + '/av_SCA_ts_{}_{}.png'.format(catchment, output_dem), dpi=300)

