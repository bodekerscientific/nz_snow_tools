# -*- coding: utf-8 -*-

"""
code to plot with the subplots the different models for each year and each station
"""

import numpy as np
from nz_snow_tools.snow.clark2009_snow_model import snow_main_simple
from nz_snow_tools.util.utils import make_regular_timeseries, convert_datetime_julian_day, convert_dt_to_hourdec, nash_sut, mean_bias, rmsd, mean_absolute_error,convert_date_hydro_DOY
import matplotlib.pylab as plt
import datetime as dt
import matplotlib.dates as mdates
import netCDF4 as nc
import pickle


def create_average(Stname, var='SWE'):
    data = dict_sin_snow[Stname][var][:]

    # remove nans
    inp_datobs = data[~np.isnan(data)]
    inp_dtobs = dict_sin_snow['dt_UTC+12'][~np.isnan(data)]

    if var == 'SWE':
        inp_datobs = inp_datobs * 1000
    # remove periods of poor data

    if Stname == 'Castle Mount' and var == 'Snow Depth':  # noisy snow depth data
        ind = inp_dtobs < dt.datetime(2018, 12, 1)

    elif Stname == 'Mueller' and var == 'SWE':  # clearly wrong as snow depth >> 1 m over this period
        ind = ~np.logical_and(inp_dtobs > dt.datetime(2012, 10, 20), inp_dtobs < dt.datetime(2012, 12, 4))

    elif Stname == 'Philistine' and var == 'SWE':  # clearly wrong as snow depth < 1 m during this period
        ind = ~np.logical_and(inp_dtobs > dt.datetime(2014, 6, 1), inp_dtobs < dt.datetime(2014, 10, 1))

    else:
        ind = np.ones(inp_datobs.shape, dtype=np.bool)
    dohy = convert_date_hydro_DOY(inp_dtobs)
    # jd = np.asarray(convert_datetime_julian_day(inp_dtobs))
    dat_jd = np.full([365], np.nan)
    for i in range(365):
        if i < 5:
            dat_jd[i] = np.median(inp_datobs[np.logical_or(dohy > i - 5 + 365, dohy <= i + 5)])
        elif i > 360:
            dat_jd[i] = np.median(inp_datobs[np.logical_or(dohy > i - 5, dohy <= i + 5 - 365)])
        else:
            dat_jd[i] = np.median(inp_datobs[np.logical_and(dohy > i - 5, dohy <= i + 5)])
    dt_to_plot = [dt.datetime(2000, 4, 1,0,0) + dt.timedelta(days=j) for j in range(365)]

    return dt_to_plot, dat_jd


def plot_year(j, Stname, plot_overlay=True, var='SWE', color='grey'):
    """
    options for var = Snow Depth and SWE
    """

    year = 2008 + j

    # load csv file
    # inp_datobs = np.genfromtxt(csv_file, delimiter=',', usecols=(1), skip_header=4) * 1000
    # inp_timeobs = np.genfromtxt(csv_file, usecols=(0), dtype=(str), delimiter=',', skip_header=4)
    # inp_dtobs = np.asarray([dt.datetime.strptime(t, '%d/%m/%Y %H:%M') for t in inp_timeobs])

    data = dict_sin_snow[Stname][var][:]
    inp_datobs = data[~np.isnan(data)]
    inp_dtobs = dict_sin_snow['dt_UTC+12'][~np.isnan(data)]
    ind = np.logical_and(inp_dtobs >= dt.datetime(year, 4, 1), inp_dtobs <= dt.datetime(year + 1, 4, 1))

    if Stname == 'Castle Mount' and var == 'Snow Depth':  # noisy snow depth data
        ind = np.logical_and(inp_dtobs < dt.datetime(2018, 12, 1), ind)

    if Stname == 'Mueller' and var == 'SWE':  # clearly wrong as snow depth >> 1 m over this period
        ind = np.logical_and(ind, ~np.logical_and(inp_dtobs > dt.datetime(2012, 10, 20), inp_dtobs < dt.datetime(2012, 12, 4)))

    if Stname == 'Philistine' and var == 'SWE':  # clearly wrong as snow depth < 1 m during this period
        ind = np.logical_and(ind, ~np.logical_and(inp_dtobs > dt.datetime(2014, 6, 1), inp_dtobs < dt.datetime(2014, 10, 1)))

    datobs_year = inp_datobs[ind]
    dtobs_year = inp_dtobs[ind]
    if var == 'SWE':
        datobs_year *= 1000

    if plot_overlay == True and len(dtobs_year) > 0:
        dtobs_year2 = dtobs_year - (dt.datetime(dtobs_year[0].year, 1, 1) - dt.datetime(2000, 1, 1))

    # plot data in new frame
    # plt.sca(axs[j])  # set the plot to the correct subplot
    # plt.title('Year : April {} - March {}'.format(year, year+1))
    # plot

    try:
        if plot_overlay == True:
            plt.plot(dtobs_year2, datobs_year, 'o', markersize=2, color=color),  # alpha=0.5
        else:
            plt.plot(dtobs_year, datobs_year, 'o', markersize=2, color=color)  # , alpha=0.5
    except:
        print('No obs for {}'.format(year))

    plt.gcf().autofmt_xdate()
    ax = plt.gca()
    if plot_overlay == True:
        months = mdates.MonthLocator()  # every month
        monthsFmt = mdates.DateFormatter('%b')

        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(monthsFmt)
        ax.set_xlabel('Month')
    else:
        ax.set_xlabel('Date')

    if var == 'SWE':
        ax.set_ylabel('{} (mm)'.format(var))
    else:
        ax.set_ylabel('{} (m)'.format(var))
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6)

    # ax.set_ylim(bottom=0)
    # plt.ylim(bottom=0)
    # plt.xlim
    plt.tight_layout()  # makes the axis fill the available space


dpi = 300
infile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/DSC Hydro/sin_snow_data/sin data June2019.pkl'
plot_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/DSC Hydro/all_years_plots'

sites = ['Castle Mount', 'Larkins', 'Mahanga', 'Mueller', 'Murchison', 'Philistine']
years_print = ['2012-18', '2013-18', '2009-18', '2010-18', '2012-   18', '2010-18']
# maxswe = np.asarray([1000, 1000, 1000, 2000, 600, 1500])/1000.

# var = 'SWE'#'Proportion' 'Snow Depth' 'SWE"
dict_sin_snow = pickle.load(open(infile, 'rb'))

font = {'family': 'normal',
        'weight': 'normal',
        'size': 14}

plt.rc('font', **font)

cmap = plt.cm.get_cmap('Dark2',6)
plt.figure(figsize=[6, 4])
for i, Stname in enumerate(sites):
    dt_to_plot, dat_jd = create_average(Stname, var='SWE')
    plt.plot(dt_to_plot, dat_jd,label=Stname)
plt.xlim([dt.datetime(2000, 4, 1), dt.datetime(2001, 4, 1)])
plt.ylim(bottom=0)
plt.legend()
plt.ylabel("SWE (mm)")
plt.gcf().autofmt_xdate()
ax = plt.gca()
months = mdates.MonthLocator()  # every month
monthsFmt = mdates.DateFormatter('%b')
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
ax.set_xlabel('Month')
plt.tight_layout()
plt.savefig(plot_folder + "/SWE_allsites_median.png", dpi=dpi)
plt.close()

plt.figure(figsize=[6, 4])
for i, Stname in enumerate(sites):
    dt_to_plot, dat_jd = create_average(Stname, var='Snow Depth')
    plt.plot(dt_to_plot, dat_jd,label=Stname)
plt.xlim([dt.datetime(2000, 4, 1), dt.datetime(2001, 4, 1)])
plt.ylim(bottom=0)
plt.legend()
plt.ylabel("Snow Depth (m)")
plt.gcf().autofmt_xdate()
ax = plt.gca()
months = mdates.MonthLocator()  # every month
monthsFmt = mdates.DateFormatter('%b')
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
ax.set_xlabel('Month')
plt.tight_layout()
plt.savefig(plot_folder + "/Snow Depth_allsites_median.png", dpi=dpi)
plt.close()

# for Stname, ymax in zip(sites, maxswe):
for i, Stname in enumerate(sites):

    plt.figure(figsize=[6, 4])
    var = 'SWE'
    for j in range(11):
        plot_year(j, Stname, var=var)
    dt_to_plot, dat_jd = create_average(Stname, var=var)
    plt.plot(dt_to_plot, dat_jd, color='blue')
    ax = plt.gca()
    ax.lines[0].set_label('SWE ' + years_print[i])
    ax.lines[-1].set_label('median')
    plt.xlim([dt.datetime(2000, 4, 1), dt.datetime(2001, 4, 1)])
    plt.ylim(bottom=0)
    # plt.title('{} @ {}'.format(var, Stname))
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_folder + "/{}_{}_allyears_overlay.png".format(var, Stname), dpi=dpi)  # save the figure once it’s done.
    plt.close()

    plt.figure(figsize=[12, 4])
    for j in range(12):
        plot_year(j, Stname, plot_overlay=False, var=var)
    plt.ylim(bottom=0)
    plt.title('{} @ {}'.format(var, Stname))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(plot_folder + "/{}_{}_allyears_ts.png".format(var, Stname), dpi=dpi)
    plt.close()

    plt.figure(figsize=[6, 4])
    var = 'Snow Depth'
    for j in range(11):
        plot_year(j, Stname, var=var)
    dt_to_plot, dat_jd = create_average(Stname, var=var)
    plt.plot(dt_to_plot, dat_jd, color='orange')
    ax = plt.gca()
    ax.lines[0].set_label('Snow depth ' + years_print[i])
    ax.lines[-1].set_label('median')
    plt.xlim([dt.datetime(2000, 4, 1), dt.datetime(2001, 4, 1)])
    plt.ylim(bottom=0)
    # plt.title('{} @ {}'.format(var, Stname))

    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_folder + "/{}_{}_allyears_overlay.png".format(var, Stname), dpi=dpi)  # save the figure once it’s done.
    plt.close()

    plt.figure(figsize=[12, 4])
    for j in range(12):
        plot_year(j, Stname, plot_overlay=False, var=var)
    plt.ylim(bottom=0)
    plt.title('{} @ {}'.format(var, Stname))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(plot_folder + "/{}_{}_allyears_ts.png".format(var, Stname), dpi=dpi)
    plt.close()

    var = 'Proportion'
    plt.figure(figsize=[6, 4])
    for j in range(11):
        plot_year(j, Stname, var=var)
    # plt.ylim(bottom=0)
    plt.xlim([dt.datetime(2000, 4, 1), dt.datetime(2001, 4, 1)])
    plt.title('{} @ {}'.format(var, Stname))
    plt.tight_layout()
    plt.savefig(plot_folder + "/{}_{}_allyears_overlay.png".format(var, Stname), dpi=dpi)  # save the figure once it’s done.
    plt.close()

    plt.figure(figsize=[12, 4])
    for j in range(12):
        plot_year(j, Stname, plot_overlay=False, var=var)
    plt.ylim(bottom=0)
    plt.title('{} @ {}'.format(var, Stname))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(plot_folder + "/{}_{}_allyears_ts.png".format(var, Stname), dpi=dpi)
    plt.close()

    # plot SWE and snow depth together
    plt.figure(figsize=[12, 4])
    for j in range(12):
        plot_year(j, Stname, plot_overlay=False, var='Snow Depth', color='red')
        plot_year(j, Stname, plot_overlay=False, var="SWE", color='grey')
    plt.ylim(bottom=0)
    plt.legend(['Snow Depth', 'SWE'])
    plt.title('{}'.format(Stname))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(plot_folder + "/SD_SWE_{}_allyears_ts.png".format(Stname), dpi=dpi)

    plt.close()
